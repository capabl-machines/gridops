"""OpenEnv-backed GRPO scaffold for GridOps.

This file intentionally starts as a guarded scaffold rather than a long-running
training script. It defines the task sampler and reward contract we want before
we spend GPU time on RL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.policies import oracle_policy
from gridops.prompting import (
    extract_action_json,
    format_reason_action_observation,
    validate_reason_action_completion,
)
from gridops.server.environment import GridOpsEnvironment
from scripts.build_gridops_v4_reasoning_traces import (
    derive_context,
    previous_outcome_from_obs,
)


TASK_SEEDS = {
    "task_1_normal": range(15000, 15080),
    "task_2_heatwave": range(16000, 16080),
    "task_3_crisis": range(17000, 17080),
}


@dataclass
class RewardBreakdown:
    total: float
    format_reward: float
    env_reward: float
    regret_reward: float
    blackout_penalty: float
    diesel_context_reward: float
    brevity_reward: float
    valid: bool
    reason: str
    action: dict[str, Any] | None


def replay_to_state(task_id: str, seed: int, hour: int) -> tuple[GridOpsEnvironment, dict[str, Any], dict[str, Any], dict[str, Any]]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    previous_action = GridOpsAction()
    previous_outcome = {
        "blackout_kwh": 0.0,
        "battery_soc_delta": 0.0,
        "diesel_used_kwh": 0.0,
        "cost": 0.0,
    }
    prior_obs: dict[str, Any] | None = None
    prior_action: GridOpsAction | None = None
    for _ in range(hour):
        obs_dict = obs.model_dump()
        if prior_obs is not None and prior_action is not None:
            previous_outcome = previous_outcome_from_obs(obs_dict, prior_obs, prior_action)
            previous_action = prior_action
        prior_obs = obs_dict
        prior_action = oracle_policy(obs_dict, task_id)
        obs = env.step(prior_action)
        if obs.done:
            break
    obs_dict = obs.model_dump()
    return env, obs_dict, previous_action.model_dump(), previous_outcome


def build_prompt(task_id: str, seed: int, hour: int) -> dict[str, Any]:
    _, obs, previous_action, previous_outcome = replay_to_state(task_id, seed, hour)
    derived = derive_context(obs, task_id)
    return {
        "task_id": task_id,
        "seed": seed,
        "hour": hour,
        "prompt": format_reason_action_observation(obs, derived, previous_action, previous_outcome),
        "observation": obs,
        "derived_context": derived,
        "previous_action": previous_action,
        "previous_outcome": previous_outcome,
    }


def sample_prompt_specs(limit: int) -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = []
    preferred_hours = {
        "task_1_normal": [10, 14, 17, 18, 19, 20, 38, 42, 62, 66],
        "task_2_heatwave": [13, 14, 17, 18, 19, 20, 37, 38, 39, 62, 63],
        "task_3_crisis": [25, 26, 29, 30, 31, 32, 33, 34, 35, 37, 38],
    }
    while len(specs) < limit:
        for task_id, seeds in TASK_SEEDS.items():
            for seed in seeds:
                for hour in preferred_hours[task_id]:
                    specs.append((task_id, int(seed), hour))
                    if len(specs) >= limit:
                        return specs
    return specs


def score_action_horizon(task_id: str, seed: int, hour: int, action: GridOpsAction, horizon: int) -> dict[str, float]:
    env, _, _, _ = replay_to_state(task_id, seed, hour)
    total_reward = 0.0
    blackout = 0.0
    diesel = 0.0
    cost = 0.0
    obs = env.step(action)
    for step_idx in range(max(1, horizon)):
        obs_dict = obs.model_dump()
        total_reward += float(obs.reward or 0.0)
        blackout += float(obs_dict.get("blackout_this_step", 0.0))
        diesel += float(obs_dict.get("flow_diesel", 0.0))
        cost += float(obs_dict.get("cost_this_step", 0.0))
        if obs.done or step_idx == horizon - 1:
            break
        obs = env.step(oracle_policy(obs_dict, task_id))
    return {
        "reward": total_reward,
        "blackout_kwh": blackout,
        "diesel_kwh": diesel,
        "cost": cost,
    }


def reward_completion(completion: str, prompt_row: dict[str, Any], horizon: int) -> RewardBreakdown:
    valid, reason = validate_reason_action_completion(completion)
    payload = extract_action_json(completion)
    if not valid or payload is None:
        return RewardBreakdown(
            total=-5.0,
            format_reward=-5.0,
            env_reward=0.0,
            regret_reward=0.0,
            blackout_penalty=0.0,
            diesel_context_reward=0.0,
            brevity_reward=0.0,
            valid=False,
            reason=reason,
            action=payload,
        )

    action = GridOpsAction(**payload)
    task_id = prompt_row["task_id"]
    seed = int(prompt_row["seed"])
    hour = int(prompt_row["hour"])
    candidate = score_action_horizon(task_id, seed, hour, action, horizon)
    oracle = score_action_horizon(task_id, seed, hour, oracle_policy(prompt_row["observation"], task_id), horizon)

    env_reward = 0.20 * float(candidate["reward"])
    regret_reward = max(-1.0, min(1.0, 0.20 * float(candidate["reward"] - oracle["reward"])))
    blackout_penalty = -0.03 * float(candidate["blackout_kwh"])
    derived = prompt_row.get("derived_context") or {}
    high_crisis_gap = task_id == "task_3_crisis" and (
        derived.get("grid_status") == "outage" or float(derived.get("max_future_supply_gap_kw", 0.0)) > 80.0
    )
    diesel = float(action.diesel_dispatch)
    if high_crisis_gap and 0.05 <= diesel <= 1.0:
        diesel_context_reward = 0.25
    elif task_id != "task_3_crisis" and diesel > 0.05:
        diesel_context_reward = -0.35 * diesel
    else:
        diesel_context_reward = 0.0
    brevity_reward = -0.001 * max(0, len(completion) - 900)
    total = 1.0 + env_reward + regret_reward + blackout_penalty + diesel_context_reward + brevity_reward
    return RewardBreakdown(
        total=float(total),
        format_reward=1.0,
        env_reward=env_reward,
        regret_reward=regret_reward,
        blackout_penalty=blackout_penalty,
        diesel_context_reward=diesel_context_reward,
        brevity_reward=brevity_reward,
        valid=True,
        reason="ok",
        action=payload,
    )


def smoke_reward_contract(output: Path, horizon: int, limit: int) -> None:
    rows = []
    for task_id, seed, hour in sample_prompt_specs(limit):
        prompt_row = build_prompt(task_id, seed, hour)
        oracle_action = oracle_policy(prompt_row["observation"], task_id)
        oracle_completion = (
            "<think>\n"
            "time_context: smoke oracle reference.\n"
            "1st_order: valid action is required.\n"
            "2nd_order: short-horizon environment reward decides quality.\n"
            "previous_action: use the provided feedback.\n"
            "decision: emit bounded JSON.\n"
            "</think>\n<action>\n"
            + json.dumps(oracle_action.model_dump(), separators=(",", ":"))
            + "\n</action>"
        )
        invalid_completion = "<think>bad</think>\n<action>\n{\"battery_dispatch\": 2.0, \"diesel_dispatch\": 0, \"demand_shedding\": 0}\n</action>"
        rows.append(
            {
                "task_id": task_id,
                "seed": seed,
                "hour": hour,
                "oracle_reward": reward_completion(oracle_completion, prompt_row, horizon).__dict__,
                "invalid_reward": reward_completion(invalid_completion, prompt_row, horizon).__dict__,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke_reward_contract", "train"], default="smoke_reward_contract")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output", default="evals/gridops_grpo_openenv_reward_contract_smoke.json")
    args = parser.parse_args()

    if args.mode == "smoke_reward_contract":
        smoke_reward_contract(Path(args.output), args.horizon, args.limit)
        return

    raise SystemExit(
        "GRPO training mode is intentionally gated. First run the reward-contract smoke, "
        "then wire TRL GRPOTrainer using reward_completion as the reward function."
    )


if __name__ == "__main__":
    main()
