"""Build tool-corrected GridOps SFT traces from deterministic rollouts.

The current model (or a lightweight candidate policy) proposes an action, the
runtime optimizer/validator/simulator guard selects the final action, and the
SFT label teaches the model to imitate that selected action with bounded
reasoning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.prompting import messages_for_reason_action_observation, validate_reason_action_completion
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import (
    PlanInputs,
    action_dict,
    derive_control_context,
    optimize_action,
    plan_action,
    previous_outcome_from_observation,
    tool_corrected_completion,
)


def _candidate_action(obs: dict[str, Any], task_id: str, policy: str, rng: random.Random) -> GridOpsAction:
    if policy == "do_nothing":
        return GridOpsAction()
    if policy == "invalid":
        return GridOpsAction()  # invalid candidates are injected as raw dicts by caller.
    if policy == "price_greedy":
        if float(obs.get("grid_price", 0.0)) <= 5.0 and float(obs.get("battery_soc", 0.0)) < 0.95:
            return GridOpsAction(battery_dispatch=-1.0)
        if float(obs.get("grid_price", 0.0)) >= 12.0 and float(obs.get("battery_soc", 0.0)) > 0.15:
            return GridOpsAction(battery_dispatch=1.0)
        return GridOpsAction()
    if policy == "noisy_optimizer":
        action, _ = optimize_action(obs, task_id, previous_outcome=previous_outcome_from_observation(obs), horizon=8)
        return GridOpsAction(
            battery_dispatch=max(-1.0, min(1.0, action.battery_dispatch + rng.uniform(-0.25, 0.25))),
            diesel_dispatch=max(0.0, min(1.0, action.diesel_dispatch + rng.uniform(-0.2, 0.2))),
            demand_shedding=max(0.0, min(1.0, action.demand_shedding + rng.uniform(-0.1, 0.1))),
        )
    raise ValueError(f"Unknown candidate policy: {policy}")


def _raw_candidate(obs: dict[str, Any], task_id: str, policy: str, rng: random.Random) -> dict[str, Any]:
    if policy == "invalid":
        return {"battery_dispatch": 2.0, "diesel_dispatch": -1.0, "demand_shedding": 0.0}
    return _candidate_action(obs, task_id, policy, rng).model_dump()


def _difficulty(task_id: str, obs: dict[str, Any], plan: dict[str, Any]) -> str:
    if task_id == "task_3_crisis":
        return "hard"
    if task_id == "task_2_heatwave" or plan.get("selected_source") == "optimizer":
        return "medium"
    return "easy"


def build_trace(
    *,
    episode_id: str,
    task_id: str,
    seed: int,
    obs: dict[str, Any],
    plan: dict[str, Any],
    previous_action: dict[str, Any] | None,
    previous_outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    derived_context = derive_control_context(obs, task_id)
    messages = messages_for_reason_action_observation(
        obs,
        derived_context,
        previous_action,
        previous_outcome,
    )
    completion = tool_corrected_completion(
        obs=obs,
        task_id=task_id,
        plan=plan,
        previous_action=previous_action,
        previous_outcome=previous_outcome,
    )
    valid, reason = validate_reason_action_completion(completion)
    trace_id = f"tool_corrected_{episode_id}_{int(float(obs['hour'])):03d}"
    return {
        "id": trace_id,
        "trace_id": trace_id,
        "task_id": task_id,
        "seed": seed,
        "hour": int(float(obs["hour"])),
        "difficulty": _difficulty(task_id, obs, plan),
        "messages": messages,
        "prompt": messages[-1]["content"],
        "completion": completion,
        "raw": {
            "source": "tool_corrected_sft",
            "prompt_mode": "reason_action",
            "episode_id": episode_id,
            "observation": obs,
            "derived_context": derived_context,
            "previous_action": previous_action,
            "previous_outcome": previous_outcome,
            "selected_action": plan["selected_action"],
            "selected_source": plan["selected_source"],
            "selection_reason": plan["selection_reason"],
            "model_candidate": plan["model_candidate"],
            "optimizer_candidate": plan["optimizer_candidate"],
            "comparison": plan["comparison"],
            "validation": {"valid": valid, "reason": reason},
        },
    }


def rollout(task_id: str, seed: int, stride: int, candidate_policy: str, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    rows: list[dict[str, Any]] = []
    previous_action: dict[str, Any] | None = None
    previous_outcome: dict[str, Any] | None = None
    episode_id = f"{task_id}_{seed}"
    counts: Counter[str] = Counter()

    while not obs.done:
        obs_dict = obs.model_dump()
        model_action = _raw_candidate(obs_dict, task_id, candidate_policy, rng)
        plan = plan_action(
            env,
            PlanInputs(
                task_id=task_id,
                observation=obs_dict,
                previous_action=previous_action,
                previous_outcome=previous_outcome,
                model_action=model_action,
            ),
        )
        selected_action = GridOpsAction(**plan["selected_action"])
        if int(float(obs_dict["hour"])) % stride == 0:
            row = build_trace(
                episode_id=episode_id,
                task_id=task_id,
                seed=seed,
                obs=obs_dict,
                plan=plan,
                previous_action=previous_action,
                previous_outcome=previous_outcome,
            )
            if row["raw"]["validation"]["valid"]:
                rows.append(row)
                counts[row["difficulty"]] += 1
                counts[f"selected_{plan['selected_source']}"] += 1
                counts[f"reason_{plan['selection_reason']}"] += 1
        prior_obs = obs_dict
        obs = env.step(selected_action)
        previous_action = action_dict(selected_action)
        previous_outcome = previous_outcome_from_observation(obs.model_dump())
        previous_outcome["battery_soc_delta"] = round(float(obs.battery_soc) - float(prior_obs["battery_soc"]), 4)

    return rows, {
        "task_id": task_id,
        "seed": seed,
        "rows": len(rows),
        "grade": env.state.grade or {},
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="7301,7302,7303")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--candidate-policy", choices=["do_nothing", "price_greedy", "noisy_optimizer", "invalid"], default="price_greedy")
    parser.add_argument("--output", default="sft_traces/gridops_tool_corrected_sft_v1.jsonl")
    parser.add_argument("--summary-output", default="evals/gridops_tool_corrected_sft_v1_summary.json")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    rng = random.Random(1234)
    task_ids = [x.strip() for x in args.tasks.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    episodes = []
    for task_id in task_ids:
        for seed in seeds:
            episode_rows, summary = rollout(task_id, seed, args.stride, args.candidate_policy, rng)
            rows.extend(episode_rows)
            episodes.append(summary)
            print(json.dumps({"task_id": task_id, "seed": seed, "rows": len(episode_rows), "score": summary["grade"].get("score")}), flush=True)

    if args.shuffle:
        rng.shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    aggregate = Counter()
    for row in rows:
        aggregate[row["difficulty"]] += 1
        aggregate[f"selected_{row['raw']['selected_source']}"] += 1
        aggregate[f"reason_{row['raw']['selection_reason']}"] += 1
    summary = {
        "rows": len(rows),
        "output": str(output),
        "candidate_policy": args.candidate_policy,
        "tasks": task_ids,
        "seeds": seeds,
        "stride": args.stride,
        "counts": dict(aggregate),
        "episodes": episodes,
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["rows", "output", "candidate_policy", "counts"]}, indent=2))


if __name__ == "__main__":
    main()
