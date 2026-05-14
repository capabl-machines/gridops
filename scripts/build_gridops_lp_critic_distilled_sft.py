#!/usr/bin/env python3
"""Build clean LP-critic-distilled GridOps SFT traces.

The trace target is clean operator reasoning plus final bounded action:

    <think>...</think>
    <action>{"battery_dispatch": ..., "diesel_dispatch": ..., "demand_shedding": ...}</action>

LP critic/tool details are stored only in ``raw`` metadata.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from gridops.critics.lp_critic import (
    build_clean_operator_completion,
    score_action_against_lp,
    validate_clean_reasoning_completion,
)
from gridops.models import GridOpsAction, GridOpsObservation
from gridops.prompting import messages_for_reason_action_observation
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import action_dict, derive_control_context, optimize_action, previous_outcome_from_observation


DEFAULT_POLICIES = (
    "do_nothing",
    "price_greedy",
    "noisy_lp",
    "under_diesel_crisis",
    "over_discharge",
    "invalid",
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _bounded_action(
    battery_dispatch: float = 0.0,
    diesel_dispatch: float = 0.0,
    demand_shedding: float = 0.0,
) -> GridOpsAction:
    return GridOpsAction(
        battery_dispatch=max(-1.0, min(1.0, battery_dispatch)),
        diesel_dispatch=max(0.0, min(1.0, diesel_dispatch)),
        demand_shedding=max(0.0, min(1.0, demand_shedding)),
    )


def candidate_for_policy(
    policy: str,
    observation: GridOpsObservation,
    task_id: str,
    previous_outcome: dict[str, Any] | None,
    *,
    rng: random.Random,
) -> GridOpsAction | dict[str, Any]:
    obs_dict = observation.model_dump()
    lp_action, _ = optimize_action(obs_dict, task_id, previous_outcome=previous_outcome, horizon=12)

    if policy == "do_nothing":
        return GridOpsAction()

    if policy == "price_greedy":
        if observation.grid_price <= 0.22 and observation.battery_soc < 0.85:
            return _bounded_action(battery_dispatch=-0.75)
        if observation.grid_price >= 0.34 and observation.battery_soc > 0.18:
            return _bounded_action(battery_dispatch=0.85)
        return GridOpsAction()

    if policy == "noisy_lp":
        return _bounded_action(
            battery_dispatch=lp_action.battery_dispatch + rng.uniform(-0.28, 0.28),
            diesel_dispatch=lp_action.diesel_dispatch + rng.uniform(-0.18, 0.18),
            demand_shedding=lp_action.demand_shedding + rng.uniform(-0.05, 0.05),
        )

    if policy == "under_diesel_crisis":
        outage_hours = set(TASKS[task_id].grid_outage_hours or [])
        hour = int(observation.hour)
        if task_id == "task_3_crisis" and (hour in outage_hours or hour + 1 in outage_hours):
            return _bounded_action(
                battery_dispatch=min(1.0, max(0.0, lp_action.battery_dispatch + 0.25)),
                diesel_dispatch=max(0.0, lp_action.diesel_dispatch - 0.65),
                demand_shedding=0.0,
            )
        return _bounded_action(
            battery_dispatch=lp_action.battery_dispatch,
            diesel_dispatch=max(0.0, lp_action.diesel_dispatch - 0.35),
            demand_shedding=0.0,
        )

    if policy == "over_discharge":
        return _bounded_action(
            battery_dispatch=1.0 if observation.battery_soc > 0.08 else 0.0,
            diesel_dispatch=0.0,
            demand_shedding=0.0,
        )

    if policy == "invalid":
        return {"battery_dispatch": 2.0, "diesel_dispatch": -0.25, "demand_shedding": 0.0}

    raise ValueError(f"unknown candidate policy: {policy}")


def difficulty_for(task_id: str, observation: GridOpsObservation, critic_reason: str) -> str:
    outage_hours = set(TASKS[task_id].grid_outage_hours or [])
    hour = int(observation.hour)
    if task_id == "task_3_crisis" or hour in outage_hours or hour + 1 in outage_hours:
        return "hard"
    if task_id == "task_2_heatwave" or critic_reason in {"candidate_higher_blackout", "candidate_higher_cost"}:
        return "medium"
    return "easy"


def build_rows(
    *,
    tasks: list[str],
    seeds: list[int],
    stride: int,
    candidate_policies: list[str],
    compare_horizon: int,
    optimizer_horizon: int,
    max_rows: int | None,
    shuffle: bool,
    rng_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(rng_seed)
    counters: Counter[str] = Counter()
    validation_failures: list[dict[str, Any]] = []

    for task_id in tasks:
        for seed in seeds:
            env = GridOpsEnvironment()
            observation = env.reset(task_id=task_id, seed=seed)
            previous_action = GridOpsAction()
            previous_outcome = previous_outcome_from_observation(None)

            done = False
            while not done:
                hour_int = int(observation.hour)
                should_record = hour_int % stride == 0
                obs_dict = observation.model_dump()
                lp_rollout_action, _ = optimize_action(
                    obs_dict,
                    task_id,
                    previous_outcome=previous_outcome,
                    horizon=optimizer_horizon,
                )

                if should_record:
                    for policy in candidate_policies:
                        candidate = candidate_for_policy(
                            policy,
                            observation,
                            task_id,
                            previous_outcome,
                            rng=rng,
                        )
                        critic = score_action_against_lp(
                            env,
                            task_id,
                            observation,
                            candidate,
                            previous_outcome=previous_outcome,
                            optimizer_horizon=optimizer_horizon,
                            compare_horizon=compare_horizon,
                        )
                        chosen_action = GridOpsAction.model_validate(critic["chosen_action"])
                        derived_context = derive_control_context(obs_dict, task_id)
                        messages = messages_for_reason_action_observation(
                            obs_dict,
                            derived_context=derived_context,
                            previous_action=action_dict(previous_action),
                            previous_outcome=previous_outcome,
                        )
                        completion = build_clean_operator_completion(
                            observation,
                            task_id,
                            chosen_action,
                            critic,
                            previous_action=previous_action,
                            previous_outcome=previous_outcome,
                        )
                        clean_ok, clean_reason = validate_clean_reasoning_completion(completion)
                        if not clean_ok:
                            validation_failures.append(
                                {
                                    "task_id": task_id,
                                    "seed": seed,
                                    "hour": hour_int,
                                    "policy": policy,
                                    "reason": clean_reason,
                                }
                            )
                            continue

                        difficulty = difficulty_for(task_id, observation, critic["reason"])
                        row_id = (
                            "gridops-lp-critic-v1-"
                            f"{task_id}-{seed:05d}-h{hour_int:03d}-{policy}"
                        )
                        rows.append(
                            {
                                "id": row_id,
                                "task_id": task_id,
                                "difficulty": difficulty,
                                "seed": seed,
                                "hour": hour_int,
                                "prompt": messages[-1]["content"],
                                "messages": messages,
                                "completion": completion,
                                "raw": {
                                    "source": "lp_critic_distilled_sft_v1",
                                    "prompt_mode": "reason_action",
                                    "task_id": task_id,
                                    "seed": seed,
                                    "hour": hour_int,
                                    "observation": observation.model_dump(),
                                    "candidate_policy": policy,
                                    "candidate_action": critic["candidate_action"],
                                    "lp_action": critic["lp_action"],
                                    "chosen_action": critic["chosen_action"],
                                    "lp_critic": critic,
                                    "derived_context": derived_context,
                                    "previous_action": action_dict(previous_action),
                                    "previous_outcome": previous_outcome,
                                    "difficulty": difficulty,
                                    "validation_status": "ok",
                                },
                            }
                        )
                        counters.update(
                            [
                                f"task:{task_id}",
                                f"policy:{policy}",
                                f"reason:{critic['reason']}",
                                f"difficulty:{difficulty}",
                                f"chosen_source:{critic['chosen_source']}",
                            ]
                        )

                step_result = env.step(lp_rollout_action)
                previous_action = lp_rollout_action
                previous_outcome = previous_outcome_from_observation(step_result.model_dump())
                observation = step_result
                done = step_result.done

    if shuffle:
        rng.shuffle(rows)
    if max_rows is not None:
        rows = rows[:max_rows]

    summary = {
        "rows": len(rows),
        "tasks": tasks,
        "seeds": seeds,
        "stride": stride,
        "candidate_policies": candidate_policies,
        "compare_horizon": compare_horizon,
        "optimizer_horizon": optimizer_horizon,
        "counts": dict(counters),
        "validation_failures": validation_failures,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="task_1_normal,task_2_heatwave,task_3_crisis")
    parser.add_argument("--seeds", default="7401,7402,7403,7404,7405,7406")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--candidate-policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--compare-horizon", type=int, default=4)
    parser.add_argument("--optimizer-horizon", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=61)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--output", default="sft_traces/gridops_lp_critic_distilled_sft_v1.jsonl")
    parser.add_argument("--summary", default="evals/gridops_lp_critic_distilled_sft_v1_summary.json")
    args = parser.parse_args()

    rows, summary = build_rows(
        tasks=parse_csv(args.tasks),
        seeds=[int(seed) for seed in parse_csv(args.seeds)],
        stride=args.stride,
        candidate_policies=parse_csv(args.candidate_policies),
        compare_horizon=args.compare_horizon,
        optimizer_horizon=args.optimizer_horizon,
        max_rows=args.max_rows,
        shuffle=not args.no_shuffle,
        rng_seed=args.rng_seed,
    )

    if summary["validation_failures"]:
        raise SystemExit(f"clean trace validation failed: {summary['validation_failures'][:3]}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
