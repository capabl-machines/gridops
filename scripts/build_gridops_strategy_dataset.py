#!/usr/bin/env python3
"""Build strict strategy-json traces for GridOps v7."""

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
from gridops.server.environment import GridOpsEnvironment
from gridops.strategy import (
    derive_strategy,
    messages_for_strategy_observation,
    plan_strategy_action,
    strategy_to_json,
    validate_strategy_completion,
)
from gridops.tool_agent import derive_control_context, previous_outcome_from_observation


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def difficulty_for(task_id: str, strategy: dict[str, str]) -> str:
    if task_id == "task_3_crisis" or strategy["risk_level"] in {"high", "critical"}:
        return "hard"
    if task_id == "task_2_heatwave" or strategy["mode"] in {"peak_shaving", "recovery"}:
        return "medium"
    return "easy"


def build_rows(
    *,
    tasks: list[str],
    seeds: list[int],
    stride: int,
    max_rows: int | None,
    shuffle: bool,
    rng_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    rng = random.Random(rng_seed)

    for task_id in tasks:
        for seed in seeds:
            env = GridOpsEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            previous_outcome = previous_outcome_from_observation(None)
            done = False
            while not done:
                obs_dict = obs.model_dump()
                hour = int(float(obs_dict["hour"]))
                strategy = derive_strategy(obs_dict, task_id, previous_outcome)
                plan = plan_strategy_action(
                    env,
                    task_id,
                    obs_dict,
                    previous_outcome=previous_outcome,
                    strategy=strategy,
                )
                action = plan["action"]
                derived_context = derive_control_context(obs_dict, task_id)

                if hour % stride == 0:
                    messages = messages_for_strategy_observation(obs_dict, derived_context, previous_outcome)
                    completion = strategy_to_json(strategy)
                    valid, reason = validate_strategy_completion(completion)
                    if not valid:
                        failures.append({"task_id": task_id, "seed": seed, "hour": hour, "reason": reason})
                    difficulty = difficulty_for(task_id, plan["strategy"])
                    row_id = f"gridops-strategy-v7-{task_id}-{seed:05d}-h{hour:03d}"
                    rows.append(
                        {
                            "id": row_id,
                            "task_id": task_id,
                            "difficulty": difficulty,
                            "seed": seed,
                            "hour": hour,
                            "prompt": messages[-1]["content"],
                            "messages": messages,
                            "completion": completion,
                            "raw": {
                                "source": "gridops_strategy_v7",
                                "prompt_mode": "strategy_json",
                                "task_id": task_id,
                                "seed": seed,
                                "hour": hour,
                                "observation": obs_dict,
                                "derived_context": derived_context,
                                "previous_outcome": previous_outcome,
                                "strategy": plan["strategy"],
                                "strategy_source": plan["strategy_source"],
                                "optimizer_config": plan["optimizer_config"],
                                "lp_action": action,
                                "difficulty": difficulty,
                                "validation_status": "ok",
                            },
                        }
                    )
                    counts.update(
                        [
                            f"task:{task_id}",
                            f"difficulty:{difficulty}",
                            f"mode:{plan['strategy']['mode']}",
                            f"risk:{plan['strategy']['risk_level']}",
                            f"battery:{plan['strategy']['battery_bias']}",
                            f"diesel:{plan['strategy']['diesel_policy']}",
                        ]
                    )

                obs = env.step(GridOpsAction(**action))
                if hour % stride == 0 and rows:
                    rows[-1]["raw"]["outcome"] = {
                        "blackout_kwh": round(float(obs.blackout_this_step), 4),
                        "cost": round(float(obs.cost_this_step), 4),
                        "grid_kw": round(float(obs.grid_kw_this_step), 4),
                        "battery_soc": round(float(obs.battery_soc), 4),
                        "diesel_used_kwh": round(float(obs.flow_diesel), 4),
                    }
                previous_outcome = previous_outcome_from_observation(obs.model_dump())
                done = obs.done

    if shuffle:
        rng.shuffle(rows)
    if max_rows is not None:
        rows = rows[:max_rows]

    return rows, {
        "rows": len(rows),
        "tasks": tasks,
        "seeds": seeds,
        "stride": stride,
        "counts": dict(counts),
        "validation_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="task_1_normal,task_2_heatwave,task_3_crisis")
    parser.add_argument("--seeds", default="7601,7602,7603,7604,7605,7606")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--output", default="sft_traces/gridops_strategy_v7.jsonl")
    parser.add_argument("--summary", default="evals/gridops_strategy_v7_summary.json")
    args = parser.parse_args()

    rows, summary = build_rows(
        tasks=parse_csv(args.tasks),
        seeds=[int(seed) for seed in parse_csv(args.seeds)],
        stride=args.stride,
        max_rows=args.max_rows,
        shuffle=not args.no_shuffle,
        rng_seed=args.rng_seed,
    )
    if summary["validation_failures"]:
        raise SystemExit(f"strategy validation failed: {summary['validation_failures'][:3]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
