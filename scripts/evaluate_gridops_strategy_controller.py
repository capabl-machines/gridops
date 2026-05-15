#!/usr/bin/env python3
"""Evaluate deterministic GridOps v7 strategy-controller harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.server.environment import GridOpsEnvironment
from gridops.strategy import plan_strategy_action
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import action_dict, optimize_action, previous_outcome_from_observation

LP_CEILING = {
    "average_score": 0.8233,
    "task_1_normal": 0.8372,
    "task_2_heatwave": 0.8416,
    "task_3_crisis": 0.7912,
}
V51_BASELINE = {
    "average_score": 0.7354,
    "task_1_normal": 0.7896,
    "task_2_heatwave": 0.7681,
    "task_3_crisis": 0.6484,
}
HYBRID_GUARD_BASELINE = {
    "average_score": 0.7946,
    "task_1_normal": 0.8182,
    "task_2_heatwave": 0.8226,
    "task_3_crisis": 0.7428,
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def rollout(task_id: str, seed: int, mode: str, horizon: int) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    previous_outcome = previous_outcome_from_observation(None)
    rows = []
    while not obs.done:
        obs_dict = obs.model_dump()
        if mode == "optimizer":
            action, info = optimize_action(obs_dict, task_id, previous_outcome=previous_outcome, horizon=horizon)
            row = {
                "hour": int(float(obs_dict["hour"])),
                "action": action_dict(action),
                "optimizer_info": info,
            }
        else:
            plan = plan_strategy_action(
                env,
                task_id,
                obs_dict,
                previous_outcome=previous_outcome,
                optimizer_horizon=horizon,
            )
            action = GridOpsAction(**plan["action"])
            row = {
                "hour": int(float(obs_dict["hour"])),
                "strategy": plan["strategy"],
                "strategy_source": plan["strategy_source"],
                "optimizer_config": plan["optimizer_config"],
                "action": action_dict(action),
                "optimizer_info": plan["optimizer_info"],
            }
        rows.append(row)
        obs = env.step(action)
        previous_outcome = previous_outcome_from_observation(obs.model_dump())

    return {
        "task_id": task_id,
        "seed": seed,
        "score": (env.state.grade or {}).get("score", 0.0),
        "valid_actions": len(rows),
        "total_actions": len(rows),
        "valid_action_rate": 1.0,
        "grade": env.state.grade or {},
        "samples": rows[:8],
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {}
    for task_id in [task for task in TASKS if any(row["task_id"] == task for row in rows)]:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        task_score = sum(row["score"] for row in task_rows) / max(len(task_rows), 1)
        by_task[task_id] = {
            "score": round(task_score, 4),
            "valid_action_rate": round(
                sum(row["valid_actions"] for row in task_rows) / max(sum(row["total_actions"] for row in task_rows), 1),
                4,
            ),
            "lp_ceiling_capture": round(task_score / LP_CEILING[task_id], 4),
            "blackout_kwh": round(
                sum((row["grade"] or {}).get("total_blackout_kwh", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
            "diesel_kwh": round(
                sum((row["grade"] or {}).get("total_diesel_kwh", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
            "cost": round(
                sum((row["grade"] or {}).get("actual_cost", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
        }
    average_score = sum(row["score"] for row in rows) / max(len(rows), 1)
    return {
        "name": name,
        "average_score": round(average_score, 4),
        "valid_action_rate": round(
            sum(row["valid_actions"] for row in rows) / max(sum(row["total_actions"] for row in rows), 1),
            4,
        ),
        "lp_ceiling_capture": round(average_score / LP_CEILING["average_score"], 4),
        "by_task": by_task,
        "rows": rows,
        "baselines": {
            "v51_model_only": V51_BASELINE,
            "hybrid_guard": HYBRID_GUARD_BASELINE,
            "full_episode_lp_ceiling": LP_CEILING,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["optimizer", "strategy"], default="strategy")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="7001,7002,7003")
    parser.add_argument("--optimizer-horizon", type=int, default=12)
    parser.add_argument("--output", default="evals/gridops_strategy_controller_eval.json")
    args = parser.parse_args()

    rows = []
    for task_id in parse_csv(args.tasks):
        for seed in [int(seed) for seed in parse_csv(args.seeds)]:
            row = rollout(task_id, seed, args.mode, args.optimizer_horizon)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "task_id": task_id,
                        "seed": seed,
                        "score": row["score"],
                        "valid_action_rate": row["valid_action_rate"],
                    }
                ),
                flush=True,
            )

    report = summarize(f"gridops_v7_{args.mode}", rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ["name", "average_score", "valid_action_rate", "lp_ceiling_capture", "by_task", "baselines"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
