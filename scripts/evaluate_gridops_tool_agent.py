"""Evaluate the GridOps optimizer and hybrid guarded tool-agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import (
    PlanInputs,
    action_dict,
    optimize_action,
    plan_action,
    previous_outcome_from_observation,
)


def _rollout_optimizer(task_id: str, seed: int, horizon: int) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    actions = []
    previous_outcome = previous_outcome_from_observation(obs.model_dump())
    for _ in range(72):
        obs_dict = obs.model_dump()
        action, info = optimize_action(obs_dict, task_id, previous_outcome=previous_outcome, horizon=horizon)
        actions.append({"hour": int(obs_dict["hour"]), "action": action_dict(action), "info": info})
        obs = env.step(action)
        previous_outcome = previous_outcome_from_observation(obs.model_dump())
        if obs.done:
            break
    return {
        "task_id": task_id,
        "seed": seed,
        "score": (env.state.grade or {}).get("score", 0.0),
        "valid_actions": len(actions),
        "total_actions": len(actions),
        "valid_action_rate": 1.0,
        "grade": env.state.grade or {},
        "samples": actions[:5],
    }


def _rollout_hybrid_guard(task_id: str, seed: int, horizon: int, compare_horizon: int) -> dict[str, Any]:
    """Guard a conservative model candidate with the optimizer.

    This mode validates the runtime guard path without requiring a GPU model.
    When a model candidate is externally supplied in production, the same
    `plan_action` selection policy is used.
    """
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    rows = []
    for _ in range(72):
        obs_dict = obs.model_dump()
        plan = plan_action(
            env,
            PlanInputs(
                task_id=task_id,
                observation=obs_dict,
                model_action=GridOpsAction().model_dump(),
                optimizer_horizon=horizon,
                compare_horizon=compare_horizon,
            ),
        )
        action = GridOpsAction(**plan["selected_action"])
        rows.append(
            {
                "hour": int(obs_dict["hour"]),
                "selected_source": plan["selected_source"],
                "selection_reason": plan["selection_reason"],
                "action": action_dict(action),
            }
        )
        obs = env.step(action)
        if obs.done:
            break
    return {
        "task_id": task_id,
        "seed": seed,
        "score": (env.state.grade or {}).get("score", 0.0),
        "valid_actions": len(rows),
        "total_actions": len(rows),
        "valid_action_rate": 1.0,
        "grade": env.state.grade or {},
        "samples": rows[:5],
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {}
    for task_id in [task for task in TASKS if any(row["task_id"] == task for row in rows)]:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = {
            "score": round(sum(row["score"] for row in task_rows) / max(len(task_rows), 1), 4),
            "valid_action_rate": round(
                sum(row["valid_actions"] for row in task_rows) / max(sum(row["total_actions"] for row in task_rows), 1),
                4,
            ),
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
    total_valid = sum(row["valid_actions"] for row in rows)
    total_actions = sum(row["total_actions"] for row in rows)
    return {
        "name": name,
        "average_score": round(sum(row["score"] for row in rows) / max(len(rows), 1), 4),
        "valid_action_rate": round(total_valid / max(total_actions, 1), 4),
        "by_task": by_task,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["optimizer", "hybrid_guard"], default="optimizer")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="7001,7002,7003")
    parser.add_argument("--optimizer-horizon", type=int, default=12)
    parser.add_argument("--compare-horizon", type=int, default=4)
    parser.add_argument("--output", default="evals/gridops_tool_agent_eval.json")
    args = parser.parse_args()

    rows = []
    for task_id in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        for seed in [int(x.strip()) for x in args.seeds.split(",") if x.strip()]:
            if args.mode == "optimizer":
                row = _rollout_optimizer(task_id, seed, args.optimizer_horizon)
            else:
                row = _rollout_hybrid_guard(task_id, seed, args.optimizer_horizon, args.compare_horizon)
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

    report = summarize(f"gridops_tool_agent_{args.mode}", rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["name", "average_score", "valid_action_rate", "by_task"]}, indent=2))


if __name__ == "__main__":
    main()
