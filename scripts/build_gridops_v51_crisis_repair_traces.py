"""Build the GridOps v5.1 crisis repair curriculum.

v5 improved average score and heatwave behavior, but crisis still under-captures
the LP ceiling. This builder creates a deliberately small continuation dataset:

- mostly task_3 crisis states around pre-outage, outage, and post-outage hours;
- labels from the v5 causal LP teacher, not the older heuristic oracle;
- a small normal/heatwave anchor set to reduce catastrophic forgetting.

The intent is a low-LR repair run initialized from the v5 adapter, not a new
from-scratch model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.server.environment import GridOpsEnvironment
from scripts.build_gridops_v4_reasoning_traces import (
    action_dict,
    derive_context,
    make_trace,
    validate_rows,
)
from scripts.build_gridops_v5_causal_teacher_traces import (
    causal_lp_teacher_action,
    previous_outcome_v5,
)


CRISIS_WINDOWS = {
    "pre_outage_soc": range(22, 30),
    "active_outage": range(30, 36),
    "post_outage_recovery": range(36, 43),
    "late_crisis_evening": list(range(44, 48)) + list(range(54, 60)),
}


def bucket_for_state(task_id: str, hour: int, action: GridOpsAction, previous_outcome: dict[str, Any]) -> str:
    if task_id != "task_3_crisis":
        return "stability_anchor"
    if float(previous_outcome.get("blackout_kwh", 0.0)) > 0.01:
        return "previous_blackout_repair"
    for bucket, hours in CRISIS_WINDOWS.items():
        if hour in hours:
            return bucket
    if float(action.diesel_dispatch) > 0.05:
        return "diesel_timing_repair"
    return "crisis_background"


def patch_metadata(
    row: dict[str, Any],
    *,
    action: GridOpsAction,
    teacher_info: dict[str, Any],
    bucket: str,
) -> dict[str, Any]:
    raw = row.setdefault("raw", {})
    raw["policy"] = "causal_lp_teacher_v51_crisis_repair"
    raw["teacher_action"] = action_dict(action)
    raw["teacher_info"] = teacher_info
    raw["bucket"] = bucket
    raw["source_labels"] = sorted(set(raw.get("source_labels", [])) | {"v51_crisis_repair", bucket})
    raw["focus_tags"] = sorted(set(raw.get("focus_tags", [])) | {"v51_crisis_repair", bucket})
    return row


def collect_crisis_rows(seed_start: int, seeds: int, horizon: int, max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []

    for seed in range(seed_start, seed_start + seeds):
        env = GridOpsEnvironment()
        task_id = "task_3_crisis"
        obs = env.reset(seed=seed, task_id=task_id)
        previous_action = action_dict(GridOpsAction())
        previous_outcome: dict[str, float] = {
            "blackout_kwh": 0.0,
            "battery_soc_delta": 0.0,
            "diesel_used_kwh": 0.0,
            "shed_kwh": 0.0,
            "grid_kw": 0.0,
            "cost": 0.0,
        }
        prior_obs: dict[str, Any] | None = None
        prior_action: GridOpsAction | None = None

        for hour in range(72):
            obs_dict = obs.model_dump()
            if prior_obs is not None and prior_action is not None:
                previous_outcome = previous_outcome_v5(obs_dict, prior_obs, prior_action)
                previous_action = action_dict(prior_action)

            action, teacher_info = causal_lp_teacher_action(
                obs_dict,
                task_id,
                previous_outcome=previous_outcome,
                horizon=horizon,
            )
            bucket = bucket_for_state(task_id, hour, action, previous_outcome)
            should_keep = bucket != "crisis_background" or float(action.diesel_dispatch) > 0.05
            if should_keep and len(rows) < max_rows:
                trace = make_trace(
                    trace_id=f"gridops_v51_{bucket}_{task_id}_seed{seed}_h{hour:02d}",
                    task_id=task_id,
                    seed=seed,
                    hour=hour,
                    obs=obs_dict,
                    action=action,
                    previous_action=previous_action,
                    previous_outcome=previous_outcome,
                    bucket=bucket,
                    source="v51_crisis_causal_teacher_rollout",
                    source_labels=["v51_crisis_repair", f"horizon_{horizon}", bucket],
                )
                rows.append(patch_metadata(trace, action=action, teacher_info=teacher_info, bucket=bucket))

            prior_obs = obs_dict
            prior_action = action
            obs = env.step(action)
            if obs.done:
                break

        grade = env.state.grade or {}
        rollout_rows.append(
            {
                "task_id": task_id,
                "seed": seed,
                "score": grade.get("score", 0.0),
                "reliability": grade.get("reliability", 0.0),
                "cost_efficiency": grade.get("cost_efficiency", 0.0),
                "green_score": grade.get("green_score", 0.0),
                "blackout_kwh": grade.get("total_blackout_kwh", 0.0),
                "diesel_kwh": grade.get("total_diesel_kwh", 0.0),
                "cost": grade.get("actual_cost", 0.0),
            }
        )
        if len(rows) >= max_rows:
            break

    return rows, rollout_rows


def collect_anchor_rows(seed_start: int, seeds_per_task: int, horizon: int, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in ["task_1_normal", "task_2_heatwave"]:
        for seed in range(seed_start, seed_start + seeds_per_task):
            env = GridOpsEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            previous_action = action_dict(GridOpsAction())
            previous_outcome: dict[str, float] = {
                "blackout_kwh": 0.0,
                "battery_soc_delta": 0.0,
                "diesel_used_kwh": 0.0,
                "shed_kwh": 0.0,
                "grid_kw": 0.0,
                "cost": 0.0,
            }
            prior_obs: dict[str, Any] | None = None
            prior_action: GridOpsAction | None = None

            for hour in [0, 8, 14, 18, 20, 36, 42, 60]:
                while int(obs.hour) < hour and not obs.done:
                    obs_dict = obs.model_dump()
                    action, _ = causal_lp_teacher_action(obs_dict, task_id, previous_outcome=previous_outcome, horizon=horizon)
                    prior_obs = obs_dict
                    prior_action = action
                    obs = env.step(action)
                    if prior_obs is not None and prior_action is not None:
                        previous_outcome = previous_outcome_v5(obs.model_dump(), prior_obs, prior_action)
                        previous_action = action_dict(prior_action)
                if obs.done:
                    break
                obs_dict = obs.model_dump()
                action, teacher_info = causal_lp_teacher_action(obs_dict, task_id, previous_outcome=previous_outcome, horizon=horizon)
                bucket = "stability_anchor"
                trace = make_trace(
                    trace_id=f"gridops_v51_{bucket}_{task_id}_seed{seed}_h{hour:02d}",
                    task_id=task_id,
                    seed=seed,
                    hour=hour,
                    obs=obs_dict,
                    action=action,
                    previous_action=previous_action,
                    previous_outcome=previous_outcome,
                    bucket=bucket,
                    source="v51_anchor_causal_teacher_rollout",
                    source_labels=["v51_stability_anchor", f"horizon_{horizon}"],
                )
                rows.append(patch_metadata(trace, action=action, teacher_info=teacher_info, bucket=bucket))
                if len(rows) >= max_rows:
                    return rows
    return rows[:max_rows]


def maybe_sample_existing_base(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), limit))]


def summarize(rows: list[dict[str, Any]], crisis_rollouts: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter((row.get("raw") or {}).get("bucket", "unknown") for row in rows)
    action_counts = {
        "diesel_positive": sum(1 for row in rows if float(row["action"]["diesel_dispatch"]) > 0.05),
        "battery_charge": sum(1 for row in rows if float(row["action"]["battery_dispatch"]) < -0.05),
        "battery_discharge": sum(1 for row in rows if float(row["action"]["battery_dispatch"]) > 0.05),
        "shedding_positive": sum(1 for row in rows if float(row["action"]["demand_shedding"]) > 0.05),
    }
    rollout_summary: dict[str, float] = {}
    if crisis_rollouts:
        for key in ["score", "reliability", "cost_efficiency", "green_score", "blackout_kwh", "diesel_kwh", "cost"]:
            rollout_summary[key] = round(sum(float(row[key]) for row in crisis_rollouts) / len(crisis_rollouts), 4)
    return {
        "rows": len(rows),
        "task_counts": dict(Counter(row["task_id"] for row in rows)),
        "bucket_counts": dict(bucket_counts),
        "action_counts": action_counts,
        "crisis_teacher_rollout": rollout_summary,
        "validation_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sft_traces/gridops_curriculum_v51_crisis_repair.jsonl")
    parser.add_argument("--summary-output", default="evals/gridops_curriculum_v51_crisis_repair_summary.json")
    parser.add_argument("--base-trace", default="sft_traces/gridops_curriculum_v4_kimi_reason_action_500.jsonl")
    parser.add_argument("--base-sample-limit", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=21000)
    parser.add_argument("--crisis-seeds", type=int, default=18)
    parser.add_argument("--anchor-seeds", type=int, default=4)
    parser.add_argument("--max-crisis-rows", type=int, default=950)
    parser.add_argument("--max-anchor-rows", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--sample-seed", type=int, default=20260512)
    args = parser.parse_args()

    crisis_rows, crisis_rollouts = collect_crisis_rows(args.seed_start, args.crisis_seeds, args.horizon, args.max_crisis_rows)
    anchor_rows = collect_anchor_rows(args.seed_start + 3000, args.anchor_seeds, args.horizon, args.max_anchor_rows)
    base_rows = maybe_sample_existing_base(Path(args.base_trace), args.base_sample_limit, args.sample_seed)
    rows = base_rows + anchor_rows + crisis_rows
    failures = validate_rows(rows)
    summary = summarize(rows, crisis_rollouts, failures)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
