"""Build GridOps v5 causal-teacher reasoning traces.

v5 uses the v4 reason-then-action prompt format, but replaces hand-written
repair labels with actions from a causal receding-horizon LP teacher. The
teacher sees only the current observation, short forecasts, task outage rules,
and previous feedback. It does not use the full 72-hour LP oracle.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.policies import oracle_policy
from gridops.simulation.physics import (
    BATTERY_CAPACITY_KWH,
    BATTERY_CHARGE_EFF,
    BATTERY_DEGRADATION_RS,
    BATTERY_DISCHARGE_EFF,
    BATTERY_MAX_POWER_KW,
    DEMAND_SHED_MAX_FRAC,
    DIESEL_COST_PER_KWH,
    DIESEL_MAX_KW,
    DIESEL_TANK_KWH,
    GRID_MAX_KW,
    VOLL,
)
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS
from scripts.build_gridops_v4_reasoning_traces import (
    action_dict,
    classify_bucket,
    derive_context,
    make_trace,
    previous_outcome_from_obs,
    validate_rows,
)


LP_VARS = ["imp", "exp", "ch", "dis", "diesel", "shed", "blackout", "curtail"]


def _idx(kind: str, h: int, horizon: int) -> int:
    if kind == "soc":
        return len(LP_VARS) * horizon + h
    if kind == "rebound":
        return len(LP_VARS) * horizon + (horizon + 1) + h
    return LP_VARS.index(kind) * horizon + h


def _soc_deficit_idx(horizon: int) -> int:
    return len(LP_VARS) * horizon + 2 * (horizon + 1)


def _fuel_deficit_idx(horizon: int) -> int:
    return _soc_deficit_idx(horizon) + 1


def _series_from_observation(obs: dict[str, Any], horizon: int) -> tuple[list[float], list[float], list[float]]:
    demand = [float(obs["demand_kw"])] + [float(x) for x in obs.get("demand_forecast_4h", [])]
    solar = [float(obs["solar_kw"])] + [float(x) for x in obs.get("solar_forecast_4h", [])]
    price = [float(obs["grid_price"])] + [float(x) for x in obs.get("price_forecast_4h", [])]
    while len(demand) < horizon:
        demand.append(demand[-1])
    while len(solar) < horizon:
        solar.append(solar[-1])
    while len(price) < horizon:
        price.append(price[-1])
    return demand[:horizon], solar[:horizon], price[:horizon]


def _soc_target(obs: dict[str, Any], task_id: str, horizon: int) -> float:
    hour = int(obs["hour"])
    terminal_hour = hour + horizon
    hour_of_day = (hour + 6) % 24
    if task_id == "task_3_crisis" and 24 <= terminal_hour <= 35:
        return 0.88
    if task_id == "task_3_crisis" and 30 <= hour <= 35:
        return 0.25
    if 10 <= hour_of_day < 18:
        return 0.78
    if 18 <= hour_of_day < 23:
        return 0.25
    return 0.42


def _fuel_target(obs: dict[str, Any], task_id: str, horizon: int) -> float:
    hour = int(obs["hour"])
    terminal_hour = hour + horizon
    if task_id == "task_3_crisis" and terminal_hour < 36:
        return 0.12 * DIESEL_TANK_KWH
    return 0.0


def _initial_rebound_kwh(previous_outcome: dict[str, Any] | None) -> float:
    if not previous_outcome:
        return 0.0
    return max(0.0, float(previous_outcome.get("shed_kwh", 0.0)))


def causal_lp_teacher_action(
    obs: dict[str, Any],
    task_id: str,
    previous_outcome: dict[str, Any] | None = None,
    horizon: int = 12,
    blackout_weight: float = 2.0,
    diesel_green_weight: float = 8.0,
    soc_deficit_weight: float = 18.0,
    fuel_deficit_weight: float = 8.0,
) -> tuple[GridOpsAction, dict[str, Any]]:
    """Return the first action from a short-horizon causal LP teacher."""
    horizon = max(1, min(int(horizon), 12))
    demand, solar, price = _series_from_observation(obs, horizon)
    hour = int(obs["hour"])
    outage_hours = set(TASKS[task_id].grid_outage_hours or [])
    initial_soc = float(obs["battery_soc"]) * BATTERY_CAPACITY_KWH
    initial_fuel = float(obs["diesel_fuel_remaining"]) * DIESEL_TANK_KWH
    initial_rebound = _initial_rebound_kwh(previous_outcome)
    n = len(LP_VARS) * horizon + 2 * (horizon + 1) + 2
    c = np.zeros(n)

    for h in range(horizon):
        c[_idx("imp", h, horizon)] = price[h]
        c[_idx("exp", h, horizon)] = -price[h]
        c[_idx("ch", h, horizon)] = BATTERY_DEGRADATION_RS
        c[_idx("dis", h, horizon)] = BATTERY_DEGRADATION_RS
        c[_idx("diesel", h, horizon)] = DIESEL_COST_PER_KWH + diesel_green_weight
        c[_idx("shed", h, horizon)] = 40.0
        c[_idx("blackout", h, horizon)] = VOLL * blackout_weight
        if h == 0 and not bool(obs.get("diesel_is_on", False)):
            c[_idx("diesel", h, horizon)] += 1.0

    c[_soc_deficit_idx(horizon)] = soc_deficit_weight
    c[_fuel_deficit_idx(horizon)] = fuel_deficit_weight

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n
    for h in range(horizon):
        grid_cap = 0.0 if (hour + h) in outage_hours else GRID_MAX_KW
        bounds[_idx("imp", h, horizon)] = (0.0, grid_cap)
        bounds[_idx("exp", h, horizon)] = (0.0, grid_cap)
        bounds[_idx("ch", h, horizon)] = (0.0, BATTERY_MAX_POWER_KW)
        bounds[_idx("dis", h, horizon)] = (0.0, BATTERY_MAX_POWER_KW)
        bounds[_idx("diesel", h, horizon)] = (0.0, DIESEL_MAX_KW)
        bounds[_idx("shed", h, horizon)] = (0.0, None)
        bounds[_idx("blackout", h, horizon)] = (0.0, None)
        bounds[_idx("curtail", h, horizon)] = (0.0, None)
    for h in range(horizon + 1):
        bounds[_idx("soc", h, horizon)] = (0.0, BATTERY_CAPACITY_KWH)
        bounds[_idx("rebound", h, horizon)] = (0.0, None)

    a_eq = []
    b_eq = []

    row = np.zeros(n)
    row[_idx("soc", 0, horizon)] = 1.0
    a_eq.append(row)
    b_eq.append(initial_soc)

    row = np.zeros(n)
    row[_idx("rebound", 0, horizon)] = 1.0
    a_eq.append(row)
    b_eq.append(initial_rebound)

    for h in range(horizon):
        row = np.zeros(n)
        row[_idx("imp", h, horizon)] = 1.0
        row[_idx("dis", h, horizon)] = BATTERY_DISCHARGE_EFF
        row[_idx("diesel", h, horizon)] = 1.0
        row[_idx("blackout", h, horizon)] = 1.0
        row[_idx("rebound", h, horizon)] = -1.0
        row[_idx("shed", h, horizon)] = 1.0
        row[_idx("ch", h, horizon)] = -1.0
        row[_idx("exp", h, horizon)] = -1.0
        row[_idx("curtail", h, horizon)] = -1.0
        a_eq.append(row)
        b_eq.append(float(demand[h] - solar[h]))

        row = np.zeros(n)
        row[_idx("soc", h + 1, horizon)] = 1.0
        row[_idx("soc", h, horizon)] = -1.0
        row[_idx("ch", h, horizon)] = -BATTERY_CHARGE_EFF
        row[_idx("dis", h, horizon)] = 1.0
        a_eq.append(row)
        b_eq.append(0.0)

        row = np.zeros(n)
        row[_idx("rebound", h + 1, horizon)] = 1.0
        row[_idx("shed", h, horizon)] = -1.0
        a_eq.append(row)
        b_eq.append(0.0)

    a_ub = []
    b_ub = []
    for h in range(horizon):
        row = np.zeros(n)
        row[_idx("shed", h, horizon)] = 1.0
        row[_idx("rebound", h, horizon)] = -DEMAND_SHED_MAX_FRAC
        a_ub.append(row)
        b_ub.append(DEMAND_SHED_MAX_FRAC * float(demand[h]))

    row = np.zeros(n)
    for h in range(horizon):
        row[_idx("diesel", h, horizon)] = 1.0
    a_ub.append(row)
    b_ub.append(initial_fuel)

    target_soc_kwh = _soc_target(obs, task_id, horizon) * BATTERY_CAPACITY_KWH
    row = np.zeros(n)
    row[_idx("soc", horizon, horizon)] = -1.0
    row[_soc_deficit_idx(horizon)] = -1.0
    a_ub.append(row)
    b_ub.append(-target_soc_kwh)

    target_fuel_kwh = _fuel_target(obs, task_id, horizon)
    row = np.zeros(n)
    for h in range(horizon):
        row[_idx("diesel", h, horizon)] = 1.0
    row[_fuel_deficit_idx(horizon)] = -1.0
    a_ub.append(row)
    b_ub.append(initial_fuel - target_fuel_kwh)

    result = linprog(
        c,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(a_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        fallback = oracle_policy(obs, task_id)
        return fallback, {
            "teacher": "causal_lp_v5",
            "status": f"lp_failed:{result.message}",
            "fallback": "oracle_policy",
            "horizon": horizon,
        }

    charge = float(result.x[_idx("ch", 0, horizon)])
    discharge = float(result.x[_idx("dis", 0, horizon)])
    diesel = float(result.x[_idx("diesel", 0, horizon)])
    shed = float(result.x[_idx("shed", 0, horizon)])
    actual_demand = max(float(demand[0]) + initial_rebound, 1.0)

    if discharge > charge and discharge > 1e-5:
        battery_dispatch = discharge / BATTERY_MAX_POWER_KW
    elif charge > 1e-5:
        battery_dispatch = -charge / BATTERY_MAX_POWER_KW
    else:
        battery_dispatch = 0.0

    action = GridOpsAction(
        battery_dispatch=float(np.clip(battery_dispatch, -1.0, 1.0)),
        diesel_dispatch=float(np.clip(diesel / DIESEL_MAX_KW, 0.0, 1.0)),
        demand_shedding=float(np.clip(shed / max(actual_demand * DEMAND_SHED_MAX_FRAC, 1.0), 0.0, 1.0)),
    )
    info = {
        "teacher": "causal_lp_v5",
        "status": "ok",
        "horizon": horizon,
        "objective": round(float(result.fun), 4),
        "soc_target": round(target_soc_kwh / BATTERY_CAPACITY_KWH, 4),
        "fuel_target_kwh": round(target_fuel_kwh, 2),
        "initial_rebound_kwh": round(initial_rebound, 4),
        "first_step": {
            "grid_import_kw": round(float(result.x[_idx("imp", 0, horizon)]), 4),
            "grid_export_kw": round(float(result.x[_idx("exp", 0, horizon)]), 4),
            "charge_kw": round(charge, 4),
            "discharge_kw": round(discharge, 4),
            "diesel_kw": round(diesel, 4),
            "shed_kwh": round(shed, 4),
            "blackout_kwh": round(float(result.x[_idx("blackout", 0, horizon)]), 4),
        },
    }
    return action, info


def previous_outcome_v5(
    current_obs: dict[str, Any],
    prior_obs: dict[str, Any] | None,
    previous_action: GridOpsAction | None,
) -> dict[str, float]:
    outcome = previous_outcome_from_obs(current_obs, prior_obs, previous_action)
    outcome["shed_kwh"] = round(float(current_obs.get("flow_shed", 0.0)), 4) if prior_obs is not None else 0.0
    outcome["grid_kw"] = round(float(current_obs.get("grid_kw_this_step", 0.0)), 4) if prior_obs is not None else 0.0
    return outcome


def patch_teacher_metadata(
    row: dict[str, Any],
    action: GridOpsAction,
    teacher_info: dict[str, Any],
    task_id: str,
    obs: dict[str, Any],
) -> dict[str, Any]:
    raw = row.setdefault("raw", {})
    raw["policy"] = "causal_lp_teacher_v5"
    raw["teacher_action"] = action_dict(action)
    raw["heuristic_oracle_action"] = action_dict(oracle_policy(obs, task_id))
    raw["teacher_info"] = teacher_info
    raw["source_labels"] = sorted(set(raw.get("source_labels", [])) | {"causal_lp_teacher_v5"})
    raw["focus_tags"] = sorted(set(raw.get("focus_tags", [])) | {"causal_lp_teacher_v5"})
    return row


def collect_teacher_rows(
    *,
    seed_start: int,
    seeds_per_task: int,
    stride: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []

    for task_index, task_id in enumerate(TASKS):
        for seed in range(seed_start + task_index * 1000, seed_start + task_index * 1000 + seeds_per_task):
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

            for step in range(72):
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
                if step % stride == 0:
                    derived = derive_context(obs_dict, task_id)
                    bucket = classify_bucket(task_id, obs_dict, action, derived, previous_outcome)
                    trace = make_trace(
                        trace_id=f"gridops_v5_causal_teacher_{bucket}_{task_id}_seed{seed}_h{step:02d}",
                        task_id=task_id,
                        seed=seed,
                        hour=step,
                        obs=obs_dict,
                        action=action,
                        previous_action=previous_action,
                        previous_outcome=previous_outcome,
                        bucket=bucket,
                        source="causal_lp_teacher_rollout",
                        source_labels=["causal_lp_teacher_v5", f"horizon_{horizon}"],
                    )
                    rows.append(patch_teacher_metadata(trace, action, teacher_info, task_id, obs_dict))

                prior_obs = obs_dict
                prior_action = action
                obs = env.step(action)
                if obs.done:
                    break

            grade = env.state.grade or {}
            rollouts.append(
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

    return rows, rollouts


def load_base_rows(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def summarize(
    rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_counts = Counter(row.get("task_id", "unknown") for row in rows)
    bucket_counts = Counter((row.get("raw") or {}).get("bucket", "unknown") for row in rows)
    source_counts = Counter((row.get("raw") or {}).get("source", "unknown") for row in rows)
    action_counts = {
        "diesel_positive": sum(1 for row in rows if float(row["action"]["diesel_dispatch"]) > 0.05),
        "battery_charge": sum(1 for row in rows if float(row["action"]["battery_dispatch"]) < -0.05),
        "battery_discharge": sum(1 for row in rows if float(row["action"]["battery_dispatch"]) > 0.05),
        "shedding_positive": sum(1 for row in rows if float(row["action"]["demand_shedding"]) > 0.05),
    }
    rollout_by_task: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        grouped[str(row["task_id"])].append(row)
    for task_id, task_rows in grouped.items():
        rollout_by_task[task_id] = {
            key: round(sum(float(row[key]) for row in task_rows) / max(len(task_rows), 1), 4)
            for key in ["score", "reliability", "cost_efficiency", "green_score"]
        }
        rollout_by_task[task_id]["blackout_kwh"] = round(
            sum(float(row["blackout_kwh"]) for row in task_rows) / max(len(task_rows), 1),
            2,
        )
        rollout_by_task[task_id]["diesel_kwh"] = round(
            sum(float(row["diesel_kwh"]) for row in task_rows) / max(len(task_rows), 1),
            2,
        )
        rollout_by_task[task_id]["cost"] = round(
            sum(float(row["cost"]) for row in task_rows) / max(len(task_rows), 1),
            2,
        )
    return {
        "rows": len(rows),
        "teacher_rows": len(teacher_rows),
        "base_rows": len(rows) - len(teacher_rows),
        "config": {
            "seed_start": args.seed_start,
            "seeds_per_task": args.seeds_per_task,
            "stride": args.stride,
            "horizon": args.horizon,
            "base_trace": args.base_trace,
            "base_sample_limit": args.base_sample_limit,
        },
        "task_counts": dict(task_counts),
        "bucket_counts": dict(bucket_counts),
        "source_counts": dict(source_counts),
        "action_counts": action_counts,
        "teacher_rollout_by_task": rollout_by_task,
        "teacher_rollouts": rollouts,
        "validation_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sft_traces/gridops_curriculum_v5_causal_teacher.jsonl")
    parser.add_argument("--summary-output", default="evals/gridops_curriculum_v5_causal_teacher_summary.json")
    parser.add_argument("--base-trace", default="sft_traces/gridops_curriculum_v4_kimi_reason_action_500.jsonl")
    parser.add_argument("--base-sample-limit", type=int, default=1800)
    parser.add_argument("--seed-start", type=int, default=16000)
    parser.add_argument("--seeds-per-task", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--sample-seed", type=int, default=20260511)
    args = parser.parse_args()

    teacher_rows, rollouts = collect_teacher_rows(
        seed_start=args.seed_start,
        seeds_per_task=args.seeds_per_task,
        stride=max(1, args.stride),
        horizon=args.horizon,
    )
    base_rows = load_base_rows(Path(args.base_trace), args.base_sample_limit, args.sample_seed)
    rows = base_rows + teacher_rows
    failures = validate_rows(rows)
    summary = summarize(rows, teacher_rows, rollouts, failures, args)

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
