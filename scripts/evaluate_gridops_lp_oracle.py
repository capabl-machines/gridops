"""Evaluate a linear-programming oracle for GridOps.

The GridOps physics is mostly linear: grid import/export, charge/discharge,
diesel, shedding, blackout, battery SOC, and rebound can be expressed as a
linear program over the full 72-hour episode.

This script solves that full-episode relaxed control problem, converts the LP
trajectory back into `GridOpsAction`s, and runs those actions through the real
OpenEnv environment/grader. It is a stronger ceiling test than a model or a
short-horizon MPC planner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.server.environment import GridOpsEnvironment
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
from gridops.tasks.definitions import TASKS
from gridops.tasks.graders import compute_dumb_baseline_cost


VARS = ["imp", "exp", "ch", "dis", "diesel", "shed", "blackout", "curtail"]


def idx(kind: str, h: int, horizon: int) -> int:
    if kind == "soc":
        return len(VARS) * horizon + h
    if kind == "rebound":
        return len(VARS) * horizon + (horizon + 1) + h
    return VARS.index(kind) * horizon + h


def solve_lp(task_id: str, seed: int) -> dict[str, Any]:
    env = GridOpsEnvironment()
    env.reset(seed=seed, task_id=task_id)
    demand = np.array(env._demand, dtype=float)  # noqa: SLF001 - ceiling experiment.
    solar = np.array(env._solar, dtype=float)  # noqa: SLF001
    price = np.array(env._price, dtype=float)  # noqa: SLF001
    cfg = TASKS[task_id]
    outages = set(cfg.grid_outage_hours or [])
    horizon = len(demand)
    n = len(VARS) * horizon + 2 * (horizon + 1)
    baseline = compute_dumb_baseline_cost(demand, solar, price, cfg.grid_outage_hours)
    raw_total_demand = max(float(demand.sum()), 1.0)

    c = np.zeros(n)
    for h in range(horizon):
        # Cost-efficiency part of the final grader.
        c[idx("imp", h, horizon)] = 0.50 * price[h] / baseline
        c[idx("exp", h, horizon)] = -0.50 * price[h] / baseline
        c[idx("ch", h, horizon)] = 0.50 * BATTERY_DEGRADATION_RS / baseline
        c[idx("dis", h, horizon)] = 0.50 * BATTERY_DEGRADATION_RS / baseline
        c[idx("diesel", h, horizon)] = 0.50 * DIESEL_COST_PER_KWH / baseline + 0.25 / raw_total_demand
        c[idx("shed", h, horizon)] = 0.50 * 40.0 / baseline
        c[idx("blackout", h, horizon)] = 0.50 * VOLL / baseline + 0.25 / raw_total_demand

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n
    for h in range(horizon):
        grid_cap = 0.0 if h in outages else GRID_MAX_KW
        bounds[idx("imp", h, horizon)] = (0.0, grid_cap)
        bounds[idx("exp", h, horizon)] = (0.0, grid_cap)
        bounds[idx("ch", h, horizon)] = (0.0, BATTERY_MAX_POWER_KW)
        bounds[idx("dis", h, horizon)] = (0.0, BATTERY_MAX_POWER_KW)
        bounds[idx("diesel", h, horizon)] = (0.0, DIESEL_MAX_KW)
        bounds[idx("shed", h, horizon)] = (0.0, None)
        bounds[idx("blackout", h, horizon)] = (0.0, None)
        bounds[idx("curtail", h, horizon)] = (0.0, None)
    for h in range(horizon + 1):
        bounds[idx("soc", h, horizon)] = (0.0, BATTERY_CAPACITY_KWH)
        bounds[idx("rebound", h, horizon)] = (0.0, None)

    a_eq = []
    b_eq = []

    # Initial SOC and rebound.
    row = np.zeros(n)
    row[idx("soc", 0, horizon)] = 1.0
    a_eq.append(row)
    b_eq.append(BATTERY_CAPACITY_KWH * 0.5)

    row = np.zeros(n)
    row[idx("rebound", 0, horizon)] = 1.0
    a_eq.append(row)
    b_eq.append(0.0)

    for h in range(horizon):
        # Energy balance:
        # solar + import + discharge*eff + diesel + blackout
        # = demand + rebound - shed + charge + export + curtail
        row = np.zeros(n)
        row[idx("imp", h, horizon)] = 1.0
        row[idx("dis", h, horizon)] = BATTERY_DISCHARGE_EFF
        row[idx("diesel", h, horizon)] = 1.0
        row[idx("blackout", h, horizon)] = 1.0
        row[idx("rebound", h, horizon)] = -1.0
        row[idx("shed", h, horizon)] = 1.0
        row[idx("ch", h, horizon)] = -1.0
        row[idx("exp", h, horizon)] = -1.0
        row[idx("curtail", h, horizon)] = -1.0
        a_eq.append(row)
        b_eq.append(float(demand[h] - solar[h]))

        # Battery SOC.
        row = np.zeros(n)
        row[idx("soc", h + 1, horizon)] = 1.0
        row[idx("soc", h, horizon)] = -1.0
        row[idx("ch", h, horizon)] = -BATTERY_CHARGE_EFF
        row[idx("dis", h, horizon)] = 1.0
        a_eq.append(row)
        b_eq.append(0.0)

        # 100% shedding rebound next hour.
        row = np.zeros(n)
        row[idx("rebound", h + 1, horizon)] = 1.0
        row[idx("shed", h, horizon)] = -1.0
        a_eq.append(row)
        b_eq.append(0.0)

    a_ub = []
    b_ub = []
    for h in range(horizon):
        # shed <= 20% * (demand + rebound)
        row = np.zeros(n)
        row[idx("shed", h, horizon)] = 1.0
        row[idx("rebound", h, horizon)] = -DEMAND_SHED_MAX_FRAC
        a_ub.append(row)
        b_ub.append(DEMAND_SHED_MAX_FRAC * float(demand[h]))

    # Diesel fuel tank.
    row = np.zeros(n)
    for h in range(horizon):
        row[idx("diesel", h, horizon)] = 1.0
    a_ub.append(row)
    b_ub.append(cfg.diesel_fuel_capacity * DIESEL_TANK_KWH)

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
        raise RuntimeError(f"LP failed for {task_id}/{seed}: {result.message}")

    actions: list[GridOpsAction] = []
    for h in range(horizon):
        ch = float(result.x[idx("ch", h, horizon)])
        dis = float(result.x[idx("dis", h, horizon)])
        diesel = float(result.x[idx("diesel", h, horizon)])
        rebound = float(result.x[idx("rebound", h, horizon)])
        actual_demand = max(float(demand[h]) + rebound, 1.0)
        shed = float(result.x[idx("shed", h, horizon)])
        if dis > ch and dis > 1e-5:
            battery_dispatch = dis / BATTERY_MAX_POWER_KW
        elif ch > 1e-5:
            battery_dispatch = -ch / BATTERY_MAX_POWER_KW
        else:
            battery_dispatch = 0.0
        actions.append(
            GridOpsAction(
                battery_dispatch=battery_dispatch,
                diesel_dispatch=diesel / DIESEL_MAX_KW,
                demand_shedding=shed / max(actual_demand * DEMAND_SHED_MAX_FRAC, 1.0),
            )
        )

    return {
        "actions": actions,
        "objective": float(result.fun),
        "lp_cost": float(c @ result.x),
        "lp_status": result.message,
    }


def rollout_lp(task_id: str, seed: int) -> dict[str, Any]:
    solution = solve_lp(task_id, seed)
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    trace = []
    for action in solution["actions"]:
        obs_dict = obs.model_dump()
        if len(trace) < 12:
            trace.append(
                {
                    "hour": int(obs_dict["hour"]),
                    "action": action.model_dump(),
                    "battery_soc": obs_dict["battery_soc"],
                    "demand_kw": obs_dict["demand_kw"],
                    "solar_kw": obs_dict["solar_kw"],
                    "grid_price": obs_dict["grid_price"],
                }
            )
        obs = env.step(action)
        if obs.done:
            break
    grade = env.state.grade or {}
    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade.get("score", 0.0),
        "grade": grade,
        "lp_objective": solution["objective"],
        "trace_samples": trace,
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {}
    for task_id in [task for task in TASKS if any(row["task_id"] == task for row in rows)]:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = {
            "score": round(sum(float(row["score"]) for row in task_rows) / max(len(task_rows), 1), 4),
            "blackout_kwh": round(
                sum(float((row["grade"] or {}).get("total_blackout_kwh", 0.0)) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
            "diesel_kwh": round(
                sum(float((row["grade"] or {}).get("total_diesel_kwh", 0.0)) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
            "cost": round(
                sum(float((row["grade"] or {}).get("actual_cost", 0.0)) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
        }
    return {
        "name": name,
        "average_score": round(sum(float(row["score"]) for row in rows) / max(len(rows), 1), 4),
        "by_task": by_task,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="7001,7002,7003")
    parser.add_argument("--output", default="evals/gridops_lp_oracle_holdout.json")
    args = parser.parse_args()

    rows = []
    for task_id in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        for seed in [int(x.strip()) for x in args.seeds.split(",") if x.strip()]:
            row = rollout_lp(task_id, seed)
            rows.append(row)
            print(json.dumps({"task_id": task_id, "seed": seed, "score": row["score"]}), flush=True)

    report = summarize("lp_full_episode_oracle_relaxed", rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["name", "average_score", "by_task"]}, indent=2))


if __name__ == "__main__":
    main()
