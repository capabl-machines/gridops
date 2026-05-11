"""Evaluate a search-based MPC planner in the GridOps environment.

This is not a model. It is a ceiling-finding controller:

1. At each hour, sample candidate action sequences.
2. Simulate each sequence in a copied OpenEnv environment.
3. Execute only the first action from the best sequence.
4. Repeat until the 72-hour episode ends.

If this planner cannot approach the desired score, model training alone is
unlikely to reach it without changing the environment, action space, or reward.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.policies import oracle_policy
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS


def action_key(action: GridOpsAction) -> tuple[float, float, float]:
    return (
        round(float(action.battery_dispatch), 4),
        round(float(action.diesel_dispatch), 4),
        round(float(action.demand_shedding), 4),
    )


def action_dict(action: GridOpsAction) -> dict[str, float]:
    return {
        "battery_dispatch": round(float(action.battery_dispatch), 4),
        "diesel_dispatch": round(float(action.diesel_dispatch), 4),
        "demand_shedding": round(float(action.demand_shedding), 4),
    }


def dedupe_actions(actions: list[GridOpsAction]) -> list[GridOpsAction]:
    seen: set[tuple[float, float, float]] = set()
    result: list[GridOpsAction] = []
    for action in actions:
        key = action_key(action)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def derived_risk(obs: dict[str, Any], task_id: str) -> dict[str, Any]:
    hour = int(obs["hour"])
    demand = float(obs["demand_kw"])
    solar = float(obs["solar_kw"])
    demand_fc = [float(x) for x in obs.get("demand_forecast_4h", [])]
    solar_fc = [float(x) for x in obs.get("solar_forecast_4h", [])]
    in_outage = task_id == "task_3_crisis" and 30 <= hour <= 35
    outage_soon = task_id == "task_3_crisis" and 25 <= hour <= 35
    grid_cap = 0.0 if in_outage else 200.0
    gap = demand - solar - grid_cap
    future_gaps = [
        d - s - (0.0 if in_outage else 200.0)
        for d, s in zip(demand_fc, solar_fc)
    ]
    max_gap = max([gap] + future_gaps) if future_gaps else gap
    return {
        "hour": hour,
        "in_outage": in_outage,
        "outage_soon": outage_soon,
        "gap": gap,
        "max_gap": max_gap,
        "soc": float(obs["battery_soc"]),
        "fuel": float(obs["diesel_fuel_remaining"]),
        "price": float(obs["grid_price"]),
    }


def action_pool(obs: dict[str, Any], task_id: str, rng: random.Random, max_actions: int) -> list[GridOpsAction]:
    risk = derived_risk(obs, task_id)
    oracle = oracle_policy(obs, task_id)
    actions: list[GridOpsAction] = [
        GridOpsAction(),
        oracle,
        GridOpsAction(battery_dispatch=-1.0),
        GridOpsAction(battery_dispatch=-0.8),
        GridOpsAction(battery_dispatch=-0.5),
        GridOpsAction(battery_dispatch=0.5),
        GridOpsAction(battery_dispatch=0.8),
        GridOpsAction(battery_dispatch=1.0),
    ]

    # Focused backup actions only when useful. This avoids diesel spam in normal
    # states while still searching crisis/outage combinations.
    backup_needed = bool(risk["in_outage"] or risk["outage_soon"] or risk["max_gap"] > 50)
    if backup_needed:
        for battery in [0.0, 0.4, 0.7, 1.0]:
            for diesel in [0.25, 0.5, 0.75, 1.0]:
                actions.append(GridOpsAction(battery_dispatch=battery, diesel_dispatch=diesel))
        for battery in [0.4, 0.7, 1.0]:
            for diesel in [0.5, 1.0]:
                for shed in [0.25, 0.5, 1.0]:
                    actions.append(GridOpsAction(battery_dispatch=battery, diesel_dispatch=diesel, demand_shedding=shed))

    # Continuous random proposals give the planner a chance to find non-oracle
    # fractional actions.
    for _ in range(max(0, max_actions * 2)):
        if backup_needed:
            battery = rng.uniform(0.0, 1.0) if risk["gap"] > 0 else rng.uniform(-0.4, 0.8)
            diesel = rng.uniform(0.0, 1.0)
            shed = rng.choice([0.0, 0.0, 0.0, rng.uniform(0.0, 1.0)])
        else:
            battery = rng.uniform(-1.0, 1.0)
            diesel = 0.0 if rng.random() < 0.95 else rng.uniform(0.0, 0.25)
            shed = 0.0
        actions.append(GridOpsAction(battery_dispatch=battery, diesel_dispatch=diesel, demand_shedding=shed))

    actions = dedupe_actions(actions)
    anchored = actions[:8]
    rest = actions[8:]
    rng.shuffle(rest)
    return dedupe_actions(anchored + rest[: max(0, max_actions - len(anchored))])


def sequence_value(
    env: GridOpsEnvironment,
    sequence: list[GridOpsAction],
    blackout_weight: float,
    diesel_weight: float,
    shed_weight: float,
    soc_target: float,
    soc_weight: float,
) -> dict[str, Any]:
    trial = copy.deepcopy(env)
    start_state = trial.state
    start_history_len = len(start_state.history)
    start_cost = float((start_state.history[-1]["cost"] if start_state.history else 0.0))
    # Cumulative values are available inside history only as step entries, so
    # derive deltas from observations as we roll forward.
    total_reward = 0.0
    blackout = 0.0
    diesel = 0.0
    shed = 0.0
    final_soc = 0.5
    cost = 0.0

    for action in sequence:
        obs = trial.step(action)
        obs_dict = obs.model_dump()
        total_reward += float(obs.reward or 0.0)
        blackout += float(obs_dict.get("blackout_this_step", 0.0))
        diesel += float(obs_dict.get("flow_diesel", 0.0))
        shed += float(obs_dict.get("flow_shed", 0.0))
        cost += float(obs_dict.get("cost_this_step", 0.0))
        final_soc = float(obs_dict.get("battery_soc", final_soc))
        if obs.done:
            break

    value = (
        total_reward
        - blackout_weight * blackout
        - diesel_weight * diesel
        - shed_weight * shed
        - 0.002 * cost
        + soc_weight * min(0.0, final_soc - soc_target)
    )
    if trial.state.done and trial.state.grade:
        value += 20.0 * float(trial.state.grade.get("score", 0.0))

    return {
        "value": float(value),
        "reward": round(total_reward, 4),
        "blackout_kwh": round(blackout, 4),
        "diesel_kwh": round(diesel, 4),
        "shed_kwh": round(shed, 4),
        "cost": round(cost, 2),
        "final_soc": round(final_soc, 4),
        "history_steps": len(trial.state.history) - start_history_len,
        "start_step_cost": start_cost,
    }


def soc_target_for_state(obs: dict[str, Any], task_id: str, risk: dict[str, Any]) -> float:
    hour_of_day = (int(obs["hour"]) + 6) % 24
    if task_id == "task_3_crisis" and risk["outage_soon"]:
        return 0.85 if not risk["in_outage"] else 0.35
    if 10 <= hour_of_day < 18:
        return 0.82
    if 18 <= hour_of_day < 23:
        return 0.35
    if risk["max_gap"] > 50:
        return 0.55
    return 0.25


def make_sequences(pool: list[GridOpsAction], rng: random.Random, horizon: int, sequence_count: int) -> list[list[GridOpsAction]]:
    if not pool:
        return [[GridOpsAction()]]
    sequences: list[list[GridOpsAction]] = []
    # Include constant-action sequences for interpretability.
    for action in pool[: min(len(pool), max(4, sequence_count // 8))]:
        sequences.append([action for _ in range(horizon)])
    # Include oracle-like first actions mixed with random continuations.
    while len(sequences) < sequence_count:
        sequences.append([rng.choice(pool) for _ in range(horizon)])
    return sequences[:sequence_count]


def choose_mpc_action(
    env: GridOpsEnvironment,
    obs: dict[str, Any],
    task_id: str,
    rng: random.Random,
    horizon: int,
    sequence_count: int,
    max_actions: int,
    blackout_weight: float,
    diesel_weight: float,
    shed_weight: float,
    soc_weight: float,
) -> tuple[GridOpsAction, dict[str, Any]]:
    risk = derived_risk(obs, task_id)
    pool = action_pool(obs, task_id, rng, max_actions=max_actions)
    sequences = make_sequences(pool, rng, horizon=horizon, sequence_count=sequence_count)
    soc_target = soc_target_for_state(obs, task_id, risk)
    best_value = -math.inf
    best_action = sequences[0][0]
    best_info: dict[str, Any] = {}

    for sequence in sequences:
        info = sequence_value(
            env,
            sequence,
            blackout_weight=blackout_weight,
            diesel_weight=diesel_weight,
            shed_weight=shed_weight,
            soc_target=soc_target,
            soc_weight=soc_weight,
        )
        if float(info["value"]) > best_value:
            best_value = float(info["value"])
            best_action = sequence[0]
            best_info = info

    best_info = {
        **best_info,
        "candidate_pool": len(pool),
        "sequence_count": len(sequences),
        "soc_target": soc_target,
        "action": action_dict(best_action),
    }
    return best_action, best_info


def rollout_mpc(
    task_id: str,
    seed: int,
    horizon: int,
    sequence_count: int,
    max_actions: int,
    rng_seed: int,
    blackout_weight: float,
    diesel_weight: float,
    shed_weight: float,
    soc_weight: float,
    trace_limit: int,
) -> dict[str, Any]:
    rng = random.Random(rng_seed + seed + hash(task_id) % 10000)
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    traces: list[dict[str, Any]] = []

    for _ in range(72):
        obs_dict = obs.model_dump()
        action, info = choose_mpc_action(
            env,
            obs_dict,
            task_id,
            rng,
            horizon=horizon,
            sequence_count=sequence_count,
            max_actions=max_actions,
            blackout_weight=blackout_weight,
            diesel_weight=diesel_weight,
            shed_weight=shed_weight,
            soc_weight=soc_weight,
        )
        if len(traces) < trace_limit:
            traces.append(
                {
                    "hour": int(obs_dict["hour"]),
                    "observation": {
                        "demand_kw": obs_dict["demand_kw"],
                        "solar_kw": obs_dict["solar_kw"],
                        "battery_soc": obs_dict["battery_soc"],
                        "grid_price": obs_dict["grid_price"],
                    },
                    "chosen": info,
                    "oracle_action": action_dict(oracle_policy(obs_dict, task_id)),
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
        "trace_samples": traces,
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {}
    present_tasks = [task_id for task_id in TASKS if any(row["task_id"] == task_id for row in rows)]
    for task_id in present_tasks:
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
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--sequence-count", type=int, default=96)
    parser.add_argument("--max-actions", type=int, default=48)
    parser.add_argument("--rng-seed", type=int, default=20260511)
    parser.add_argument("--blackout-weight", type=float, default=0.75)
    parser.add_argument("--diesel-weight", type=float, default=0.3)
    parser.add_argument("--shed-weight", type=float, default=0.25)
    parser.add_argument("--soc-weight", type=float, default=12.0)
    parser.add_argument("--trace-limit", type=int, default=8)
    parser.add_argument("--output", default="evals/gridops_mpc_planner_holdout.json")
    args = parser.parse_args()

    task_ids = [x.strip() for x in args.tasks.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for task_id in task_ids:
        for seed in seeds:
            result = rollout_mpc(
                task_id=task_id,
                seed=seed,
                horizon=args.horizon,
                sequence_count=args.sequence_count,
                max_actions=args.max_actions,
                rng_seed=args.rng_seed,
                blackout_weight=args.blackout_weight,
                diesel_weight=args.diesel_weight,
                shed_weight=args.shed_weight,
                soc_weight=args.soc_weight,
                trace_limit=args.trace_limit,
            )
            rows.append(result)
            print(json.dumps({"task_id": task_id, "seed": seed, "score": result["score"]}), flush=True)

    report = summarize(
        f"mpc_h{args.horizon}_seq{args.sequence_count}_actions{args.max_actions}",
        rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["name", "average_score", "by_task"]}, indent=2))


if __name__ == "__main__":
    main()
