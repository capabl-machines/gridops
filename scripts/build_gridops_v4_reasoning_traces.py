"""Build the GridOps v4 reasoning-action SFT curriculum.

v4 teaches the model a compact operator reasoning loop before the final action:
time context, 1st-order effect, 2nd-order consequence, previous feedback, and
decision. The final action remains the same GridOps JSON contract, wrapped in
an <action> block for robust parsing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.policies import oracle_policy
from gridops.prompting import (
    REASON_ACTION_SYSTEM_PROMPT,
    action_to_json,
    format_reason_action_observation,
    messages_for_reason_action_observation,
    validate_reason_action_completion,
)
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS


GRID_MAX_KW = 200.0
BATTERY_MAX_KW = 100.0
DIESEL_MAX_KW = 100.0

TASK_DIFFICULTY = {
    "task_1_normal": "easy",
    "task_2_heatwave": "medium",
    "task_3_crisis": "hard",
}

DEFAULT_TARGETS = {
    "crisis_diesel_positive": 900,
    "normal_no_diesel": 700,
    "heatwave_rebound": 600,
    "previous_action_correction": 500,
    "low_resource_edges": 400,
    "time_context_mix": 600,
    "format_anchors": 300,
}


def action_dict(action: GridOpsAction | dict[str, Any]) -> dict[str, float]:
    if isinstance(action, GridOpsAction):
        payload = action.model_dump()
    else:
        payload = action
    return {
        "battery_dispatch": round(float(payload.get("battery_dispatch", 0.0)), 4),
        "diesel_dispatch": round(float(payload.get("diesel_dispatch", 0.0)), 4),
        "demand_shedding": round(float(payload.get("demand_shedding", 0.0)), 4),
    }


def trend(values: list[float], current: float, tolerance: float) -> str:
    if not values:
        return "unknown"
    future = float(values[-1])
    if future > current + tolerance:
        return "rising"
    if future < current - tolerance:
        return "falling"
    return "steady"


def time_phase(hour: int) -> str:
    hour_of_day = (hour + 6) % 24
    if 0 <= hour_of_day < 6:
        return "overnight_low_demand"
    if 6 <= hour_of_day < 10:
        return "morning_ramp"
    if 10 <= hour_of_day < 15:
        return "midday_solar_window"
    if 15 <= hour_of_day < 18:
        return "pre_evening_charge_window"
    if 18 <= hour_of_day < 22:
        return "evening_ramp"
    return "late_evening"


def derive_context(obs: dict[str, Any], task_id: str) -> dict[str, Any]:
    hour = int(obs["hour"])
    demand = float(obs["demand_kw"])
    solar = float(obs["solar_kw"])
    soc = float(obs["battery_soc"])
    fuel = float(obs["diesel_fuel_remaining"])
    price = float(obs["grid_price"])
    demand_fc = [float(x) for x in obs.get("demand_forecast_4h", [])]
    solar_fc = [float(x) for x in obs.get("solar_forecast_4h", [])]
    price_fc = [float(x) for x in obs.get("price_forecast_4h", [])]
    in_outage = task_id == "task_3_crisis" and 30 <= hour <= 35
    outage_soon = task_id == "task_3_crisis" and 26 <= hour <= 35
    grid_capacity = 0.0 if in_outage else GRID_MAX_KW
    supply_gap = demand - solar - grid_capacity
    future_supply_gaps = [
        d - s - (0.0 if in_outage else GRID_MAX_KW)
        for d, s in zip(demand_fc, solar_fc)
    ]
    max_future_supply_gap = max([supply_gap] + future_supply_gaps) if future_supply_gaps else supply_gap
    no_solar_soon = max([solar] + solar_fc) < 20.0
    high_gap = max_future_supply_gap > 50.0
    scarcity_risk = "high" if in_outage or (high_gap and (soc < 0.45 or no_solar_soon)) else "medium" if outage_soon or high_gap else "low"
    return {
        "time_phase": time_phase(hour),
        "hour_of_day": (hour + 6) % 24,
        "grid_status": "outage" if in_outage else "available",
        "outage_risk": "active" if in_outage else "near" if outage_soon else "normal",
        "solar_trend": trend(solar_fc, solar, 25.0),
        "demand_trend": trend(demand_fc, demand, 30.0),
        "price_trend": trend(price_fc, price, 2.0),
        "supply_gap_kw": round(supply_gap, 2),
        "max_future_supply_gap_kw": round(max_future_supply_gap, 2),
        "battery_state": "low" if soc < 0.25 else "medium" if soc < 0.75 else "high",
        "fuel_state": "scarce" if fuel < 0.18 else "limited" if fuel < 0.45 else "available",
        "scarcity_risk": scarcity_risk,
    }


def previous_outcome_from_obs(
    current_obs: dict[str, Any],
    prior_obs: dict[str, Any] | None,
    previous_action: GridOpsAction | None,
) -> dict[str, float]:
    if prior_obs is None or previous_action is None:
        return {
            "blackout_kwh": 0.0,
            "battery_soc_delta": 0.0,
            "diesel_used_kwh": 0.0,
            "cost": 0.0,
        }
    return {
        "blackout_kwh": round(float(current_obs.get("blackout_this_step", 0.0)), 4),
        "battery_soc_delta": round(float(current_obs.get("battery_soc", 0.0)) - float(prior_obs.get("battery_soc", 0.0)), 4),
        "diesel_used_kwh": round(float(current_obs.get("flow_diesel", 0.0)), 4),
        "cost": round(float(current_obs.get("cost_this_step", 0.0)), 2),
    }


def focus_tags(
    obs: dict[str, Any],
    action: GridOpsAction,
    task_id: str,
    derived: dict[str, Any],
    previous_outcome: dict[str, Any],
    bucket: str,
) -> list[str]:
    tags = {bucket, task_id, derived["time_phase"], derived["scarcity_risk"]}
    if float(action.diesel_dispatch) > 0.05:
        tags.add("diesel_positive")
    else:
        tags.add("diesel_zero")
    if float(action.battery_dispatch) > 0.25:
        tags.add("battery_discharge")
    elif float(action.battery_dispatch) < -0.25:
        tags.add("battery_charge")
    if float(action.demand_shedding) > 0.05:
        tags.add("shedding_positive")
    if derived["grid_status"] == "outage":
        tags.add("outage")
    if derived["battery_state"] == "low":
        tags.add("low_soc")
    if derived["fuel_state"] in {"scarce", "limited"}:
        tags.add("fuel_constraint")
    if float(previous_outcome.get("blackout_kwh", 0.0)) > 0.01:
        tags.add("previous_blackout")
    if "Rebound" in str(obs.get("narration", "")) or float(obs.get("flow_shed", 0.0)) > 0.01:
        tags.add("rebound")
    return sorted(tags)


def reasoning_lines(
    obs: dict[str, Any],
    action: GridOpsAction,
    task_id: str,
    derived: dict[str, Any],
    previous_action: dict[str, float],
    previous_outcome: dict[str, float],
) -> list[str]:
    phase = str(derived["time_phase"]).replace("_", " ")
    grid_status = str(derived["grid_status"])
    supply_gap = float(derived["supply_gap_kw"])
    max_gap = float(derived["max_future_supply_gap_kw"])
    battery = float(action.battery_dispatch)
    diesel = float(action.diesel_dispatch)
    shedding = float(action.demand_shedding)
    blackout = float(previous_outcome.get("blackout_kwh", 0.0))
    soc_delta = float(previous_outcome.get("battery_soc_delta", 0.0))
    prev_diesel = float(previous_action.get("diesel_dispatch", 0.0))

    time_context = (
        f"{phase}; solar is {derived['solar_trend']}, demand is {derived['demand_trend']}, "
        f"grid is {grid_status}, and scarcity risk is {derived['scarcity_risk']}."
    )

    if supply_gap > 20:
        first_order = (
            f"Demand exceeds immediate grid plus solar by about {supply_gap:.0f} kW, "
            "so flexible supply is needed now."
        )
    elif supply_gap < -40:
        first_order = (
            f"Available grid and solar exceed demand by about {-supply_gap:.0f} kW, "
            "so diesel is unnecessary and charging can be considered."
        )
    else:
        first_order = "Immediate supply is close to demand, so avoid emergency tools unless forecasts justify them."

    if max_gap > 50 and derived["solar_trend"] != "rising":
        second_order = (
            f"The next 4 hours can still face a gap near {max_gap:.0f} kW with weak solar recovery, "
            "so blackout prevention matters more than a zero-diesel habit."
        )
    elif derived["time_phase"] == "midday_solar_window":
        second_order = "Midday solar can support demand and recharge SOC, so preserve diesel for later stress."
    elif derived["time_phase"] in {"pre_evening_charge_window", "evening_ramp"}:
        second_order = "Evening demand can stay elevated while solar fades, so SOC and backup fuel must be managed deliberately."
    else:
        second_order = "Forecast risk is manageable, so keep the action economical and avoid unnecessary diesel or shedding."

    if blackout > 0.01:
        previous = (
            f"Last action caused {blackout:.1f} kWh blackout with diesel at {prev_diesel:.2f}, "
            "so the policy should correct instead of repeating it."
        )
    elif abs(soc_delta) > 0.03:
        previous = f"Last action changed SOC by {soc_delta:+.2f}, so the current action should account for that battery movement."
    else:
        previous = "Last action did not create a major penalty, so current conditions and forecast drive the decision."

    if diesel > 0.05:
        decision = (
            f"Use battery at {battery:.2f} and diesel at {diesel:.2f} to reduce blackout risk; "
            f"shedding stays {shedding:.2f} unless supply remains insufficient."
        )
    elif battery < -0.05:
        decision = f"Charge the battery at {battery:.2f} while keeping diesel off because there is no emergency gap."
    elif battery > 0.05:
        decision = f"Discharge battery at {battery:.2f} while keeping diesel off because the gap is manageable without backup fuel."
    else:
        decision = "Hold dispatch near neutral and keep diesel off because the state does not justify emergency resources."

    return [
        f"time_context: {time_context}",
        f"1st_order: {first_order}",
        f"2nd_order: {second_order}",
        f"previous_action: {previous}",
        f"decision: {decision}",
    ]


def make_completion(
    obs: dict[str, Any],
    action: GridOpsAction,
    task_id: str,
    derived: dict[str, Any],
    previous_action: dict[str, float],
    previous_outcome: dict[str, float],
) -> str:
    lines = reasoning_lines(obs, action, task_id, derived, previous_action, previous_outcome)
    return "<think>\n" + "\n".join(lines) + "\n</think>\n<action>\n" + action_to_json(action) + "\n</action>"


def classify_bucket(
    task_id: str,
    obs: dict[str, Any],
    action: GridOpsAction,
    derived: dict[str, Any],
    previous_outcome: dict[str, Any],
) -> str:
    diesel = float(action.diesel_dispatch)
    battery = float(action.battery_dispatch)
    if float(previous_outcome.get("blackout_kwh", 0.0)) > 0.01:
        return "previous_action_correction"
    if task_id == "task_3_crisis" and diesel > 0.05:
        return "crisis_diesel_positive"
    if derived["battery_state"] == "low" or derived["fuel_state"] in {"scarce", "limited"}:
        return "low_resource_edges"
    if task_id == "task_2_heatwave" and (
        "Rebound" in str(obs.get("narration", "")) or derived["max_future_supply_gap_kw"] > 20
    ):
        return "heatwave_rebound"
    if task_id == "task_1_normal" and diesel <= 0.01:
        return "normal_no_diesel"
    if abs(battery) < 0.05 and diesel <= 0.01:
        return "format_anchors"
    return "time_context_mix"


def make_trace(
    *,
    trace_id: str,
    task_id: str,
    seed: int,
    hour: int,
    obs: dict[str, Any],
    action: GridOpsAction,
    previous_action: dict[str, float],
    previous_outcome: dict[str, float],
    bucket: str,
    source: str,
    source_labels: list[str] | None = None,
) -> dict[str, Any]:
    derived = derive_context(obs, task_id)
    completion = make_completion(obs, action, task_id, derived, previous_action, previous_outcome)
    valid, reason = validate_reason_action_completion(completion)
    messages = messages_for_reason_action_observation(obs, derived, previous_action, previous_outcome)
    prompt = format_reason_action_observation(obs, derived, previous_action, previous_outcome)
    tags = focus_tags(obs, action, task_id, derived, previous_outcome, bucket)
    if source_labels:
        tags = sorted(set(tags) | set(source_labels))
    return {
        "id": trace_id,
        "task_id": task_id,
        "difficulty": TASK_DIFFICULTY.get(task_id, "medium") if bucket != "previous_action_correction" else "hard",
        "seed": seed,
        "hour": hour,
        "messages": messages,
        "prompt": prompt,
        "completion": completion,
        "action": json.loads(action_to_json(action)),
        "raw": {
            "observation": obs,
            "derived_context": derived,
            "previous_action": previous_action,
            "previous_outcome": previous_outcome,
            "oracle_action": action_dict(action),
            "prompt_mode": "reason_action",
            "policy": "oracle_reasoning_v4",
            "bucket": bucket,
            "source": source,
            "source_labels": source_labels or [],
            "focus_tags": tags,
            "validation": {"valid": valid, "reason": reason},
        },
    }


def collect_oracle_rows(targets: dict[str, int], seed_start: int, max_seeds_per_task: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_states: set[tuple[str, int, int, str]] = set()

    for task_index, task_id in enumerate(TASKS):
        for seed in range(seed_start + task_index * 1000, seed_start + task_index * 1000 + max_seeds_per_task):
            env = GridOpsEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            previous_action: dict[str, float] = action_dict(GridOpsAction())
            previous_outcome = {
                "blackout_kwh": 0.0,
                "battery_soc_delta": 0.0,
                "diesel_used_kwh": 0.0,
                "cost": 0.0,
            }
            prior_obs: dict[str, Any] | None = None
            prior_action: GridOpsAction | None = None

            for step in range(72):
                obs_dict = obs.model_dump()
                if prior_obs is not None and prior_action is not None:
                    previous_outcome = previous_outcome_from_obs(obs_dict, prior_obs, prior_action)
                    previous_action = action_dict(prior_action)

                action = oracle_policy(obs_dict, task_id)
                derived = derive_context(obs_dict, task_id)
                bucket = classify_bucket(task_id, obs_dict, action, derived, previous_outcome)
                state_key = (task_id, seed, step, bucket)
                if counts[bucket] < targets.get(bucket, 0) and state_key not in seen_states:
                    rows.append(
                        make_trace(
                            trace_id=f"gridops_v4_{bucket}_{task_id}_seed{seed}_h{step:02d}",
                            task_id=task_id,
                            seed=seed,
                            hour=step,
                            obs=obs_dict,
                            action=action,
                            previous_action=previous_action,
                            previous_outcome=previous_outcome,
                            bucket=bucket,
                            source="oracle_rollout",
                        )
                    )
                    counts[bucket] += 1
                    seen_states.add(state_key)

                prior_obs = obs_dict
                prior_action = action
                obs = env.step(action)
                if obs.done:
                    break

            if all(counts[bucket] >= target for bucket, target in targets.items() if bucket != "previous_action_correction"):
                break

    return rows


def replay_to_hour(task_id: str, seed: int, hour: int) -> tuple[GridOpsEnvironment, dict[str, Any], GridOpsAction | None]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    prior_action: GridOpsAction | None = None
    for _ in range(hour):
        obs_dict = obs.model_dump()
        prior_action = oracle_policy(obs_dict, task_id)
        obs = env.step(prior_action)
        if obs.done:
            break
    return env, obs.model_dump(), prior_action


def collect_failure_corrections(paths: list[Path], target: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip() or len(rows) >= target:
                    continue
                failure = json.loads(line)
                task_id = str(failure["task_id"])
                seed = int(failure["seed"])
                hour = int(failure["hour"])
                if hour >= 71 or (task_id, seed, hour) in seen:
                    continue
                labels = [str(x) for x in failure.get("labels", [])]
                if not any(label in labels for label in ["blackout_after_action", "missed_diesel", "diesel_oracle_gap"]):
                    continue
                try:
                    bad_action = GridOpsAction(**failure.get("model_action", {}))
                    env, current_obs, _ = replay_to_hour(task_id, seed, hour)
                    next_obs = env.step(bad_action)
                    if next_obs.done:
                        continue
                    next_obs_dict = next_obs.model_dump()
                    target_action = oracle_policy(next_obs_dict, task_id)
                except Exception:
                    continue
                previous_outcome = previous_outcome_from_obs(next_obs_dict, current_obs, bad_action)
                if float(previous_outcome.get("blackout_kwh", 0.0)) <= 0.01 and "missed_diesel" not in labels:
                    continue
                rows.append(
                    make_trace(
                        trace_id=f"gridops_v4_previous_action_correction_{task_id}_seed{seed}_h{hour + 1:02d}",
                        task_id=task_id,
                        seed=seed,
                        hour=hour + 1,
                        obs=next_obs_dict,
                        action=target_action,
                        previous_action=action_dict(bad_action),
                        previous_outcome=previous_outcome,
                        bucket="previous_action_correction",
                        source=f"failure_bank:{path}",
                        source_labels=labels,
                    )
                )
                seen.add((task_id, seed, hour))
        if len(rows) >= target:
            break

    return rows


def stress_policy(obs: dict[str, Any], task_id: str) -> GridOpsAction:
    """Create hard-but-realistic crisis states for diesel correction traces.

    This intentionally behaves like the v3 failure mode before the target
    example: it avoids diesel and often underuses the battery. The target label
    for the resulting state still comes from the oracle.
    """
    hour = int(obs["hour"])
    task_is_crisis = task_id == "task_3_crisis"
    if task_is_crisis and 24 <= hour < 30:
        return GridOpsAction(battery_dispatch=0.4, diesel_dispatch=0.0, demand_shedding=0.0)
    if task_is_crisis and 30 <= hour <= 35:
        return GridOpsAction(battery_dispatch=0.0, diesel_dispatch=0.0, demand_shedding=0.0)
    return GridOpsAction()


def collect_stressed_crisis_rows(seed_start: int, target: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = seed_start
    seen: set[tuple[int, int]] = set()

    while len(rows) < target and seed < seed_start + 2000:
        task_id = "task_3_crisis"
        env = GridOpsEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        previous_action: dict[str, float] = action_dict(GridOpsAction())
        previous_outcome = {
            "blackout_kwh": 0.0,
            "battery_soc_delta": 0.0,
            "diesel_used_kwh": 0.0,
            "cost": 0.0,
        }
        prior_obs: dict[str, Any] | None = None
        prior_action: GridOpsAction | None = None

        for step in range(72):
            obs_dict = obs.model_dump()
            if prior_obs is not None and prior_action is not None:
                previous_outcome = previous_outcome_from_obs(obs_dict, prior_obs, prior_action)
                previous_action = action_dict(prior_action)

            target_action = oracle_policy(obs_dict, task_id)
            if (
                30 <= step <= 40
                and float(target_action.diesel_dispatch) > 0.05
                and (seed, step) not in seen
                and len(rows) < target
            ):
                rows.append(
                    make_trace(
                        trace_id=f"gridops_v4_stressed_crisis_diesel_positive_{task_id}_seed{seed}_h{step:02d}",
                        task_id=task_id,
                        seed=seed,
                        hour=step,
                        obs=obs_dict,
                        action=target_action,
                        previous_action=previous_action,
                        previous_outcome=previous_outcome,
                        bucket="crisis_diesel_positive",
                        source="stressed_crisis_rollout",
                        source_labels=["v3_never_diesel_counterexample"],
                    )
                )
                seen.add((seed, step))

            prior_obs = obs_dict
            if step < 30:
                prior_action = stress_policy(obs_dict, task_id)
                obs = env.step(prior_action)
            else:
                prior_action = target_action
                obs = env.step(target_action)
            if obs.done:
                break
        seed += 1

    return rows


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            failures.append({"id": row_id, "reason": "duplicate_id"})
        seen_ids.add(row_id)
        valid, reason = validate_reason_action_completion(row.get("completion", ""))
        if not valid:
            failures.append({"id": row_id, "reason": reason})
        raw = row.get("raw") or {}
        obs = raw.get("observation")
        if row.get("messages", [{}])[0].get("content") != REASON_ACTION_SYSTEM_PROMPT:
            failures.append({"id": row_id, "reason": "system_prompt_mismatch"})
        if obs and row.get("prompt") != format_reason_action_observation(
            obs,
            raw.get("derived_context"),
            raw.get("previous_action"),
            raw.get("previous_outcome"),
        ):
            failures.append({"id": row_id, "reason": "prompt_mismatch"})
    return failures


def summarize(rows: list[dict[str, Any]], failures: list[dict[str, Any]], targets: dict[str, int]) -> dict[str, Any]:
    task_counts = Counter(row["task_id"] for row in rows)
    bucket_counts = Counter((row.get("raw") or {}).get("bucket", "unknown") for row in rows)
    focus_counts = Counter(tag for row in rows for tag in (row.get("raw") or {}).get("focus_tags", []))
    action_counts = {
        "diesel_positive": sum(1 for row in rows if float(row["action"]["diesel_dispatch"]) > 0.05),
        "diesel_zero": sum(1 for row in rows if float(row["action"]["diesel_dispatch"]) <= 0.05),
        "battery_charge": sum(1 for row in rows if float(row["action"]["battery_dispatch"]) < -0.05),
        "battery_discharge": sum(1 for row in rows if float(row["action"]["battery_dispatch"]) > 0.05),
        "shedding_positive": sum(1 for row in rows if float(row["action"]["demand_shedding"]) > 0.05),
    }
    return {
        "rows": len(rows),
        "targets": targets,
        "task_counts": dict(task_counts),
        "bucket_counts": dict(bucket_counts),
        "action_counts": action_counts,
        "top_focus_tags": dict(focus_counts.most_common(30)),
        "validation_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sft_traces/gridops_curriculum_v4_reason_action.jsonl")
    parser.add_argument("--summary-output", default="evals/gridops_curriculum_v4_reason_action_summary.json")
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument("--max-seeds-per-task", type=int, default=80)
    parser.add_argument(
        "--failure-bank",
        action="append",
        default=[
            "evals/failure_mining_v2_7101_7110/failure_bank.jsonl",
            "evals/failure_mining_7101_7110/failure_bank.jsonl",
        ],
        help="Failure-bank JSONL to turn into previous-action correction traces. Can be repeated.",
    )
    parser.add_argument("--previous-correction-target", type=int, default=DEFAULT_TARGETS["previous_action_correction"])
    args = parser.parse_args()

    targets = dict(DEFAULT_TARGETS)
    targets["previous_action_correction"] = max(0, int(args.previous_correction_target))

    correction_rows = collect_failure_corrections(
        [Path(path) for path in args.failure_bank],
        targets["previous_action_correction"],
    )
    targets_for_oracle = dict(targets)
    targets_for_oracle["previous_action_correction"] = max(
        0,
        targets["previous_action_correction"] - len(correction_rows),
    )
    oracle_rows = collect_oracle_rows(targets_for_oracle, args.seed_start, args.max_seeds_per_task)
    current_bucket_counts = Counter((row.get("raw") or {}).get("bucket", "unknown") for row in oracle_rows + correction_rows)
    missing_crisis_diesel = max(0, targets["crisis_diesel_positive"] - current_bucket_counts["crisis_diesel_positive"])
    stressed_crisis_rows = collect_stressed_crisis_rows(args.seed_start + 5000, missing_crisis_diesel)
    rows = oracle_rows + correction_rows + stressed_crisis_rows
    failures = validate_rows(rows)
    summary = summarize(rows, failures, targets)

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
