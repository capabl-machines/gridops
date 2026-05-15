"""Build a v4.1 repair curriculum for GridOps reasoning-action SFT.

v4 improved policy quality but still produced about 2.6% invalid actions on
heldout heatwave/crisis rollouts. The saved invalids were Pydantic validation
errors, not missing JSON, which points to out-of-bound action values.

This builder creates a small continuation dataset that over-teaches:

- fractional action scale, not kW;
- hard bounds for all three knobs;
- crisis diesel use without exceeding 1.0;
- heatwave/evening-ramp battery use without exceeding 1.0;
- concise `<think>` so `<action>` is never crowded out.
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
from scripts.build_gridops_v4_reasoning_traces import (
    action_dict,
    derive_context,
    previous_outcome_from_obs,
)


DEFAULT_REPAIR_TARGETS = {
    "heatwave_bound_repair": 220,
    "crisis_bound_repair": 260,
    "previous_blackout_bound_repair": 140,
    "charge_bound_repair": 90,
    "format_bound_anchor": 90,
}


def clamp_action(action: GridOpsAction) -> GridOpsAction:
    payload = action.model_dump()
    return GridOpsAction(
        battery_dispatch=max(-1.0, min(1.0, float(payload["battery_dispatch"]))),
        diesel_dispatch=max(0.0, min(1.0, float(payload["diesel_dispatch"]))),
        demand_shedding=max(0.0, min(1.0, float(payload["demand_shedding"]))),
    )


def max_bound_pressure(obs: dict[str, Any], task_id: str) -> float:
    derived = derive_context(obs, task_id)
    return max(abs(float(derived["supply_gap_kw"])), abs(float(derived["max_future_supply_gap_kw"])))


def bounded_completion(obs: dict[str, Any], action: GridOpsAction, task_id: str, derived: dict[str, Any], previous_outcome: dict[str, Any]) -> str:
    battery = float(action.battery_dispatch)
    diesel = float(action.diesel_dispatch)
    shedding = float(action.demand_shedding)
    gap = float(derived["supply_gap_kw"])
    max_gap = float(derived["max_future_supply_gap_kw"])
    scale_note = "Action values are fractions: battery -1..1, diesel 0..1, shedding 0..1; never write kW."

    if derived["grid_status"] == "outage" or diesel > 0.05:
        decision = (
            f"Clamp backup dispatch to valid fractions: battery={battery:.2f}, diesel={diesel:.2f}, "
            f"shedding={shedding:.2f}; use diesel only inside 0..1 to reduce blackout."
        )
    elif battery < -0.05:
        decision = f"Charge within bounds at battery={battery:.2f}; keep diesel={diesel:.2f} and shedding={shedding:.2f}."
    elif battery > 0.05:
        decision = f"Discharge within bounds at battery={battery:.2f}; keep diesel={diesel:.2f} unless backup is required."
    else:
        decision = f"Hold near neutral with valid fractions: battery={battery:.2f}, diesel={diesel:.2f}, shedding={shedding:.2f}."

    previous = "Previous feedback is clean, so follow the current bounded dispatch."
    if float(previous_outcome.get("blackout_kwh", 0.0)) > 0.01:
        previous = (
            f"Previous step had {float(previous_outcome['blackout_kwh']):.1f} kWh blackout, "
            "so correct with bounded battery/diesel instead of emitting out-of-range values."
        )

    lines = [
        (
            f"time_context: {derived['time_phase']} with grid {derived['grid_status']}, "
            f"scarcity {derived['scarcity_risk']}, gap {gap:.0f} kW, max future gap {max_gap:.0f} kW."
        ),
        f"1st_order: {scale_note}",
        "2nd_order: If the physical gap is above 100 kW, the action still caps at 1.00 and the grid/diesel/shedding handle the remainder.",
        f"previous_action: {previous}",
        f"decision: {decision}",
    ]
    return "<think>\n" + "\n".join(lines) + "\n</think>\n<action>\n" + action_to_json(action) + "\n</action>"


def make_repair_trace(
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
) -> dict[str, Any]:
    action = clamp_action(action)
    derived = derive_context(obs, task_id)
    completion = bounded_completion(obs, action, task_id, derived, previous_outcome)
    valid, reason = validate_reason_action_completion(completion)
    prompt = format_reason_action_observation(obs, derived, previous_action, previous_outcome)
    return {
        "id": trace_id,
        "task_id": task_id,
        "difficulty": "hard" if "crisis" in task_id or "blackout" in bucket else "medium",
        "seed": seed,
        "hour": hour,
        "messages": messages_for_reason_action_observation(obs, derived, previous_action, previous_outcome),
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
            "policy": "oracle_reasoning_v41_bound_repair",
            "bucket": bucket,
            "source": source,
            "source_labels": ["v41_action_bound_repair", bucket],
            "focus_tags": [
                "v41_action_bound_repair",
                bucket,
                task_id,
                derived["time_phase"],
                derived["scarcity_risk"],
                "valid_fractional_action",
            ],
            "validation": {"valid": valid, "reason": reason},
        },
    }


def collect_policy_rows(seed_start: int, max_seeds: int, targets: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, int, int, str]] = set()

    for task_id in ["task_2_heatwave", "task_3_crisis", "task_1_normal"]:
        for seed in range(seed_start, seed_start + max_seeds):
            if all(counts[k] >= v for k, v in targets.items() if k != "previous_blackout_bound_repair"):
                break
            env = GridOpsEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            previous_action = action_dict(GridOpsAction())
            previous_outcome = {"blackout_kwh": 0.0, "battery_soc_delta": 0.0, "diesel_used_kwh": 0.0, "cost": 0.0}
            prior_obs: dict[str, Any] | None = None
            prior_action: GridOpsAction | None = None

            for hour in range(72):
                obs_dict = obs.model_dump()
                if prior_obs is not None and prior_action is not None:
                    previous_outcome = previous_outcome_from_obs(obs_dict, prior_obs, prior_action)
                    previous_action = action_dict(prior_action)

                action = oracle_policy(obs_dict, task_id)
                derived = derive_context(obs_dict, task_id)
                bucket = ""
                if task_id == "task_2_heatwave" and derived["time_phase"] in {"evening_ramp", "late_evening"} and max_bound_pressure(obs_dict, task_id) > 80:
                    bucket = "heatwave_bound_repair"
                elif task_id == "task_3_crisis" and (derived["grid_status"] == "outage" or float(action.diesel_dispatch) > 0.05):
                    bucket = "crisis_bound_repair"
                elif float(action.battery_dispatch) < -0.5:
                    bucket = "charge_bound_repair"
                elif abs(float(action.battery_dispatch)) < 0.05 and float(action.diesel_dispatch) <= 0.01:
                    bucket = "format_bound_anchor"

                key = (task_id, seed, hour, bucket)
                if bucket and counts[bucket] < targets[bucket] and key not in seen:
                    rows.append(
                        make_repair_trace(
                            f"gridops_v41_{bucket}_{task_id}_seed{seed}_h{hour:02d}",
                            task_id,
                            seed,
                            hour,
                            obs_dict,
                            action,
                            previous_action,
                            previous_outcome,
                            bucket,
                            "fresh_oracle_rollout",
                        )
                    )
                    counts[bucket] += 1
                    seen.add(key)

                prior_obs = obs_dict
                prior_action = action
                obs = env.step(action)
                if obs.done:
                    break
    return rows


def collect_previous_blackout_rows(seed_start: int, max_seeds: int, target: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + max_seeds):
        if len(rows) >= target:
            break
        task_id = "task_3_crisis"
        env = GridOpsEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        prior_obs: dict[str, Any] | None = None
        prior_action: GridOpsAction | None = None
        for hour in range(72):
            obs_dict = obs.model_dump()
            if 30 <= hour <= 36:
                bad_action = GridOpsAction(battery_dispatch=0.0, diesel_dispatch=0.0, demand_shedding=0.0)
                next_obs = env.step(bad_action)
                if next_obs.done:
                    break
                next_obs_dict = next_obs.model_dump()
                previous_outcome = previous_outcome_from_obs(next_obs_dict, obs_dict, bad_action)
                if float(previous_outcome["blackout_kwh"]) > 0.01:
                    target_action = oracle_policy(next_obs_dict, task_id)
                    rows.append(
                        make_repair_trace(
                            f"gridops_v41_previous_blackout_bound_repair_{task_id}_seed{seed}_h{hour + 1:02d}",
                            task_id,
                            seed,
                            hour + 1,
                            next_obs_dict,
                            target_action,
                            action_dict(bad_action),
                            previous_outcome,
                            "previous_blackout_bound_repair",
                            "zero_action_failure_replay",
                        )
                    )
                obs = next_obs
            else:
                action = oracle_policy(obs_dict, task_id)
                prior_obs = obs_dict
                prior_action = action
                obs = env.step(action)
            if len(rows) >= target or obs.done:
                break
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen:
            failures.append({"id": row_id, "reason": "duplicate_id"})
        seen.add(row_id)
        valid, reason = validate_reason_action_completion(row.get("completion", ""))
        if not valid:
            failures.append({"id": row_id, "reason": reason})
        raw = row.get("raw") or {}
        if raw.get("prompt_mode") == "reason_action" and row.get("messages", [{}])[0].get("content") != REASON_ACTION_SYSTEM_PROMPT:
            failures.append({"id": row_id, "reason": "system_prompt_mismatch"})
    return failures


def summarize(rows: list[dict[str, Any]], repair_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "repair_rows": len(repair_rows),
        "bucket_counts": dict(Counter((row.get("raw") or {}).get("bucket", "unknown") for row in repair_rows)),
        "task_counts": dict(Counter(row.get("task_id") for row in repair_rows)),
        "action_counts": {
            "diesel_positive": sum(1 for row in repair_rows if float(row["action"]["diesel_dispatch"]) > 0.05),
            "battery_charge": sum(1 for row in repair_rows if float(row["action"]["battery_dispatch"]) < -0.05),
            "battery_discharge": sum(1 for row in repair_rows if float(row["action"]["battery_dispatch"]) > 0.05),
            "shedding_positive": sum(1 for row in repair_rows if float(row["action"]["demand_shedding"]) > 0.05),
        },
        "validation_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="sft_traces/gridops_curriculum_v4_kimi_reason_action_500.jsonl")
    parser.add_argument("--output", default="sft_traces/gridops_curriculum_v41_bound_repair_mix.jsonl")
    parser.add_argument("--repair-output", default="sft_traces/gridops_curriculum_v41_bound_repair_only.jsonl")
    parser.add_argument("--summary-output", default="evals/gridops_curriculum_v41_bound_repair_summary.json")
    parser.add_argument("--seed-start", type=int, default=14000)
    parser.add_argument("--max-seeds", type=int, default=240)
    parser.add_argument("--duplicate-repair", type=int, default=1)
    args = parser.parse_args()

    targets = dict(DEFAULT_REPAIR_TARGETS)
    repair_rows = collect_policy_rows(args.seed_start, args.max_seeds, targets)
    missing_blackout = max(0, targets["previous_blackout_bound_repair"] - sum(1 for row in repair_rows if (row.get("raw") or {}).get("bucket") == "previous_blackout_bound_repair"))
    repair_rows.extend(collect_previous_blackout_rows(args.seed_start + 5000, args.max_seeds, missing_blackout))

    base_rows = load_jsonl(Path(args.base))
    expanded_repair = []
    for copy_idx in range(max(1, args.duplicate_repair)):
        for row in repair_rows:
            copied = json.loads(json.dumps(row))
            if copy_idx:
                copied["id"] = f"{copied['id']}_dup{copy_idx}"
                copied["raw"]["source_labels"] = sorted(set(copied["raw"].get("source_labels", [])) | {f"repair_duplicate_{copy_idx}"})
            expanded_repair.append(copied)
    rows = base_rows + expanded_repair
    failures = validate_rows(rows)
    summary = summarize(rows, expanded_repair, failures)

    for path, payload in [
        (Path(args.output), rows),
        (Path(args.repair_output), repair_rows),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in payload:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
