"""Causal LP critic used for clean GridOps imitation data.

The critic compares a proposed action with the causal LP action on copied
environment state. It is intentionally runtime-safe: full 72-hour LP remains an
offline ceiling, while this module only uses the current observation, forecasts,
task rules, SOC/fuel, and previous feedback exposed to the agent.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from gridops.models import GridOpsAction, GridOpsObservation
from gridops.prompting import action_to_json, validate_reason_action_completion
from gridops.simulation.physics import GRID_MAX_KW
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import action_dict, optimize_action, rollout_candidate


CLEAN_REASONING_FORBIDDEN_TERMS = (
    "optimizer_delta",
    "model_delta",
    "candidate_delta",
    "lp_delta",
    "guard_reason",
    "selected_source",
    "tool-selected",
    "validated tool",
    "python",
    "traceback",
    "critic_reward",
    "regret",
    "_delta",
    "tool",
    "candidate",
    "optimizer",
    "critic",
)


def _clean_number(value: float, digits: int = 2) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == -0.0 else rounded


def _delta_dict(delta: dict[str, Any]) -> dict[str, float]:
    return {
        "cost": _clean_number(float(delta.get("cost", 0.0)), 4),
        "blackout_kwh": _clean_number(float(delta.get("blackout_kwh", 0.0)), 4),
        "diesel_kwh": _clean_number(float(delta.get("diesel_kwh", 0.0)), 4),
    }


def _validate_candidate(candidate_action: GridOpsAction | dict[str, Any]) -> GridOpsAction | None:
    if isinstance(candidate_action, GridOpsAction):
        return candidate_action
    if not isinstance(candidate_action, dict):
        return None
    try:
        return GridOpsAction.model_validate(candidate_action)
    except Exception:
        return None


def _obs_dict(observation: GridOpsObservation | dict[str, Any]) -> dict[str, Any]:
    if isinstance(observation, dict):
        return observation
    return observation.model_dump()


def _weighted_delta_gap(candidate_delta: dict[str, float], lp_delta: dict[str, float]) -> float:
    blackout_gap = candidate_delta["blackout_kwh"] - lp_delta["blackout_kwh"]
    cost_gap = (candidate_delta["cost"] - lp_delta["cost"]) / 1000.0
    diesel_gap = 0.15 * (candidate_delta["diesel_kwh"] - lp_delta["diesel_kwh"])
    return blackout_gap + cost_gap + diesel_gap


def _reason_for_choice(
    *,
    valid_candidate: bool,
    candidate_delta: dict[str, float],
    lp_delta: dict[str, float],
    task_id: str,
) -> tuple[str, bool]:
    """Return reason and whether to choose the candidate action."""

    if not valid_candidate:
        return "candidate_invalid", False

    blackout_margin = 0.02
    cost_margin = 15.0
    diesel_margin = 0.05

    if candidate_delta["blackout_kwh"] > lp_delta["blackout_kwh"] + blackout_margin:
        return "candidate_higher_blackout", False
    if candidate_delta["cost"] > lp_delta["cost"] + cost_margin:
        return "candidate_higher_cost", False
    if task_id == "task_3_crisis" and candidate_delta["diesel_kwh"] > lp_delta["diesel_kwh"] + diesel_margin:
        return "candidate_higher_diesel_use", False

    gap = _weighted_delta_gap(candidate_delta, lp_delta)
    if gap <= 0.01:
        return "candidate_matches_or_beats_lp", True
    return "candidate_lower_reward", False


def score_action_against_lp(
    env: Any,
    task_id: str,
    observation: GridOpsObservation | dict[str, Any],
    candidate_action: GridOpsAction | dict[str, Any],
    *,
    previous_outcome: dict[str, Any] | None = None,
    optimizer_horizon: int = 12,
    compare_horizon: int = 4,
) -> dict[str, Any]:
    """Compare a candidate action against the causal LP action.

    The supplied environment is deep-copied before any rollout, so callers can
    safely invoke this from data builders, API handlers, and tests without
    advancing live OpenEnv state.
    """

    candidate = _validate_candidate(candidate_action)
    obs_dict = _obs_dict(observation)
    lp_action, _ = optimize_action(
        obs_dict,
        task_id,
        previous_outcome=previous_outcome,
        horizon=optimizer_horizon,
    )

    lp_delta = _delta_dict(rollout_candidate(copy.deepcopy(env), task_id, lp_action, horizon=compare_horizon)["delta"])
    if candidate is None:
        candidate_delta = {"cost": 1_000_000_000.0, "blackout_kwh": 1_000_000_000.0, "diesel_kwh": 1_000_000_000.0}
        reason, choose_candidate = _reason_for_choice(
            valid_candidate=False,
            candidate_delta=candidate_delta,
            lp_delta=lp_delta,
            task_id=task_id,
        )
        chosen_action = lp_action
    else:
        candidate_delta = _delta_dict(
            rollout_candidate(copy.deepcopy(env), task_id, candidate, horizon=compare_horizon)["delta"]
        )
        reason, choose_candidate = _reason_for_choice(
            valid_candidate=True,
            candidate_delta=candidate_delta,
            lp_delta=lp_delta,
            task_id=task_id,
        )
        chosen_action = candidate if choose_candidate else lp_action

    regret = 0.0 if candidate is None else max(0.0, _weighted_delta_gap(candidate_delta, lp_delta))
    if candidate is None:
        critic_reward = -5.0
    elif reason == "candidate_matches_or_beats_lp":
        critic_reward = 1.0
    else:
        critic_reward = max(-5.0, -regret)

    return {
        "candidate_action": candidate_action if candidate is None else action_dict(candidate),
        "lp_action": action_dict(lp_action),
        "chosen_action": action_dict(chosen_action),
        "candidate_delta": candidate_delta,
        "lp_delta": lp_delta,
        "regret": _clean_number(regret, 6),
        "critic_reward": _clean_number(critic_reward, 6),
        "reason": reason,
        "chosen_source": "candidate" if choose_candidate else "lp",
    }


def _hour_context(observation: GridOpsObservation, task_id: str) -> str:
    hour = int(observation.hour)
    hour_of_day = hour % 24
    demand = observation.demand_kw
    solar = observation.solar_kw
    price = observation.grid_price
    outage_hours = set(TASKS[task_id].grid_outage_hours or [])
    outage = hour in outage_hours or (hour + 1) in outage_hours
    if hour_of_day < 6:
        window = "overnight"
    elif hour_of_day < 11:
        window = "morning ramp"
    elif hour_of_day < 16:
        window = "solar window"
    elif hour_of_day < 21:
        window = "evening peak"
    else:
        window = "late evening"
    outage_text = "with outage risk active" if outage else "with grid support available"
    return (
        f"time_context: Hour {hour} is in the {window}, demand is "
        f"{demand:.1f} kW, solar is {solar:.1f} kW, price is {price:.2f}, "
        f"and {task_id} is operating {outage_text}."
    )


def _action_decision_text(action: GridOpsAction) -> tuple[str, str]:
    battery = action.battery_dispatch
    diesel = action.diesel_dispatch
    shed = action.demand_shedding
    if battery > 0.05:
        battery_text = f"discharge the battery at {battery:.2f} kW"
    elif battery < -0.05:
        battery_text = f"charge the battery at {abs(battery):.2f} kW"
    else:
        battery_text = "hold the battery nearly flat"

    if diesel > 0.05:
        diesel_text = f"run diesel at {diesel:.2f} kW"
    else:
        diesel_text = "avoid diesel"

    if shed > 0.01:
        shed_text = f"shed {shed:.2f} kW only as a last-resort reliability action"
    else:
        shed_text = "avoid demand shedding"

    return battery_text, f"{battery_text}, {diesel_text}, and {shed_text}"


def build_clean_operator_completion(
    observation: GridOpsObservation,
    task_id: str,
    chosen_action: GridOpsAction | dict[str, Any],
    critic_result: dict[str, Any],
    *,
    previous_action: GridOpsAction | dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> str:
    """Create clean reasoning plus final action JSON, without tool logs."""

    action = _validate_candidate(chosen_action)
    if action is None:
        action = GridOpsAction.model_validate(critic_result["chosen_action"])

    battery_text, full_action_text = _action_decision_text(action)
    soc = observation.battery_soc
    fuel = observation.diesel_fuel_remaining
    previous_action_dict = action_dict(previous_action) if isinstance(previous_action, GridOpsAction) else (
        previous_action or {"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0}
    )
    previous_outcome = previous_outcome or {}
    previous_blackout = float(previous_outcome.get("blackout_kwh", 0.0))
    previous_grid = float(previous_outcome.get("grid_import_kw", previous_outcome.get("grid_kw", 0.0)))
    outage_hours = set(TASKS[task_id].grid_outage_hours or [])
    outage_now = int(observation.hour) in outage_hours
    outage_next = int(observation.hour) + 1 in outage_hours

    if outage_now:
        first_order = "1st_order: The grid is unavailable now, so reliability dominates price timing."
    elif outage_next:
        first_order = "1st_order: The next hour has outage risk, so the action must not waste stored energy."
    elif observation.solar_kw > observation.demand_kw * 0.65 and observation.grid_price < 0.26:
        first_order = "1st_order: Solar support is strong, so charging or holding energy can prepare for the later peak."
    elif observation.demand_kw > GRID_MAX_KW * 0.85:
        first_order = "1st_order: Demand is pressing against grid support, so stored energy should protect service quality."
    else:
        first_order = "1st_order: Current supply can cover most load, so the action should avoid unnecessary fuel and shedding."

    if task_id == "task_3_crisis" or outage_now or outage_next:
        second_order = (
            "2nd_order: Preserve enough SOC and fuel for outage continuity while using diesel only when it prevents blackout."
        )
    elif task_id == "task_2_heatwave":
        second_order = (
            "2nd_order: Heatwave load can rebound later, so the battery move should not create a deeper evening shortage."
        )
    else:
        second_order = (
            "2nd_order: Normal operation rewards low-cost timing, stable SOC, and avoiding needless generator starts."
        )

    previous_text = (
        "previous_action: The prior action was "
        f"battery {previous_action_dict.get('battery_dispatch', 0.0):.2f}, "
        f"diesel {previous_action_dict.get('diesel_dispatch', 0.0):.2f}, "
        f"shedding {previous_action_dict.get('demand_shedding', 0.0):.2f}; "
        f"previous blackout was {previous_blackout:.2f} kWh and grid import was {previous_grid:.2f} kW."
    )

    decision = (
        f"decision: With SOC at {soc:.2f} and fuel at {fuel:.2f}, {full_action_text}; "
        f"the battery stance is to {battery_text}."
    )

    think = "\n".join(
        [
            _hour_context(observation, task_id),
            first_order,
            second_order,
            previous_text,
            decision,
        ]
    )
    return f"<think>\n{think}\n</think>\n<action>\n{action_to_json(action)}\n</action>"


def _extract_think(completion: str) -> str | None:
    match = re.search(r"<think>\s*(.*?)\s*</think>", completion, flags=re.DOTALL)
    return match.group(1) if match else None


def validate_clean_reasoning_completion(completion: str) -> tuple[bool, str]:
    """Validate reason/action format and reject tool-log style reasoning."""

    valid, reason = validate_reason_action_completion(completion)
    if not valid:
        return False, reason
    think = _extract_think(completion)
    if think is None:
        return False, "missing_think"
    if "{" in think or "}" in think:
        return False, "dict_or_json_inside_think"
    lowered = think.lower()
    for term in CLEAN_REASONING_FORBIDDEN_TERMS:
        if term in lowered:
            return False, f"forbidden_clean_reasoning_term:{term}"
    return True, "ok"
