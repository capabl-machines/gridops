"""Runtime optimizer, validator, and guarded planner for GridOps.

The full-episode LP oracle remains an offline ceiling. This module exposes a
causal short-horizon optimizer that only uses the current observation, short
forecasts, task rules, and previous feedback, so it is suitable as a runtime
tool for the demo/API path.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ValidationError
from scipy.optimize import linprog

from gridops.models import GridOpsAction
from gridops.policies import oracle_policy
from gridops.prompting import (
    action_to_json,
    extract_action_json,
    messages_for_reason_action_observation,
    validate_reason_action_completion,
)
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
from gridops.strategy import plan_strategy_action


LP_VARS = ["imp", "exp", "ch", "dis", "diesel", "shed", "blackout", "curtail"]
DEFAULT_OPTIMIZER_HORIZON = 12
DEFAULT_COMPARE_HORIZON = 4


def action_dict(action: GridOpsAction) -> dict[str, float]:
    """Serialize an action with stable compact floats."""
    def clean(value: float) -> float:
        rounded = round(float(value), 4)
        return 0.0 if abs(rounded) < 0.00005 else rounded

    return {
        "battery_dispatch": clean(action.battery_dispatch),
        "diesel_dispatch": clean(action.diesel_dispatch),
        "demand_shedding": clean(action.demand_shedding),
    }


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


def previous_outcome_from_observation(obs: dict[str, Any] | None) -> dict[str, float]:
    """Return compact last-step feedback from an observation."""
    if not obs:
        return {
            "blackout_kwh": 0.0,
            "battery_soc_delta": 0.0,
            "diesel_used_kwh": 0.0,
            "shed_kwh": 0.0,
            "grid_kw": 0.0,
            "cost": 0.0,
        }
    return {
        "blackout_kwh": round(float(obs.get("blackout_this_step", 0.0)), 4),
        "battery_soc_delta": 0.0,
        "diesel_used_kwh": round(float(obs.get("flow_diesel", 0.0)), 4),
        "shed_kwh": round(float(obs.get("flow_shed", 0.0)), 4),
        "grid_kw": round(float(obs.get("grid_kw_this_step", 0.0)), 4),
        "cost": round(float(obs.get("cost_this_step", 0.0)), 4),
    }


def derive_control_context(obs: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Small inference-time context block for the model/tool trace."""
    hour = int(obs.get("hour", 0))
    hour_of_day = (hour + 6) % 24
    outage_hours = set(TASKS[task_id].grid_outage_hours or [])
    return {
        "task_id": task_id,
        "hour_of_day": hour_of_day,
        "is_outage_now": hour in outage_hours,
        "outage_in_next_4h": any((hour + i) in outage_hours for i in range(1, 5)),
        "soc_pct": round(float(obs.get("battery_soc", 0.0)) * 100, 2),
        "diesel_fuel_pct": round(float(obs.get("diesel_fuel_remaining", 0.0)) * 100, 2),
        "net_load_kw": round(float(obs.get("demand_kw", 0.0)) - float(obs.get("solar_kw", 0.0)), 2),
        "price": round(float(obs.get("grid_price", 0.0)), 2),
    }


def optimize_action(
    obs: dict[str, Any],
    task_id: str,
    previous_outcome: dict[str, Any] | None = None,
    horizon: int = DEFAULT_OPTIMIZER_HORIZON,
    blackout_weight: float = 2.0,
    diesel_green_weight: float = 8.0,
    soc_deficit_weight: float = 18.0,
    fuel_deficit_weight: float = 8.0,
) -> tuple[GridOpsAction, dict[str, Any]]:
    """Return the first action from a short-horizon causal LP optimizer."""
    horizon = max(1, min(int(horizon), DEFAULT_OPTIMIZER_HORIZON))
    if task_id not in TASKS:
        task_id = "task_1_normal"
    demand, solar, price = _series_from_observation(obs, horizon)
    hour = int(obs["hour"])
    outage_hours = set(TASKS[task_id].grid_outage_hours or [])
    initial_soc = float(obs["battery_soc"]) * BATTERY_CAPACITY_KWH
    initial_fuel = float(obs["diesel_fuel_remaining"]) * DIESEL_TANK_KWH
    initial_rebound = max(0.0, float((previous_outcome or {}).get("shed_kwh", 0.0)))
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
            "tool": "causal_lp_optimizer",
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
    return action, {
        "tool": "causal_lp_optimizer",
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


def validate_action_payload(payload: Any) -> dict[str, Any]:
    """Validate a dict or completion string into a GridOpsAction."""
    raw_payload = extract_action_json(payload) if isinstance(payload, str) else payload
    if not isinstance(raw_payload, dict):
        return {
            "valid": False,
            "reason": "missing_action_json",
            "action": action_dict(GridOpsAction()),
            "raw_action": raw_payload,
        }
    try:
        action = GridOpsAction(**raw_payload)
    except (TypeError, ValueError, ValidationError) as exc:
        return {
            "valid": False,
            "reason": f"invalid_action:{type(exc).__name__}",
            "action": action_dict(GridOpsAction()),
            "raw_action": raw_payload,
            "errors": exc.errors() if hasattr(exc, "errors") else str(exc),
        }
    return {
        "valid": True,
        "reason": "ok",
        "action": action_dict(action),
        "raw_action": raw_payload,
    }


def _env_metrics(env: GridOpsEnvironment) -> dict[str, float]:
    micro = env._micro  # noqa: SLF001 - guarded rollout diagnostics.
    return {
        "hour": float(micro.hour),
        "cost": float(micro.cumulative_cost),
        "blackout_kwh": float(micro.cumulative_blackout_kwh),
        "diesel_kwh": float(micro.cumulative_diesel_kwh),
        "battery_throughput_kwh": float(micro.cumulative_battery_throughput_kwh),
    }


def rollout_candidate(
    env: GridOpsEnvironment,
    task_id: str,
    first_action: GridOpsAction,
    horizon: int = DEFAULT_COMPARE_HORIZON,
) -> dict[str, Any]:
    """Simulate one candidate, then let the optimizer handle the remaining horizon."""
    sim_env = copy.deepcopy(env)
    before = _env_metrics(sim_env)
    obs = sim_env.step(first_action)
    actions = [action_dict(first_action)]
    rewards = [float(obs.reward)]
    previous_outcome = previous_outcome_from_observation(obs.model_dump())
    for _ in range(max(0, int(horizon) - 1)):
        if obs.done:
            break
        next_obs = obs.model_dump()
        action, _ = optimize_action(next_obs, task_id, previous_outcome=previous_outcome)
        obs = sim_env.step(action)
        actions.append(action_dict(action))
        rewards.append(float(obs.reward))
        previous_outcome = previous_outcome_from_observation(obs.model_dump())

    after = _env_metrics(sim_env)
    delta = {
        "cost": round(after["cost"] - before["cost"], 4),
        "blackout_kwh": round(after["blackout_kwh"] - before["blackout_kwh"], 4),
        "diesel_kwh": round(after["diesel_kwh"] - before["diesel_kwh"], 4),
        "battery_throughput_kwh": round(after["battery_throughput_kwh"] - before["battery_throughput_kwh"], 4),
        "reward_sum": round(sum(rewards), 6),
    }
    return {
        "horizon": max(1, int(horizon)),
        "first_action": action_dict(first_action),
        "actions": actions,
        "delta": delta,
        "done": bool(obs.done),
    }


def compare_candidates(
    env: GridOpsEnvironment,
    task_id: str,
    candidates: dict[str, GridOpsAction],
    horizon: int = DEFAULT_COMPARE_HORIZON,
) -> dict[str, Any]:
    """Roll out candidates from the current environment state."""
    rollouts = {
        name: rollout_candidate(env, task_id, action, horizon=horizon)
        for name, action in candidates.items()
    }
    return {"horizon": max(1, int(horizon)), "candidates": rollouts}


def _materially_worse(model_delta: dict[str, float], optimizer_delta: dict[str, float], task_id: str) -> tuple[bool, str]:
    blackout_margin = 0.25 if task_id == "task_3_crisis" else 1.0
    cost_margin = 250.0 if task_id == "task_3_crisis" else 750.0
    if model_delta["blackout_kwh"] > optimizer_delta["blackout_kwh"] + blackout_margin:
        return True, "model_candidate_higher_blackout"
    if model_delta["cost"] > optimizer_delta["cost"] + cost_margin:
        return True, "model_candidate_higher_cost"
    return False, "model_candidate_similar"


def _materially_better(model_delta: dict[str, float], optimizer_delta: dict[str, float], task_id: str) -> tuple[bool, str]:
    blackout_margin = 0.05 if task_id == "task_3_crisis" else 0.25
    cost_margin = 100.0 if task_id == "task_3_crisis" else 250.0
    if model_delta["blackout_kwh"] + blackout_margin < optimizer_delta["blackout_kwh"]:
        return True, "model_candidate_lower_blackout"
    if task_id == "task_3_crisis":
        return False, "optimizer_default_in_crisis_without_blackout_gain"
    if (
        model_delta["blackout_kwh"] <= optimizer_delta["blackout_kwh"] + blackout_margin
        and model_delta["cost"] + cost_margin < optimizer_delta["cost"]
    ):
        return True, "model_candidate_lower_cost"
    return False, "optimizer_default_when_model_not_better"


def select_guarded_action(
    *,
    model_action: GridOpsAction | None,
    model_valid: bool,
    optimizer_action: GridOpsAction,
    comparison: dict[str, Any] | None,
    task_id: str,
) -> dict[str, Any]:
    """Choose the final action under the safety-first hybrid policy."""
    if model_action is None or not model_valid:
        return {
            "selected_source": "optimizer",
            "selected_action": action_dict(optimizer_action),
            "reason": "model_candidate_invalid_or_missing",
        }
    if not comparison or "model" not in comparison.get("candidates", {}):
        return {
            "selected_source": "optimizer",
            "selected_action": action_dict(optimizer_action),
            "reason": "comparison_unavailable",
        }

    model_delta = comparison["candidates"]["model"]["delta"]
    optimizer_delta = comparison["candidates"]["optimizer"]["delta"]
    worse, reason = _materially_worse(model_delta, optimizer_delta, task_id)
    if worse:
        return {
            "selected_source": "optimizer",
            "selected_action": action_dict(optimizer_action),
            "reason": reason,
        }
    better, reason = _materially_better(model_delta, optimizer_delta, task_id)
    if not better:
        return {
            "selected_source": "optimizer",
            "selected_action": action_dict(optimizer_action),
            "reason": reason,
        }
    return {
        "selected_source": "model",
        "selected_action": action_dict(model_action),
        "reason": reason,
    }


def _configured_llm_client() -> tuple[Any, str] | None:
    model = os.environ.get("GRIDOPS_LLM_MODEL")
    api_key = os.environ.get("GRIDOPS_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not model or not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    base_url = os.environ.get("GRIDOPS_LLM_BASE_URL")
    if not base_url and os.environ.get("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), model


def call_configured_llm(
    obs: dict[str, Any],
    task_id: str,
    previous_action: dict[str, Any] | None,
    previous_outcome: dict[str, Any] | None,
    max_tokens: int = 220,
) -> dict[str, Any]:
    """Call an optional OpenAI-compatible endpoint for a model candidate."""
    configured = _configured_llm_client()
    if configured is None:
        return {"available": False, "reason": "llm_not_configured"}
    client, model = configured
    messages = messages_for_reason_action_observation(
        obs,
        derive_control_context(obs, task_id),
        previous_action,
        previous_outcome,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return {"available": False, "reason": f"llm_call_failed:{type(exc).__name__}"}
    text = response.choices[0].message.content or ""
    valid, reason = validate_reason_action_completion(text)
    return {
        "available": True,
        "model": model,
        "completion": text,
        "validation": validate_action_payload(text),
        "format_valid": valid,
        "format_reason": reason,
    }


@dataclass
class PlanInputs:
    task_id: str
    observation: dict[str, Any]
    previous_action: dict[str, Any] | None = None
    previous_outcome: dict[str, Any] | None = None
    model_action: dict[str, Any] | None = None
    model_completion: str | None = None
    strategy: dict[str, Any] | str | None = None
    use_llm: bool = False
    optimizer_horizon: int = DEFAULT_OPTIMIZER_HORIZON
    compare_horizon: int = DEFAULT_COMPARE_HORIZON


def plan_action(env: GridOpsEnvironment, inputs: PlanInputs) -> dict[str, Any]:
    """Plan one action from the current environment state."""
    task_id = inputs.task_id if inputs.task_id in TASKS else "task_1_normal"
    previous_outcome = inputs.previous_outcome or previous_outcome_from_observation(inputs.observation)
    strategy_plan = plan_strategy_action(
        env,
        task_id,
        inputs.observation,
        previous_outcome=previous_outcome,
        strategy=inputs.strategy,
        optimizer_horizon=inputs.optimizer_horizon,
    )
    optimizer_action = GridOpsAction(**strategy_plan["action"])
    optimizer_info = strategy_plan["optimizer_info"]

    llm_result: dict[str, Any] = {"available": False, "reason": "not_requested"}
    raw_model_candidate: Any = inputs.model_action
    if inputs.model_completion:
        raw_model_candidate = inputs.model_completion
    elif inputs.use_llm:
        llm_result = call_configured_llm(
            inputs.observation,
            task_id,
            inputs.previous_action,
            previous_outcome,
        )
        if llm_result.get("available"):
            raw_model_candidate = llm_result.get("completion")

    if raw_model_candidate is None:
        model_validation = {
            "valid": False,
            "reason": "missing_model_candidate",
            "action": action_dict(GridOpsAction()),
        }
    else:
        model_validation = validate_action_payload(raw_model_candidate)
    model_action = GridOpsAction(**model_validation["action"]) if model_validation["valid"] else None
    candidates = {"optimizer": optimizer_action}
    if model_action is not None:
        candidates["model"] = model_action
    comparison = compare_candidates(env, task_id, candidates, horizon=inputs.compare_horizon)
    selection = select_guarded_action(
        model_action=model_action,
        model_valid=bool(model_validation["valid"]),
        optimizer_action=optimizer_action,
        comparison=comparison,
        task_id=task_id,
    )
    return {
        "task_id": task_id,
        "hour": int(float(inputs.observation.get("hour", 0))),
        "selected_action": selection["selected_action"],
        "selected_source": selection["selected_source"],
        "selection_reason": selection["reason"],
        "model_candidate": {
            "validation": model_validation,
            "llm": llm_result,
        },
        "strategy_candidate": {
            "strategy": strategy_plan["strategy"],
            "source": strategy_plan["strategy_source"],
            "validation": strategy_plan["strategy_validation"],
        },
        "optimizer_candidate": {
            "action": action_dict(optimizer_action),
            "info": optimizer_info,
        },
        "optimizer_config": strategy_plan["optimizer_config"],
        "comparison": comparison,
    }


def tool_corrected_completion(
    *,
    obs: dict[str, Any],
    task_id: str,
    plan: dict[str, Any],
    previous_action: dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> str:
    """Build a compact reason-action completion from a tool-selected plan."""
    context = derive_control_context(obs, task_id)
    selected_action = GridOpsAction(**action_dict(GridOpsAction(**plan["selected_action"])))
    selected = plan.get("selected_source", "optimizer")
    reason = plan.get("selection_reason", "tool_selected")
    optimizer_delta = (
        plan.get("comparison", {})
        .get("candidates", {})
        .get("optimizer", {})
        .get("delta", {})
    )
    model_delta = (
        plan.get("comparison", {})
        .get("candidates", {})
        .get("model", {})
        .get("delta", {})
    )
    previous_action = previous_action or {
        "battery_dispatch": 0.0,
        "diesel_dispatch": 0.0,
        "demand_shedding": 0.0,
    }
    previous_outcome = previous_outcome or previous_outcome_from_observation(None)
    return (
        "<think>\n"
        f"time_context: hour {int(float(obs.get('hour', 0)))}; "
        f"hour_of_day {context['hour_of_day']}; outage_now={context['is_outage_now']}; "
        f"outage_in_next_4h={context['outage_in_next_4h']}; soc={context['soc_pct']:.1f}%.\n"
        f"1st_order: selected {selected} action {action_dict(selected_action)} to balance net_load={context['net_load_kw']:.1f} kW.\n"
        f"2nd_order: optimizer_delta={optimizer_delta}; model_delta={model_delta}; guard_reason={reason}.\n"
        f"previous_action: {previous_action}; previous_outcome: {previous_outcome}.\n"
        "decision: follow the validated tool-selected action and keep the final answer as bounded JSON.\n"
        "</think>\n"
        "<action>\n"
        f"{action_to_json(selected_action)}\n"
        "</action>"
    )
