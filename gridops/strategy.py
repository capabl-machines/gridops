"""Strategy-level planning for GridOps.

The strategy layer is intentionally above the OpenEnv action contract. It lets
an LLM or deterministic teacher choose operating intent, while the causal
optimizer remains responsible for exact dispatch floats.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from gridops.models import GridOpsAction
from gridops.prompting import format_observation
from gridops.tasks.definitions import TASKS


StrategyMode = Literal[
    "cost_saving",
    "peak_shaving",
    "outage_prepare",
    "reliability",
    "recovery",
    "fuel_conservation",
]
RiskLevel = Literal["low", "medium", "high", "critical"]
BatteryBias = Literal["charge", "preserve", "discharge", "neutral"]
DieselPolicy = Literal["avoid", "allow_if_blackout", "prewarm", "conserve"]
SheddingPolicy = Literal["never", "last_resort"]


class GridOpsStrategy(BaseModel):
    """High-level operating intent for a GridOps controller."""

    mode: StrategyMode
    risk_level: RiskLevel
    battery_bias: BatteryBias
    diesel_policy: DieselPolicy
    shedding_policy: SheddingPolicy


DEFAULT_STRATEGY = GridOpsStrategy(
    mode="cost_saving",
    risk_level="low",
    battery_bias="neutral",
    diesel_policy="avoid",
    shedding_policy="never",
)

STRATEGY_SYSTEM_PROMPT = """\
You are a microgrid strategy selector.

Read the GridOps observation and choose the high-level operating strategy.
Do not output dispatch floats. Do not output prose. Do not use tools.

Respond ONLY with strict JSON:
{"mode":"cost_saving|peak_shaving|outage_prepare|reliability|recovery|fuel_conservation","risk_level":"low|medium|high|critical","battery_bias":"charge|preserve|discharge|neutral","diesel_policy":"avoid|allow_if_blackout|prewarm|conserve","shedding_policy":"never|last_resort"}"""


def strategy_dict(strategy: GridOpsStrategy) -> dict[str, str]:
    return strategy.model_dump()


def strategy_to_json(strategy: GridOpsStrategy) -> str:
    return json.dumps(strategy_dict(strategy), sort_keys=True, separators=(",", ":"))


def extract_strategy_json(text: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    stripped = (text or "").strip()
    if not stripped:
        return None
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def validate_strategy_payload(payload: Any) -> dict[str, Any]:
    raw_payload = extract_strategy_json(payload)
    if not isinstance(raw_payload, dict):
        return {
            "valid": False,
            "reason": "missing_strategy_json",
            "strategy": strategy_dict(DEFAULT_STRATEGY),
            "raw_strategy": raw_payload,
        }
    try:
        strategy = GridOpsStrategy.model_validate(raw_payload)
    except ValidationError as exc:
        return {
            "valid": False,
            "reason": "invalid_strategy:ValidationError",
            "strategy": strategy_dict(DEFAULT_STRATEGY),
            "raw_strategy": raw_payload,
            "errors": exc.errors(),
        }
    return {
        "valid": True,
        "reason": "ok",
        "strategy": strategy_dict(strategy),
        "raw_strategy": raw_payload,
    }


def parse_strategy(text: str | dict[str, Any] | None, default: GridOpsStrategy | None = None) -> GridOpsStrategy:
    validation = validate_strategy_payload(text)
    if not validation["valid"]:
        return default or DEFAULT_STRATEGY
    return GridOpsStrategy.model_validate(validation["strategy"])


def validate_strategy_completion(text: str) -> tuple[bool, str]:
    stripped = (text or "").strip()
    validation = validate_strategy_payload(stripped)
    if not validation["valid"]:
        return False, validation["reason"]
    expected_compact = strategy_to_json(GridOpsStrategy.model_validate(validation["strategy"]))
    if stripped != expected_compact and stripped != json.dumps(validation["strategy"], sort_keys=True):
        before = stripped[: stripped.find("{")].strip()
        after = stripped[stripped.rfind("}") + 1 :].strip()
        if before or after:
            return False, "prose_outside_strategy_json"
    return True, "ok"


def _hour_of_day(obs: dict[str, Any]) -> int:
    return (int(float(obs.get("hour", 0))) + 6) % 24


def _outage_flags(obs: dict[str, Any], task_id: str) -> tuple[bool, bool]:
    hour = int(float(obs.get("hour", 0)))
    outage_hours = set(TASKS[task_id].grid_outage_hours or [])
    outage_now = hour in outage_hours
    outage_soon = any(hour + i in outage_hours for i in range(1, 5))
    return outage_now, outage_soon


def derive_strategy(
    obs: dict[str, Any],
    task_id: str,
    previous_outcome: dict[str, Any] | None = None,
) -> GridOpsStrategy:
    """Deterministically label an operating strategy from visible state."""

    if task_id not in TASKS:
        task_id = "task_1_normal"
    previous_outcome = previous_outcome or {}
    hod = _hour_of_day(obs)
    demand = float(obs.get("demand_kw", 0.0))
    solar = float(obs.get("solar_kw", 0.0))
    price = float(obs.get("grid_price", 0.0))
    soc = float(obs.get("battery_soc", 0.0))
    fuel = float(obs.get("diesel_fuel_remaining", 0.0))
    net_load = demand - solar
    prev_blackout = float(previous_outcome.get("blackout_kwh", 0.0))
    outage_now, outage_soon = _outage_flags(obs, task_id)

    if prev_blackout > 0.05 or soc < 0.18:
        return GridOpsStrategy(
            mode="recovery",
            risk_level="critical" if prev_blackout > 1.0 else "high",
            battery_bias="charge" if not outage_now else "preserve",
            diesel_policy="allow_if_blackout" if fuel > 0.12 else "conserve",
            shedding_policy="last_resort",
        )

    if fuel < 0.15 and (task_id == "task_3_crisis" or outage_now or outage_soon):
        return GridOpsStrategy(
            mode="fuel_conservation",
            risk_level="high",
            battery_bias="discharge" if outage_now and soc > 0.3 else "preserve",
            diesel_policy="conserve",
            shedding_policy="last_resort",
        )

    if outage_now:
        return GridOpsStrategy(
            mode="reliability",
            risk_level="critical",
            battery_bias="discharge" if soc > 0.25 else "preserve",
            diesel_policy="allow_if_blackout" if fuel > 0.08 else "conserve",
            shedding_policy="last_resort",
        )

    if outage_soon:
        return GridOpsStrategy(
            mode="outage_prepare",
            risk_level="high",
            battery_bias="charge" if soc < 0.82 else "preserve",
            diesel_policy="conserve" if fuel < 0.3 else "allow_if_blackout",
            shedding_policy="last_resort",
        )

    if 18 <= hod <= 22 and (price >= 10.0 or net_load >= 150.0):
        return GridOpsStrategy(
            mode="peak_shaving",
            risk_level="medium" if soc >= 0.35 else "high",
            battery_bias="discharge" if soc > 0.25 else "preserve",
            diesel_policy="avoid",
            shedding_policy="never",
        )

    if price <= 7.5 and soc < 0.85:
        return GridOpsStrategy(
            mode="cost_saving",
            risk_level="low",
            battery_bias="charge",
            diesel_policy="avoid",
            shedding_policy="never",
        )

    if net_load >= 190.0 and soc > 0.4:
        return GridOpsStrategy(
            mode="peak_shaving",
            risk_level="medium",
            battery_bias="discharge",
            diesel_policy="avoid",
            shedding_policy="never",
        )

    return GridOpsStrategy(
        mode="cost_saving",
        risk_level="low",
        battery_bias="neutral",
        diesel_policy="avoid",
        shedding_policy="never",
    )


def strategy_to_optimizer_config(
    strategy: GridOpsStrategy,
    obs: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    """Map strategy intent into causal LP optimizer parameters."""

    config: dict[str, Any] = {
        "horizon": 12,
        "blackout_weight": 2.0,
        "diesel_green_weight": 8.0,
        "soc_deficit_weight": 18.0,
        "fuel_deficit_weight": 8.0,
        "shedding_policy": strategy.shedding_policy,
    }
    mode_updates = {
        "cost_saving": {"blackout_weight": 2.0, "diesel_green_weight": 12.0, "soc_deficit_weight": 20.0},
        "peak_shaving": {"blackout_weight": 2.5, "diesel_green_weight": 10.0, "soc_deficit_weight": 14.0},
        "outage_prepare": {"blackout_weight": 3.0, "diesel_green_weight": 8.0, "soc_deficit_weight": 32.0, "fuel_deficit_weight": 16.0},
        "reliability": {"blackout_weight": 5.0, "diesel_green_weight": 2.0, "soc_deficit_weight": 20.0, "fuel_deficit_weight": 6.0},
        "recovery": {"blackout_weight": 5.5, "diesel_green_weight": 4.0, "soc_deficit_weight": 28.0, "fuel_deficit_weight": 8.0},
        "fuel_conservation": {"blackout_weight": 3.5, "diesel_green_weight": 20.0, "soc_deficit_weight": 20.0, "fuel_deficit_weight": 28.0},
    }
    config.update(mode_updates[strategy.mode])

    risk_blackout_adjust = {"low": -0.25, "medium": 0.0, "high": 1.0, "critical": 2.0}
    config["blackout_weight"] += risk_blackout_adjust[strategy.risk_level]

    if strategy.battery_bias == "charge":
        config["soc_deficit_weight"] += 8.0
    elif strategy.battery_bias == "preserve":
        config["soc_deficit_weight"] += 5.0
    elif strategy.battery_bias == "discharge":
        config["soc_deficit_weight"] = max(4.0, config["soc_deficit_weight"] - 6.0)

    if strategy.diesel_policy == "avoid":
        config["diesel_green_weight"] += 6.0
    elif strategy.diesel_policy == "allow_if_blackout":
        config["diesel_green_weight"] = max(1.0, config["diesel_green_weight"] - 4.0)
    elif strategy.diesel_policy == "prewarm":
        config["diesel_green_weight"] = max(1.0, config["diesel_green_weight"] - 6.0)
    elif strategy.diesel_policy == "conserve":
        config["diesel_green_weight"] += 8.0
        config["fuel_deficit_weight"] += 8.0

    # Keep optimizer parameters inside conservative numeric bounds.
    config["horizon"] = max(1, min(int(config["horizon"]), 12))
    for key in ["blackout_weight", "diesel_green_weight", "soc_deficit_weight", "fuel_deficit_weight"]:
        config[key] = round(float(max(0.5, min(config[key], 60.0))), 4)
    config["strategy"] = strategy_dict(strategy)
    config["task_id"] = task_id
    config["hour"] = int(float(obs.get("hour", 0)))
    return config


def format_strategy_observation(
    obs: dict[str, Any],
    derived_context: dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> str:
    derived_context = derived_context or {}
    previous_outcome = previous_outcome or {}
    return (
        format_observation(obs).replace("\nWhat action? Reply with JSON only.", "")
        + "\nDerived control context:\n"
        + json.dumps(derived_context, sort_keys=True, separators=(",", ":"))
        + "\nPrevious outcome:\n"
        + json.dumps(previous_outcome, sort_keys=True, separators=(",", ":"))
        + "\n\nWhat strategy? Reply with strategy JSON only."
    )


def messages_for_strategy_observation(
    obs: dict[str, Any],
    derived_context: dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
        {"role": "user", "content": format_strategy_observation(obs, derived_context, previous_outcome)},
    ]


def plan_strategy_action(
    env: Any,
    task_id: str,
    observation: dict[str, Any],
    previous_outcome: dict[str, Any] | None = None,
    strategy: dict[str, Any] | str | GridOpsStrategy | None = None,
    optimizer_horizon: int | None = None,
) -> dict[str, Any]:
    """Plan an action by selecting/validating strategy then invoking LP.

    The environment argument is accepted for parity with other planner APIs; it
    is not stepped or mutated.
    """

    del env
    if task_id not in TASKS:
        task_id = "task_1_normal"
    previous_outcome = previous_outcome or {}
    if isinstance(strategy, GridOpsStrategy):
        validation = {"valid": True, "reason": "ok", "strategy": strategy_dict(strategy), "raw_strategy": strategy_dict(strategy)}
        chosen_strategy = strategy
        source = "provided"
    elif strategy is not None:
        validation = validate_strategy_payload(strategy)
        if validation["valid"]:
            chosen_strategy = GridOpsStrategy.model_validate(validation["strategy"])
            source = "provided"
        else:
            chosen_strategy = derive_strategy(observation, task_id, previous_outcome)
            source = "derived_fallback"
    else:
        validation = {
            "valid": False,
            "reason": "missing_strategy",
            "strategy": strategy_dict(DEFAULT_STRATEGY),
            "raw_strategy": None,
        }
        chosen_strategy = derive_strategy(observation, task_id, previous_outcome)
        source = "derived"

    config = strategy_to_optimizer_config(chosen_strategy, observation, task_id)
    if optimizer_horizon is not None:
        config["horizon"] = max(1, min(int(optimizer_horizon), 12))
    from gridops.tool_agent import action_dict, optimize_action  # Local import avoids circular dependency.

    action, info = optimize_action(
        observation,
        task_id,
        previous_outcome=previous_outcome,
        horizon=config["horizon"],
        blackout_weight=config["blackout_weight"],
        diesel_green_weight=config["diesel_green_weight"],
        soc_deficit_weight=config["soc_deficit_weight"],
        fuel_deficit_weight=config["fuel_deficit_weight"],
    )
    return {
        "strategy": strategy_dict(chosen_strategy),
        "strategy_source": source,
        "strategy_validation": validation,
        "optimizer_config": config,
        "action": action_dict(action),
        "optimizer_info": info,
    }
