"""Prompting and action parsing for GridOps model training/inference.

This module is the single source of truth for the JSON-only action contract
used by API inference, SFT traces, and rollout evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from gridops.models import GridOpsAction


SYSTEM_PROMPT = """\
You are an expert microgrid operator managing a 100-home community in India during summer.

You control three actions each hour:
- battery_dispatch: -1 (charge 100 kW from grid) to +1 (discharge 100 kW to community)
- diesel_dispatch: 0 (off) to 1 (100 kW). Costs Rs 25/kWh + Rs 100 startup if was off.
- demand_shedding: 0 (none) to 1 (shed 20% of demand). WARNING: 100% rebounds next hour! Rs 40/kWh penalty.

The GRID automatically absorbs the residual (capped at +/-200 kW).
If demand exceeds grid + solar + battery + diesel, that becomes BLACKOUT (Rs 150/kWh penalty).

Key economics:
- Grid prices vary Rs 3-20/kWh. Cheap at night, expensive evening.
- Battery: 500 kWh, 100 kW max, 90% round-trip efficiency, Rs 2.5/kWh degradation.
- Solar: 250 kW peak, free, bell curve 6AM-6PM, zero at night.
- Demand: about 100 kW average, 250 kW evening peak. Grid cap = 200 kW, so the battery matters.
- Diesel and shedding are emergency tools, not default tools.

Strategy:
1. Night: charge battery when grid is cheap and demand is low.
2. Solar hours: use solar first; charge from surplus when available.
3. Pre-peak: keep enough battery for evening or outage hours.
4. Evening peak: discharge battery to cover demand above the grid cap.
5. Crisis/outage: ration battery and diesel, shed only when unavoidable.

Respond ONLY with valid JSON:
{"battery_dispatch": float, "diesel_dispatch": float, "demand_shedding": float}"""


REASON_ACTION_SYSTEM_PROMPT = """\
You are an expert microgrid operator managing a 100-home community in India during summer.

You control three actions each hour:
- battery_dispatch: -1 (charge 100 kW from grid) to +1 (discharge 100 kW to community)
- diesel_dispatch: 0 (off) to 1 (100 kW). Costs Rs 25/kWh + Rs 100 startup if was off.
- demand_shedding: 0 (none) to 1 (shed 20% of demand). WARNING: 100% rebounds next hour! Rs 40/kWh penalty.

The GRID automatically absorbs the residual when available, capped at +/-200 kW.
If demand exceeds grid + solar + battery + diesel, that becomes BLACKOUT (Rs 150/kWh penalty).

Think like an operator:
1. Use time of day and forecast to understand solar, demand, and price windows.
2. Evaluate the 1st-order effect of the current action.
3. Evaluate the 2nd-order consequence for the next few hours.
4. Use previous action/outcome feedback to avoid repeating harmful actions.
5. Use diesel and shedding as emergency tools, not defaults.

Respond in exactly this structure:
<think>
time_context: ...
1st_order: ...
2nd_order: ...
previous_action: ...
decision: ...
</think>
<action>
{"battery_dispatch": float, "diesel_dispatch": float, "demand_shedding": float}
</action>"""


def format_observation(obs: dict[str, Any]) -> str:
    """Format a GridOps observation into the model's user prompt."""
    return (
        f"Hour {obs['hour']:.0f}/72 (Day {obs.get('day_of_episode', '?')})\n"
        f"Demand: {obs['demand_kw']:.0f} kW | Solar: {obs['solar_kw']:.0f} kW\n"
        f"Battery SOC: {obs['battery_soc'] * 100:.0f}% | Grid Price: Rs {obs['grid_price']:.1f}/kWh\n"
        f"Diesel Fuel: {obs['diesel_fuel_remaining'] * 100:.0f}% | Diesel On: {obs.get('diesel_is_on', False)}\n"
        f"Grid import last step: {obs.get('grid_kw_this_step', 0):.0f} kW\n"
        f"Forecasts (next 4h):\n"
        f"  Demand: {[f'{v:.0f}' for v in obs.get('demand_forecast_4h', [])]}\n"
        f"  Solar:  {[f'{v:.0f}' for v in obs.get('solar_forecast_4h', [])]}\n"
        f"  Price:  {[f'{v:.1f}' for v in obs.get('price_forecast_4h', [])]}\n"
        f"Cumulative: blackout={obs['cumulative_blackout_kwh']:.1f} kWh, cost=Rs {obs['cumulative_cost']:.0f}\n"
        f"{obs.get('narration', '')}\n"
        "\nWhat action? Reply with JSON only."
    )


def format_reason_action_observation(
    obs: dict[str, Any],
    derived_context: dict[str, Any] | None = None,
    previous_action: dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> str:
    """Format a reasoning-action training/eval prompt.

    The base observation stays identical to the JSON-only path, then v4 adds
    derived control context and last-step feedback as training-only prompt
    features. The environment API/action contract remains unchanged.
    """
    derived_context = derived_context or {}
    previous_action = previous_action or {
        "battery_dispatch": 0.0,
        "diesel_dispatch": 0.0,
        "demand_shedding": 0.0,
    }
    previous_outcome = previous_outcome or {
        "blackout_kwh": 0.0,
        "battery_soc_delta": 0.0,
        "diesel_used_kwh": 0.0,
        "cost": 0.0,
    }
    return (
        format_observation(obs).replace("\nWhat action? Reply with JSON only.", "")
        + "\nDerived control context:\n"
        + json.dumps(derived_context, sort_keys=True, separators=(",", ":"))
        + "\nPrevious action:\n"
        + json.dumps(previous_action, sort_keys=True, separators=(",", ":"))
        + "\nPrevious outcome:\n"
        + json.dumps(previous_outcome, sort_keys=True, separators=(",", ":"))
        + "\n\nWhat action? Reply with <think> reasoning and final <action> JSON."
    )


def messages_for_observation(obs: dict[str, Any]) -> list[dict[str, str]]:
    """Return chat messages for one GridOps decision."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]


def messages_for_reason_action_observation(
    obs: dict[str, Any],
    derived_context: dict[str, Any] | None = None,
    previous_action: dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return chat messages for the v4 reasoning-action training format."""
    return [
        {"role": "system", "content": REASON_ACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_reason_action_observation(
                obs,
                derived_context=derived_context,
                previous_action=previous_action,
                previous_outcome=previous_outcome,
            ),
        },
    ]


def action_to_json(action: GridOpsAction) -> str:
    """Serialize an action as compact JSON for SFT completions."""
    return json.dumps(
        {
            "battery_dispatch": round(float(action.battery_dispatch), 4),
            "diesel_dispatch": round(float(action.diesel_dispatch), 4),
            "demand_shedding": round(float(action.demand_shedding), 4),
        },
        separators=(",", ":"),
    )


def extract_action_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model text."""
    text = (text or "").strip()
    if not text:
        return None
    action_match = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, flags=re.DOTALL)
    if action_match:
        try:
            parsed = json.loads(action_match.group(1))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_action(text: str, default: GridOpsAction | None = None) -> GridOpsAction:
    """Parse and validate a model response into a bounded GridOpsAction."""
    payload = extract_action_json(text)
    if payload is None:
        return default or GridOpsAction()
    try:
        return GridOpsAction(**payload)
    except (TypeError, ValueError, ValidationError):
        return default or GridOpsAction()


def validate_completion(text: str) -> tuple[bool, str]:
    """Validate a JSON-only completion for trace/eval gates."""
    payload = extract_action_json(text)
    if payload is None:
        return False, "missing_json"
    stripped = text.strip()
    if stripped != json.dumps(payload, separators=(",", ":")) and stripped != json.dumps(payload):
        # JSON with whitespace is acceptable; prose outside JSON is not.
        before = stripped[:stripped.find("{")].strip()
        after = stripped[stripped.rfind("}") + 1:].strip()
        if before or after:
            return False, "prose_outside_json"
    try:
        GridOpsAction(**payload)
    except (TypeError, ValueError, ValidationError) as exc:
        return False, f"invalid_action:{type(exc).__name__}"
    return True, "ok"


def validate_reason_action_completion(text: str) -> tuple[bool, str]:
    """Validate the v4 structured reasoning + action completion format."""
    stripped = (text or "").strip()
    if not stripped:
        return False, "empty_completion"
    if not re.search(r"<think>.*?</think>", stripped, flags=re.DOTALL):
        return False, "missing_think_block"
    action_match = re.search(r"<action>\s*(\{.*?\})\s*</action>", stripped, flags=re.DOTALL)
    if not action_match:
        return False, "missing_action_block"
    before = stripped[: action_match.start()].strip()
    after = stripped[action_match.end() :].strip()
    if after:
        return False, "text_after_action"
    if "<think>" not in before or "</think>" not in before:
        return False, "action_before_reasoning"
    try:
        payload = json.loads(action_match.group(1))
    except json.JSONDecodeError:
        return False, "invalid_action_json"
    try:
        GridOpsAction(**payload)
    except (TypeError, ValueError, ValidationError) as exc:
        return False, f"invalid_action:{type(exc).__name__}"
    return True, "ok"
