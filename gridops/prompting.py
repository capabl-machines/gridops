"""Prompting and action parsing for GridOps model training/inference.

This module is the single source of truth for the JSON-only action contract
used by API inference, SFT traces, and rollout evaluation.
"""

from __future__ import annotations

import json
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


def messages_for_observation(obs: dict[str, Any]) -> list[dict[str, str]]:
    """Return chat messages for one GridOps decision."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
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
