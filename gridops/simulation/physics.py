"""
Energy balance engine — the core physics of the microgrid.

KEY DESIGN (per Gemini review):
  - Agent controls: battery dispatch, diesel, demand shedding
  - Grid is the SLACK variable (absorbs residual, capped at ±200 kW)
  - VoLL penalty (Rs 150/kWh) replaces hard reliability gate
  - Battery degradation cost (Rs 2.5/kWh throughput)
  - Diesel startup cost (Rs 100 if was off last step)
  - Demand shedding rebound (50% of shed kWh added to next hour)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────

BATTERY_CAPACITY_KWH = 500.0
BATTERY_MAX_POWER_KW = 100.0
BATTERY_EFFICIENCY = 0.90          # round-trip (applied as √0.9 each way)
BATTERY_CHARGE_EFF = 0.949         # √0.90 ≈ 0.949
BATTERY_DISCHARGE_EFF = 0.949      # agent gets 94.9% of what battery releases
BATTERY_DEGRADATION_RS = 2.5       # Rs per kWh of throughput (charge or discharge)
GRID_MAX_KW = 200.0
DIESEL_MAX_KW = 100.0
DIESEL_COST_PER_KWH = 25.0
DIESEL_STARTUP_COST = 100.0        # Rs, one-time when turning on from off
DEMAND_SHED_MAX_FRAC = 0.20
SHED_REBOUND_FRAC = 1.00           # 100% of shed energy rebounds next hour (deferred, not destroyed)
DIESEL_TANK_KWH = 2400.0           # total fuel capacity
VOLL = 150.0                       # Value of Lost Load (Rs/kWh)
DT = 1.0                           # 1 hour per step


@dataclass
class MicrogridState:
    """Mutable internal state of the microgrid."""

    hour: int = 0
    battery_soc_kwh: float = 250.0   # start half-charged
    diesel_fuel_kwh: float = 2400.0
    diesel_was_on: bool = False       # for startup cost
    shed_rebound_kwh: float = 0.0     # deferred load from previous shedding
    cumulative_cost: float = 0.0
    cumulative_blackout_kwh: float = 0.0
    cumulative_diesel_kwh: float = 0.0
    cumulative_battery_throughput_kwh: float = 0.0
    total_demand_kwh: float = 0.0

    # Per-step bookkeeping
    last_blackout_kwh: float = 0.0
    last_cost: float = 0.0
    last_reward: float = 0.0
    last_grid_kw: float = 0.0


@dataclass
class StepFlows:
    """Detailed energy flows for one step (all in kW)."""

    # Supply side (positive = providing power to the bus)
    solar_kw: float = 0.0
    grid_import_kw: float = 0.0       # grid importing INTO community
    battery_discharge_kw: float = 0.0  # power delivered from battery (after efficiency loss)
    diesel_kw: float = 0.0

    # Demand side (positive = consuming power from the bus)
    effective_demand_kw: float = 0.0   # demand after shedding + rebound
    grid_export_kw: float = 0.0       # surplus exported to grid
    battery_charge_kw: float = 0.0    # power consumed to charge battery (before efficiency)
    blackout_kw: float = 0.0          # unmet demand
    curtailed_kw: float = 0.0         # excess supply that goes nowhere

    # Derived
    total_supply_kw: float = 0.0
    total_consumption_kw: float = 0.0
    shed_kw: float = 0.0              # how much was shed
    rebound_kw: float = 0.0           # how much rebounded from last step


@dataclass
class StepResult:
    """What physics.step() returns to the environment."""

    state: MicrogridState
    reward: float
    done: bool
    narration: str
    flows: StepFlows = None


def step(
    state: MicrogridState,
    battery_dispatch_norm: float,
    diesel_norm: float,
    shed_norm: float,
    solar_kw: float,
    demand_kw: float,
    grid_price: float,
    diesel_fuel_cap: float = DIESEL_TANK_KWH,
    grid_available: bool = True,
) -> StepResult:
    """
    Advance the microgrid by one hour.

    Actions (agent controls):
      battery_dispatch_norm: -1 (charge 100kW) to +1 (discharge 100kW)
      diesel_norm:           0 (off) to 1 (100kW)
      shed_norm:             0 (none) to 1 (shed 20%)

    Grid is the SLACK — absorbs residual up to ±200 kW.
    """
    # ── Scale actions ────────────────────────────────────────────────
    battery_cmd_kw = float(np.clip(battery_dispatch_norm, -1, 1)) * BATTERY_MAX_POWER_KW
    diesel_kw = float(np.clip(diesel_norm, 0, 1)) * DIESEL_MAX_KW
    shed_frac = float(np.clip(shed_norm, 0, 1)) * DEMAND_SHED_MAX_FRAC

    # ── Demand (with shedding rebound from last step) ────────────────
    actual_demand = demand_kw + state.shed_rebound_kwh / DT  # rebound is kWh, convert to kW
    effective_demand = actual_demand * (1.0 - shed_frac)
    shed_kwh = actual_demand * shed_frac * DT
    state.shed_rebound_kwh = shed_kwh * SHED_REBOUND_FRAC  # 50% rebounds next hour

    # ── Diesel fuel constraint ───────────────────────────────────────
    available_diesel_kwh = state.diesel_fuel_kwh
    diesel_kw = min(diesel_kw, available_diesel_kwh / DT)
    diesel_kw = max(0.0, diesel_kw)

    # ── Battery physics ──────────────────────────────────────────────
    if battery_cmd_kw > 0:
        # Discharge: agent wants power FROM battery
        max_discharge = min(battery_cmd_kw, state.battery_soc_kwh / DT)
        battery_kw = max(0.0, max_discharge)
        delivered_kw = battery_kw * BATTERY_DISCHARGE_EFF
        state.battery_soc_kwh -= battery_kw * DT
    else:
        # Charge: agent wants to push power INTO battery
        charge_cmd = abs(battery_cmd_kw)
        headroom = (BATTERY_CAPACITY_KWH - state.battery_soc_kwh) / BATTERY_CHARGE_EFF
        max_charge = min(charge_cmd, headroom / DT)
        battery_kw = -max(0.0, max_charge)  # negative = charging
        delivered_kw = battery_kw  # charging consumes power (negative delivery)
        state.battery_soc_kwh += abs(battery_kw) * BATTERY_CHARGE_EFF * DT

    state.battery_soc_kwh = float(np.clip(state.battery_soc_kwh, 0, BATTERY_CAPACITY_KWH))
    battery_throughput = abs(battery_kw) * DT
    state.cumulative_battery_throughput_kwh += battery_throughput

    # ── Grid as slack variable ───────────────────────────────────────
    # residual = what the community still needs after solar + battery + diesel
    # positive → grid must import; negative → surplus exported
    grid_cap = GRID_MAX_KW if grid_available else 0.0
    residual = effective_demand - solar_kw - delivered_kw - diesel_kw
    grid_kw = float(np.clip(residual, -grid_cap, grid_cap))

    # ── Blackout / curtailment detection ─────────────────────────────
    blackout_kwh = 0.0
    curtailed_kw = 0.0
    if residual > grid_cap:
        blackout_kwh = (residual - grid_cap) * DT
    elif residual < -grid_cap:
        curtailed_kw = abs(residual) - grid_cap  # excess that can't be exported

    # ── Build flow snapshot ──────────────────────────────────────────
    grid_import = max(0.0, grid_kw)
    grid_export = max(0.0, -grid_kw)
    batt_discharge = max(0.0, delivered_kw)
    batt_charge = max(0.0, -delivered_kw)  # power drawn from bus to charge

    flows = StepFlows(
        solar_kw=solar_kw,
        grid_import_kw=grid_import,
        battery_discharge_kw=batt_discharge,
        diesel_kw=diesel_kw,
        effective_demand_kw=effective_demand,
        grid_export_kw=grid_export,
        battery_charge_kw=batt_charge,
        blackout_kw=blackout_kwh / DT,
        curtailed_kw=curtailed_kw,
        total_supply_kw=solar_kw + grid_import + batt_discharge + diesel_kw,
        total_consumption_kw=effective_demand + grid_export + batt_charge,
        shed_kw=actual_demand * shed_frac,
        rebound_kw=state.shed_rebound_kwh / SHED_REBOUND_FRAC if shed_frac == 0 else 0,
    )

    # ── Cost accounting ──────────────────────────────────────────────
    step_cost = 0.0

    # Grid cost (import costs money, export earns revenue)
    if grid_kw > 0:
        step_cost += grid_price * grid_kw * DT
    else:
        step_cost -= grid_price * abs(grid_kw) * DT  # revenue

    # Diesel cost
    step_cost += DIESEL_COST_PER_KWH * diesel_kw * DT

    # Diesel startup cost
    if diesel_kw > 0 and not state.diesel_was_on:
        step_cost += DIESEL_STARTUP_COST
    state.diesel_was_on = (diesel_kw > 0)

    # Battery degradation cost
    step_cost += BATTERY_DEGRADATION_RS * battery_throughput

    # VoLL penalty (replaces hard reliability gate)
    step_cost += VOLL * blackout_kwh

    # Shedding penalty (comfort + political cost — Rs 40/kWh shed)
    # More expensive than diesel (Rs 25), so only used as true emergency
    step_cost += 40.0 * shed_kwh

    # ── Fuel accounting ──────────────────────────────────────────────
    state.diesel_fuel_kwh -= diesel_kw * DT
    state.diesel_fuel_kwh = max(0.0, state.diesel_fuel_kwh)

    # ── Cumulative tracking ──────────────────────────────────────────
    state.cumulative_cost += step_cost
    state.cumulative_blackout_kwh += blackout_kwh
    state.cumulative_diesel_kwh += diesel_kw * DT
    state.total_demand_kwh += effective_demand * DT
    state.last_blackout_kwh = blackout_kwh
    state.last_cost = step_cost
    state.last_grid_kw = grid_kw

    # ── Per-step reward (negative cost, normalized) ──────────────────
    # Simple: reward = -cost / normalization. Agent minimizes total cost.
    reward = -step_cost / 500.0  # normalize to roughly [-2, 0] range
    state.last_reward = reward

    # ── Advance clock ────────────────────────────────────────────────
    state.hour += 1
    done = state.hour >= 72

    # ── Narration ────────────────────────────────────────────────────
    narration = _narrate(state, solar_kw, actual_demand, grid_price, blackout_kwh,
                         diesel_kw, shed_frac, grid_kw, delivered_kw, grid_available)

    return StepResult(state=state, reward=reward, done=done, narration=narration, flows=flows)


def _narrate(
    s: MicrogridState,
    solar: float,
    demand: float,
    price: float,
    blackout: float,
    diesel: float,
    shed: float,
    grid_kw: float,
    battery_kw: float,
    grid_available: bool = True,
) -> str:
    """Generate a short human-readable situation summary."""
    START_HOUR = 6
    clock = (s.hour - 1) + START_HOUR  # absolute hour since midnight Day 1
    hour_of_day = clock % 24
    day = clock // 24 + 1
    soc_pct = s.battery_soc_kwh / BATTERY_CAPACITY_KWH * 100

    parts = [f"Day {day}, {hour_of_day:02d}:00."]

    if not grid_available:
        parts.append("GRID OUTAGE — islanding mode! No grid import/export.")

    if blackout > 0:
        parts.append(f"BLACKOUT: {blackout:.0f} kWh unmet!")
    elif demand > 200:
        parts.append("Peak demand period.")
    elif solar > 150:
        parts.append("Strong solar generation.")
    elif hour_of_day >= 18:
        parts.append("Evening approaching — solar fading.")
    elif hour_of_day < 6:
        parts.append("Night — low demand, no solar.")

    if grid_kw > 150:
        parts.append(f"Grid import near limit ({grid_kw:.0f}/{GRID_MAX_KW:.0f} kW).")
    elif grid_kw < -50:
        parts.append(f"Exporting {abs(grid_kw):.0f} kW to grid at Rs {price:.1f}.")

    if price > 12:
        parts.append(f"Grid price high (Rs {price:.1f}/kWh).")
    elif price < 5:
        parts.append(f"Grid price low (Rs {price:.1f}/kWh).")

    if soc_pct < 20:
        parts.append(f"Battery low ({soc_pct:.0f}%).")
    elif soc_pct > 80:
        parts.append(f"Battery well-charged ({soc_pct:.0f}%).")

    if battery_kw > 10:
        parts.append(f"Battery discharging {battery_kw:.0f} kW.")
    elif battery_kw < -10:
        parts.append(f"Battery charging {abs(battery_kw):.0f} kW.")

    if diesel > 0:
        fuel_pct = s.diesel_fuel_kwh / DIESEL_TANK_KWH * 100
        parts.append(f"Diesel running ({fuel_pct:.0f}% fuel left).")

    if shed > 0:
        parts.append(f"Demand response active ({shed * 100:.0f}% shed).")
        if s.shed_rebound_kwh > 1:
            parts.append(f"Rebound: +{s.shed_rebound_kwh:.0f} kW next hour.")

    return " ".join(parts)
