"""
Perfect-foresight MILP baseline for GridOps.

Solves the microgrid dispatch problem with full knowledge of demand, solar,
and price curves — gives the absolute cost ceiling for any causal policy.

Use as the grader's denominator: any RL agent beating dumb-heuristic and
approaching this LP is measurably close to optimal.

Physics replicated verbatim from gridops.simulation.physics:
  - Battery: 500 kWh cap, ±100 kW, √0.9 = 0.949 charge/discharge eff
  - Grid: ±200 kW slack, outage hours forced to 0
  - Diesel: 100 kW max, Rs 25/kWh + Rs 100 startup (binary)
  - Shedding: up to 20% of actual demand, Rs 40/kWh, 30% rebound
    over 3 hours (15% / 10% / 5%)
  - VoLL Rs 150/kWh on blackout
  - Battery degradation Rs 1.0/kWh throughput
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from gridops.simulation.physics import (
    BATTERY_CAPACITY_KWH,
    BATTERY_CHARGE_EFF,
    BATTERY_DEGRADATION_RS,
    BATTERY_DISCHARGE_EFF,
    BATTERY_MAX_POWER_KW,
    DEMAND_SHED_MAX_FRAC,
    DIESEL_COST_PER_KWH,
    DIESEL_MAX_KW,
    DIESEL_STARTUP_COST,
    DIESEL_TANK_KWH,
    DT,
    GRID_MAX_KW,
    SHED_REBOUND_PROFILE,
    VOLL,
)


@dataclass
class LPResult:
    cost: float
    blackout_kwh: float
    diesel_kwh: float
    battery_throughput_kwh: float
    shed_kwh: float
    status: str


def solve_optimal(
    demand: np.ndarray,
    solar: np.ndarray,
    price: np.ndarray,
    grid_outage_hours: list[int] | None = None,
    initial_soc_kwh: float = 250.0,
    diesel_fuel_cap_kwh: float = DIESEL_TANK_KWH,
    include_startup_binary: bool = True,
) -> LPResult:
    """Solve the perfect-foresight dispatch problem.

    Args:
        demand, solar, price: length-72 arrays (the 'true' curves, not noisy forecasts)
        grid_outage_hours: hours with zero grid capacity
        initial_soc_kwh: starting battery state
        diesel_fuel_cap_kwh: total available diesel energy
        include_startup_binary: if True, solve MILP with diesel startup cost.
            If False, pure LP (no startup cost, faster).
    """
    T = len(demand)
    assert len(solar) == T and len(price) == T
    outages = set(grid_outage_hours or [])
    grid_cap = np.array([0.0 if h in outages else GRID_MAX_KW for h in range(T)])

    # ── Decision variables ────────────────────────────────────────────
    # Battery: bus-side charge draw, SOC-side discharge (matches physics.step)
    p_charge = cp.Variable(T, nonneg=True)      # bus draws this much to charge
    p_discharge = cp.Variable(T, nonneg=True)   # SOC releases this much (bus gets 0.949×)
    soc = cp.Variable(T + 1)

    # Grid (split into import/export to price them separately — same price)
    g_import = cp.Variable(T, nonneg=True)
    g_export = cp.Variable(T, nonneg=True)

    # Diesel
    diesel = cp.Variable(T, nonneg=True)
    if include_startup_binary:
        diesel_on = cp.Variable(T, boolean=True)
        startup = cp.Variable(T, nonneg=True)
    else:
        diesel_on = cp.Variable(T, nonneg=True)
        startup = cp.Variable(T, nonneg=True)

    # Shedding
    shed = cp.Variable(T, nonneg=True)

    # Slack variables
    blackout = cp.Variable(T, nonneg=True)
    curtailed = cp.Variable(T, nonneg=True)

    constraints = []

    # ── Rebound convolution (linear in prior shed) ────────────────────
    # rebound[h] = Σ_k profile[k] * shed[h-1-k]  for k in 0..len(profile)-1
    rebound = []
    for h in range(T):
        terms = []
        for k, frac in enumerate(SHED_REBOUND_PROFILE):
            idx = h - 1 - k
            if idx >= 0:
                terms.append(frac * shed[idx])
        rebound.append(sum(terms) if terms else cp.Constant(0.0))

    # ── Shedding upper bound: ≤ 20% of actual demand (demand + rebound) ─
    for h in range(T):
        constraints.append(shed[h] <= DEMAND_SHED_MAX_FRAC * (demand[h] + rebound[h]))

    # ── Battery SOC dynamics ──────────────────────────────────────────
    constraints.append(soc[0] == initial_soc_kwh)
    for h in range(T):
        constraints.append(
            soc[h + 1] == soc[h] + BATTERY_CHARGE_EFF * p_charge[h] * DT - p_discharge[h] * DT
        )
    constraints.append(soc >= 0)
    constraints.append(soc <= BATTERY_CAPACITY_KWH)
    constraints.append(p_charge <= BATTERY_MAX_POWER_KW)
    constraints.append(p_discharge <= BATTERY_MAX_POWER_KW)

    # ── Grid constraints ──────────────────────────────────────────────
    for h in range(T):
        constraints.append(g_import[h] <= grid_cap[h])
        constraints.append(g_export[h] <= grid_cap[h])

    # ── Diesel constraints ────────────────────────────────────────────
    for h in range(T):
        if include_startup_binary:
            constraints.append(diesel[h] <= DIESEL_MAX_KW * diesel_on[h])
        else:
            constraints.append(diesel[h] <= DIESEL_MAX_KW)
            constraints.append(diesel_on[h] <= 1)
    # Startup: startup[h] >= diesel_on[h] - diesel_on[h-1]
    constraints.append(startup[0] >= diesel_on[0])  # treat prior as off
    for h in range(1, T):
        constraints.append(startup[h] >= diesel_on[h] - diesel_on[h - 1])
    # Total fuel budget
    constraints.append(cp.sum(diesel) * DT <= diesel_fuel_cap_kwh)

    # ── Energy balance at the bus (per hour) ──────────────────────────
    for h in range(T):
        supply = (
            solar[h]
            + g_import[h]
            + BATTERY_DISCHARGE_EFF * p_discharge[h]
            + diesel[h]
            + blackout[h]
        )
        effective_demand = demand[h] + rebound[h] - shed[h]
        consumption = effective_demand + g_export[h] + p_charge[h] + curtailed[h]
        constraints.append(supply == consumption)

    # ── Objective: total cost ─────────────────────────────────────────
    # Throughput cost (matches physics): |battery_kw| where battery_kw is
    # bus-draw for charge, SOC-drop for discharge. So degradation bills
    # p_charge + p_discharge (both are physics.step's battery_kw magnitudes).
    throughput = cp.sum(p_charge + p_discharge) * DT
    grid_cost = cp.sum(cp.multiply(price, g_import - g_export)) * DT
    diesel_cost = cp.sum(DIESEL_COST_PER_KWH * diesel) * DT
    startup_cost = DIESEL_STARTUP_COST * cp.sum(startup)
    deg_cost = BATTERY_DEGRADATION_RS * throughput
    voll_cost = VOLL * cp.sum(blackout) * DT
    shed_cost = 40.0 * cp.sum(shed) * DT

    total_cost = grid_cost + diesel_cost + startup_cost + deg_cost + voll_cost + shed_cost

    prob = cp.Problem(cp.Minimize(total_cost), constraints)
    solver = cp.HIGHS if include_startup_binary else cp.HIGHS
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP did not solve: status={prob.status}")

    return LPResult(
        cost=float(prob.value),
        blackout_kwh=float(np.sum(blackout.value) * DT),
        diesel_kwh=float(np.sum(diesel.value) * DT),
        battery_throughput_kwh=float(np.sum(p_charge.value + p_discharge.value) * DT),
        shed_kwh=float(np.sum(shed.value) * DT),
        status=prob.status,
    )
