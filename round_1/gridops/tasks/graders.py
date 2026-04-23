"""
Episode graders — deterministic 0.0-1.0 score at end of episode.

Scoring (post-LP baseline):
  - cost_efficiency is now LP-normalized: 0.0 = as bad as dumb (grid-max) baseline,
    1.0 = matches perfect-foresight MILP optimal. This is a far more
    informative signal than the old baseline-only ratio.
  - The old baseline-normalized cost_efficiency is kept as `cost_efficiency_dumb`
    for back-compat and comparison.
  - Reliability and green_score unchanged.
  - Composite: 50% cost_efficiency (LP) + 25% reliability + 25% green_score.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gridops.simulation.physics import (
    BATTERY_CAPACITY_KWH,
    DIESEL_COST_PER_KWH,
    DIESEL_TANK_KWH,
    GRID_MAX_KW,
    VOLL,
    MicrogridState,
)


# Module-level cache: (curves_hash, outages_tuple, fuel_cap) → optimal_cost
# LP is deterministic in its inputs, so this is safe forever.
_OPTIMAL_CACHE: dict = {}


def _curves_key(
    demand: np.ndarray,
    solar: np.ndarray,
    price: np.ndarray,
    outages: Optional[list[int]],
    fuel_cap: float,
) -> tuple:
    """Hashable cache key for LP inputs."""
    return (
        demand.tobytes(),
        solar.tobytes(),
        price.tobytes(),
        tuple(sorted(outages)) if outages else (),
        round(fuel_cap, 3),
    )


def _get_optimal_cost(
    demand: np.ndarray,
    solar: np.ndarray,
    price: np.ndarray,
    outages: Optional[list[int]],
    fuel_cap: float,
) -> float:
    """Cached LP-optimal cost for the given curves. First call: ~0.2s. Subsequent: ~0s."""
    key = _curves_key(demand, solar, price, outages, fuel_cap)
    cached = _OPTIMAL_CACHE.get(key)
    if cached is not None:
        return cached

    # Import locally to avoid cvxpy cost on modules that don't grade
    from gridops.tasks.lp_baseline import solve_optimal

    result = solve_optimal(
        demand, solar, price,
        grid_outage_hours=outages,
        diesel_fuel_cap_kwh=fuel_cap,
        include_startup_binary=True,
    )
    _OPTIMAL_CACHE[key] = result.cost
    return result.cost


def compute_dumb_baseline_cost(
    demand_curve: np.ndarray,
    solar_curve: np.ndarray,
    price_curve: np.ndarray,
    grid_outage_hours: list[int] | None = None,
) -> float:
    """Cost of a dumb baseline: import max grid, no battery/diesel/shedding.

    Where demand > grid + solar, apply VoLL for the blackout.
    During grid outages, all non-solar demand is blackout.
    """
    outages = set(grid_outage_hours or [])
    total_cost = 0.0
    for h in range(len(demand_curve)):
        demand = demand_curve[h]
        solar = solar_curve[h]
        price = price_curve[h]
        grid_cap = 0.0 if h in outages else GRID_MAX_KW

        needed_from_grid = max(0.0, demand - solar)
        grid_import = min(needed_from_grid, grid_cap)
        total_cost += price * grid_import

        unmet = max(0.0, needed_from_grid - grid_cap)
        total_cost += VOLL * unmet

    return float(total_cost)


def grade_episode(
    state: MicrogridState,
    demand_curve: np.ndarray,
    solar_curve: np.ndarray,
    price_curve: np.ndarray,
    grid_outage_hours: list[int] | None = None,
    diesel_fuel_cap_kwh: float = DIESEL_TANK_KWH,
) -> dict:
    """
    Grade a completed episode. Returns dict with score 0.0-1.0.

    cost_efficiency is LP-normalized: (dumb - actual) / (dumb - optimal),
    so 1.0 means the agent matches perfect-foresight MILP optimal.
    """
    total_demand = max(state.total_demand_kwh, 1.0)
    total_blackout = state.cumulative_blackout_kwh
    actual_cost = state.cumulative_cost

    # Reliability: fraction of demand met (0-1)
    reliability = float(np.clip((total_demand - total_blackout) / total_demand, 0, 1))

    # Dumb baseline (grid-max, no battery/diesel/shed)
    baseline_cost = compute_dumb_baseline_cost(
        demand_curve, solar_curve, price_curve, grid_outage_hours
    )

    # LP optimal (perfect foresight). Cached per curve-set.
    optimal_cost = _get_optimal_cost(
        demand_curve, solar_curve, price_curve, grid_outage_hours, diesel_fuel_cap_kwh
    )

    # Old-style cost_efficiency (vs dumb baseline only) — kept for continuity
    if baseline_cost > 0:
        cost_eff_dumb = 1.0 - (actual_cost / baseline_cost)
    else:
        cost_eff_dumb = 0.0
    cost_eff_dumb = float(np.clip(cost_eff_dumb, 0, 1))

    # LP-normalized: 0 = dumb baseline, 1 = perfect-foresight LP
    denom = baseline_cost - optimal_cost
    if denom > 1.0:
        cost_efficiency = (baseline_cost - actual_cost) / denom
    else:
        cost_efficiency = 0.0
    cost_efficiency = float(np.clip(cost_efficiency, 0, 1))

    # Green score: 1 - diesel fraction of total energy
    green_score = float(np.clip(1.0 - (state.cumulative_diesel_kwh / total_demand), 0, 1))

    # Composite score uses LP-normalized cost_efficiency
    score = 0.50 * cost_efficiency + 0.25 * reliability + 0.25 * green_score
    score = float(np.clip(score, 0, 1))

    return {
        "score": round(score, 4),
        "reliability": round(reliability, 4),
        "cost_efficiency": round(cost_efficiency, 4),
        "cost_efficiency_dumb": round(cost_eff_dumb, 4),
        "green_score": round(green_score, 4),
        "baseline_cost": round(baseline_cost, 2),
        "optimal_cost": round(optimal_cost, 2),
        "actual_cost": round(actual_cost, 2),
        "total_blackout_kwh": round(total_blackout, 2),
        "total_diesel_kwh": round(state.cumulative_diesel_kwh, 2),
        "total_demand_kwh": round(total_demand, 2),
        "battery_throughput_kwh": round(state.cumulative_battery_throughput_kwh, 2),
    }
