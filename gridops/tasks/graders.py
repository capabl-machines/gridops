"""
Episode graders — deterministic 0.0-1.0 score at end of episode.

Scoring (post-Gemini review):
  - No hard reliability gate. VoLL (Rs 150/kWh) in the physics makes
    blackouts extremely expensive, creating a smooth gradient.
  - Score = how much better than a dumb heuristic baseline.
  - Baseline: "import max grid every hour, no battery/diesel/shedding"
  - score = clip(1 - agent_cost / baseline_cost, 0, 1)
  - Weighted: 50% cost efficiency + 25% reliability + 25% green score
"""

from __future__ import annotations

import numpy as np

from gridops.simulation.physics import (
    BATTERY_CAPACITY_KWH,
    DIESEL_COST_PER_KWH,
    DIESEL_TANK_KWH,
    GRID_MAX_KW,
    VOLL,
    MicrogridState,
)


def compute_dumb_baseline_cost(
    demand_curve: np.ndarray,
    solar_curve: np.ndarray,
    price_curve: np.ndarray,
) -> float:
    """Cost of a dumb baseline: import max grid, no battery/diesel/shedding.

    Where demand > grid + solar, apply VoLL for the blackout.
    This is a realistic "no-intelligence" baseline.
    """
    total_cost = 0.0
    for h in range(len(demand_curve)):
        demand = demand_curve[h]
        solar = solar_curve[h]
        price = price_curve[h]

        # Grid covers what it can (up to 200 kW)
        needed_from_grid = max(0.0, demand - solar)
        grid_import = min(needed_from_grid, GRID_MAX_KW)
        total_cost += price * grid_import  # grid cost

        # Any excess demand is a blackout
        unmet = max(0.0, needed_from_grid - GRID_MAX_KW)
        total_cost += VOLL * unmet  # VoLL penalty

    return float(total_cost)


def grade_episode(
    state: MicrogridState,
    demand_curve: np.ndarray,
    solar_curve: np.ndarray,
    price_curve: np.ndarray,
) -> dict:
    """
    Grade a completed episode. Returns dict with score 0.0-1.0.

    score = 0.50 × cost_efficiency + 0.25 × reliability + 0.25 × green_score
    """
    total_demand = max(state.total_demand_kwh, 1.0)
    total_blackout = state.cumulative_blackout_kwh

    # Reliability: fraction of demand met (0-1)
    reliability = (total_demand - total_blackout) / total_demand
    reliability = float(np.clip(reliability, 0, 1))

    # Cost efficiency: how much better than dumb baseline
    baseline_cost = compute_dumb_baseline_cost(demand_curve, solar_curve, price_curve)
    actual_cost = state.cumulative_cost
    if baseline_cost > 0:
        cost_efficiency = 1.0 - (actual_cost / baseline_cost)
    else:
        cost_efficiency = 0.0
    cost_efficiency = float(np.clip(cost_efficiency, 0, 1))

    # Green score: 1 - diesel fraction of total energy
    green_score = 1.0 - (state.cumulative_diesel_kwh / max(total_demand, 1.0))
    green_score = float(np.clip(green_score, 0, 1))

    # Composite score (smooth, no gate)
    score = 0.50 * cost_efficiency + 0.25 * reliability + 0.25 * green_score
    score = float(np.clip(score, 0, 1))

    return {
        "score": round(score, 4),
        "reliability": round(reliability, 4),
        "cost_efficiency": round(cost_efficiency, 4),
        "green_score": round(green_score, 4),
        "baseline_cost": round(baseline_cost, 2),
        "actual_cost": round(actual_cost, 2),
        "total_blackout_kwh": round(total_blackout, 2),
        "total_diesel_kwh": round(state.cumulative_diesel_kwh, 2),
        "total_demand_kwh": round(total_demand, 2),
        "battery_throughput_kwh": round(state.cumulative_battery_throughput_kwh, 2),
    }
