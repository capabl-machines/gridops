"""
Data models for the GridOps Microgrid Environment.

Action:  3 continuous controls (battery_dispatch, diesel_dispatch, demand_shedding)
         Grid is the SLACK variable — absorbs the residual up to ±200 kW.
Observation: partial observation of the microgrid state + forecasts.
"""

from typing import List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class GridOpsAction(Action):
    """Agent action — three continuous knobs each step.

    The grid connection is NOT an action. It passively absorbs whatever
    the community needs after solar + battery + diesel - demand, clamped
    to the ±200 kW transformer limit. If the grid can't cover the
    residual, that's a blackout (or curtailment).
    """

    battery_dispatch: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Battery: -1 (charge 100 kW) to +1 (discharge 100 kW)",
    )
    diesel_dispatch: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Diesel generator: 0 (off) to 1 (100 kW). Rs 100 startup cost if was off.",
    )
    demand_shedding: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Demand response: 0 (none) to 1 (shed 20%). 100% rebounds next hour. Rs 40/kWh penalty.",
    )


class GridOpsObservation(Observation):
    """What the agent sees each hour."""

    # Current state
    hour: float = Field(default=0.0, description="Hour in episode (0-72)")
    demand_kw: float = Field(default=0.0, description="Current aggregate demand (kW)")
    solar_kw: float = Field(default=0.0, description="Current solar generation (kW)")
    battery_soc: float = Field(default=0.0, description="Battery state-of-charge (0-1)")
    grid_price: float = Field(default=0.0, description="Current IEX price (Rs/kWh)")
    diesel_fuel_remaining: float = Field(default=1.0, description="Diesel fuel level (0-1)")
    diesel_is_on: bool = Field(default=False, description="Whether diesel was running last step")

    # Noisy 4-hour forecasts
    demand_forecast_4h: List[float] = Field(default_factory=list, description="Demand forecast next 4h")
    solar_forecast_4h: List[float] = Field(default_factory=list, description="Solar forecast next 4h")
    price_forecast_4h: List[float] = Field(default_factory=list, description="Price forecast next 4h")

    # Cumulative metrics
    cumulative_blackout_kwh: float = Field(default=0.0, description="Total unmet demand (kWh)")
    cumulative_cost: float = Field(default=0.0, description="Net cost so far (Rs)")
    day_of_episode: int = Field(default=1, description="Current day (1-3)")

    # Step-level feedback
    blackout_this_step: float = Field(default=0.0, description="Blackout kWh this step")
    cost_this_step: float = Field(default=0.0, description="Cost incurred this step (Rs)")
    grid_kw_this_step: float = Field(default=0.0, description="Grid import(+)/export(-) this step")
    narration: str = Field(default="", description="Human-readable situation summary")

    # Detailed energy flows (kW, this step)
    flow_solar: float = Field(default=0.0, description="Solar supply kW")
    flow_grid_import: float = Field(default=0.0, description="Grid import kW")
    flow_grid_export: float = Field(default=0.0, description="Grid export kW")
    flow_battery_discharge: float = Field(default=0.0, description="Battery discharge kW (delivered)")
    flow_battery_charge: float = Field(default=0.0, description="Battery charge kW (consumed)")
    flow_diesel: float = Field(default=0.0, description="Diesel supply kW")
    flow_demand: float = Field(default=0.0, description="Effective demand kW")
    flow_blackout: float = Field(default=0.0, description="Unmet demand kW")
    flow_shed: float = Field(default=0.0, description="Demand shed kW")
    flow_total_supply: float = Field(default=0.0, description="Total supply kW")
    flow_total_consumption: float = Field(default=0.0, description="Total consumption kW")
