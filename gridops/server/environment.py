"""
GridOps OpenEnv Environment — wires physics + scenarios + grading together.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from gridops.models import GridOpsAction, GridOpsObservation
from gridops.simulation import physics, scenarios
from gridops.simulation.physics import (
    BATTERY_CAPACITY_KWH,
    DIESEL_TANK_KWH,
    MicrogridState,
)
from gridops.simulation.scenarios import ScenarioConfig, make_forecast
from gridops.tasks.definitions import TASKS
from gridops.tasks.graders import grade_episode


class GridOpsState(State):
    """Extended state exposed via GET /state."""

    task_id: str = "task_1_normal"
    hour: int = 0
    done: bool = False
    grade: dict | None = None
    history: list[dict] = []


class GridOpsEnvironment(Environment):
    """Community microgrid RL environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._task_id = "task_1_normal"
        self._cfg: ScenarioConfig = TASKS[self._task_id]
        self._rng = np.random.default_rng(42)
        self._demand = np.zeros(72)
        self._solar = np.zeros(72)
        self._price = np.zeros(72)
        self._micro = MicrogridState()
        self._episode_id = str(uuid4())
        self._history: list[dict] = []
        self._grade: dict | None = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GridOpsObservation:
        task_id = kwargs.get("task_id", "task_1_normal")
        if task_id not in TASKS:
            task_id = "task_1_normal"

        self._task_id = task_id
        self._cfg = TASKS[task_id]
        self._episode_id = episode_id or str(uuid4())

        s = seed if seed is not None else 42
        self._rng = np.random.default_rng(s)

        self._demand = scenarios.generate_demand(self._cfg, self._rng)
        self._solar = scenarios.generate_solar(self._cfg, self._rng)
        self._price = scenarios.generate_price(self._cfg, self._rng)

        self._micro = MicrogridState(
            diesel_fuel_kwh=self._cfg.diesel_fuel_capacity * DIESEL_TANK_KWH,
        )
        self._history = []
        self._grade = None

        return self._make_observation(reward=0.0, done=False, narration="Episode started. Day 1 begins.")

    def step(
        self,
        action: GridOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GridOpsObservation:
        if self._micro.hour >= 72:
            return self._make_observation(
                reward=0.0, done=True,
                narration="Episode already finished.",
            )

        h = self._micro.hour
        result = physics.step(
            self._micro,
            battery_dispatch_norm=action.battery_dispatch,
            diesel_norm=action.diesel_dispatch,
            shed_norm=action.demand_shedding,
            solar_kw=float(self._solar[h]),
            demand_kw=float(self._demand[h]),
            grid_price=float(self._price[h]),
            diesel_fuel_cap=self._cfg.diesel_fuel_capacity * DIESEL_TANK_KWH,
        )

        self._history.append({
            "hour": h,
            "demand": float(self._demand[h]),
            "solar": float(self._solar[h]),
            "price": float(self._price[h]),
            "battery_soc": self._micro.battery_soc_kwh / BATTERY_CAPACITY_KWH,
            "blackout": result.state.last_blackout_kwh,
            "cost": result.state.last_cost,
            "reward": result.reward,
            "grid_kw": result.state.last_grid_kw,
            "battery_dispatch": action.battery_dispatch,
            "diesel": action.diesel_dispatch,
            "shedding": action.demand_shedding,
        })

        if result.done:
            self._grade = grade_episode(
                self._micro, self._demand, self._solar, self._price
            )

        obs = self._make_observation(
            reward=result.reward,
            done=result.done,
            narration=result.narration,
        )
        if result.done and self._grade:
            obs.metadata["grade"] = self._grade
        return obs

    @property
    def state(self) -> GridOpsState:
        return GridOpsState(
            episode_id=self._episode_id,
            step_count=self._micro.hour,
            task_id=self._task_id,
            hour=self._micro.hour,
            done=self._micro.hour >= 72,
            grade=self._grade,
            history=self._history,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="GridOps",
            description="Community microgrid bridge operator — balance solar, battery, diesel, and grid across 3-day episodes.",
            version="0.2.0",
        )

    def _make_observation(self, reward: float, done: bool, narration: str) -> GridOpsObservation:
        h = min(self._micro.hour, 71)
        rng = self._rng

        return GridOpsObservation(
            hour=float(self._micro.hour),
            demand_kw=float(self._demand[h]),
            solar_kw=float(self._solar[h]),
            battery_soc=self._micro.battery_soc_kwh / BATTERY_CAPACITY_KWH,
            grid_price=float(self._price[h]),
            diesel_fuel_remaining=self._micro.diesel_fuel_kwh / DIESEL_TANK_KWH,
            diesel_is_on=self._micro.diesel_was_on,
            demand_forecast_4h=make_forecast(self._demand, h, 4, self._cfg.forecast_noise, rng),
            solar_forecast_4h=make_forecast(self._solar, h, 4, self._cfg.forecast_noise, rng),
            price_forecast_4h=make_forecast(self._price, h, 4, self._cfg.forecast_noise, rng),
            cumulative_blackout_kwh=self._micro.cumulative_blackout_kwh,
            cumulative_cost=self._micro.cumulative_cost,
            day_of_episode=(self._micro.hour // 24) + 1,
            blackout_this_step=self._micro.last_blackout_kwh,
            cost_this_step=self._micro.last_cost,
            grid_kw_this_step=self._micro.last_grid_kw,
            narration=narration,
            done=done,
            reward=reward,
        )
