"""Pydantic models — the OpenEnv contract.

Action / Observation / State inherit from openenv-core base classes so
the FastAPI server (`create_app`) can introspect the schemas, serve them
at `/schema`, and validate inputs/outputs at the WebSocket / HTTP boundary.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from openenv.core.env_server.types import Action, Observation, State

from .constants import N_ASSETS
from .inflation import Regime


class PortfolioAction(Action):
    """What the agent outputs each quarter (single-turn flattened MDP).

    Inherits `metadata` field from openenv `Action`.
    """

    weights: list[float] = Field(
        ..., min_length=N_ASSETS, max_length=N_ASSETS,
        description='Allocation across [TECH, OIL, GREEN, REAL_ESTATE, BONDS]. Auto-normalized to sum to 1.',
    )
    infra_commit: float = Field(
        default=0.0, ge=0.0, le=0.2,
        description='4-quarter irreversible lockup. Payoff conditional on transition shocks during lockup.',
    )
    carbon_offset_buy: float = Field(
        default=0.0, ge=0.0, le=0.1,
        description='Buy carbon offsets. 1 unit NAV → 10 kg CO₂ offset.',
    )
    put_hedge: float = Field(
        default=0.0, ge=0.0, le=0.05,
        description='Protective put. 2% premium per quarter. Caps portfolio downside at −5% if return < −15%.',
    )
    tech_bet: Literal['status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'] = Field(
        default='status_quo',
        description='Q1-only macro thesis. Tilts shock distribution for remainder of episode.',
    )

    @field_validator('weights')
    @classmethod
    def _normalize_weights(cls, v: list[float]) -> list[float]:
        # clamp to [0, 1], renormalize to sum=1. If all zero, use equal.
        v = [max(0.0, x) for x in v]
        s = sum(v)
        if s <= 1e-9:
            return [1.0 / N_ASSETS] * N_ASSETS
        return [x / s for x in v]


class PortfolioObs(Observation):
    """What the agent sees each quarter.

    Inherits `done`, `reward`, `metadata` fields from openenv `Observation`.
    """

    # Time
    quarter: int = Field(ge=0, le=11)
    difficulty_tier: str = 'easy'  # 'easy' | 'ambiguous' | 'hard' — for curriculum visibility

    # Current state
    current_weights: list[float] = Field(min_length=N_ASSETS, max_length=N_ASSETS)
    infra_locked_fraction: float = 0.0
    infra_unlock_quarters: int = 0            # quarters until unlock (0 = nothing locked)
    carbon_offsets_held: float = 0.0          # accumulated offset credits (kg CO₂)
    active_put_hedge: bool = False
    tech_bet_chosen: str = 'status_quo'

    # Financials (real, inflation-adjusted)
    portfolio_nav_nominal: float = 1.0
    portfolio_nav_real: float = 1.0
    baseline_nav_real: float = 1.0
    cumulative_real_return_pct: float = 0.0

    # Inflation state
    current_inflation_rate: float = 0.010
    current_regime: Regime = 'normal'
    cumulative_inflation_multiplier: float = 1.0

    # Sustainability
    carbon_footprint_accumulated: float = 0.0
    carbon_budget_remaining: float = 120.0

    # The reasoning signal
    news: str = ''

    # Feedback
    last_quarter_returns_nominal: list[float] = Field(default_factory=lambda: [0.0] * N_ASSETS)
    last_quarter_returns_real: list[float] = Field(default_factory=lambda: [0.0] * N_ASSETS)
    last_quarter_regret: float = 0.0

    # Narration for dashboard
    narration: str = ''


class PortfolioState(State):
    """Episode-level state exposed at `/state`. Inherits `episode_id`, `step_count`."""

    phase: int = 3
    quarter: int = 0
    done: bool = False
    final_grade: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
