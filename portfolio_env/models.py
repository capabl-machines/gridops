"""Pydantic models — the OpenEnv contract.

Action and Observation are what the LLM sees/emits. Everything else is
derived from these.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import N_ASSETS
from .inflation import Regime


class PortfolioAction(BaseModel):
    """What the agent outputs each quarter."""

    model_config = ConfigDict(extra='forbid', validate_assignment=True)

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
        description='Protective put. 2% premium per quarter. Caps downside at −5% if worst asset < −15%.',
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


class PortfolioObs(BaseModel):
    """What the agent sees each quarter."""

    model_config = ConfigDict(extra='forbid', validate_assignment=True)

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
