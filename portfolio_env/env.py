"""PortfolioEnv — reset / step with path-dependent state.

Single-agent env. One LLM inference per quarter. 12 quarters per episode.

Not yet OpenEnv-wrapped. Exposes a minimal Gym-like interface so we can
smoke-test logic; OpenEnv server wrapper comes in a later pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import (
    ASSETS,
    BASE_QUARTERLY_RETURN,
    BASE_QUARTERLY_VOL,
    BASE_RETURN_NOISE,
    BASELINE_WEIGHTS,
    CARBON_CAP,
    CARBON_INTENSITY,
    CARBON_OFFSET_RATIO,
    EPISODE_LENGTH,
    INFRA_LOCKUP_QUARTERS,
    INFRA_RETURN_PER_TRANSITION_SHOCK,
    N_ASSETS,
    PUT_HEDGE_DOWNSIDE_CAP,
    PUT_HEDGE_PREMIUM,
    PUT_HEDGE_TRIGGER_RETURN,
    STARTING_NAV,
    TRANSACTION_COST_RATE,
)
from .inflation import (
    REGIME_ASSET_ADJUST,
    REGIME_INFLATION_RATE,
    Regime,
    real_return,
)
from .models import PortfolioAction, PortfolioObs
from .rewards import Trajectory
from .shocks import Shock, shocks_available


@dataclass
class _EpisodePlan:
    """Generated at reset(). Tells env which shocks fire at which quarter."""
    shocks_by_quarter: dict[int, Shock] = field(default_factory=dict)


@dataclass
class _PathState:
    """Mutable state tracked across quarters."""
    quarter: int = 0
    nav_nominal: float = STARTING_NAV
    nav_real: float = STARTING_NAV
    baseline_nav_real: float = STARTING_NAV
    cumulative_inflation_multiplier: float = 1.0
    current_regime: Regime = 'normal'

    current_weights: list[float] = field(default_factory=lambda: list(BASELINE_WEIGHTS))

    infra_locked_fraction: float = 0.0
    infra_unlock_quarter: int = -1           # quarter at which infra payout fires (-1 = inactive)
    transition_shocks_during_lockup: int = 0

    carbon_offsets_held: float = 0.0
    active_put_hedge: bool = False
    tech_bet_chosen: str = 'status_quo'

    # Trajectory accumulator
    traj: Trajectory = field(default_factory=Trajectory)


class PortfolioEnv:
    """Single-agent portfolio env.

    Usage:
        env = PortfolioEnv(phase=1, seed=42)
        obs = env.reset()
        for _ in range(EPISODE_LENGTH):
            action = PortfolioAction(weights=[...], ...)
            obs, reward_components, done, info = env.step(action)
    """

    def __init__(self, phase: int = 3, seed: int | None = None):
        self.phase = phase
        self.rng = np.random.default_rng(seed)
        self._state: _PathState | None = None
        self._plan: _EpisodePlan | None = None
        self._last_completion: str = ''

    # ──────────────────────────── reset ─────────────────────────────

    def reset(self, seed: int | None = None) -> PortfolioObs:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._state = _PathState()
        self._state.traj.nav_nominal_series = [STARTING_NAV]
        self._state.traj.nav_real_series = [STARTING_NAV]
        self._state.traj.baseline_nav_real_series = [STARTING_NAV]

        self._plan = self._generate_episode_plan()
        return self._current_obs(news=self._news_for_quarter(0))

    def _generate_episode_plan(self) -> _EpisodePlan:
        """Sample shocks for this episode respecting curriculum phase + regime coherence."""
        pool = shocks_available(self.phase)
        # Phase-dependent number of shocks
        n_shocks = {1: 2, 2: 3, 3: 5}.get(self.phase, 5)
        # Simple uniform sample without replacement, random quarters (no Q0)
        if len(pool) < n_shocks:
            n_shocks = len(pool)
        chosen_shocks = self.rng.choice(pool, size=n_shocks, replace=False)
        # Avoid Q0; spread across remaining quarters
        quarters = list(self.rng.choice(
            np.arange(1, EPISODE_LENGTH),
            size=n_shocks,
            replace=False,
        ))
        return _EpisodePlan(shocks_by_quarter=dict(zip(quarters, chosen_shocks)))

    # ──────────────────────────── step ──────────────────────────────

    def step(self, action: PortfolioAction, completion: str = '') -> tuple[PortfolioObs, dict, bool, dict]:
        """Advance one quarter.

        Returns (obs, reward_components, done, info). reward_components
        is a dict with the 5 named rewards for monitoring; GRPOTrainer
        expects the reward functions themselves to be called externally
        on the completion + trajectory after episode end, but we expose
        per-step component snapshots for dashboarding.
        """
        s = self._state
        assert s is not None, 'call reset() first'
        s.traj.completions.append(completion)
        self._last_completion = completion

        q = s.quarter

        # Q1-only tech_bet commit
        if q == 0:
            s.tech_bet_chosen = action.tech_bet

        # 1. Transaction cost on weight changes
        turnover = sum(abs(new - old) for new, old in zip(action.weights, s.current_weights))
        tc = TRANSACTION_COST_RATE * turnover
        s.nav_nominal *= (1.0 - tc)

        # 2. Resolve current quarter's shock (if any)
        shock = self._plan.shocks_by_quarter.get(q)
        shock_impacts = shock.impacts if shock else {a: 0.0 for a in ASSETS}

        # 3. Inflation regime — apply shift at END of this quarter if shock says so
        # (so the current quarter's real calc uses the EXISTING regime)
        existing_regime = s.current_regime

        # 4. Compute nominal returns per asset
        returns_nominal = []
        for i, asset in enumerate(ASSETS):
            base = BASE_QUARTERLY_RETURN[asset]
            vol = BASE_QUARTERLY_VOL[asset]
            shock_adj = shock_impacts.get(asset, 0.0)
            regime_adj = REGIME_ASSET_ADJUST[existing_regime][asset]
            noise = float(self.rng.normal(0, vol * BASE_RETURN_NOISE / vol if vol else 0))
            r = base + shock_adj + regime_adj + noise
            returns_nominal.append(r)

        # 5. Apply returns to the LIQUID portion only (infra_locked is off-limits)
        liquid_fraction = 1.0 - s.infra_locked_fraction
        liquid_portfolio_return = float(np.dot(action.weights, returns_nominal)) * liquid_fraction
        s.nav_nominal *= (1.0 + liquid_portfolio_return)

        # 6. Handle infra_commit action (new lockup)
        if action.infra_commit > 0 and s.infra_locked_fraction == 0:
            s.infra_locked_fraction = action.infra_commit
            s.infra_unlock_quarter = q + INFRA_LOCKUP_QUARTERS
            s.transition_shocks_during_lockup = 0

        # 7. Track transition/physical shocks during lockup for infra payoff calc (v0.7)
        if shock and s.infra_locked_fraction > 0 and q < s.infra_unlock_quarter:
            if 'transition_risk' in shock.tags or 'fragmentation' in shock.tags:
                s.transition_shocks_during_lockup += 1
            elif 'physical_risk' in shock.tags:
                s.physical_shocks_during_lockup = getattr(s, 'physical_shocks_during_lockup', 0) + 1

        # 8. Resolve infra payoff if unlocking this quarter (v0.7: physical-risk counter-penalty)
        if s.infra_locked_fraction > 0 and q == s.infra_unlock_quarter:
            phys = getattr(s, 'physical_shocks_during_lockup', 0)
            infra_return = (INFRA_RETURN_PER_TRANSITION_SHOCK * s.transition_shocks_during_lockup
                            - 0.08 * phys)  # v0.7: -8% per physical-risk shock (matches transition-risk gain; makes infra a true bet) (Gemini finding)
            s.nav_nominal += s.infra_locked_fraction * s.nav_nominal * infra_return  # v0.7: return only (principal was always in NAV — fixes double-count)
            s.infra_locked_fraction = 0.0
            s.infra_unlock_quarter = -1
            s.transition_shocks_during_lockup = 0
            s.physical_shocks_during_lockup = 0

        # 9. Put hedge payoff — v0.7: triggers on PORTFOLIO return, not single-asset
        if s.active_put_hedge and liquid_portfolio_return < PUT_HEDGE_TRIGGER_RETURN:
            # Cap the portfolio return at DOWNSIDE_CAP. Claw back the difference.
            realized = liquid_portfolio_return
            cap = PUT_HEDGE_DOWNSIDE_CAP
            if realized < cap:
                s.nav_nominal *= (1.0 + cap) / (1.0 + realized)

        # 10. Put premium (always pays, whether it triggered or not)
        if action.put_hedge > 0:
            s.nav_nominal *= (1.0 - action.put_hedge * PUT_HEDGE_PREMIUM / 0.02)  # scale premium to amount bought
            s.active_put_hedge = True
        else:
            s.active_put_hedge = False

        # 11. Carbon emissions this quarter (based on weights × NAV × intensity)
        carbon_this_quarter = sum(
            action.weights[i] * CARBON_INTENSITY[asset] * s.nav_nominal
            for i, asset in enumerate(ASSETS)
        )
        s.traj.carbon_footprint_accumulated += carbon_this_quarter

        # 12. Carbon offset purchase
        if action.carbon_offset_buy > 0:
            offset_cost = action.carbon_offset_buy * s.nav_nominal
            offset_kg = offset_cost * CARBON_OFFSET_RATIO
            s.nav_nominal -= offset_cost
            s.carbon_offsets_held += offset_kg
            # Offsets burn automatically against accumulated footprint
            burn = min(offset_kg, max(0.0, s.traj.carbon_footprint_accumulated))
            s.traj.carbon_footprint_accumulated -= burn
            s.traj.carbon_offsets_used += burn

        # 13. Inflation — accumulate, then apply regime shift if shock says so
        inflation_rate = REGIME_INFLATION_RATE[existing_regime]
        s.cumulative_inflation_multiplier *= (1.0 + inflation_rate)
        s.nav_real = s.nav_nominal / s.cumulative_inflation_multiplier

        # Baseline (equal-weighted, same regime + shock)
        baseline_return = float(np.dot(BASELINE_WEIGHTS, returns_nominal))
        s.baseline_nav_real *= (1.0 + baseline_return) / (1.0 + inflation_rate)

        # 14. Apply regime shift from shock AT END of step (next quarter uses new regime)
        if shock and shock.regime_shift is not None:
            s.current_regime = shock.regime_shift

        # 15. Record + advance
        returns_real = [real_return(r, inflation_rate) for r in returns_nominal]
        s.current_weights = list(action.weights)
        s.traj.nav_nominal_series.append(s.nav_nominal)
        s.traj.nav_real_series.append(s.nav_real)
        s.traj.baseline_nav_real_series.append(s.baseline_nav_real)
        s.traj.quarterly_real_returns.append(
            float(np.dot(action.weights, returns_real)) * liquid_fraction
        )

        s.quarter += 1
        done = s.quarter >= EPISODE_LENGTH

        # Build observation for next quarter
        next_news = self._news_for_quarter(s.quarter) if not done else ''
        obs = self._current_obs(
            news=next_news,
            last_returns_nominal=returns_nominal,
            last_returns_real=returns_real,
        )

        # Per-step reward components snapshot (for monitoring only)
        reward_snapshot = {
            'carbon_accumulated': s.traj.carbon_footprint_accumulated,
            'nav_real': s.nav_real,
            'baseline_nav_real': s.baseline_nav_real,
            'regret_so_far': s.nav_real / STARTING_NAV - s.baseline_nav_real / STARTING_NAV,
            'quarterly_return_real': s.traj.quarterly_real_returns[-1],
        }
        info = {
            'shock_fired': shock.id if shock else None,
            'regime': s.current_regime,
        }
        return obs, reward_snapshot, done, info

    # ──────────────────────────── helpers ───────────────────────────

    def _news_for_quarter(self, q: int) -> str:
        if q >= EPISODE_LENGTH:
            return ''
        if not self._plan:
            return '(news pending plan)'
        shock = self._plan.shocks_by_quarter.get(q)
        if shock:
            return shock.news
        return f'Q{q + 1}: routine quarter. No significant macro news.'

    def _current_obs(
        self,
        news: str,
        last_returns_nominal: list[float] | None = None,
        last_returns_real: list[float] | None = None,
    ) -> PortfolioObs:
        s = self._state
        assert s is not None
        current_shock = self._plan.shocks_by_quarter.get(s.quarter) if self._plan else None
        tier = current_shock.tier if current_shock else 'easy'

        # Regret so far
        regret = (s.nav_real / STARTING_NAV) - (s.baseline_nav_real / STARTING_NAV)

        carbon_remaining = max(0.0, CARBON_CAP - s.traj.carbon_footprint_accumulated)

        return PortfolioObs(
            quarter=min(s.quarter, EPISODE_LENGTH - 1),
            difficulty_tier=tier,
            current_weights=list(s.current_weights),
            infra_locked_fraction=s.infra_locked_fraction,
            infra_unlock_quarters=max(0, s.infra_unlock_quarter - s.quarter) if s.infra_unlock_quarter >= 0 else 0,
            carbon_offsets_held=s.carbon_offsets_held,
            active_put_hedge=s.active_put_hedge,
            tech_bet_chosen=s.tech_bet_chosen,
            portfolio_nav_nominal=s.nav_nominal,
            portfolio_nav_real=s.nav_real,
            baseline_nav_real=s.baseline_nav_real,
            cumulative_real_return_pct=(s.nav_real / STARTING_NAV - 1.0) * 100.0,
            current_inflation_rate=REGIME_INFLATION_RATE[s.current_regime],
            current_regime=s.current_regime,
            cumulative_inflation_multiplier=s.cumulative_inflation_multiplier,
            carbon_footprint_accumulated=s.traj.carbon_footprint_accumulated,
            carbon_budget_remaining=carbon_remaining,
            news=news,
            last_quarter_returns_nominal=last_returns_nominal or [0.0] * N_ASSETS,
            last_quarter_returns_real=last_returns_real or [0.0] * N_ASSETS,
            last_quarter_regret=float(regret),
        )

    @property
    def trajectory(self) -> Trajectory:
        assert self._state is not None
        return self._state.traj
