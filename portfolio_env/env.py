"""PortfolioEnv — reset / step with path-dependent state.

Single-agent env. One LLM action per episode (flattened MDP — held for
all 12 quarters). Inherits from OpenEnv `Environment` so it works with
`create_app` for FastAPI server + WebSocket protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

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
from .models import PortfolioAction, PortfolioObs, PortfolioState
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


class PortfolioEnv(Environment):
    """Single-agent portfolio env, OpenEnv-compliant.

    Usage as a library:
        env = PortfolioEnv(phase=1)
        obs = env.reset(seed=42)
        for _ in range(EPISODE_LENGTH):
            action = PortfolioAction(weights=[...], ...)
            obs = env.step(action)
            if obs.done: break

    OpenEnv contract: `reset` / `step` / `state` / `get_metadata`.
    Each WebSocket session gets its own `PortfolioEnv` instance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, phase: int = 3, seed: int | None = None):
        super().__init__()
        self.phase = phase
        self.rng = np.random.default_rng(seed)
        self._state: _PathState | None = None
        self._plan: _EpisodePlan | None = None
        self._last_completion: str = ''
        self._episode_id: str = str(uuid4())
        self._final_grade: dict[str, Any] | None = None

    # ──────────────────────────── reset ─────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PortfolioObs:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if 'phase' in kwargs:
            self.phase = int(kwargs['phase'])
        self._episode_id = episode_id or str(uuid4())
        self._final_grade = None

        self._state = _PathState()
        self._state.traj.nav_nominal_series = [STARTING_NAV]
        self._state.traj.nav_real_series = [STARTING_NAV]
        self._state.traj.baseline_nav_real_series = [STARTING_NAV]

        self._plan = self._generate_episode_plan()
        return self._current_obs(
            news=self._news_for_quarter(0),
            reward=0.0,
            done=False,
            narration='Episode start. Q0: commit your tech_bet thesis.',
        )

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

    def step(
        self,
        action: PortfolioAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PortfolioObs:
        """Advance one quarter. Returns the next Observation.

        `obs.reward` is set to this quarter's regret (agent_real_return −
        baseline_real_return) — useful per-step signal even though our
        composite reward functions operate on the full trajectory at
        episode end.

        `obs.metadata['snapshot']` includes carbon, NAV, regret-so-far for
        dashboard consumption.
        """
        s = self._state
        assert s is not None, 'call reset() first'
        completion: str = kwargs.get('completion', '')
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

        # Per-step reward = regret this quarter (real)
        agent_real_q = s.traj.quarterly_real_returns[-1]
        baseline_real_q = (1.0 + baseline_return) / (1.0 + inflation_rate) - 1.0
        step_reward = float(agent_real_q - baseline_real_q)

        # On episode end, compute final grade (composite of 5 rewards) for /state
        if done:
            from .rewards import r_format, r_regret, r_sharpe, r_carbon, r_drawdown
            traj = s.traj
            self._final_grade = {
                'r_format':   float(r_format(self._last_completion)),
                'r_regret':   float(r_regret(traj)),
                'r_sharpe':   float(r_sharpe(traj)),
                'r_carbon':   float(r_carbon(traj, phase_weight=1.0)),
                'r_drawdown': float(r_drawdown(traj)),
                'final_nav_real': float(s.nav_real),
                'baseline_nav_real': float(s.baseline_nav_real),
            }

        # Build observation
        next_news = self._news_for_quarter(s.quarter) if not done else ''
        narration_parts = []
        if shock:
            narration_parts.append(f'shock fired: {shock.id} ({shock.tier})')
        if done:
            narration_parts.append('Episode complete.')
        narration = ' | '.join(narration_parts) or f'Q{s.quarter}: {next_news[:80]}'

        obs = self._current_obs(
            news=next_news,
            last_returns_nominal=returns_nominal,
            last_returns_real=returns_real,
            reward=step_reward,
            done=done,
            narration=narration,
            metadata={
                'snapshot': {
                    'carbon_accumulated': float(s.traj.carbon_footprint_accumulated),
                    'nav_real': float(s.nav_real),
                    'baseline_nav_real': float(s.baseline_nav_real),
                    'regret_so_far': float(s.nav_real / STARTING_NAV - s.baseline_nav_real / STARTING_NAV),
                    'quarterly_return_real': float(s.traj.quarterly_real_returns[-1]),
                },
                'shock_fired': shock.id if shock else None,
                'regime': s.current_regime,
                'grade': self._final_grade if done else None,
            },
        )
        return obs

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
        reward: float | None = None,
        done: bool = False,
        narration: str = '',
        metadata: dict[str, Any] | None = None,
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
            narration=narration,
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    @property
    def trajectory(self) -> Trajectory:
        assert self._state is not None
        return self._state.traj

    # ─────────────────────── OpenEnv interface ──────────────────────

    @property
    def state(self) -> PortfolioState:
        s = self._state
        if s is None:
            return PortfolioState(episode_id=self._episode_id, step_count=0, phase=self.phase)
        return PortfolioState(
            episode_id=self._episode_id,
            step_count=s.quarter,
            phase=self.phase,
            quarter=s.quarter,
            done=s.quarter >= EPISODE_LENGTH,
            final_grade=self._final_grade,
            history=[],  # leave empty for now; trajectory accessed separately
        )

    def get_metadata(self) -> EnvironmentMetadata:
        from pathlib import Path
        readme = None
        try:
            readme_path = Path(__file__).parent.parent / 'README.md'
            if readme_path.exists():
                readme = readme_path.read_text()
        except Exception:
            pass
        return EnvironmentMetadata(
            name='portfolio-env',
            description=('Reasoning-Under-Constraints OpenEnv: LLM acts as a climate-aware '
                         'portfolio manager over a 12-quarter macro cycle, trained via GRPO '
                         'to reason about ambiguous shocks, path-dependent decisions, and '
                         'competing objectives (return vs carbon vs risk).'),
            version='0.7.0',
            readme_content=readme,
        )
