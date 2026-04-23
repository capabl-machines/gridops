"""Smoke test — runs one random-policy episode end-to-end, checks invariants.

Run:  python -m tests.test_env_smoke
"""

from __future__ import annotations

import numpy as np

from portfolio_env import (
    PortfolioAction,
    PortfolioEnv,
    ALL_REWARDS,
    r_format,
    r_regret,
    r_sharpe,
    r_carbon,
    r_drawdown,
)
from portfolio_env.constants import EPISODE_LENGTH, N_ASSETS


def random_action(rng: np.random.Generator) -> PortfolioAction:
    return PortfolioAction(
        weights=list(rng.dirichlet([1.0] * N_ASSETS)),
        infra_commit=float(rng.uniform(0, 0.2)),
        carbon_offset_buy=float(rng.uniform(0, 0.05)),
        put_hedge=float(rng.uniform(0, 0.03)),
        tech_bet=rng.choice(['status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation']),
    )


def run_episode(env: PortfolioEnv, rng: np.random.Generator) -> dict:
    obs = env.reset(seed=int(rng.integers(10000)))
    steps = 0
    dummy_completion = '<think>random.</think>{"weights": [0.2,0.2,0.2,0.2,0.2]}'
    while True:
        action = random_action(rng)
        obs, snapshot, done, info = env.step(action, completion=dummy_completion)
        steps += 1
        if done:
            break
    return {
        'steps': steps,
        'final_nav_real': obs.portfolio_nav_real,
        'baseline_nav_real': obs.baseline_nav_real,
        'carbon': obs.carbon_footprint_accumulated,
        'traj': env.trajectory,
    }


def main():
    rng = np.random.default_rng(0)
    print(f'{"phase":<6}{"steps":>6}{"agent_real":>14}{"baseline_real":>14}{"carbon":>10}{"r_regret":>12}{"r_format":>12}')
    print('-' * 80)
    for phase in (1, 2, 3):
        env = PortfolioEnv(phase=phase, seed=42)
        out = run_episode(env, rng)
        assert out['steps'] == EPISODE_LENGTH, f'expected {EPISODE_LENGTH} steps, got {out["steps"]}'
        assert len(out['traj'].nav_real_series) == EPISODE_LENGTH + 1
        assert len(out['traj'].baseline_nav_real_series) == EPISODE_LENGTH + 1

        rf = r_format(out['traj'].completions[-1])
        rr = r_regret(out['traj'])
        rs = r_sharpe(out['traj'])
        rc = r_carbon(out['traj'], phase_weight=1.0)
        rd = r_drawdown(out['traj'])
        total = rf + rr + rs + rc + rd

        print(
            f'{phase:<6}{out["steps"]:>6}{out["final_nav_real"]:>14.4f}'
            f'{out["baseline_nav_real"]:>14.4f}{out["carbon"]:>10.1f}'
            f'{rr:>12.4f}{rf:>12.4f}'
        )
        print(f'       reward components — format={rf:+.3f} regret={rr:+.3f} sharpe={rs:+.3f} carbon={rc:+.3f} drawdown={rd:+.3f} total={total:+.3f}')

    # Sanity: ALL_REWARDS is the right length
    assert len(ALL_REWARDS) == 5, f'expected 5 reward fns, got {len(ALL_REWARDS)}'

    print('\nsmoke test OK.')


if __name__ == '__main__':
    main()
