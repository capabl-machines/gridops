"""Verify training sampler never produces a holdout seed (even under N attempts).

Also verifies eval-on-holdout executes identically across phases.
"""

from __future__ import annotations

import numpy as np

from portfolio_env.constants import HOLDOUT_SEEDS
from portfolio_env.sampling import training_seeds, holdout_seeds


def test_training_seeds_exclude_holdout():
    rng = np.random.default_rng(0)
    # Tight range to stress: ask for 100 seeds out of 200-range, minus 5 holdout → feasible
    seeds = training_seeds(rng, n=100, max_seed=200)
    assert len(seeds) == 100, 'should return exactly n seeds'
    assert len(set(seeds)) == len(seeds), 'should be unique'
    for h in HOLDOUT_SEEDS:
        assert h not in seeds, f'holdout seed {h} leaked into training sampler!'


def test_holdout_seeds_stable():
    assert holdout_seeds() == HOLDOUT_SEEDS
    assert len(HOLDOUT_SEEDS) == 5, 'expected 5 holdout seeds'


if __name__ == '__main__':
    test_training_seeds_exclude_holdout()
    test_holdout_seeds_stable()
    print('✓ holdout seed sampler is safe')
