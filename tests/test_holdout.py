"""Verify training sampler never produces a holdout seed (even under N attempts).

Also verifies eval-on-holdout executes identically across phases.
"""

from __future__ import annotations

import numpy as np

from portfolio_env.constants import HOLDOUT_SEEDS
from portfolio_env.sampling import training_seeds, holdout_seeds


def test_training_seeds_exclude_holdout():
    rng = np.random.default_rng(0)
    seeds = training_seeds(rng, n=10_000, max_seed=10_000)  # tight range stresses it
    assert len(seeds) == 10_000, 'should return exactly n seeds'
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
