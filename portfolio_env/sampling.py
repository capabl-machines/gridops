"""Seed samplers for training vs eval.

Training must NEVER sample a seed in HOLDOUT_SEEDS so we can measure
generalization cleanly on those seeds at eval time.
"""

from __future__ import annotations

from typing import Iterable, Iterator
import numpy as np

from .constants import HOLDOUT_SEEDS


def training_seeds(
    rng: np.random.Generator,
    n: int,
    max_seed: int = 10_000_000,
) -> list[int]:
    """Return n seeds drawn without replacement from [0, max_seed) \\ HOLDOUT_SEEDS."""
    holdout = set(HOLDOUT_SEEDS)
    out: list[int] = []
    seen: set[int] = set()
    while len(out) < n:
        candidate = int(rng.integers(0, max_seed))
        if candidate in holdout or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def holdout_seeds() -> tuple[int, ...]:
    """The immutable holdout set used for eval."""
    return HOLDOUT_SEEDS
