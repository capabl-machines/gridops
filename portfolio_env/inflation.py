"""Inflation regime dynamics — see design doc §5.1.

Three regimes. Each quarter has a rate and a per-asset return adjustment
that stacks on top of nominal base returns.
"""

from __future__ import annotations

from typing import Literal

Regime = Literal['normal', 'stagflationary', 'deflationary']

REGIME_INFLATION_RATE: dict[Regime, float] = {
    'normal':         0.010,   # 1.0% quarterly  → ~4% annual
    'stagflationary': 0.025,   # 2.5% quarterly  → ~10% annual
    'deflationary':  -0.003,   # -0.3% quarterly → ~-1.2% annual
}

REGIME_ASSET_ADJUST: dict[Regime, dict[str, float]] = {
    'normal': {
        'TECH': 0.0, 'OIL': 0.0, 'GREEN': 0.0,
        'REAL_ESTATE': 0.0, 'BONDS': 0.0,
    },
    'stagflationary': {
        'TECH':        -0.020,   # long-duration crushed by real rates
        'OIL':          0.030,   # commodity inflation hedge + supply response
        'GREEN':       -0.030,   # long-duration + policy uncertainty
        'REAL_ESTATE':  0.005,   # paces inflation
        'BONDS':       -0.010,   # duration hit AND real bleed
    },
    'deflationary': {
        'TECH':         0.010,   # duration benefit mutes demand destruction
        'OIL':         -0.020,   # supply glut + demand collapse
        'GREEN':        0.005,
        'REAL_ESTATE': -0.010,   # asset deflation
        'BONDS':        0.003,   # deflation friend + flight to quality
    },
}


def real_return(nominal: float, inflation_rate: float) -> float:
    """Convert nominal return to real using (1+nom)/(1+inf) − 1."""
    return (1.0 + nominal) / (1.0 + inflation_rate) - 1.0


def apply_regime(base_return: float, asset: str, regime: Regime) -> float:
    """Stack regime adjustment on top of nominal base return."""
    return base_return + REGIME_ASSET_ADJUST[regime][asset]
