"""All the magic numbers. One place, reviewed with brother."""

from __future__ import annotations

ASSETS: tuple[str, ...] = ('TECH', 'OIL', 'GREEN', 'REAL_ESTATE', 'BONDS')
N_ASSETS = len(ASSETS)

EPISODE_LENGTH = 12           # quarters = 3-year full bull-bear cycle
STARTING_NAV = 1.0

BASE_QUARTERLY_RETURN: dict[str, float] = {
    'TECH':         0.030,
    'OIL':          0.020,
    'GREEN':        0.015,
    'REAL_ESTATE':  0.010,
    'BONDS':        0.005,
}

BASE_QUARTERLY_VOL: dict[str, float] = {
    'TECH':         0.08,
    'OIL':          0.05,
    'GREEN':        0.05,
    'REAL_ESTATE':  0.02,
    'BONDS':        0.005,
}

CARBON_INTENSITY: dict[str, float] = {  # kg CO₂ per $ of exposure per quarter
    'TECH':         0.05,
    'OIL':          2.50,
    'GREEN':        0.01,
    'REAL_ESTATE':  0.10,
    'BONDS':        0.00,
}

CARBON_CAP = 25.0   # v0.7 fix: tightened from 120 (all_oil exploit) — equal-weighted uses ~6, all-OIL ~30            # kg CO₂ total per episode (scaled to 12Q)
TRANSACTION_COST_RATE = 0.005 # 0.5% × turnover

INFRA_LOCKUP_QUARTERS = 4
INFRA_RETURN_PER_TRANSITION_SHOCK = 0.08
INFRA_MAX_FRACTION = 0.20

CARBON_OFFSET_RATIO = 10.0    # kg CO₂ offset per $1 of NAV spent
CARBON_OFFSET_MAX = 0.10

PUT_HEDGE_PREMIUM = 0.02      # 2% of NAV per quarter
PUT_HEDGE_DOWNSIDE_CAP = -0.05
PUT_HEDGE_TRIGGER_RETURN = -0.15
PUT_HEDGE_MAX = 0.05

TECH_BET_OPTIONS = ('status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation')

BASELINE_WEIGHTS = [0.2] * N_ASSETS  # equal-weighted benchmark

# v0.7: Reserve seeds for eval-only. Training seed sampler MUST skip these
# so we can measure generalization cleanly. (FAQ #44, #52)
HOLDOUT_SEEDS: tuple[int, ...] = (100, 200, 300, 400, 500)

# Forecast / observation noise
BASE_RETURN_NOISE = 0.02

# Reward weights
REWARD_WEIGHT_FORMAT = 0.15
REWARD_WEIGHT_REGRET = 1.0
REWARD_WEIGHT_SHARPE = 0.3
REWARD_WEIGHT_CARBON = 1.0    # scaled by phase (0, 0.3, 1.0)
REWARD_WEIGHT_DRAWDOWN = 2.0
