"""portfolio_env — OpenEnv Round 2 submission.

Reasoning-Under-Constraints Environment: LLM acts as a climate-aware
portfolio manager over a 12-quarter (3-year) full market cycle, trained
with GRPO + Unsloth. See portfolio_env_design.md for the spec.
"""

from .env import PortfolioEnv
from .models import PortfolioAction, PortfolioObs, PortfolioState
from .rewards import (
    Trajectory,
    r_carbon,
    r_drawdown,
    r_format,
    r_regret,
    r_sharpe,
    parse_json_action,
    extract_think,
    ALL_REWARDS,
)
from .sampling import training_seeds, holdout_seeds
from .shocks import Shock, SHOCKS_BY_ID, shocks_available

__all__ = [
    'PortfolioEnv',
    'PortfolioAction',
    'PortfolioObs',
    'PortfolioState',
    'Trajectory',
    'Shock',
    'SHOCKS_BY_ID',
    'shocks_available',
    'r_format',
    'r_regret',
    'r_sharpe',
    'r_carbon',
    'r_drawdown',
    'parse_json_action',
    'extract_think',
    'ALL_REWARDS',
]
