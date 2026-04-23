"""Five independent reward functions — GRPOTrainer takes a list of these.

Each is a pure function that takes the episode trajectory + the LLM
completion and returns a scalar. Together they:
1. Teach JSON shape (format)
2. Incentivize beating the benchmark in real terms (regret — primary)
3. Penalize volatility (sharpe secondary)
4. Enforce carbon cap (non-linear, phase-weighted)
5. Penalize drawdowns
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    REWARD_WEIGHT_CARBON,
    REWARD_WEIGHT_DRAWDOWN,
    REWARD_WEIGHT_FORMAT,
    REWARD_WEIGHT_REGRET,
    REWARD_WEIGHT_SHARPE,
    CARBON_CAP,
)


@dataclass
class Trajectory:
    """Collected over one episode. Passed to reward functions at end-of-episode."""
    nav_nominal_series: list[float] = field(default_factory=list)
    nav_real_series: list[float] = field(default_factory=list)
    baseline_nav_real_series: list[float] = field(default_factory=list)
    quarterly_real_returns: list[float] = field(default_factory=list)
    carbon_footprint_accumulated: float = 0.0
    carbon_offsets_used: float = 0.0
    completions: list[str] = field(default_factory=list)  # per-quarter LLM text


# ══════════════════════════════════════════════════════════════════════
# Parsing helpers — shared with env.step()
# ══════════════════════════════════════════════════════════════════════

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def extract_think(completion: str) -> str | None:
    """Return the <think> body, or None if missing / malformed."""
    m = _THINK_RE.search(completion)
    return m.group(1).strip() if m else None


def parse_json_action(completion: str) -> dict | None:
    """Extract a JSON block from the completion. None on failure."""
    # Find first balanced JSON object
    start = completion.find('{')
    end = completion.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(completion[start:end + 1])
    except json.JSONDecodeError:
        return None


# ══════════════════════════════════════════════════════════════════════
# 1. FORMAT — per-completion, immediate
# ══════════════════════════════════════════════════════════════════════

def r_format(completion: str) -> float:
    """+0.05 for <think>, +0.10 for valid JSON action. Max 0.15."""
    score = 0.0
    if extract_think(completion) is not None:
        score += 0.05
    if parse_json_action(completion) is not None:
        score += 0.10
    return score * REWARD_WEIGHT_FORMAT / 0.15  # normalize so max ≈ weight


# ══════════════════════════════════════════════════════════════════════
# 2. REGRET vs EQUAL-WEIGHTED BASELINE (primary, REAL returns)
# ══════════════════════════════════════════════════════════════════════

def r_regret(traj: Trajectory) -> float:
    """Total real return minus baseline's real return. Positive = beat benchmark."""
    if len(traj.nav_real_series) < 2 or len(traj.baseline_nav_real_series) < 2:
        return 0.0
    agent_ret = traj.nav_real_series[-1] / traj.nav_real_series[0] - 1.0
    base_ret = traj.baseline_nav_real_series[-1] / traj.baseline_nav_real_series[0] - 1.0
    return REWARD_WEIGHT_REGRET * float(agent_ret - base_ret)


# ══════════════════════════════════════════════════════════════════════
# 3. SHARPE (secondary)
# ══════════════════════════════════════════════════════════════════════

def r_sharpe(traj: Trajectory) -> float:
    if len(traj.quarterly_real_returns) < 2:
        return 0.0
    r = np.asarray(traj.quarterly_real_returns, dtype=float)
    sharpe = float(r.mean() / (r.std() + 1e-6))
    return REWARD_WEIGHT_SHARPE * sharpe


# ══════════════════════════════════════════════════════════════════════
# 4. CARBON PENALTY — non-linear above cap, phase-weighted
# ══════════════════════════════════════════════════════════════════════

def r_carbon(traj: Trajectory, phase_weight: float = 1.0) -> float:
    """Quadratic penalty on overshoot above CARBON_CAP.

    phase_weight: 0.0 (Phase 1) / 0.3 (Phase 2) / 1.0 (Phase 3).
    """
    net_carbon = traj.carbon_footprint_accumulated - traj.carbon_offsets_used * 1.0  # offsets subtract
    overshoot = max(0.0, net_carbon - CARBON_CAP)
    return -phase_weight * REWARD_WEIGHT_CARBON * 5.0 * (overshoot ** 2) / 100.0


# ══════════════════════════════════════════════════════════════════════
# 5. MAX DRAWDOWN PENALTY
# ══════════════════════════════════════════════════════════════════════

def r_drawdown(traj: Trajectory) -> float:
    if not traj.nav_real_series:
        return 0.0
    peak = 0.0
    max_dd = 0.0
    for v in traj.nav_real_series:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
    return -REWARD_WEIGHT_DRAWDOWN * float(max_dd)


# ══════════════════════════════════════════════════════════════════════
# Bundle — GRPOTrainer wants a list of callables
# ══════════════════════════════════════════════════════════════════════

ALL_REWARDS = [r_format, r_regret, r_sharpe, r_carbon, r_drawdown]
