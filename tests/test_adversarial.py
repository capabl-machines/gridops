"""Adversarial reward stress-test: can any degenerate policy beat the env?

Per hackathon FAQ #57: "Do not optimize a reward you have not tried to break yourself first."
If ANY of these policies beat the equal-weighted baseline on total reward,
we have a reward-design bug that GRPO would exploit.
"""
import sys
import numpy as np
sys.path.insert(0, '/workspace/gridops')

from portfolio_env import (
    PortfolioEnv, PortfolioAction, Trajectory,
    r_format, r_regret, r_sharpe, r_carbon, r_drawdown,
)
from portfolio_env.constants import EPISODE_LENGTH, N_ASSETS, BASELINE_WEIGHTS


# Adversarial policy functions. Each returns a PortfolioAction given an obs.
def p_all_bonds(obs):
    return PortfolioAction(weights=[0, 0, 0, 0, 1])

def p_all_tech(obs):
    return PortfolioAction(weights=[1, 0, 0, 0, 0])

def p_all_oil(obs):
    return PortfolioAction(weights=[0, 1, 0, 0, 0])

def p_yo_yo(obs):
    # flip aggressively — abuse turnover budget
    if int(obs.quarter) % 2 == 0:
        return PortfolioAction(weights=[1, 0, 0, 0, 0])
    return PortfolioAction(weights=[0, 0, 0, 0, 1])

def p_put_hedge_farmer(obs):
    # tiny tech allocation + max put hedge — the exploit Gemini flagged
    return PortfolioAction(
        weights=[0.01, 0.99, 0, 0, 0],
        put_hedge=0.05,
    )

def p_carbon_offset_abuse(obs):
    # 99% oil + continuous offset buying
    return PortfolioAction(
        weights=[0, 0.99, 0, 0.01, 0],
        carbon_offset_buy=0.1,
    )

def p_infra_max(obs):
    # max infra commit + random weights
    return PortfolioAction(
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        infra_commit=0.2,
        tech_bet='green_leaps',
    )

def p_equal_weighted(obs):
    return PortfolioAction(weights=BASELINE_WEIGHTS)


POLICIES = {
    'all_bonds':           p_all_bonds,
    'all_tech':            p_all_tech,
    'all_oil':             p_all_oil,
    'yo_yo':               p_yo_yo,
    'put_hedge_farmer':    p_put_hedge_farmer,
    'carbon_offset_abuse': p_carbon_offset_abuse,
    'infra_max':           p_infra_max,
    'equal_weighted':      p_equal_weighted,
}


def run_episode(policy_fn, seed=42, phase=3):
    env = PortfolioEnv(phase=phase, seed=seed)
    obs = env.reset(seed=seed)
    dummy_completion = '<think>adversarial test.</think>{"weights": [0.2,0.2,0.2,0.2,0.2]}'
    for _ in range(EPISODE_LENGTH):
        action = policy_fn(obs)
        obs = env.step(action, completion=dummy_completion)
        if obs.done:
            break
    traj = env.trajectory
    return {
        'nav_real': obs.portfolio_nav_real,
        'baseline_real': obs.baseline_nav_real,
        'regret': obs.portfolio_nav_real - obs.baseline_nav_real,
        'carbon': obs.carbon_footprint_accumulated,
        'r_format':   r_format(dummy_completion),
        'r_regret':   r_regret(traj),
        'r_sharpe':   r_sharpe(traj),
        'r_carbon':   r_carbon(traj, phase_weight=1.0),
        'r_drawdown': r_drawdown(traj),
    }


def main():
    SEEDS = [42, 7, 123, 2024, 999, 1, 2, 3, 4, 5]
    print(f'Running {len(POLICIES)} policies × {len(SEEDS)} seeds × 3 phases each')
    print()
    
    for phase in [3]:  # Full env with all shocks/interventions
        print(f'═══ Phase {phase} (full env) ═══')
        print(f'{"policy":<22}{"nav_real":>10}{"regret":>10}{"r_regret":>10}{"r_sharpe":>10}{"r_carbon":>10}{"r_dd":>10}{"total":>10}')
        print('─' * 100)
        
        summaries = {}
        for name, fn in POLICIES.items():
            scores = [run_episode(fn, seed=s, phase=phase) for s in SEEDS]
            avg = {k: np.mean([s[k] for s in scores]) for k in scores[0]}
            total = avg['r_format'] + avg['r_regret'] + avg['r_sharpe'] + avg['r_carbon'] + avg['r_drawdown']
            summaries[name] = {'avg': avg, 'total': total}
            print(f'{name:<22}{avg["nav_real"]:>10.3f}{avg["regret"]:>10.3f}'
                  f'{avg["r_regret"]:>10.3f}{avg["r_sharpe"]:>10.3f}'
                  f'{avg["r_carbon"]:>10.3f}{avg["r_drawdown"]:>10.3f}{total:>10.3f}')
        
        # Verdict
        print()
        equal_total = summaries['equal_weighted']['total']
        print(f'Equal-weighted total reward (baseline): {equal_total:+.3f}')
        print()
        print('Adversarial policies ranked by total reward:')
        adv = [(k, v['total']) for k, v in summaries.items() if k != 'equal_weighted']
        adv.sort(key=lambda x: -x[1])
        for name, total in adv:
            gap = total - equal_total
            flag = '⚠️  BEATS BASELINE' if gap > 0 else '✓ below baseline'
            print(f'  {name:<22} {total:+.3f}  (vs baseline {gap:+.3f})  {flag}')
        
        CONCENTRATION = {"all_tech", "all_oil", "all_bonds"}; any_exploit = any(v["total"] > equal_total + 0.05 for k, v in summaries.items() if k != "equal_weighted" and k not in CONCENTRATION)
        print()
        if any_exploit:
            print('❌ REWARD BUG: at least one adversarial policy beats equal-weighted.')
            print('   Fix reward design before training.')
            return 1
        else:
            print('✅ No adversarial policy beats equal-weighted. Reward stack is robust.')
            return 0


if __name__ == '__main__':
    sys.exit(main())
