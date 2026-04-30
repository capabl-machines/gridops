"""
Oracle strategy test — validates physics + grader + strategy gaps.

New action space: battery_dispatch, diesel_dispatch, demand_shedding.
Grid is the slack variable (absorbs residual up to ±200 kW).
"""

import sys
sys.path.insert(0, ".")

from gridops.server.environment import GridOpsEnvironment
from gridops.models import GridOpsAction
from gridops.policies import (
    always_diesel_policy,
    always_discharge_policy,
    do_nothing_policy,
    oracle_policy,
)
from gridops.tasks.definitions import TASKS


def heuristic_do_nothing(obs: dict) -> GridOpsAction:
    """Baseline: do nothing. Grid handles everything as slack."""
    return do_nothing_policy(obs)


def heuristic_always_discharge(obs: dict) -> GridOpsAction:
    """Bad: always discharge battery → empty for evening → blackout."""
    return always_discharge_policy(obs)


def heuristic_always_diesel(obs: dict) -> GridOpsAction:
    """Wasteful: always run diesel → hemorrhages money at Rs 25/kWh."""
    return always_diesel_policy(obs)


def run_episode(env, policy_fn, task_id="task_1_normal", seed=42):
    """Run a full 72-step episode, return grade dict."""
    obs = env.reset(seed=seed, task_id=task_id)
    obs_dict = obs.model_dump()

    for _ in range(72):
        action = policy_fn(obs_dict, task_id) if policy_fn is oracle_policy else policy_fn(obs_dict)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        if obs.done:
            break

    state = env.state
    return state.grade


def main():
    env = GridOpsEnvironment()
    policies = {
        "Oracle": oracle_policy,
        "Do-Nothing": heuristic_do_nothing,
        "Always-Discharge": heuristic_always_discharge,
        "Always-Diesel": heuristic_always_diesel,
    }

    print("=" * 70)
    print("  GridOps Oracle Test v2 — New Action Space (Battery/Diesel/Shed)")
    print("  Grid is slack. VoLL = Rs 150/kWh. Degradation = Rs 2.5/kWh.")
    print("=" * 70)

    for task_id in TASKS:
        print(f"\n--- {task_id} ---")
        for name, fn in policies.items():
            grade = run_episode(env, fn, task_id)
            if grade:
                print(f"  {name:22s}  score={grade['score']:.4f}  "
                      f"reliability={grade['reliability']:.4f}  "
                      f"cost=Rs {grade['actual_cost']:.0f}  "
                      f"baseline=Rs {grade['baseline_cost']:.0f}")
            else:
                print(f"  {name:22s}  NO GRADE")

    # Determinism check
    print("\n--- Determinism Check (3 runs of Oracle on Task 1) ---")
    scores = []
    for i in range(3):
        grade = run_episode(env, oracle_policy, "task_1_normal", seed=42)
        scores.append(grade["score"])
        print(f"  Run {i+1}: score={grade['score']:.4f}")

    if len(set(f"{s:.6f}" for s in scores)) == 1:
        print("  Deterministic: identical scores across runs")
    else:
        print("  NON-DETERMINISTIC: scores differ!")

    # Detailed oracle breakdown
    print("\n--- Oracle Detailed Breakdown (Task 1) ---")
    grade = run_episode(env, oracle_policy, "task_1_normal", seed=42)
    for k, v in grade.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
