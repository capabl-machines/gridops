"""
Oracle strategy test — validates physics + grader + strategy gaps.

New action space: battery_dispatch, diesel_dispatch, demand_shedding.
Grid is the slack variable (absorbs residual up to ±200 kW).
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from gridops.server.environment import GridOpsEnvironment
from gridops.models import GridOpsAction
from gridops.tasks.definitions import TASKS


def oracle_policy(obs: dict) -> GridOpsAction:
    """
    Smart oracle: manages battery for arbitrage + evening peak coverage.

    Strategy:
      - Night (cheap grid): charge battery
      - Solar midday: let solar cover demand, charge battery from surplus
      - Pre-peak (15-17h): top up battery
      - Evening peak (18-22h): discharge battery to reduce expensive grid import
      - Use diesel only when grid is at capacity AND battery is depleted
      - Shed demand only as last resort during extreme peaks
    """
    hour_of_day = int(obs["hour"]) % 24
    soc = obs["battery_soc"]
    price = obs["grid_price"]
    demand = obs["demand_kw"]
    solar = obs["solar_kw"]
    fuel = obs["diesel_fuel_remaining"]

    battery = 0.0   # -1=charge, +1=discharge
    diesel = 0.0
    shedding = 0.0

    # Net demand after solar
    net = demand - solar

    if hour_of_day < 6:
        # Night: cheap power, charge battery aggressively
        if soc < 0.9:
            battery = -0.8  # charge
        else:
            battery = 0.0

    elif 6 <= hour_of_day < 15:
        # Solar hours: if solar > demand, charge battery from surplus
        if solar > demand:
            # Surplus — charge battery (grid absorbs the rest as export)
            if soc < 0.95:
                battery = -min(1.0, (solar - demand) / 100.0)
            else:
                battery = 0.0  # battery full, surplus exports to grid
        else:
            # Deficit — grid covers it. Charge battery if cheap.
            if soc < 0.7 and price < 6:
                battery = -0.5
            else:
                battery = 0.0

    elif 15 <= hour_of_day < 18:
        # Pre-peak: ensure battery is charged for evening
        if soc < 0.8:
            battery = -0.8  # charge hard
        else:
            battery = 0.0

    elif 18 <= hour_of_day < 23:
        # Evening peak: discharge battery to cover demand beyond grid cap
        if net > GRID_MAX_KW and soc > 0.1:
            # Need battery to cover the gap
            gap = net - GRID_MAX_KW
            battery = min(1.0, gap / 100.0)

            # If battery can't cover full gap, use diesel
            remaining = gap - battery * 100
            if remaining > 0 and fuel > 0.05:
                diesel = min(1.0, remaining / 100.0)

            # If still short, shed demand
            remaining2 = remaining - diesel * 100
            if remaining2 > 0:
                shedding = min(1.0, remaining2 / (demand * 0.20 + 1))
        elif price > 10 and soc > 0.5:
            # Expensive grid: discharge battery to save money
            battery = min(0.6, (price - 8) / 10.0)
        else:
            battery = 0.0

    else:
        # Hour 23: low demand, recharge if depleted
        if soc < 0.4:
            battery = -0.5
        else:
            battery = 0.0

    return GridOpsAction(
        battery_dispatch=float(np.clip(battery, -1, 1)),
        diesel_dispatch=float(np.clip(diesel, 0, 1)),
        demand_shedding=float(np.clip(shedding, 0, 1)),
    )


GRID_MAX_KW = 200.0  # for oracle calculations


def heuristic_do_nothing(obs: dict) -> GridOpsAction:
    """Baseline: do nothing. Grid handles everything as slack."""
    return GridOpsAction(battery_dispatch=0.0, diesel_dispatch=0.0, demand_shedding=0.0)


def heuristic_always_discharge(obs: dict) -> GridOpsAction:
    """Bad: always discharge battery → empty for evening → blackout."""
    return GridOpsAction(battery_dispatch=1.0, diesel_dispatch=0.0, demand_shedding=0.0)


def heuristic_always_diesel(obs: dict) -> GridOpsAction:
    """Wasteful: always run diesel → hemorrhages money at Rs 25/kWh."""
    return GridOpsAction(battery_dispatch=0.0, diesel_dispatch=1.0, demand_shedding=0.0)


def run_episode(env, policy_fn, task_id="task_1_normal", seed=42):
    """Run a full 72-step episode, return grade dict."""
    obs = env.reset(seed=seed, task_id=task_id)
    obs_dict = obs.model_dump()

    for _ in range(72):
        action = policy_fn(obs_dict)
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
