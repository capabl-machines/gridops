"""
Three task configurations with escalating difficulty.

Task 1: Normal Summer (easy) — basic arbitrage
Task 2: Heatwave + Price Spike (medium) — forces forecasting / temporal planning
Task 3: Extreme Crisis + Grid Outage (hard) — forces islanding / constraint management
"""

from gridops.simulation.scenarios import ScenarioConfig

TASK_1_NORMAL = ScenarioConfig(
    demand_multiplier=1.0,
    solar_multiplier=1.0,
    price_floor=3.0,
    price_ceiling=12.0,
    price_spike_hour=None,
    price_spike_value=12.0,
    heatwave_start_hour=72,      # no heatwave
    cloud_hours=None,
    diesel_fuel_capacity=1.0,
    forecast_noise=0.15,
    grid_outage_hours=None,
)

# Task 2: Heatwave + severe evening price spikes on Day 2-3.
# Mid-day prices dip (solar glut), then spike to Rs 18-20 at evening.
# A greedy agent discharges battery mid-day; an RL agent reads the
# 4-hour forecast and HOLDS charge for the evening spike.
TASK_2_HEATWAVE = ScenarioConfig(
    demand_multiplier=1.3,
    solar_multiplier=1.0,
    price_floor=3.0,
    price_ceiling=15.0,
    price_spike_hour=36,         # Day 2, ~18:00 (step 36 = hour 12 of day 2)
    price_spike_value=20.0,      # severe spike — hold battery for this
    heatwave_start_hour=24,      # Day 2 start
    cloud_hours=[30, 31, 36, 37, 54, 55],
    diesel_fuel_capacity=1.0,
    forecast_noise=0.15,
    grid_outage_hours=None,
)

# Task 3: Full crisis + 6-hour grid outage on Day 2 evening.
# Grid cap drops to 0 kW during outage — agent must survive on
# battery + diesel + shedding alone. Tests true islanding capability.
TASK_3_CRISIS = ScenarioConfig(
    demand_multiplier=1.5,
    solar_multiplier=0.7,
    price_floor=8.0,
    price_ceiling=20.0,
    price_spike_hour=44,
    price_spike_value=20.0,
    heatwave_start_hour=0,
    cloud_hours=list(range(8, 16)) + list(range(32, 40)) + list(range(56, 64)),
    diesel_fuel_capacity=0.33,   # 800 kWh ≈ 8 hrs at 100 kW
    forecast_noise=0.15,
    grid_outage_hours=list(range(30, 36)),  # 6-hour outage: Day 2, ~12:00-18:00
)

TASKS = {
    "task_1_normal": TASK_1_NORMAL,
    "task_2_heatwave": TASK_2_HEATWAVE,
    "task_3_crisis": TASK_3_CRISIS,
}
