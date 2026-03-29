"""
Three task configurations with escalating difficulty.

Task 1: Normal Summer (easy)
Task 2: Heatwave + Clouds (medium)
Task 3: Extreme Crisis (hard)
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
)

TASK_2_HEATWAVE = ScenarioConfig(
    demand_multiplier=1.3,
    solar_multiplier=1.0,
    price_floor=5.0,
    price_ceiling=15.0,
    price_spike_hour=44,         # Day 2, 20:00 (hour 44)
    price_spike_value=18.0,
    heatwave_start_hour=24,      # Day 2 start
    cloud_hours=[30, 31, 36, 37, 54, 55],  # intermittent Day 2-3
    diesel_fuel_capacity=1.0,
    forecast_noise=0.15,
)

TASK_3_CRISIS = ScenarioConfig(
    demand_multiplier=1.5,
    solar_multiplier=0.7,
    price_floor=8.0,
    price_ceiling=20.0,
    price_spike_hour=44,
    price_spike_value=20.0,
    heatwave_start_hour=0,       # heatwave from the start
    cloud_hours=list(range(8, 16)) + list(range(32, 40)) + list(range(56, 64)),
    diesel_fuel_capacity=0.33,   # 800 kWh ≈ 8 hrs at 100 kW
    forecast_noise=0.15,
)

TASKS = {
    "task_1_normal": TASK_1_NORMAL,
    "task_2_heatwave": TASK_2_HEATWAVE,
    "task_3_crisis": TASK_3_CRISIS,
}
