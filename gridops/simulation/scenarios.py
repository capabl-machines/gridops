"""
Scenario generators — demand, solar, and price curves for 72-hour episodes.

Each generator returns a numpy array of length 72 (one value per hour).
Scenarios are seeded for deterministic replay.
"""

from dataclasses import dataclass

import numpy as np


START_HOUR = 6  # Episodes begin at 6 AM


@dataclass
class ScenarioConfig:
    """Knobs that define a task's difficulty."""

    demand_multiplier: float = 1.0      # 1.0 = normal, 1.3 = heatwave
    solar_multiplier: float = 1.0       # 1.0 = clear, 0.7 = haze
    price_floor: float = 4.0            # Rs/kWh
    price_ceiling: float = 8.0          # Rs/kWh
    price_spike_hour: int | None = None # hour at which an evening spike occurs
    price_spike_value: float = 15.0     # Rs/kWh at spike
    heatwave_start_hour: int = 24       # when heatwave kicks in (0 = Day 1 start)
    cloud_hours: list[int] | None = None  # hours with intermittent clouds
    diesel_fuel_capacity: float = 1.0   # 1.0 = full tank (800 kWh worth)
    forecast_noise: float = 0.15        # ±15 % Gaussian noise on forecasts
    grid_outage_hours: list[int] | None = None  # hours where grid cap drops to 0 (islanding)


# ── Demand ───────────────────────────────────────────────────────────────

def _base_demand_curve() -> np.ndarray:
    """24-hour demand template for a 100-home Indian summer community.

    Realistic Indian residential: ~15-20 kWh/home/day → ~100 kW avg, 250 kW peak.
    Grid (200 kW) covers off-peak easily. Solar creates midday surplus.
    Evening 18-22h is THE bottleneck: demand > grid cap → need battery + diesel.
    """
    hourly = np.array([
         50,  45,  40,  40,  45,  55,   # 0-5   night (deep trough, grid surplus)
         70,  85, 100, 110, 115, 120,   # 6-11  morning ramp (solar kicks in)
        125, 130, 130, 120, 115, 140,   # 12-17 afternoon (solar covers, charge battery)
        200, 220, 250, 230, 180, 100,   # 18-23 evening peak → 250 kW at 20:00
    ], dtype=np.float64)
    return hourly


def _rotate(curve_24h: np.ndarray) -> np.ndarray:
    """Rotate a 24-hour curve so index 0 = START_HOUR (6 AM)."""
    return np.roll(curve_24h, -START_HOUR)


def generate_demand(cfg: ScenarioConfig, rng: np.random.Generator) -> np.ndarray:
    """72-hour demand with heatwave multiplier and stochastic noise."""
    base = np.tile(_rotate(_base_demand_curve()), 3)  # 3 days starting at 6 AM
    demand = base.copy()

    # Apply heatwave multiplier after start hour
    hw = cfg.heatwave_start_hour
    if cfg.demand_multiplier != 1.0 and hw < 72:
        demand[hw:] *= cfg.demand_multiplier

    # ±10 % base noise
    noise = 1.0 + rng.normal(0, 0.05, size=72)
    demand *= noise
    return np.clip(demand, 20, 500)


# ── Solar ────────────────────────────────────────────────────────────────

def _base_solar_curve() -> np.ndarray:
    """24-hour solar bell curve peaking at noon, 250 kW capacity."""
    hours = np.arange(24)
    solar = np.maximum(0, 250 * np.sin(np.pi * (hours - 6) / 12))
    solar[:6] = 0
    solar[18:] = 0
    return solar


def generate_solar(cfg: ScenarioConfig, rng: np.random.Generator) -> np.ndarray:
    """72-hour solar with optional haze reduction and cloud dips."""
    base = np.tile(_rotate(_base_solar_curve()), 3)
    solar = base * cfg.solar_multiplier

    # Cloud cover — 50 % drop during listed hours
    if cfg.cloud_hours:
        for h in cfg.cloud_hours:
            if 0 <= h < 72:
                solar[h] *= 0.5

    # Small stochastic variation
    noise = 1.0 + rng.normal(0, 0.03, size=72)
    solar *= noise
    return np.clip(solar, 0, 200)


# ── Grid price ───────────────────────────────────────────────────────────

def _base_price_curve(floor: float, ceiling: float) -> np.ndarray:
    """24-hour IEX-like price curve with clear cheap/expensive periods.

    Night (0-6): near floor (cheap — battery charging window)
    Morning (7-11): moderate
    Afternoon (12-16): moderate-high (solar competes)
    Evening (17-22): near ceiling (peak demand, solar gone — sell window)
    Late night (23): dropping back
    """
    hours = np.arange(24)
    mid = (floor + ceiling) / 2
    amp = (ceiling - floor) / 2
    # Strong evening peak, cheap night
    price = mid + amp * (
        0.6 * np.sin(np.pi * (hours - 4) / 20)       # base daily shape
        + 0.4 * np.exp(-0.5 * ((hours - 20) / 2.5)**2) # evening spike
        - 0.3 * np.exp(-0.5 * ((hours - 3) / 2)**2)    # night trough
    )
    return np.clip(price, floor, ceiling)


def generate_price(cfg: ScenarioConfig, rng: np.random.Generator) -> np.ndarray:
    """72-hour grid price with optional spikes."""
    base = np.tile(_rotate(_base_price_curve(cfg.price_floor, cfg.price_ceiling)), 3)
    price = base.copy()

    # Evening spike
    if cfg.price_spike_hour is not None:
        spike_h = cfg.price_spike_hour
        # Spread spike over 3 hours centered on spike_h
        for offset in range(-1, 2):
            h = spike_h + offset
            if 0 <= h < 72:
                price[h] = max(price[h], cfg.price_spike_value * (1.0 - 0.2 * abs(offset)))

    # Small noise
    noise = 1.0 + rng.normal(0, 0.02, size=72)
    price *= noise
    return np.clip(price, 3, 20)


# ── Forecasts ────────────────────────────────────────────────────────────

def make_forecast(
    true_values: np.ndarray,
    current_hour: int,
    horizon: int,
    noise_frac: float,
    rng: np.random.Generator,
) -> list[float]:
    """Return a noisy forecast for the next `horizon` hours."""
    forecasts = []
    for i in range(1, horizon + 1):
        h = current_hour + i
        if h < len(true_values):
            val = true_values[h]
        else:
            val = true_values[-1]
        noisy = val * (1.0 + rng.normal(0, noise_frac))
        forecasts.append(max(0.0, float(noisy)))
    return forecasts
