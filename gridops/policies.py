"""Reusable GridOps policies for traces, baselines, and adversarial tests."""

from __future__ import annotations

import numpy as np

from gridops.models import GridOpsAction


GRID_MAX_KW = 200.0


def oracle_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    """Price/SOC/outage-aware operator policy used as SFT expert labels."""
    hour_of_day = (int(obs["hour"]) + 6) % 24
    hour = int(obs["hour"])
    soc = float(obs["battery_soc"])
    price = float(obs["grid_price"])
    demand = float(obs["demand_kw"])
    solar = float(obs["solar_kw"])
    fuel = float(obs["diesel_fuel_remaining"])
    demand_fc = [float(v) for v in obs.get("demand_forecast_4h", [])]
    solar_fc = [float(v) for v in obs.get("solar_forecast_4h", [])]
    price_fc = [float(v) for v in obs.get("price_forecast_4h", [])]

    battery = 0.0
    diesel = 0.0
    shedding = 0.0
    net = demand - solar
    future_net_peak = max([net] + [d - s for d, s in zip(demand_fc, solar_fc)])
    future_price_peak = max([price] + price_fc)
    outage_soon = task_id == "task_3_crisis" and 26 <= hour <= 35
    in_outage = task_id == "task_3_crisis" and 30 <= hour <= 35

    if in_outage:
        gap = max(0.0, demand - solar)
        if soc > 0.18:
            battery = min(1.0, gap / 100.0)
            gap -= battery * 100.0
        if gap > 0 and fuel > 0.04:
            diesel = min(1.0, gap / 100.0)
            gap -= diesel * 100.0
        if gap > 0:
            shedding = min(1.0, gap / max(demand * 0.20, 1.0))
    elif outage_soon:
        if soc < 0.9:
            battery = -0.9
        else:
            battery = 0.0
    elif hour_of_day < 6:
        if soc < 0.9:
            battery = -0.8
    elif 6 <= hour_of_day < 15:
        if solar > demand and soc < 0.95:
            battery = -min(1.0, (solar - demand) / 100.0)
        elif soc < 0.72 and (price < 6.0 or future_net_peak > GRID_MAX_KW):
            battery = -0.5
    elif 15 <= hour_of_day < 18:
        if soc < 0.82 or future_price_peak > 14.0 or future_net_peak > GRID_MAX_KW:
            battery = -0.8
    elif 18 <= hour_of_day < 23:
        if net > GRID_MAX_KW and soc > 0.1:
            gap = net - GRID_MAX_KW
            battery = min(1.0, gap / 100.0)
            gap -= battery * 100.0
            if gap > 0 and fuel > 0.05:
                diesel = min(1.0, gap / 100.0)
                gap -= diesel * 100.0
            if gap > 0:
                shedding = min(1.0, gap / max(demand * 0.20, 1.0))
        elif price > 10.0 and soc > 0.5:
            battery = min(0.6, (price - 8.0) / 10.0)
    else:
        if soc < 0.4:
            battery = -0.5

    return GridOpsAction(
        battery_dispatch=float(np.clip(battery, -1, 1)),
        diesel_dispatch=float(np.clip(diesel, 0, 1)),
        demand_shedding=float(np.clip(shedding, 0, 1)),
    )


def do_nothing_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction()


def always_charge_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction(battery_dispatch=-1.0)


def always_discharge_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction(battery_dispatch=1.0)


def always_diesel_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction(diesel_dispatch=1.0)


def shed_farmer_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction(demand_shedding=1.0)


def diesel_chatter_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction(diesel_dispatch=1.0 if int(obs["hour"]) % 2 == 0 else 0.0)


def blackout_acceptor_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction(battery_dispatch=0.0, diesel_dispatch=0.0, demand_shedding=0.0)


def price_greedy_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    price = float(obs["grid_price"])
    if price > 8.0:
        return GridOpsAction(battery_dispatch=1.0)
    if price < 5.0:
        return GridOpsAction(battery_dispatch=-1.0)
    return GridOpsAction()


def grid_only_policy(obs: dict, task_id: str | None = None) -> GridOpsAction:
    return GridOpsAction(battery_dispatch=0.0, diesel_dispatch=0.0, demand_shedding=0.0)


POLICIES = {
    "oracle": oracle_policy,
    "do_nothing": do_nothing_policy,
    "always_charge": always_charge_policy,
    "always_discharge": always_discharge_policy,
    "always_diesel": always_diesel_policy,
    "shed_farmer": shed_farmer_policy,
    "diesel_chatter": diesel_chatter_policy,
    "blackout_acceptor": blackout_acceptor_policy,
    "price_greedy": price_greedy_policy,
    "grid_only": grid_only_policy,
}
