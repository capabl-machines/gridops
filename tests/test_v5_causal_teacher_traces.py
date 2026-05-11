from gridops.models import GridOpsAction
from scripts.build_gridops_v4_reasoning_traces import validate_rows
from scripts.build_gridops_v5_causal_teacher_traces import (
    causal_lp_teacher_action,
    collect_teacher_rows,
)


def test_causal_lp_teacher_action_is_valid():
    obs = {
        "hour": 30.0,
        "day_of_episode": 2,
        "demand_kw": 360.0,
        "solar_kw": 0.0,
        "battery_soc": 0.55,
        "grid_price": 18.0,
        "diesel_fuel_remaining": 0.25,
        "diesel_is_on": False,
        "demand_forecast_4h": [340.0, 300.0, 260.0, 180.0],
        "solar_forecast_4h": [0.0, 0.0, 0.0, 0.0],
        "price_forecast_4h": [18.0, 17.0, 16.0, 15.0],
        "cumulative_blackout_kwh": 0.0,
        "cumulative_cost": 0.0,
        "blackout_this_step": 0.0,
        "cost_this_step": 0.0,
        "grid_kw_this_step": 0.0,
        "narration": "Islanded crisis hour.",
        "flow_shed": 0.0,
        "flow_diesel": 0.0,
    }
    action, info = causal_lp_teacher_action(
        obs,
        "task_3_crisis",
        previous_outcome={"shed_kwh": 0.0},
        horizon=5,
    )

    assert isinstance(action, GridOpsAction)
    assert info["teacher"] == "causal_lp_v5"
    assert info["status"] == "ok"
    assert action.battery_dispatch >= 0.0
    assert action.diesel_dispatch > 0.0


def test_v5_teacher_trace_contract():
    rows, rollouts = collect_teacher_rows(
        seed_start=17000,
        seeds_per_task=1,
        stride=24,
        horizon=3,
    )

    assert len(rows) == 9
    assert len(rollouts) == 3
    assert validate_rows(rows) == []
    assert all((row["raw"] or {})["policy"] == "causal_lp_teacher_v5" for row in rows)
