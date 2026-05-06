from gridops.models import GridOpsAction
from scripts.build_gridops_v4_reasoning_traces import (
    derive_context,
    make_trace,
    previous_outcome_from_obs,
    validate_rows,
)


def test_v4_reasoning_trace_contract():
    obs = {
        "hour": 30.0,
        "day_of_episode": 2,
        "demand_kw": 360.0,
        "solar_kw": 0.0,
        "battery_soc": 0.5,
        "grid_price": 18.0,
        "diesel_fuel_remaining": 0.4,
        "diesel_is_on": False,
        "demand_forecast_4h": [340.0, 300.0, 260.0, 180.0],
        "solar_forecast_4h": [0.0, 0.0, 0.0, 0.0],
        "price_forecast_4h": [18.0, 17.0, 16.0, 15.0],
        "cumulative_blackout_kwh": 0.0,
        "cumulative_cost": 0.0,
        "blackout_this_step": 42.0,
        "cost_this_step": 9000.0,
        "grid_kw_this_step": 0.0,
        "narration": "Islanded crisis hour.",
        "flow_shed": 0.0,
        "flow_diesel": 40.0,
    }
    prior = dict(obs, hour=29.0, battery_soc=0.7, blackout_this_step=0.0, flow_diesel=0.0)
    previous_action = GridOpsAction(battery_dispatch=1.0, diesel_dispatch=0.0, demand_shedding=0.0)
    previous_outcome = previous_outcome_from_obs(obs, prior, previous_action)
    action = GridOpsAction(battery_dispatch=1.0, diesel_dispatch=0.6, demand_shedding=0.0)

    row = make_trace(
        trace_id="test_v4_reasoning",
        task_id="task_3_crisis",
        seed=9999,
        hour=30,
        obs=obs,
        action=action,
        previous_action=previous_action.model_dump(),
        previous_outcome=previous_outcome,
        bucket="previous_action_correction",
        source="test",
    )

    assert row["messages"][1]["content"] == row["prompt"]
    assert row["raw"]["prompt_mode"] == "reason_action"
    assert row["raw"]["derived_context"] == derive_context(obs, "task_3_crisis")
    assert row["raw"]["validation"]["valid"] is True
    assert row["completion"].startswith("<think>")
    assert "<action>" in row["completion"]
    assert validate_rows([row]) == []
