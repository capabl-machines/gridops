from fastapi.testclient import TestClient

from gridops.models import GridOpsAction
from gridops.server.app import app
from gridops.server.environment import GridOpsEnvironment
from gridops.tool_agent import (
    PlanInputs,
    compare_candidates,
    optimize_action,
    plan_action,
    tool_corrected_completion,
    validate_action_payload,
)
from gridops.prompting import parse_action, validate_reason_action_completion


def _obs_for(task_id: str):
    env = GridOpsEnvironment()
    obs = env.reset(seed=7001, task_id=task_id)
    return env, obs.model_dump()


def test_optimizer_returns_valid_actions_for_all_tasks():
    for task_id in ["task_1_normal", "task_2_heatwave", "task_3_crisis"]:
        _, obs = _obs_for(task_id)
        action, info = optimize_action(obs, task_id, horizon=4)
        assert isinstance(action, GridOpsAction)
        assert info["status"] == "ok"


def test_validator_rejects_out_of_bounds_action():
    result = validate_action_payload({"battery_dispatch": 2.0, "diesel_dispatch": -1.0, "demand_shedding": 0.0})
    assert result["valid"] is False
    assert result["reason"].startswith("invalid_action")


def test_hybrid_selector_falls_back_to_optimizer_for_invalid_model_action():
    env, obs = _obs_for("task_3_crisis")
    result = plan_action(
        env,
        PlanInputs(
            task_id="task_3_crisis",
            observation=obs,
            model_action={"battery_dispatch": 9, "diesel_dispatch": 0, "demand_shedding": 0},
            compare_horizon=2,
        ),
    )
    assert result["selected_source"] == "optimizer"
    assert result["selection_reason"] == "model_candidate_invalid_or_missing"
    assert validate_action_payload(result["selected_action"])["valid"] is True


def test_tool_corrected_completion_is_valid_reason_action():
    env, obs = _obs_for("task_2_heatwave")
    result = plan_action(
        env,
        PlanInputs(
            task_id="task_2_heatwave",
            observation=obs,
            model_action={"battery_dispatch": 0, "diesel_dispatch": 0, "demand_shedding": 0},
            compare_horizon=2,
        ),
    )
    completion = tool_corrected_completion(obs=obs, task_id="task_2_heatwave", plan=result)
    valid, reason = validate_reason_action_completion(completion)
    assert valid, reason
    assert parse_action(completion) == GridOpsAction(**result["selected_action"])


def test_compare_candidates_exposes_blackout_risk():
    env = GridOpsEnvironment()
    obs = env.reset(seed=7001, task_id="task_3_crisis")
    while int(obs.hour) < 30:
        action, _ = optimize_action(obs.model_dump(), "task_3_crisis", horizon=4)
        obs = env.step(action)
    optimizer_action, _ = optimize_action(obs.model_dump(), "task_3_crisis", horizon=4)
    comparison = compare_candidates(
        env,
        "task_3_crisis",
        {
            "do_nothing": GridOpsAction(),
            "optimizer": optimizer_action,
        },
        horizon=2,
    )
    assert comparison["candidates"]["do_nothing"]["delta"]["blackout_kwh"] >= 0
    assert comparison["candidates"]["optimizer"]["delta"]["blackout_kwh"] >= 0


def test_tool_agent_api_endpoints_do_not_step_environment():
    client = TestClient(app)
    reset = client.post("/api/reset", json={"seed": 7001, "task_id": "task_1_normal"})
    assert reset.status_code == 200
    state_before = client.get("/api/state").json()

    optimize = client.post("/api/tools/optimize", json={})
    assert optimize.status_code == 200
    assert validate_action_payload(optimize.json()["action"])["valid"] is True

    validate = client.post(
        "/api/tools/validate",
        json={"action": {"battery_dispatch": 0, "diesel_dispatch": 0, "demand_shedding": 0}},
    )
    assert validate.status_code == 200
    assert validate.json()["valid"] is True

    plan = client.post("/api/plan", json={"model_action": {"battery_dispatch": 2, "diesel_dispatch": 0, "demand_shedding": 0}})
    assert plan.status_code == 200
    assert plan.json()["selected_source"] == "optimizer"

    state_after = client.get("/api/state").json()
    assert state_after["hour"] == state_before["hour"]
