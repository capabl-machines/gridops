import copy
import json

from fastapi.testclient import TestClient
from pydantic import ValidationError

from gridops.models import GridOpsAction
from gridops.server.app import app
from gridops.server.environment import GridOpsEnvironment
from gridops.strategy import (
    GridOpsStrategy,
    derive_strategy,
    plan_strategy_action,
    strategy_to_json,
    strategy_to_optimizer_config,
    validate_strategy_completion,
    validate_strategy_payload,
)
from gridops.tool_agent import validate_action_payload
from scripts.build_gridops_strategy_dataset import build_rows
from scripts.validate_traces import validate_file


def _obs_for(task_id: str = "task_3_crisis"):
    env = GridOpsEnvironment()
    obs = env.reset(seed=7601, task_id=task_id)
    return env, obs.model_dump()


def test_strategy_schema_accepts_only_fixed_enums():
    strategy = GridOpsStrategy(
        mode="reliability",
        risk_level="critical",
        battery_bias="discharge",
        diesel_policy="allow_if_blackout",
        shedding_policy="last_resort",
    )
    assert strategy.mode == "reliability"
    try:
        GridOpsStrategy(
            mode="freeform",
            risk_level="critical",
            battery_bias="discharge",
            diesel_policy="allow_if_blackout",
            shedding_policy="last_resort",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("invalid strategy enum was accepted")


def test_strategy_json_completion_is_strict_and_validated():
    strategy = GridOpsStrategy(
        mode="cost_saving",
        risk_level="low",
        battery_bias="charge",
        diesel_policy="avoid",
        shedding_policy="never",
    )
    valid, reason = validate_strategy_completion(strategy_to_json(strategy))
    assert valid, reason

    valid, reason = validate_strategy_completion(f"Use this: {strategy_to_json(strategy)}")
    assert valid is False
    assert reason == "prose_outside_strategy_json"


def test_invalid_strategy_payload_falls_back_safely():
    env, obs = _obs_for("task_3_crisis")
    plan = plan_strategy_action(
        env,
        "task_3_crisis",
        obs,
        strategy={"mode": "bad", "risk_level": "low"},
        optimizer_horizon=4,
    )
    assert plan["strategy_source"] == "derived_fallback"
    assert plan["strategy_validation"]["valid"] is False
    assert validate_action_payload(plan["action"])["valid"] is True


def test_every_strategy_maps_to_valid_optimizer_config():
    _, obs = _obs_for("task_1_normal")
    strategies = [
        GridOpsStrategy(mode="cost_saving", risk_level="low", battery_bias="charge", diesel_policy="avoid", shedding_policy="never"),
        GridOpsStrategy(mode="peak_shaving", risk_level="medium", battery_bias="discharge", diesel_policy="avoid", shedding_policy="never"),
        GridOpsStrategy(mode="outage_prepare", risk_level="high", battery_bias="preserve", diesel_policy="conserve", shedding_policy="last_resort"),
        GridOpsStrategy(mode="reliability", risk_level="critical", battery_bias="discharge", diesel_policy="allow_if_blackout", shedding_policy="last_resort"),
        GridOpsStrategy(mode="recovery", risk_level="high", battery_bias="charge", diesel_policy="allow_if_blackout", shedding_policy="last_resort"),
        GridOpsStrategy(mode="fuel_conservation", risk_level="high", battery_bias="preserve", diesel_policy="conserve", shedding_policy="last_resort"),
    ]
    for strategy in strategies:
        config = strategy_to_optimizer_config(strategy, obs, "task_1_normal")
        assert 1 <= config["horizon"] <= 12
        for key in ["blackout_weight", "diesel_green_weight", "soc_deficit_weight", "fuel_deficit_weight"]:
            assert 0.5 <= config[key] <= 60.0


def test_strategy_controller_returns_bounded_action_without_mutating_env():
    env, obs = _obs_for("task_3_crisis")
    before = copy.deepcopy(env.state.model_dump())
    plan = plan_strategy_action(env, "task_3_crisis", obs, optimizer_horizon=4)
    action = GridOpsAction(**plan["action"])
    after = env.state.model_dump()

    assert before == after
    assert -1.0 <= action.battery_dispatch <= 1.0
    assert 0.0 <= action.diesel_dispatch <= 1.0
    assert 0.0 <= action.demand_shedding <= 1.0
    assert plan["strategy_source"] == "derived"


def test_strategy_api_plan_accepts_valid_and_invalid_strategy_without_stepping():
    client = TestClient(app)
    assert client.post("/api/reset", json={"seed": 7601, "task_id": "task_3_crisis"}).status_code == 200
    before = client.get("/api/state").json()
    valid_strategy = {
        "mode": "reliability",
        "risk_level": "critical",
        "battery_bias": "discharge",
        "diesel_policy": "allow_if_blackout",
        "shedding_policy": "last_resort",
    }

    plan = client.post("/api/plan", json={"strategy": valid_strategy, "optimizer_horizon": 4})
    assert plan.status_code == 200
    body = plan.json()
    assert body["strategy_candidate"]["source"] == "provided"
    assert body["optimizer_config"]["strategy"] == valid_strategy
    assert validate_action_payload(body["selected_action"])["valid"] is True

    fallback = client.post("/api/strategy/plan", json={"strategy": {"mode": "bad"}})
    assert fallback.status_code == 200
    assert fallback.json()["strategy_candidate"]["source"] == "derived_fallback"

    after = client.get("/api/state").json()
    assert after["hour"] == before["hour"]


def test_strategy_dataset_rows_are_strict_and_validate(tmp_path):
    output = tmp_path / "strategy_smoke.jsonl"
    rows, summary = build_rows(
        tasks=["task_1_normal", "task_3_crisis"],
        seeds=[7601],
        stride=24,
        max_rows=None,
        shuffle=False,
        rng_seed=7,
    )
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert summary["validation_failures"] == []
    assert rows
    for row in rows:
        payload = json.loads(row["completion"])
        assert validate_strategy_payload(payload)["valid"] is True
        assert row["raw"]["prompt_mode"] == "strategy_json"
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = validate_file(output)
    assert report["failures"] == []
