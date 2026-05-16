import copy
import json

from gridops.server.environment import GridOpsEnvironment
from gridops.strategy import validate_strategy_payload
from scripts.build_gridops_strategy_dpo_pairs import (
    build_pairs,
    canonical_strategy_candidates,
    parse_seed_values,
    parse_task_seed_map,
    score_strategy_candidate,
)


def test_dpo_candidate_generator_returns_unique_valid_strategies():
    env = GridOpsEnvironment()
    obs = env.reset(seed=7701, task_id="task_3_crisis").model_dump()
    strategies = canonical_strategy_candidates(obs, "task_3_crisis", {"blackout_kwh": 0.0})
    payloads = [strategy.model_dump() for strategy in strategies]
    serialized = [json.dumps(payload, sort_keys=True) for payload in payloads]

    assert len(strategies) >= 6
    assert len(serialized) == len(set(serialized))
    for payload in payloads:
        assert validate_strategy_payload(payload)["valid"] is True


def test_score_strategy_candidate_does_not_mutate_env():
    env = GridOpsEnvironment()
    obs = env.reset(seed=7701, task_id="task_2_heatwave").model_dump()
    strategy = canonical_strategy_candidates(obs, "task_2_heatwave", {"blackout_kwh": 0.0})[0]
    before = copy.deepcopy(env.state.model_dump())

    scored = score_strategy_candidate(
        env,
        "task_2_heatwave",
        strategy,
        {"blackout_kwh": 0.0, "shed_kwh": 0.0},
        horizon=3,
        optimizer_horizon=4,
    )

    assert env.state.model_dump() == before
    assert scored["completion"]
    assert scored["delta"]["cost"] >= 0.0
    assert "preference_score" in scored
    assert scored["actions"]


def test_parse_task_seed_map_supports_ranges_and_defaults():
    assert parse_seed_values("7001,7003-7005") == [7001, 7003, 7004, 7005]

    mapped = parse_task_seed_map(
        "task_3_crisis=7801-7803;task_1_normal=8001",
        tasks=["task_1_normal", "task_3_crisis"],
        seeds=[1, 2],
    )

    assert mapped["task_3_crisis"] == [7801, 7802, 7803]
    assert mapped["task_1_normal"] == [8001]


def test_build_dpo_pairs_outputs_chosen_and_rejected_strategy_json():
    rows, summary = build_pairs(
        tasks=["task_1_normal", "task_3_crisis"],
        seeds=[7701],
        task_seed_map={"task_1_normal": [7701], "task_3_crisis": [7701, 7702]},
        stride=24,
        horizon=3,
        optimizer_horizon=4,
        min_margin=0.0,
        pairs_per_state=2,
        max_pairs=None,
        rng_seed=17,
        shuffle=False,
    )

    assert rows
    assert summary["validation_failures"] == []
    assert summary["pairs_per_state"] == 2
    assert summary["task_seed_map"]["task_3_crisis"] == [7701, 7702]
    assert sum(1 for row in rows if row["task_id"] == "task_3_crisis") > sum(1 for row in rows if row["task_id"] == "task_1_normal")
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert validate_strategy_payload(row["chosen"])["valid"] is True
        assert validate_strategy_payload(row["rejected"])["valid"] is True
        assert row["chosen"] != row["rejected"]
        assert row["raw"]["chosen_score"] >= row["raw"]["rejected_score"]
        assert row["raw"]["candidate_count"] >= 2
        assert row["raw"]["pair_kind"] in {"near_miss", "plausible_contrast", "strong_contrast", "worst_contrast"}
