from gridops.models import GridOpsAction
from scripts.generate_openrouter_tool_augmented_traces import (
    candidate_schema,
    choose_trace_action,
    collect_observations,
    score_action,
    score_candidates,
)


def test_candidate_schema_is_strict_and_batch_sized():
    rows = collect_observations(per_task=1, seed_start=8100)
    schema = candidate_schema(rows[:2], candidates_per_item=5)
    body = schema["schema"]["properties"]["items"]
    candidate_body = body["items"]["properties"]["candidates"]
    assert schema["strict"] is True
    assert body["minItems"] == 2
    assert body["maxItems"] == 2
    assert candidate_body["minItems"] == 5
    assert candidate_body["maxItems"] == 5


def test_short_horizon_scoring_is_deterministic():
    row = collect_observations(per_task=1, seed_start=8200)[0]
    action = GridOpsAction(**row["oracle_action"])
    first = score_action(row["task_id"], row["seed"], row["hour"], action, horizon=6)
    second = score_action(row["task_id"], row["seed"], row["hour"], action, horizon=6)
    assert first == second
    assert first["horizon_steps"] == 6


def test_oracle_failure_replay_accepts_important_bad_candidate_state():
    row = collect_observations(per_task=10, seed_start=8300)[25]
    model_outputs = {
        "test_model": [
            {"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0, "tag": "bad_hold"},
        ]
    }
    scored = score_candidates(row, model_outputs, horizon=6)
    chosen_action, chosen_source, reason = choose_trace_action(row, scored)
    assert row["task_id"] == "task_3_crisis"
    assert chosen_action == row["oracle_action"]
    assert chosen_source == "oracle"
    assert reason == "oracle_failure_replay"
