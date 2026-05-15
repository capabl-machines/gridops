from gridops.critics.lp_critic import (
    build_clean_operator_completion,
    score_action_against_lp,
    validate_clean_reasoning_completion,
)
from gridops.models import GridOpsAction
from gridops.server.environment import GridOpsEnvironment
from gridops.tool_agent import optimize_action, previous_outcome_from_observation


def _advanced_env(task_id: str, seed: int = 7001, target_hour: int = 30):
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    previous_outcome = previous_outcome_from_observation(None)
    previous_action = GridOpsAction()
    while int(obs.hour) < target_hour:
        action, _ = optimize_action(obs.model_dump(), task_id, previous_outcome=previous_outcome, horizon=4)
        obs = env.step(action)
        previous_action = action
        previous_outcome = previous_outcome_from_observation(obs.model_dump())
    return env, obs, previous_action, previous_outcome


def test_lp_critic_returns_valid_actions_without_mutating_env():
    env, obs, _, previous_outcome = _advanced_env("task_2_heatwave", target_hour=18)
    before_hour = env.state.hour
    before_history_len = len(env.state.history)

    result = score_action_against_lp(
        env,
        "task_2_heatwave",
        obs,
        GridOpsAction(),
        previous_outcome=previous_outcome,
        compare_horizon=2,
    )

    GridOpsAction.model_validate(result["lp_action"])
    GridOpsAction.model_validate(result["chosen_action"])
    assert env.state.hour == before_hour
    assert len(env.state.history) == before_history_len


def test_lp_critic_chooses_equal_lp_candidate():
    env, obs, _, previous_outcome = _advanced_env("task_1_normal", target_hour=18)
    lp_action, _ = optimize_action(obs.model_dump(), "task_1_normal", previous_outcome=previous_outcome, horizon=4)

    result = score_action_against_lp(
        env,
        "task_1_normal",
        obs,
        lp_action,
        previous_outcome=previous_outcome,
        compare_horizon=2,
    )

    assert result["chosen_source"] == "candidate"
    assert result["reason"] == "candidate_matches_or_beats_lp"
    assert result["chosen_action"] == result["candidate_action"]


def test_lp_critic_rejects_invalid_candidate():
    env, obs, _, previous_outcome = _advanced_env("task_3_crisis", target_hour=30)

    result = score_action_against_lp(
        env,
        "task_3_crisis",
        obs,
        {"battery_dispatch": 2.0, "diesel_dispatch": -1.0, "demand_shedding": 0.0},
        previous_outcome=previous_outcome,
        compare_horizon=2,
    )

    assert result["chosen_source"] == "lp"
    assert result["reason"] == "candidate_invalid"
    GridOpsAction.model_validate(result["chosen_action"])


def test_clean_completion_validator_rejects_tool_logs():
    bad_completion = """<think>
time_context: Evening.
1st_order: candidate_delta {'blackout_kwh': 2} is worse than optimizer_delta.
2nd_order: The tool-selected action is safer.
previous_action: none.
decision: choose the tool output.
</think>
<action>
{"battery_dispatch":0,"diesel_dispatch":0,"demand_shedding":0}
</action>"""

    valid, reason = validate_clean_reasoning_completion(bad_completion)
    assert not valid
    assert reason.startswith("dict_or_json_inside_think") or reason.startswith("forbidden_clean_reasoning_term")


def test_clean_operator_completion_is_valid_and_clean():
    env, obs, previous_action, previous_outcome = _advanced_env("task_3_crisis", target_hour=30)
    result = score_action_against_lp(
        env,
        "task_3_crisis",
        obs,
        GridOpsAction(),
        previous_outcome=previous_outcome,
        compare_horizon=2,
    )

    completion = build_clean_operator_completion(
        obs,
        "task_3_crisis",
        GridOpsAction.model_validate(result["chosen_action"]),
        result,
        previous_action=previous_action,
        previous_outcome=previous_outcome,
    )
    valid, reason = validate_clean_reasoning_completion(completion)
    assert valid, reason
