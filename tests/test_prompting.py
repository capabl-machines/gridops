import json

from gridops.models import GridOpsAction
from gridops.prompting import action_to_json, parse_action, validate_completion


def test_action_json_round_trip():
    action = GridOpsAction(battery_dispatch=-0.5, diesel_dispatch=0.25, demand_shedding=0.1)
    text = action_to_json(action)
    assert json.loads(text) == {
        "battery_dispatch": -0.5,
        "diesel_dispatch": 0.25,
        "demand_shedding": 0.1,
    }
    parsed = parse_action(text)
    assert parsed == action


def test_invalid_action_falls_back_to_safe_default():
    parsed = parse_action('{"battery_dispatch": 9, "diesel_dispatch": -1, "demand_shedding": 2}')
    assert parsed == GridOpsAction()


def test_completion_validation_rejects_prose():
    valid, reason = validate_completion('Here is the action: {"battery_dispatch":0,"diesel_dispatch":0,"demand_shedding":0}')
    assert not valid
    assert reason == "prose_outside_json"


def test_completion_validation_accepts_json_only():
    valid, reason = validate_completion('{"battery_dispatch":0,"diesel_dispatch":0,"demand_shedding":0}')
    assert valid
    assert reason == "ok"
