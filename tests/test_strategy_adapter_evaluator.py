import torch

from gridops.server.environment import GridOpsEnvironment
from scripts.evaluate_gridops_strategy_adapter import generate_strategy, rollout, summarize


class _TokenBatch(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, reply: str):
        self.reply = reply

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"

    def __call__(self, prompt, return_tensors):
        assert prompt == "rendered prompt"
        assert return_tensors == "pt"
        return _TokenBatch({"input_ids": torch.tensor([[10, 11]])})

    def decode(self, _tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return self.reply


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        assert "input_ids" in kwargs
        return torch.tensor([[10, 11, 12, 13]])


def test_generate_strategy_parses_valid_reply():
    reply = (
        '{"battery_bias":"charge","diesel_policy":"avoid","mode":"cost_saving",'
        '"risk_level":"low","shedding_policy":"never"}'
    )
    env = GridOpsEnvironment()
    obs = env.reset(seed=7001, task_id="task_1_normal").model_dump()
    text, strategy, validation = generate_strategy(
        _FakeTokenizer(reply),
        _FakeModel(),
        obs,
        "task_1_normal",
        {"blackout_kwh": 0.0},
        max_new_tokens=32,
    )

    assert text == reply
    assert validation["valid"] is True
    assert strategy.mode == "cost_saving"


def test_rollout_falls_back_cleanly_on_invalid_strategy_reply():
    result = rollout(
        _FakeTokenizer("not-json"),
        _FakeModel(),
        "task_1_normal",
        seed=7001,
        max_new_tokens=16,
        sample_limit=2,
        horizon=3,
        optimizer_horizon=4,
    )

    assert result["total_steps"] == 3
    assert result["valid_strategy_rate"] == 0.0
    assert result["invalid_examples"]
    assert result["invalid_examples"][0]["reason"] == "missing_strategy_json"
    assert result["score"] >= 0.0


def test_strategy_adapter_summary_includes_baselines_and_ceiling_capture():
    rows = [
        {
            "task_id": "task_1_normal",
            "seed": 1,
            "score": 0.8,
            "valid_strategies": 72,
            "total_steps": 72,
            "grade": {"total_blackout_kwh": 0.0, "total_diesel_kwh": 0.0, "actual_cost": 100.0},
        },
        {
            "task_id": "task_2_heatwave",
            "seed": 1,
            "score": 0.82,
            "valid_strategies": 70,
            "total_steps": 72,
            "grade": {"total_blackout_kwh": 1.0, "total_diesel_kwh": 10.0, "actual_cost": 200.0},
        },
    ]

    report = summarize("fake-strategy-adapter", rows)

    assert report["name"] == "fake-strategy-adapter"
    assert report["average_score"] == 0.81
    assert report["valid_strategy_rate"] == round(142 / 144, 4)
    assert report["lp_ceiling_capture"] > 0.0
    assert "v7_deterministic_strategy_controller" in report["baselines"]
    assert report["by_task"]["task_1_normal"]["lp_ceiling_capture"] > 0.0
