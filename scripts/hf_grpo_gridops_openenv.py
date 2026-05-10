"""OpenEnv-backed GRPO scaffold for GridOps.

This file intentionally starts as a guarded scaffold rather than a long-running
training script. It defines the task sampler and reward contract we want before
we spend GPU time on RL.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.policies import oracle_policy
from gridops.prompting import (
    extract_action_json,
    format_reason_action_observation,
    messages_for_reason_action_observation,
    validate_reason_action_completion,
)
from gridops.server.environment import GridOpsEnvironment
from scripts.build_gridops_v4_reasoning_traces import (
    derive_context,
    previous_outcome_from_obs,
)

try:
    from datasets import Dataset
    from huggingface_hub import HfApi, upload_folder
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    Dataset = None
    HfApi = None
    upload_folder = None
    PeftModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    TrainerCallback = object
    GRPOConfig = None
    GRPOTrainer = None


TASK_SEEDS = {
    "task_1_normal": range(15000, 15080),
    "task_2_heatwave": range(16000, 16080),
    "task_3_crisis": range(17000, 17080),
}
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INIT_ADAPTER = "77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4"
DEFAULT_MODEL_REPO = "77ethers/gridops-models"
DEFAULT_RUN_LABEL = "grpo_qwen25_3b_gridops_openenv_v4_smoke"


@dataclass
class RewardBreakdown:
    total: float
    format_reward: float
    env_reward: float
    regret_reward: float
    blackout_penalty: float
    diesel_context_reward: float
    brevity_reward: float
    valid: bool
    reason: str
    action: dict[str, Any] | None


def replay_to_state(task_id: str, seed: int, hour: int) -> tuple[GridOpsEnvironment, dict[str, Any], dict[str, Any], dict[str, Any]]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    previous_action = GridOpsAction()
    previous_outcome = {
        "blackout_kwh": 0.0,
        "battery_soc_delta": 0.0,
        "diesel_used_kwh": 0.0,
        "cost": 0.0,
    }
    prior_obs: dict[str, Any] | None = None
    prior_action: GridOpsAction | None = None
    for _ in range(hour):
        obs_dict = obs.model_dump()
        if prior_obs is not None and prior_action is not None:
            previous_outcome = previous_outcome_from_obs(obs_dict, prior_obs, prior_action)
            previous_action = prior_action
        prior_obs = obs_dict
        prior_action = oracle_policy(obs_dict, task_id)
        obs = env.step(prior_action)
        if obs.done:
            break
    obs_dict = obs.model_dump()
    return env, obs_dict, previous_action.model_dump(), previous_outcome


def build_prompt(task_id: str, seed: int, hour: int) -> dict[str, Any]:
    _, obs, previous_action, previous_outcome = replay_to_state(task_id, seed, hour)
    derived = derive_context(obs, task_id)
    return {
        "task_id": task_id,
        "seed": seed,
        "hour": hour,
        "prompt": messages_for_reason_action_observation(obs, derived, previous_action, previous_outcome),
        "prompt_text": format_reason_action_observation(obs, derived, previous_action, previous_outcome),
        "observation": obs,
        "derived_context": derived,
        "previous_action": previous_action,
        "previous_outcome": previous_outcome,
    }


def sample_prompt_specs(limit: int) -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = []
    preferred_hours = {
        "task_1_normal": [10, 14, 17, 18, 19, 20, 38, 42, 62, 66],
        "task_2_heatwave": [13, 14, 17, 18, 19, 20, 37, 38, 39, 62, 63],
        "task_3_crisis": [25, 26, 29, 30, 31, 32, 33, 34, 35, 37, 38],
    }
    seed_offsets = range(80)
    hour_idx = 0
    while len(specs) < limit:
        for offset in seed_offsets:
            for task_id, seeds in TASK_SEEDS.items():
                seed = list(seeds)[offset % len(list(seeds))]
                hour = preferred_hours[task_id][hour_idx % len(preferred_hours[task_id])]
                specs.append((task_id, int(seed), hour))
                if len(specs) >= limit:
                    return specs
            hour_idx += 1
    return specs


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    return str(completion or "")


def score_action_horizon(task_id: str, seed: int, hour: int, action: GridOpsAction, horizon: int) -> dict[str, float]:
    env, _, _, _ = replay_to_state(task_id, seed, hour)
    total_reward = 0.0
    blackout = 0.0
    diesel = 0.0
    cost = 0.0
    obs = env.step(action)
    for step_idx in range(max(1, horizon)):
        obs_dict = obs.model_dump()
        total_reward += float(obs.reward or 0.0)
        blackout += float(obs_dict.get("blackout_this_step", 0.0))
        diesel += float(obs_dict.get("flow_diesel", 0.0))
        cost += float(obs_dict.get("cost_this_step", 0.0))
        if obs.done or step_idx == horizon - 1:
            break
        obs = env.step(oracle_policy(obs_dict, task_id))
    return {
        "reward": total_reward,
        "blackout_kwh": blackout,
        "diesel_kwh": diesel,
        "cost": cost,
    }


def reward_completion(completion: str, prompt_row: dict[str, Any], horizon: int) -> RewardBreakdown:
    valid, reason = validate_reason_action_completion(completion)
    payload = extract_action_json(completion)
    if not valid or payload is None:
        return RewardBreakdown(
            total=-25.0,
            format_reward=-25.0,
            env_reward=0.0,
            regret_reward=0.0,
            blackout_penalty=0.0,
            diesel_context_reward=0.0,
            brevity_reward=0.0,
            valid=False,
            reason=reason,
            action=payload,
        )

    action = GridOpsAction(**payload)
    task_id = prompt_row["task_id"]
    seed = int(prompt_row["seed"])
    hour = int(prompt_row["hour"])
    candidate = score_action_horizon(task_id, seed, hour, action, horizon)
    oracle = score_action_horizon(task_id, seed, hour, oracle_policy(prompt_row["observation"], task_id), horizon)

    env_reward = 0.20 * float(candidate["reward"])
    regret_reward = max(-1.0, min(1.0, 0.20 * float(candidate["reward"] - oracle["reward"])))
    blackout_penalty = -0.03 * float(candidate["blackout_kwh"])
    derived = prompt_row.get("derived_context") or {}
    high_crisis_gap = task_id == "task_3_crisis" and (
        derived.get("grid_status") == "outage" or float(derived.get("max_future_supply_gap_kw", 0.0)) > 80.0
    )
    diesel = float(action.diesel_dispatch)
    if high_crisis_gap and 0.05 <= diesel <= 1.0:
        diesel_context_reward = 0.25
    elif task_id != "task_3_crisis" and diesel > 0.05:
        diesel_context_reward = -0.35 * diesel
    else:
        diesel_context_reward = 0.0
    brevity_reward = -0.001 * max(0, len(completion) - 900)
    total = 1.0 + env_reward + regret_reward + blackout_penalty + diesel_context_reward + brevity_reward
    return RewardBreakdown(
        total=float(total),
        format_reward=1.0,
        env_reward=env_reward,
        regret_reward=regret_reward,
        blackout_penalty=blackout_penalty,
        diesel_context_reward=diesel_context_reward,
        brevity_reward=brevity_reward,
        valid=True,
        reason="ok",
        action=payload,
    )


def smoke_reward_contract(output: Path, horizon: int, limit: int) -> None:
    rows = []
    for task_id, seed, hour in sample_prompt_specs(limit):
        prompt_row = build_prompt(task_id, seed, hour)
        oracle_action = oracle_policy(prompt_row["observation"], task_id)
        oracle_completion = (
            "<think>\n"
            "time_context: smoke oracle reference.\n"
            "1st_order: valid action is required.\n"
            "2nd_order: short-horizon environment reward decides quality.\n"
            "previous_action: use the provided feedback.\n"
            "decision: emit bounded JSON.\n"
            "</think>\n<action>\n"
            + json.dumps(oracle_action.model_dump(), separators=(",", ":"))
            + "\n</action>"
        )
        invalid_completion = "<think>bad</think>\n<action>\n{\"battery_dispatch\": 2.0, \"diesel_dispatch\": 0, \"demand_shedding\": 0}\n</action>"
        rows.append(
            {
                "task_id": task_id,
                "seed": seed,
                "hour": hour,
                "oracle_reward": reward_completion(oracle_completion, prompt_row, horizon).__dict__,
                "invalid_reward": reward_completion(invalid_completion, prompt_row, horizon).__dict__,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


def build_grpo_dataset(limit: int) -> list[dict[str, Any]]:
    rows = []
    for task_id, seed, hour in sample_prompt_specs(limit):
        rows.append(build_prompt(task_id, seed, hour))
    return rows


class RewardRecorder:
    def __init__(self, horizon: int, output_dir: Path):
        self.horizon = horizon
        self.output_dir = output_dir
        self.calls: list[dict[str, Any]] = []
        # TRL's GRPOTrainer records reward function names by reading
        # `__name__`, even when the reward function is a callable object.
        self.__name__ = "gridops_openenv_reward"

    def __call__(
        self,
        completions: list[Any],
        task_id: list[str],
        seed: list[int],
        hour: list[int],
        observation: list[dict[str, Any]],
        derived_context: list[dict[str, Any]],
        log_extra=None,
        log_metric=None,
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []
        valid_count = 0
        diesel_positive = 0
        lengths: list[int] = []
        details: list[dict[str, Any]] = []
        for completion, task, one_seed, one_hour, obs, derived in zip(
            completions,
            task_id,
            seed,
            hour,
            observation,
            derived_context,
        ):
            text = completion_text(completion)
            prompt_row = {
                "task_id": task,
                "seed": int(one_seed),
                "hour": int(one_hour),
                "observation": obs,
                "derived_context": derived,
            }
            breakdown = reward_completion(text, prompt_row, self.horizon)
            rewards.append(float(breakdown.total))
            valid_count += int(breakdown.valid)
            action = breakdown.action or {}
            diesel_positive += int(float(action.get("diesel_dispatch", 0.0) or 0.0) > 0.05)
            lengths.append(len(text))
            details.append(
                {
                    "task_id": task,
                    "seed": int(one_seed),
                    "hour": int(one_hour),
                    "reward": breakdown.__dict__,
                    "completion": text[:1200],
                }
            )

        if log_metric and rewards:
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
            log_metric("gridops/reward_mean", mean_reward)
            log_metric("gridops/reward_std", variance ** 0.5)
            log_metric("gridops/valid_action_rate", valid_count / len(rewards))
            log_metric("gridops/diesel_positive_rate", diesel_positive / len(rewards))
            log_metric("gridops/completion_mean_chars", sum(lengths) / len(lengths))
            log_metric("gridops/completion_min_chars", min(lengths))
            log_metric("gridops/completion_max_chars", max(lengths))
        if log_extra:
            log_extra("gridops_task_id", list(task_id))
            log_extra("gridops_reward", rewards)
        self.calls.append(
            {
                "rewards": rewards,
                "valid_action_rate": valid_count / max(len(rewards), 1),
                "diesel_positive_rate": diesel_positive / max(len(rewards), 1),
                "mean_chars": sum(lengths) / max(len(lengths), 1),
                "details": details[:12],
            }
        )
        return rewards

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gridops_grpo_reward_calls.json").write_text(json.dumps({"calls": self.calls}, indent=2) + "\n")


class RewardSaveCallback(TrainerCallback):
    def __init__(self, recorder: RewardRecorder):
        self.recorder = recorder

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        self.recorder.save()

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        self.recorder.save()


def model_path_kwargs(path: str) -> tuple[str, dict[str, str]]:
    if Path(path).exists():
        return path, {}
    parts = path.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2]), {"subfolder": "/".join(parts[2:])}
    return path, {}


def load_trainable_model(base_model: str, init_adapter: str, token: str | None, load_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        token=token,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    adapter_id, adapter_kwargs = model_path_kwargs(init_adapter)
    model = PeftModel.from_pretrained(model, adapter_id, token=token, is_trainable=True, **adapter_kwargs)
    return tokenizer, model


def make_grpo_config(args: argparse.Namespace, output_dir: Path):
    params = inspect.signature(GRPOConfig).parameters
    requested: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "logging_steps": 1,
        "save_steps": max(10, args.max_steps),
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "use_vllm": False,
        "beta": args.beta,
        "scale_rewards": "group",
        "log_completions": True,
        "num_completions_to_print": 2,
    }
    supported = {key: value for key, value in requested.items() if key in params}
    return GRPOConfig(**supported)


def train_grpo(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in {
            "datasets": Dataset,
            "huggingface_hub": HfApi,
            "peft": PeftModel,
            "transformers": AutoModelForCausalLM,
            "trl": GRPOTrainer,
        }.items()
        if value is None
    ]
    if missing:
        raise RuntimeError(f"Missing GRPO dependencies: {', '.join(missing)}")

    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    output_dir = Path("outputs") / args.run_label
    rows = build_grpo_dataset(args.prompt_limit)
    dataset = Dataset.from_list(rows)
    tokenizer, model = load_trainable_model(args.base_model, args.init_adapter, token, load_4bit=not args.no_4bit)
    recorder = RewardRecorder(args.horizon, output_dir)
    config = make_grpo_config(args, output_dir)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=recorder,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[RewardSaveCallback(recorder)],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    recorder.save()

    metrics = {
        "base_model": args.base_model,
        "init_adapter": args.init_adapter,
        "run_label": args.run_label,
        "prompt_limit": args.prompt_limit,
        "horizon": args.horizon,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
    }
    (output_dir / "gridops_grpo_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    upload_target = ""
    if args.upload:
        api = HfApi(token=token)
        api.create_repo(args.model_repo, repo_type="model", exist_ok=True, private=False)
        upload_folder(
            folder_path=str(output_dir),
            repo_id=args.model_repo,
            repo_type="model",
            path_in_repo=args.run_label,
            token=token,
            commit_message=f"Upload GridOps GRPO adapter {args.run_label}",
        )
        upload_target = f"{args.model_repo}/{args.run_label}"
    print(json.dumps({"uploaded_to": upload_target, **metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke_reward_contract", "train"], default="smoke_reward_contract")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output", default="evals/gridops_grpo_openenv_reward_contract_smoke.json")
    parser.add_argument("--base-model", default=os.environ.get("GRIDOPS_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--init-adapter", default=os.environ.get("GRIDOPS_INIT_ADAPTER", DEFAULT_INIT_ADAPTER))
    parser.add_argument("--model-repo", default=os.environ.get("GRIDOPS_MODEL_REPO", DEFAULT_MODEL_REPO))
    parser.add_argument("--run-label", default=os.environ.get("GRIDOPS_GRPO_RUN_LABEL", DEFAULT_RUN_LABEL))
    parser.add_argument("--prompt-limit", type=int, default=int(os.environ.get("GRIDOPS_GRPO_PROMPT_LIMIT", "24")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("GRIDOPS_GRPO_STEPS", "8")))
    parser.add_argument("--num-generations", type=int, default=int(os.environ.get("GRIDOPS_GRPO_NUM_GENERATIONS", "2")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("GRIDOPS_GRPO_BATCH_SIZE", "2")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRIDOPS_GRPO_GRAD_ACCUM", "1")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("GRIDOPS_GRPO_LR", "2e-6")))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("GRIDOPS_GRPO_MAX_PROMPT_LENGTH", "1400")))
    parser.add_argument("--max-completion-length", type=int, default=int(os.environ.get("GRIDOPS_GRPO_MAX_COMPLETION_LENGTH", "220")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("GRIDOPS_GRPO_TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("GRIDOPS_GRPO_TOP_P", "0.95")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("GRIDOPS_GRPO_TOP_K", "50")))
    parser.add_argument("--beta", type=float, default=float(os.environ.get("GRIDOPS_GRPO_BETA", "0.0")))
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    parser.set_defaults(upload=os.environ.get("GRIDOPS_UPLOAD", "1").lower() not in {"0", "false", "no"})
    args = parser.parse_args()

    if args.mode == "smoke_reward_contract":
        smoke_reward_contract(Path(args.output), args.horizon, args.limit)
        return

    train_grpo(args)


if __name__ == "__main__":
    main()
