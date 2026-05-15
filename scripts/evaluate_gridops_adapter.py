"""Evaluate a local or Hub LoRA adapter in the GridOps environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.prompting import (
    extract_action_json,
    messages_for_observation,
    messages_for_reason_action_observation,
    parse_action,
    validate_completion,
    validate_reason_action_completion,
)
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import derive_control_context, previous_outcome_from_observation


def invalid_action_diagnostics(reply: str) -> dict[str, Any]:
    """Return extra debugging context for invalid model completions."""
    payload = extract_action_json(reply)
    parsed_action = parse_action(reply, default=GridOpsAction())
    diagnostics: dict[str, Any] = {
        "reply": reply,
        "reply_chars": len(reply or ""),
        "action_payload": payload,
        "parsed_action": parsed_action.model_dump(),
        "has_think": "<think>" in (reply or ""),
        "has_think_close": "</think>" in (reply or ""),
        "has_action": "<action>" in (reply or ""),
        "has_action_close": "</action>" in (reply or ""),
    }
    if payload is None:
        return diagnostics
    try:
        GridOpsAction(**payload)
    except Exception as exc:
        diagnostics["validation_error"] = str(exc)
        if hasattr(exc, "errors"):
            diagnostics["validation_errors"] = exc.errors()
    return diagnostics


def model_path_kwargs(path: str) -> tuple[str, dict[str, str]]:
    """Support local paths, normal Hub ids, and Hub repo subfolders."""
    if Path(path).exists():
        return path, {}
    parts = path.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2]), {"subfolder": "/".join(parts[2:])}
    return path, {}


def load_model(base_model: str, adapter_path: str, token: str | None, load_4bit: bool):
    adapter_id, adapter_kwargs = model_path_kwargs(adapter_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=token, **adapter_kwargs)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
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
    model = PeftModel.from_pretrained(model, adapter_id, token=token, **adapter_kwargs)
    model.eval()
    return tokenizer, model


def build_messages(
    obs: dict[str, Any],
    task_id: str,
    prompt_mode: str,
    previous_action: dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if prompt_mode == "reason_action":
        return messages_for_reason_action_observation(
            obs,
            derive_control_context(obs, task_id),
            previous_action,
            previous_outcome,
        )
    return messages_for_observation(obs)


@torch.inference_mode()
def generate_action(
    tokenizer,
    model,
    obs: dict[str, Any],
    max_new_tokens: int,
    task_id: str = "task_1_normal",
    prompt_mode: str = "json",
    previous_action: dict[str, Any] | None = None,
    previous_outcome: dict[str, Any] | None = None,
) -> tuple[str, GridOpsAction, bool, str]:
    prompt = tokenizer.apply_chat_template(
        build_messages(obs, task_id, prompt_mode, previous_action, previous_outcome),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if prompt_mode == "reason_action":
        valid, reason = validate_reason_action_completion(reply)
    else:
        valid, reason = validate_completion(reply)
    action = parse_action(reply, default=GridOpsAction())
    return reply, action, valid, reason


def rollout(
    tokenizer,
    model,
    task_id: str,
    seed: int,
    max_new_tokens: int,
    sample_limit: int,
    prompt_mode: str,
    horizon: int,
) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    valid_actions = 0
    total_actions = 0
    invalid_examples: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    previous_action: dict[str, Any] | None = None
    previous_outcome: dict[str, Any] | None = None
    prior_obs: dict[str, Any] | None = None
    prior_model_action: GridOpsAction | None = None

    for _ in range(horizon):
        obs_dict = obs.model_dump()
        if prior_obs is not None and prior_model_action is not None:
            previous_outcome = previous_outcome_from_observation(obs_dict)
            previous_action = prior_model_action.model_dump()
        reply, action, valid, reason = generate_action(
            tokenizer,
            model,
            obs_dict,
            max_new_tokens,
            task_id=task_id,
            prompt_mode=prompt_mode,
            previous_action=previous_action,
            previous_outcome=previous_outcome,
        )
        valid_actions += int(valid)
        total_actions += 1
        if valid and len(samples) < sample_limit:
            samples.append(
                {
                    "hour": obs_dict["hour"],
                    "task_id": task_id,
                    "seed": seed,
                    "reply": reply,
                    "action": action.model_dump(),
                }
            )
        if not valid and len(invalid_examples) < 10:
            invalid_examples.append(
                {
                    "hour": obs_dict["hour"],
                    "task_id": task_id,
                    "seed": seed,
                    "reason": reason,
                    **invalid_action_diagnostics(reply),
                }
            )
        prior_obs = obs_dict
        prior_model_action = action
        obs = env.step(action)
        if obs.done:
            break

    grade = env.state.grade or {}
    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade.get("score", 0.0),
        "valid_actions": valid_actions,
        "total_actions": total_actions,
        "valid_action_rate": valid_actions / max(total_actions, 1),
        "invalid_examples": invalid_examples,
        "samples": samples,
        "grade": grade,
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {}
    for task_id in TASKS:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = {
            "score": round(sum(row["score"] for row in task_rows) / max(len(task_rows), 1), 4),
            "valid_action_rate": round(
                sum(row["valid_actions"] for row in task_rows) / max(sum(row["total_actions"] for row in task_rows), 1),
                4,
            ),
            "blackout_kwh": round(
                sum((row["grade"] or {}).get("total_blackout_kwh", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
            "diesel_kwh": round(
                sum((row["grade"] or {}).get("total_diesel_kwh", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
            "cost": round(
                sum((row["grade"] or {}).get("actual_cost", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
        }
    total_valid = sum(row["valid_actions"] for row in rows)
    total_actions = sum(row["total_actions"] for row in rows)
    return {
        "name": name,
        "average_score": round(sum(row["score"] for row in rows) / max(len(rows), 1), 4),
        "valid_action_rate": round(total_valid / max(total_actions, 1), 4),
        "by_task": by_task,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.environ.get("GRIDOPS_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--adapter-path", default=os.environ.get("GRIDOPS_ADAPTER_PATH", "outputs/sft_qwen25_3b_gridops_mixed1418_v1"))
    parser.add_argument("--seeds", default="7001,7002,7003")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-mode", choices=["json", "reason_action"], default=os.environ.get("GRIDOPS_PROMPT_MODE", "json"))
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--output", default="evals/gridops_sft_adapter_eval.json")
    parser.add_argument("--invalid-output", default="")
    parser.add_argument("--samples-output", default="")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_ids = [x.strip() for x in args.tasks.split(",") if x.strip()]

    tokenizer, model = load_model(args.base_model, args.adapter_path, token, load_4bit=not args.no_4bit)
    rows = []
    invalid_output = Path(args.invalid_output) if args.invalid_output else Path(args.output).with_suffix(".invalid_examples.jsonl")
    samples_output = Path(args.samples_output) if args.samples_output else Path(args.output).with_suffix(".valid_samples.jsonl")
    invalid_output.parent.mkdir(parents=True, exist_ok=True)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    invalid_output.write_text("")
    samples_output.write_text("")
    for task_id in task_ids:
        for seed in seeds:
            result = rollout(
                tokenizer,
                model,
                task_id,
                seed,
                args.max_new_tokens,
                args.sample_limit,
                args.prompt_mode,
                args.horizon,
            )
            rows.append(result)
            with invalid_output.open("a", encoding="utf-8") as handle:
                for example in result["invalid_examples"]:
                    handle.write(json.dumps(example, sort_keys=True) + "\n")
            with samples_output.open("a", encoding="utf-8") as handle:
                for sample in result["samples"]:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
            first_invalid = result["invalid_examples"][0] if result["invalid_examples"] else None
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "score": result["score"],
                        "valid_action_rate": round(result["valid_action_rate"], 4),
                        "first_invalid_reason": first_invalid.get("reason") if first_invalid else None,
                        "first_invalid_flags": {
                            "has_think": first_invalid.get("has_think") if first_invalid else None,
                            "has_think_close": first_invalid.get("has_think_close") if first_invalid else None,
                            "has_action": first_invalid.get("has_action") if first_invalid else None,
                            "has_action_close": first_invalid.get("has_action_close") if first_invalid else None,
                            "reply_chars": first_invalid.get("reply_chars") if first_invalid else None,
                        },
                        "first_invalid_reply_prefix": first_invalid.get("reply", "")[:500] if first_invalid else None,
                    }
                ),
                flush=True,
            )

    report = summarize(args.adapter_path, rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["name", "average_score", "valid_action_rate", "by_task"]}, indent=2))


if __name__ == "__main__":
    main()
