#!/usr/bin/env python3
"""Evaluate a strategy-json LoRA adapter through the v7 controller harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.server.environment import GridOpsEnvironment
from gridops.strategy import (
    GridOpsStrategy,
    messages_for_strategy_observation,
    parse_strategy,
    plan_strategy_action,
    validate_strategy_payload,
)
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import derive_control_context, previous_outcome_from_observation

LP_CEILING = {
    "average_score": 0.8233,
    "task_1_normal": 0.8372,
    "task_2_heatwave": 0.8416,
    "task_3_crisis": 0.7912,
}
V51_BASELINE = {
    "average_score": 0.7354,
    "task_1_normal": 0.7896,
    "task_2_heatwave": 0.7681,
    "task_3_crisis": 0.6484,
}
V7_DETERMINISTIC_BASELINE = {
    "average_score": 0.7907,
    "task_1_normal": 0.7995,
    "task_2_heatwave": 0.8224,
    "task_3_crisis": 0.7503,
}


def model_path_kwargs(path: str) -> tuple[str, dict[str, str]]:
    if Path(path).exists():
        return path, {}
    parts = path.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2]), {"subfolder": "/".join(parts[2:])}
    return path, {}


def load_model(base_model: str, adapter_path: str, token: str | None, load_4bit: bool, use_adapter: bool = True):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_id, adapter_kwargs = model_path_kwargs(adapter_path)
    if use_adapter:
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=token, **adapter_kwargs)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    else:
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
    if use_adapter:
        model = PeftModel.from_pretrained(model, adapter_id, token=token, **adapter_kwargs)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_strategy(
    tokenizer,
    model,
    obs: dict[str, Any],
    task_id: str,
    previous_outcome: dict[str, Any] | None,
    max_new_tokens: int,
) -> tuple[str, GridOpsStrategy, dict[str, Any]]:
    messages = messages_for_strategy_observation(
        obs,
        derive_control_context(obs, task_id),
        previous_outcome,
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    validation = validate_strategy_payload(reply)
    strategy = parse_strategy(reply)
    return reply, strategy, validation


def rollout(
    tokenizer,
    model,
    task_id: str,
    seed: int,
    max_new_tokens: int,
    sample_limit: int,
    horizon: int,
    optimizer_horizon: int,
) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    previous_outcome = previous_outcome_from_observation(None)
    valid_strategies = 0
    total_steps = 0
    invalid_examples: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for _ in range(horizon):
        obs_dict = obs.model_dump()
        reply, strategy, validation = generate_strategy(
            tokenizer,
            model,
            obs_dict,
            task_id,
            previous_outcome,
            max_new_tokens,
        )
        valid = bool(validation["valid"])
        valid_strategies += int(valid)
        total_steps += 1
        plan = plan_strategy_action(
            env,
            task_id,
            obs_dict,
            previous_outcome=previous_outcome,
            strategy=strategy if valid else None,
            optimizer_horizon=optimizer_horizon,
        )
        action = GridOpsAction(**plan["action"])
        if valid and len(samples) < sample_limit:
            samples.append(
                {
                    "hour": obs_dict["hour"],
                    "task_id": task_id,
                    "seed": seed,
                    "reply": reply,
                    "strategy": validation["strategy"],
                    "selected_action": plan["action"],
                    "optimizer_config": plan["optimizer_config"],
                }
            )
        if not valid and len(invalid_examples) < 10:
            invalid_examples.append(
                {
                    "hour": obs_dict["hour"],
                    "task_id": task_id,
                    "seed": seed,
                    "reason": validation["reason"],
                    "reply": reply,
                    "reply_chars": len(reply or ""),
                    "fallback_strategy": plan["strategy"],
                }
            )
        obs = env.step(action)
        previous_outcome = previous_outcome_from_observation(obs.model_dump())
        if obs.done:
            break

    grade = env.state.grade or {}
    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade.get("score", 0.0),
        "valid_strategies": valid_strategies,
        "total_steps": total_steps,
        "valid_strategy_rate": valid_strategies / max(total_steps, 1),
        "invalid_examples": invalid_examples,
        "samples": samples,
        "grade": grade,
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {}
    for task_id in TASKS:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        if not task_rows:
            continue
        task_score = sum(row["score"] for row in task_rows) / len(task_rows)
        task_total = sum(row["total_steps"] for row in task_rows)
        by_task[task_id] = {
            "score": round(task_score, 4),
            "valid_strategy_rate": round(sum(row["valid_strategies"] for row in task_rows) / max(task_total, 1), 4),
            "lp_ceiling_capture": round(task_score / LP_CEILING[task_id], 4),
            "blackout_kwh": round(
                sum((row["grade"] or {}).get("total_blackout_kwh", 0.0) for row in task_rows) / len(task_rows),
                2,
            ),
            "diesel_kwh": round(
                sum((row["grade"] or {}).get("total_diesel_kwh", 0.0) for row in task_rows) / len(task_rows),
                2,
            ),
            "cost": round(sum((row["grade"] or {}).get("actual_cost", 0.0) for row in task_rows) / len(task_rows), 2),
        }
    average = sum(row["score"] for row in rows) / max(len(rows), 1)
    total_steps = sum(row["total_steps"] for row in rows)
    return {
        "name": name,
        "average_score": round(average, 4),
        "valid_strategy_rate": round(sum(row["valid_strategies"] for row in rows) / max(total_steps, 1), 4),
        "lp_ceiling_capture": round(average / LP_CEILING["average_score"], 4),
        "by_task": by_task,
        "baselines": {
            "v51_model_only": V51_BASELINE,
            "v7_deterministic_strategy_controller": V7_DETERMINISTIC_BASELINE,
            "full_episode_lp_ceiling": LP_CEILING,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.environ.get("GRIDOPS_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--adapter-path", default=os.environ.get("GRIDOPS_ADAPTER_PATH", "outputs/sft_qwen25_15b_gridops_strategy_v7"))
    parser.add_argument("--seeds", default="7001,7002,7003")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--optimizer-horizon", type=int, default=12)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--output", default="evals/gridops_strategy_adapter_eval.json")
    parser.add_argument("--invalid-output", default="")
    parser.add_argument("--samples-output", default="")
    parser.add_argument("--no-adapter", action="store_true", help="Evaluate the untouched base model without loading a LoRA adapter.")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_ids = [x.strip() for x in args.tasks.split(",") if x.strip()]

    tokenizer, model = load_model(
        args.base_model,
        args.adapter_path,
        token,
        load_4bit=not args.no_4bit,
        use_adapter=not args.no_adapter,
    )
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
                args.horizon,
                args.optimizer_horizon,
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
                        "valid_strategy_rate": round(result["valid_strategy_rate"], 4),
                        "first_invalid_reason": first_invalid.get("reason") if first_invalid else None,
                        "first_invalid_reply_prefix": first_invalid.get("reply", "")[:300] if first_invalid else None,
                    }
                ),
                flush=True,
            )

    report_name = args.base_model if args.no_adapter else args.adapter_path
    report = summarize(report_name, rows)
    report["base_model"] = args.base_model
    report["adapter_path"] = None if args.no_adapter else args.adapter_path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ["name", "average_score", "valid_strategy_rate", "lp_ceiling_capture", "by_task", "baselines"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
