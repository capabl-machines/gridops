#!/usr/bin/env python3
"""Evaluate an OpenRouter chat model through the GridOps v7 strategy harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
UNTUNED_QWEN15B_BASELINE = {
    "average_score": 0.7911,
    "task_1_normal": 0.7993,
    "task_2_heatwave": 0.8223,
    "task_3_crisis": 0.7517,
}
V73_BASELINE = {
    "average_score": 0.7888,
    "task_1_normal": 0.7993,
    "task_2_heatwave": 0.8223,
    "task_3_crisis": 0.7449,
}


STRATEGY_SCHEMA = {
    "name": "gridops_strategy",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["cost_saving", "peak_shaving", "outage_prepare", "reliability", "recovery", "fuel_conservation"],
            },
            "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "battery_bias": {"type": "string", "enum": ["charge", "preserve", "discharge", "neutral"]},
            "diesel_policy": {"type": "string", "enum": ["avoid", "allow_if_blackout", "prewarm", "conserve"]},
            "shedding_policy": {"type": "string", "enum": ["never", "last_resort"]},
        },
        "required": ["mode", "risk_level", "battery_bias", "diesel_policy", "shedding_policy"],
    },
}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def openrouter_model_catalog() -> dict[str, dict[str, Any]]:
    request = Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "GridOps OpenRouter evaluator"})
    with urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return {model["id"]: model for model in data.get("data", [])}


def call_openrouter_strategy(
    *,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    timeout: int,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "include_reasoning": False,
        "reasoning": {"effort": reasoning_effort, "exclude": True},
        "response_format": {"type": "json_schema", "json_schema": STRATEGY_SCHEMA},
        "provider": {"require_parameters": True},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/capabl-machines/gridops",
        "X-Title": "GridOps Strategy Benchmark",
    }
    payload = json.dumps(body).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request("https://openrouter.ai/api/v1/chat/completions", data=payload, headers=headers, method="POST")
            with urlopen(request, timeout=timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            message = response_body["choices"][0]["message"]
            return (message.get("content") or "").strip(), {
                "id": response_body.get("id"),
                "model": response_body.get("model", model),
                "usage": response_body.get("usage", {}),
                "finish_reason": response_body["choices"][0].get("finish_reason"),
            }
        except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"openrouter_call_failed:{type(last_error).__name__}:{last_error}")


def generate_strategy(
    obs: dict[str, Any],
    task_id: str,
    previous_outcome: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[str, GridOpsStrategy, dict[str, Any], dict[str, Any]]:
    messages = messages_for_strategy_observation(
        obs,
        derive_control_context(obs, task_id),
        previous_outcome,
    )
    reply, meta = call_openrouter_strategy(
        model=args.model,
        api_key=args.api_key,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        retries=args.retries,
    )
    validation = validate_strategy_payload(reply)
    strategy = parse_strategy(reply)
    return reply, strategy, validation, meta


def rollout(args: argparse.Namespace, task_id: str, seed: int) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    previous_outcome = previous_outcome_from_observation(None)
    valid_strategies = 0
    total_steps = 0
    invalid_examples: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for _ in range(args.horizon):
        obs_dict = obs.model_dump()
        try:
            reply, strategy, validation, meta = generate_strategy(obs_dict, task_id, previous_outcome, args)
        except Exception as exc:
            reply = ""
            strategy = parse_strategy(None)
            validation = {"valid": False, "reason": f"api_error:{type(exc).__name__}", "strategy": strategy.model_dump()}
            meta = {"error": str(exc)}
        for key in usage:
            usage[key] += int((meta.get("usage") or {}).get(key, 0) or 0)

        valid = bool(validation["valid"])
        valid_strategies += int(valid)
        total_steps += 1
        plan = plan_strategy_action(
            env,
            task_id,
            obs_dict,
            previous_outcome=previous_outcome,
            strategy=strategy if valid else None,
            optimizer_horizon=args.optimizer_horizon,
        )
        action = GridOpsAction(**plan["action"])
        if valid and len(samples) < args.sample_limit:
            samples.append(
                {
                    "hour": obs_dict["hour"],
                    "task_id": task_id,
                    "seed": seed,
                    "reply": reply,
                    "strategy": validation["strategy"],
                    "selected_action": plan["action"],
                    "optimizer_config": plan["optimizer_config"],
                    "meta": meta,
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
                    "meta": meta,
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
        "usage": usage,
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
            "blackout_kwh": round(sum((row["grade"] or {}).get("total_blackout_kwh", 0.0) for row in task_rows) / len(task_rows), 2),
            "diesel_kwh": round(sum((row["grade"] or {}).get("total_diesel_kwh", 0.0) for row in task_rows) / len(task_rows), 2),
            "cost": round(sum((row["grade"] or {}).get("actual_cost", 0.0) for row in task_rows) / len(task_rows), 2),
        }
    average = sum(row["score"] for row in rows) / max(len(rows), 1)
    total_steps = sum(row["total_steps"] for row in rows)
    usage = {
        key: sum(int((row.get("usage") or {}).get(key, 0) or 0) for row in rows)
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    return {
        "name": name,
        "average_score": round(average, 4),
        "valid_strategy_rate": round(sum(row["valid_strategies"] for row in rows) / max(total_steps, 1), 4),
        "lp_ceiling_capture": round(average / LP_CEILING["average_score"], 4),
        "usage": usage,
        "by_task": by_task,
        "baselines": {
            "v51_model_only": V51_BASELINE,
            "v7_deterministic_strategy_controller": V7_DETERMINISTIC_BASELINE,
            "untuned_qwen25_15b_strategy_harness": UNTUNED_QWEN15B_BASELINE,
            "v73_trained_strategy_selector": V73_BASELINE,
            "full_episode_lp_ceiling": LP_CEILING,
        },
        "rows": rows,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro"))
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--seeds", default="7001,7002,7003")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--optimizer-horizon", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=["none", "high", "xhigh"], default="none")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--output", default="evals/gridops_openrouter_strategy_eval.json")
    parser.add_argument("--invalid-output", default="")
    parser.add_argument("--samples-output", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        raise SystemExit("Set OPENROUTER_API_KEY in .env or environment.")

    catalog = openrouter_model_catalog()
    model_info = catalog.get(args.model)
    if model_info is None:
        raise SystemExit(f"OpenRouter model not found: {args.model}")
    print(
        json.dumps(
            {
                "model": args.model,
                "name": model_info.get("name"),
                "context_length": model_info.get("context_length"),
                "supported_parameters": model_info.get("supported_parameters", []),
                "pricing": model_info.get("pricing", {}),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_ids = [x.strip() for x in args.tasks.split(",") if x.strip()]
    rows = []
    invalid_output = Path(args.invalid_output) if args.invalid_output else Path(args.output).with_suffix(".invalid_examples.jsonl")
    samples_output = Path(args.samples_output) if args.samples_output else Path(args.output).with_suffix(".valid_samples.jsonl")
    invalid_output.parent.mkdir(parents=True, exist_ok=True)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    invalid_output.write_text("")
    samples_output.write_text("")

    for task_id in task_ids:
        for seed in seeds:
            result = rollout(args, task_id, seed)
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
                        "usage": result["usage"],
                        "first_invalid_reason": first_invalid.get("reason") if first_invalid else None,
                    }
                ),
                flush=True,
            )

    report = summarize(args.model, rows)
    report["openrouter_model_info"] = {
        "id": model_info.get("id"),
        "name": model_info.get("name"),
        "context_length": model_info.get("context_length"),
        "pricing": model_info.get("pricing", {}),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ["name", "average_score", "valid_strategy_rate", "lp_ceiling_capture", "usage", "by_task", "baselines"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
