"""Evaluate GridOps policies or API-hosted models on deterministic holdout seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.policies import POLICIES
from gridops.prompting import messages_for_observation, parse_action, validate_completion
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS


PolicyFn = Callable[[dict, str | None], GridOpsAction]


def rollout_policy(policy: PolicyFn, task_id: str, seed: int) -> dict:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    valid_actions = 0
    total_actions = 0
    for _ in range(72):
        action = policy(obs.model_dump(), task_id)
        valid_actions += 1
        total_actions += 1
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
        "grade": grade,
    }


def rollout_api_model(client, model_name: str, task_id: str, seed: int, temperature: float, max_tokens: int) -> dict:
    """Roll out a chat-completions model using the exact SFT prompt format."""
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    valid_actions = 0
    total_actions = 0
    invalid_examples = []

    for _ in range(72):
        obs_dict = obs.model_dump()
        reply = ""
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages_for_observation(obs_dict),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = completion.choices[0].message.content or ""
        except Exception as exc:  # Keep holdout runs fail-soft and inspectable.
            invalid_examples.append({"hour": obs_dict.get("hour"), "reason": f"api_error:{type(exc).__name__}"})

        valid, reason = validate_completion(reply)
        if valid:
            valid_actions += 1
        elif len(invalid_examples) < 5:
            invalid_examples.append({"hour": obs_dict.get("hour"), "reason": reason, "reply": reply[:300]})
        total_actions += 1

        action = parse_action(reply, default=GridOpsAction())
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
        "invalid_examples": invalid_examples,
        "grade": grade,
    }


def evaluate_policy(policy_name: str, seeds: list[int]) -> dict:
    policy = POLICIES[policy_name]
    rows = [rollout_policy(policy, task_id, seed) for task_id in TASKS for seed in seeds]
    return summarize(policy_name, rows)


def evaluate_api_model(model_name: str, seeds: list[int], api_base_url: str, api_key: str, temperature: float, max_tokens: int) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    rows = [
        rollout_api_model(client, model_name, task_id, seed, temperature, max_tokens)
        for task_id in TASKS
        for seed in seeds
    ]
    return summarize(model_name, rows)


def summarize(name: str, rows: list[dict]) -> dict:
    by_task = {}
    for task_id in TASKS:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = round(sum(row["score"] for row in task_rows) / max(len(task_rows), 1), 4)
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
    parser.add_argument("--policy", choices=sorted(POLICIES), default="oracle")
    parser.add_argument("--model-name", default="", help="Evaluate an OpenAI-compatible chat model instead of a policy.")
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"))
    parser.add_argument("--api-key", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seeds", default="7001,7002,7003")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.model_name:
        if not args.api_key:
            raise SystemExit("Set --api-key, HF_API_TOKEN, HF_TOKEN, or API_KEY for model evaluation.")
        report = evaluate_api_model(
            args.model_name,
            seeds,
            args.api_base_url,
            args.api_key,
            args.temperature,
            args.max_tokens,
        )
    else:
        report = evaluate_policy(args.policy, seeds)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
