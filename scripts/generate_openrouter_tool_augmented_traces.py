"""Generate GridOps tool-augmented SFT traces through OpenRouter.

LLMs propose candidate actions, but the local GridOps simulator decides which
actions are accepted. Accepted rows use the same SFT contract as the existing
curriculum: one prompt and one JSON-only action completion.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.policies import oracle_policy
from gridops.prompting import SYSTEM_PROMPT, action_to_json, format_observation, validate_completion
from gridops.server.environment import GridOpsEnvironment
from scripts.generate_sft_traces import DIFFICULTY, focus_tags, label_policy_name

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs
    OpenAI = None  # type: ignore[assignment]


DEFAULT_PROPOSERS = "deepseek/deepseek-v4-flash,google/gemma-4-26b-a4b-it"
DEFAULT_JUDGE = "deepseek/deepseek-v4-pro"
TASKS = ("task_1_normal", "task_2_heatwave", "task_3_crisis")
PILOT_HOURS = {
    "task_1_normal": [0, 10, 16, 18, 19, 20, 21, 34, 44, 68],
    "task_2_heatwave": [0, 24, 28, 30, 35, 36, 37, 44, 54, 60],
    "task_3_crisis": [0, 8, 24, 26, 29, 30, 31, 34, 35, 44],
}


def action_dict(action: GridOpsAction) -> dict[str, float]:
    return {
        "battery_dispatch": round(float(action.battery_dispatch), 4),
        "diesel_dispatch": round(float(action.diesel_dispatch), 4),
        "demand_shedding": round(float(action.demand_shedding), 4),
    }


def action_distance(left: GridOpsAction, right: GridOpsAction) -> float:
    return (
        abs(float(left.battery_dispatch) - float(right.battery_dispatch))
        + abs(float(left.diesel_dispatch) - float(right.diesel_dispatch))
        + abs(float(left.demand_shedding) - float(right.demand_shedding))
    )


def replay_to_hour(task_id: str, seed: int, hour: int) -> tuple[GridOpsEnvironment, dict[str, Any]]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    for _ in range(hour):
        obs_dict = obs.model_dump()
        obs = env.step(oracle_policy(obs_dict, task_id))
        if obs.done:
            break
    return env, obs.model_dump()


def collect_observations(per_task: int, seed_start: int, task_counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(TASKS):
        target_count = task_counts.get(task_id, per_task) if task_counts else per_task
        if target_count <= 0:
            continue
        seed = seed_start + task_index * 1000
        hours = PILOT_HOURS[task_id]
        idx = 0
        while sum(1 for row in rows if row["task_id"] == task_id) < target_count:
            hour = hours[idx % len(hours)]
            if idx and idx % len(hours) == 0:
                seed += 1
            env, obs = replay_to_hour(task_id, seed, hour)
            grade = run_oracle_episode(task_id, seed)
            tags = focus_tags(obs, task_id)
            rows.append(
                {
                    "id": f"gridops_v3_probe_{task_id}_seed{seed}_h{hour:02d}",
                    "task_id": task_id,
                    "difficulty": DIFFICULTY[task_id],
                    "seed": seed,
                    "hour": hour,
                    "observation": obs,
                    "oracle_action": action_dict(oracle_policy(obs, task_id)),
                    "oracle_episode_grade": grade,
                    "focus_tags": tags,
                    "label_policy": label_policy_name(tags),
                    "state_snapshot": {
                        "cumulative_cost": round(float(env._micro.cumulative_cost), 4),
                        "cumulative_blackout_kwh": round(float(env._micro.cumulative_blackout_kwh), 4),
                        "battery_soc_kwh": round(float(env._micro.battery_soc_kwh), 4),
                        "diesel_fuel_kwh": round(float(env._micro.diesel_fuel_kwh), 4),
                    },
                }
            )
            idx += 1
    return rows


def run_oracle_episode(task_id: str, seed: int) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    while not obs.done:
        obs_dict = obs.model_dump()
        obs = env.step(oracle_policy(obs_dict, task_id))
    return env.state.grade or {}


def candidate_schema(batch: list[dict[str, Any]], candidates_per_item: int) -> dict[str, Any]:
    ids = [row["id"] for row in batch]
    return {
        "name": "gridops_candidate_actions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": len(batch),
                    "maxItems": len(batch),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": ids},
                            "candidates": {
                                "type": "array",
                                "minItems": candidates_per_item,
                                "maxItems": candidates_per_item,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "battery_dispatch": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                        "diesel_dispatch": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "demand_shedding": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "tag": {
                                            "type": "string",
                                            "maxLength": 32,
                                            "description": "Short decision label, not prose reasoning.",
                                        },
                                    },
                                    "required": ["battery_dispatch", "diesel_dispatch", "demand_shedding", "tag"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["id", "candidates"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


def single_candidate_schema(candidates_per_item: int) -> dict[str, Any]:
    return {
        "name": "gridops_single_candidate_actions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": candidates_per_item,
                    "maxItems": candidates_per_item,
                    "items": {
                        "type": "object",
                        "properties": {
                            "battery_dispatch": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                            "diesel_dispatch": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "demand_shedding": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "tag": {"type": "string", "maxLength": 32},
                        },
                        "required": ["battery_dispatch", "diesel_dispatch", "demand_shedding", "tag"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
    }


def judge_schema(batch: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["id"] for row in batch]
    return {
        "name": "gridops_candidate_judgement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": len(batch),
                    "maxItems": len(batch),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": ids},
                            "preferred_source": {"type": "string"},
                            "risk_tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 4,
                            },
                        },
                        "required": ["id", "preferred_source", "risk_tags"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


def batch_prompt(batch: list[dict[str, Any]], candidates_per_item: int) -> str:
    items = []
    for row in batch:
        items.append(
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "difficulty": row["difficulty"],
                "focus_tags": row["focus_tags"],
                "oracle_action_reference": row["oracle_action"],
                "observation_prompt": format_observation(row["observation"]),
            }
        )
    return (
        f"Propose exactly {candidates_per_item} safe candidate actions for each GridOps observation.\n"
        "Return only the structured JSON object requested by the schema.\n"
        "Prefer diverse candidates that are physically valid and operationally plausible.\n"
        "Do not include long reasoning; use the short tag field for the intent.\n\n"
        f"Items:\n{json.dumps(items, separators=(',', ':'))}"
    )


def single_prompt(row: dict[str, Any], candidates_per_item: int) -> str:
    item = {
        "id": row["id"],
        "task_id": row["task_id"],
        "difficulty": row["difficulty"],
        "focus_tags": row["focus_tags"],
        "oracle_action_reference": row["oracle_action"],
        "observation_prompt": format_observation(row["observation"]),
    }
    return (
        f"Propose exactly {candidates_per_item} safe candidate actions for this GridOps observation.\n"
        "Return only the structured JSON object requested by the schema.\n"
        "Candidates must be diverse, physically valid, and use numeric values for every action field.\n\n"
        f"Item:\n{json.dumps(item, separators=(',', ':'))}"
    )


def judge_prompt(rows: list[dict[str, Any]]) -> str:
    payload = []
    for row in rows:
        payload.append(
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "difficulty": row["difficulty"],
                "focus_tags": row["focus_tags"],
                "oracle_action": row["oracle_action"],
                "top_candidates": [
                    {
                        "source": item["source"],
                        "action": item["action"],
                        "score_margin": item["score_margin"],
                        "horizon_blackout_kwh": item["horizon_blackout_kwh"],
                        "horizon_cost": item["horizon_cost"],
                    }
                    for item in row["scored_candidates"][:5]
                ],
            }
        )
    return (
        "Audit these simulator-scored GridOps candidates. The simulator remains the authority; "
        "you only flag the operationally preferred source and risk tags for metadata.\n"
        f"Items:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def parse_model_items(content: str, default_id: str | None = None) -> dict[str, dict[str, Any]]:
    parsed = json.loads(content or "{}")
    if default_id and isinstance(parsed.get("candidates"), list):
        return {default_id: {"id": default_id, "candidates": parsed["candidates"]}}
    items = parsed.get("items", [])
    if not isinstance(items, list):
        raise ValueError("items_not_array")
    return {str(item.get("id")): item for item in items if isinstance(item, dict)}


@contextlib.contextmanager
def api_deadline(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"api_deadline_exceeded:{seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def request_items(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    reasoning_exclude: bool,
    plugins: list[dict[str, str]],
    provider_order: list[str],
    provider_only: list[str],
    allow_fallbacks: bool,
    timeout_sleep: float,
    max_retries: int,
    request_timeout: float,
    default_id: str | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    reply = ""
    for attempt in range(1, max_retries + 1):
        try:
            provider: dict[str, Any] = {
                "require_parameters": True,
                "allow_fallbacks": allow_fallbacks,
            }
            if provider_order:
                provider["order"] = provider_order
            if provider_only:
                provider["only"] = provider_only
            with api_deadline(request_timeout + 5):
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_schema", "json_schema": schema},
                    extra_headers={
                        "HTTP-Referer": "https://huggingface.co/spaces/77ethers/GridOps",
                        "X-Title": "GridOps OpenRouter Data Factory",
                    },
                    extra_body={
                        "provider": provider,
                        "reasoning": {"effort": reasoning_effort, "exclude": reasoning_exclude},
                        "plugins": plugins,
                    },
            )
            reply = completion.choices[0].message.content or ""
            return parse_model_items(reply, default_id=default_id), reply
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "model": model,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}:{str(exc)[:160]}",
                    }
                ),
                flush=True,
            )
            if attempt == max_retries:
                raise
            time.sleep(timeout_sleep * attempt)
    return {}, reply


def score_action(task_id: str, seed: int, hour: int, action: GridOpsAction, horizon: int) -> dict[str, Any]:
    env, _ = replay_to_hour(task_id, seed, hour)
    start_cost = float(env._micro.cumulative_cost)
    start_blackout = float(env._micro.cumulative_blackout_kwh)
    total_reward = 0.0
    steps = 0
    obs = env.step(action)
    total_reward += float(obs.reward or 0.0)
    steps += 1
    while not obs.done and steps < horizon:
        obs_dict = obs.model_dump()
        obs = env.step(oracle_policy(obs_dict, task_id))
        total_reward += float(obs.reward or 0.0)
        steps += 1
    return {
        "score": round(total_reward / max(steps, 1), 6),
        "reward_sum": round(total_reward, 6),
        "horizon_steps": steps,
        "horizon_cost": round(float(env._micro.cumulative_cost) - start_cost, 4),
        "horizon_blackout_kwh": round(float(env._micro.cumulative_blackout_kwh) - start_blackout, 4),
    }


def score_candidates(row: dict[str, Any], model_outputs: dict[str, list[dict[str, Any]]], horizon: int) -> list[dict[str, Any]]:
    task_id = row["task_id"]
    seed = int(row["seed"])
    hour = int(row["hour"])
    oracle_action = GridOpsAction(**row["oracle_action"])
    oracle_score = score_action(task_id, seed, hour, oracle_action, horizon)
    scored: list[dict[str, Any]] = [
        {
            "source": "oracle",
            "candidate_index": 0,
            "tag": "oracle_reference",
            "action": row["oracle_action"],
            "valid": True,
            "validation": {"valid": True, "reason": "ok"},
            "score": oracle_score["score"],
            "score_margin": 0.0,
            "horizon_cost": oracle_score["horizon_cost"],
            "horizon_blackout_kwh": oracle_score["horizon_blackout_kwh"],
            "distance_from_oracle": 0.0,
        }
    ]
    for model, candidates in model_outputs.items():
        for idx, candidate in enumerate(candidates, start=1):
            action_payload = {
                "battery_dispatch": candidate.get("battery_dispatch"),
                "diesel_dispatch": candidate.get("diesel_dispatch"),
                "demand_shedding": candidate.get("demand_shedding"),
            }
            action_text = json.dumps(action_payload, separators=(",", ":"))
            valid, reason = validate_completion(action_text)
            if not valid:
                scored.append(
                    {
                        "source": model,
                        "candidate_index": idx,
                        "tag": str(candidate.get("tag", ""))[:80],
                        "action": action_payload,
                        "valid": False,
                        "validation": {"valid": False, "reason": reason},
                        "score": None,
                        "score_margin": None,
                    }
                )
                continue
            action = GridOpsAction(**action_payload)
            metrics = score_action(task_id, seed, hour, action, horizon)
            scored.append(
                {
                    "source": model,
                    "candidate_index": idx,
                    "tag": str(candidate.get("tag", ""))[:80],
                    "action": action_dict(action),
                    "valid": True,
                    "validation": {"valid": True, "reason": "ok"},
                    "score": metrics["score"],
                    "score_margin": round(metrics["score"] - oracle_score["score"], 6),
                    "horizon_cost": metrics["horizon_cost"],
                    "horizon_blackout_kwh": metrics["horizon_blackout_kwh"],
                    "distance_from_oracle": round(action_distance(action, oracle_action), 4),
                }
            )
    return sorted(scored, key=lambda item: (-1 if item["score"] is None else item["score"]), reverse=True)


def choose_trace_action(row: dict[str, Any], scored: list[dict[str, Any]]) -> tuple[dict[str, Any], str, str]:
    valid_candidates = [item for item in scored if item.get("valid") and item["source"] != "oracle"]
    best = valid_candidates[0] if valid_candidates else None
    tags = set(row["focus_tags"])
    important_state = bool(tags & {"outage_window", "evening_peak", "grid_cap_pressure", "low_soc", "diesel_scarcity", "rebound"})
    if best and float(best["score_margin"]) >= 0.02:
        return best["action"], best["source"], "beats_oracle"
    if best and float(best["score_margin"]) >= -0.01 and float(best.get("distance_from_oracle", 0.0)) >= 0.15:
        return best["action"], best["source"], "near_oracle_contrast"
    if important_state and (not best or float(best["score_margin"]) < -0.03):
        return row["oracle_action"], "oracle", "oracle_failure_replay"
    return {}, "", "rejected_no_useful_candidate"


def should_judge(row: dict[str, Any], scored: list[dict[str, Any]]) -> bool:
    valid = [item for item in scored if item.get("valid")]
    if row["task_id"] == "task_3_crisis":
        return True
    if len(valid) < 3:
        return False
    top_non_oracle = [item for item in valid if item["source"] != "oracle"][:2]
    if top_non_oracle and -0.03 <= float(top_non_oracle[0].get("score_margin", -1.0)) <= 0.02:
        return True
    return len({json.dumps(item["action"], sort_keys=True) for item in top_non_oracle}) > 1


def make_trace(row: dict[str, Any], chosen_action: dict[str, Any], chosen_source: str, reason: str, scored: list[dict[str, Any]], judge: dict[str, Any] | None) -> dict[str, Any]:
    action = GridOpsAction(**chosen_action)
    completion = action_to_json(action)
    valid, validation_reason = validate_completion(completion)
    return {
        "id": row["id"].replace("gridops_v3_probe_", "gridops_v3_tool_augmented_"),
        "task_id": row["task_id"],
        "difficulty": row["difficulty"],
        "seed": row["seed"],
        "hour": row["hour"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(row["observation"])},
        ],
        "prompt": format_observation(row["observation"]),
        "completion": completion,
        "action": json.loads(completion),
        "raw": {
            "observation": row["observation"],
            "policy": "openrouter_tool_augmented",
            "label_policy": label_policy_name(row["focus_tags"]),
            "focus_tags": row["focus_tags"],
            "oracle_action": row["oracle_action"],
            "chosen_source": chosen_source,
            "acceptance_reason": reason,
            "score_context": row["oracle_episode_grade"],
            "candidate_scores": scored,
            "judge": judge or {},
            "validation": {"valid": valid, "reason": validation_reason},
        },
    }


def summarize(
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    all_scored: list[dict[str, Any]],
    candidates_per_item: int,
) -> dict[str, Any]:
    by_model: dict[str, Counter] = defaultdict(Counter)
    margins: dict[str, list[float]] = defaultdict(list)
    by_task = Counter()
    response_failures = 0
    for row in accepted_rows:
        by_task[row["task_id"]] += 1
        source = row["raw"]["chosen_source"]
        by_model[source]["accepted"] += 1
        best = next((item for item in row["raw"]["candidate_scores"] if item["action"] == row["action"]), None)
        if best and best.get("score_margin") is not None:
            margins[source].append(float(best["score_margin"]))
    for row in rejected_rows:
        source = row.get("best_source", "unknown")
        by_model[source]["rejected"] += 1
        if str(row.get("reason", "")).startswith("api_error:"):
            response_failures += 1
            by_model[source]["api_or_parse_error"] += 1
            by_model[source]["invalid"] += candidates_per_item
    for row in all_scored:
        for item in row["scored_candidates"]:
            source = item["source"]
            if source == "oracle":
                continue
            if item.get("valid"):
                by_model[source]["valid"] += 1
            else:
                by_model[source]["invalid"] += 1
    model_summary = {}
    for model, counts in by_model.items():
        valid = counts.get("valid", 0)
        invalid = counts.get("invalid", 0)
        valid_rate = None if valid + invalid == 0 else round(valid / max(valid + invalid, 1), 4)
        model_summary[model] = {
            **dict(counts),
            "valid_action_rate": valid_rate,
            "average_accepted_margin": round(sum(margins[model]) / max(len(margins[model]), 1), 6),
        }
    total_candidates = sum(1 for row in all_scored for item in row["scored_candidates"] if item["source"] != "oracle")
    valid_candidates = sum(1 for row in all_scored for item in row["scored_candidates"] if item["source"] != "oracle" and item.get("valid"))
    expected_candidates_from_failed_responses = response_failures * candidates_per_item
    expected_model_slots = sum(
        max(len({item["source"] for item in row["scored_candidates"] if item["source"] != "oracle"}), 1)
        for row in all_scored
    ) * candidates_per_item
    missing_candidate_slots = max(0, expected_model_slots - total_candidates)
    effective_total_candidates = total_candidates + expected_candidates_from_failed_responses + missing_candidate_slots
    tasks_with_accepts = sorted({row["task_id"] for row in accepted_rows})
    return {
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "candidate_valid_action_rate": round(valid_candidates / max(total_candidates, 1), 4),
        "candidate_valid_action_rate_including_api_errors": round(valid_candidates / max(effective_total_candidates, 1), 4),
        "api_or_parse_error_rows": response_failures,
        "missing_candidate_slots": missing_candidate_slots,
        "accepted_by_task": dict(by_task),
        "accepted_tasks": tasks_with_accepts,
        "by_model": model_summary,
        "pilot_gate": {
            "valid_action_rate_ge_0_99": valid_candidates / max(effective_total_candidates, 1) >= 0.99,
            "accepted_all_tasks": set(tasks_with_accepts) == set(TASKS),
            "accepted_ratio_ge_0_20": len(accepted_rows) / max(len(all_scored), 1) >= 0.20,
        },
        "best_accepted_examples": sorted(
            [
                {
                    "id": row["id"],
                    "task_id": row["task_id"],
                    "chosen_source": row["raw"]["chosen_source"],
                    "acceptance_reason": row["raw"]["acceptance_reason"],
                    "action": row["action"],
                }
                for row in accepted_rows
            ],
            key=lambda item: item["id"],
        )[:10],
        "closest_rejected_examples": rejected_rows[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-output", default="sft_traces/gridops_v3_openrouter_pilot_accepted.jsonl")
    parser.add_argument("--rejected-output", default="sft_traces/gridops_v3_openrouter_pilot_rejected.jsonl")
    parser.add_argument("--summary-output", default="evals/gridops_v3_openrouter_pilot_summary.json")
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY"))
    parser.add_argument("--proposers", default=os.environ.get("OPENROUTER_PROPOSER_MODELS", DEFAULT_PROPOSERS))
    parser.add_argument("--judge-model", default=os.environ.get("OPENROUTER_JUDGE_MODEL", DEFAULT_JUDGE))
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--task-1", type=int, default=None, help="Override task_1_normal observation count.")
    parser.add_argument("--task-2", type=int, default=None, help="Override task_2_heatwave observation count.")
    parser.add_argument("--task-3", type=int, default=None, help="Override task_3_crisis observation count.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--candidates-per-item", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--judge-max-tokens", type=int, default=2000)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--judge-reasoning-effort", default="medium")
    parser.add_argument("--include-reasoning", action="store_true", help="Allow providers to return reasoning fields.")
    parser.add_argument("--disable-response-healing", action="store_true")
    parser.add_argument("--provider-order", default="", help="Comma-separated OpenRouter provider slugs to try first.")
    parser.add_argument("--provider-only", default="", help="Comma-separated OpenRouter provider slugs to allow exclusively.")
    parser.add_argument("--allow-fallbacks", action="store_true", help="Allow OpenRouter to fall back outside requested providers.")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Build observations and schemas without calling OpenRouter.")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    proposers = [model.strip() for model in args.proposers.split(",") if model.strip()]
    plugins = [] if args.disable_response_healing else [{"id": "response-healing"}]
    provider_order = [item.strip() for item in args.provider_order.split(",") if item.strip()]
    provider_only = [item.strip() for item in args.provider_only.split(",") if item.strip()]
    if not proposers:
        raise SystemExit("Provide at least one proposer model.")
    if not args.dry_run and not args.api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or API_KEY.")
    if not args.dry_run and OpenAI is None:
        raise SystemExit("Install the openai package to call OpenRouter.")

    task_counts = {
        "task_1_normal": args.per_task if args.task_1 is None else args.task_1,
        "task_2_heatwave": args.per_task if args.task_2 is None else args.task_2,
        "task_3_crisis": args.per_task if args.task_3 is None else args.task_3,
    }
    rows = collect_observations(args.per_task, args.seed_start, task_counts=task_counts)
    if args.dry_run:
        schema = candidate_schema(rows[: min(args.batch_size, len(rows))], args.candidates_per_item)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "observations": len(rows),
                    "first_ids": [row["id"] for row in rows[:3]],
                    "proposers": proposers,
                    "judge_model": args.judge_model,
                    "schema_name": schema["name"],
                    "schema_strict": schema["strict"],
                    "task_counts": task_counts,
                },
                indent=2,
            )
        )
        return

    client = OpenAI(base_url=args.api_base_url, api_key=args.api_key, timeout=args.request_timeout)
    accepted_output = Path(args.accepted_output)
    rejected_output = Path(args.rejected_output)
    summary_output = Path(args.summary_output)
    accepted_output.parent.mkdir(parents=True, exist_ok=True)
    rejected_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    all_scored: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        model_items: dict[str, dict[str, dict[str, Any]]] = {}
        for model in proposers:
            print(json.dumps({"batch_start": start, "model": model, "status": "requesting"}), flush=True)
            default_id = batch[0]["id"] if len(batch) == 1 else None
            try:
                items, _reply = request_items(
                    client=client,
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": single_prompt(batch[0], args.candidates_per_item)
                            if len(batch) == 1
                            else batch_prompt(batch, args.candidates_per_item),
                        },
                    ],
                    schema=single_candidate_schema(args.candidates_per_item)
                    if len(batch) == 1
                    else candidate_schema(batch, args.candidates_per_item),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_exclude=not args.include_reasoning,
                    plugins=plugins,
                    provider_order=provider_order,
                    provider_only=provider_only,
                    allow_fallbacks=args.allow_fallbacks,
                    timeout_sleep=args.sleep,
                    max_retries=args.max_retries,
                    request_timeout=args.request_timeout,
                    default_id=default_id,
                )
                model_items[model] = items
            except Exception as exc:
                model_items[model] = {}
                for row in batch:
                    rejected_rows.append(
                        {
                            "id": row["id"],
                            "task_id": row["task_id"],
                            "reason": f"api_error:{model}:{type(exc).__name__}",
                            "best_source": model,
                        }
                    )
        judge_rows: list[dict[str, Any]] = []
        for row in batch:
            candidates_by_model = {
                model: (model_items.get(model, {}).get(row["id"], {}).get("candidates") or [])
                for model in proposers
            }
            scored = score_candidates(row, candidates_by_model, args.horizon)
            scored_row = {**row, "scored_candidates": scored}
            all_scored.append(scored_row)
            if not args.no_judge and should_judge(row, scored):
                judge_rows.append(scored_row)
        judge_by_id: dict[str, dict[str, Any]] = {}
        if judge_rows:
            try:
                items, _reply = request_items(
                    client=client,
                    model=args.judge_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt(judge_rows)},
                    ],
                    schema=judge_schema(judge_rows),
                    max_tokens=args.judge_max_tokens,
                    temperature=args.temperature,
                    reasoning_effort=args.judge_reasoning_effort,
                    reasoning_exclude=not args.include_reasoning,
                    plugins=plugins,
                    provider_order=provider_order,
                    provider_only=provider_only,
                    allow_fallbacks=args.allow_fallbacks,
                    timeout_sleep=args.sleep,
                    max_retries=args.max_retries,
                    request_timeout=args.request_timeout,
                )
                judge_by_id = items
            except Exception as exc:
                for row in judge_rows:
                    judge_by_id[row["id"]] = {"error": f"judge_error:{type(exc).__name__}"}
        for scored_row in all_scored[-len(batch):]:
            chosen_action, chosen_source, reason = choose_trace_action(scored_row, scored_row["scored_candidates"])
            judge = judge_by_id.get(scored_row["id"])
            if chosen_action:
                accepted_rows.append(make_trace(scored_row, chosen_action, chosen_source, reason, scored_row["scored_candidates"], judge))
            else:
                best = next((item for item in scored_row["scored_candidates"] if item["source"] != "oracle"), {})
                rejected_rows.append(
                    {
                        "id": scored_row["id"],
                        "task_id": scored_row["task_id"],
                        "reason": reason,
                        "best_source": best.get("source", "none"),
                        "best_margin": best.get("score_margin"),
                        "focus_tags": scored_row["focus_tags"],
                        "scored_candidates": scored_row["scored_candidates"],
                        "judge": judge or {},
                    }
                )
        print(json.dumps({"processed": min(start + len(batch), len(rows)), "accepted": len(accepted_rows), "rejected": len(rejected_rows)}), flush=True)
        time.sleep(args.sleep)

    with accepted_output.open("w") as f:
        for row in accepted_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    with rejected_output.open("w") as f:
        for row in rejected_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = summarize(accepted_rows, rejected_rows, all_scored, args.candidates_per_item)
    summary.update(
        {
            "accepted_output": str(accepted_output),
            "rejected_output": str(rejected_output),
            "proposers": proposers,
            "judge_model": args.judge_model,
            "per_task": args.per_task,
            "task_counts": task_counts,
            "horizon": args.horizon,
            "candidates_per_item": args.candidates_per_item,
        }
    )
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
