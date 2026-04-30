"""Generate GridOps SFT traces with DeepSeek on OpenRouter.

The script batches 10 observations per API call by default, validates every
JSON action, and writes accepted/rejected rows separately. Accepted rows follow
the same trace schema as `generate_sft_traces.py` so SFT can concatenate both
datasets without prompt drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.policies import oracle_policy
from gridops.prompting import (
    SYSTEM_PROMPT,
    action_to_json,
    format_observation,
    parse_action,
    validate_completion,
)
from gridops.server.environment import GridOpsEnvironment


TASK_TARGETS = {
    "task_1_normal": 300,
    "task_2_heatwave": 400,
    "task_3_crisis": 500,
}

DIFFICULTY = {
    "task_1_normal": "easy",
    "task_2_heatwave": "medium",
    "task_3_crisis": "hard",
}


def focus_tags(obs: dict[str, Any], task_id: str) -> list[str]:
    tags: list[str] = []
    hour = int(obs["hour"])
    hour_of_day = (hour + 6) % 24
    net = float(obs["demand_kw"]) - float(obs["solar_kw"])
    if float(obs["battery_soc"]) < 0.25:
        tags.append("low_soc")
    if float(obs["grid_price"]) >= 14 or any(float(x) >= 14 for x in obs.get("price_forecast_4h", [])):
        tags.append("high_price")
    if net > 190:
        tags.append("grid_cap_pressure")
    if 18 <= hour_of_day <= 22:
        tags.append("evening_peak")
    if task_id == "task_3_crisis" and 26 <= hour <= 35:
        tags.append("outage_window")
    if float(obs.get("flow_shed", 0.0)) > 0 or "Rebound" in str(obs.get("narration", "")):
        tags.append("rebound")
    if float(obs.get("diesel_fuel_remaining", 1.0)) < 0.2:
        tags.append("diesel_scarcity")
    return tags or ["routine"]


def label_policy_name(tags: list[str]) -> str:
    if "outage_window" in tags:
        return "deepseek_outage_guard"
    if "diesel_scarcity" in tags:
        return "deepseek_diesel_rationing"
    if "low_soc" in tags:
        return "deepseek_low_soc_recovery"
    if "high_price" in tags:
        return "deepseek_high_price_arbitrage"
    if "rebound" in tags:
        return "deepseek_rebound_avoidance"
    if "grid_cap_pressure" in tags:
        return "deepseek_grid_cap_pressure"
    return "deepseek_routine_dispatch"


def collect_observations(targets: dict[str, int], seed_start: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_seed = seed_start
    for task_id, target_count in targets.items():
        collected = 0
        seed = next_seed
        while collected < target_count:
            env = GridOpsEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            episode: list[tuple[int, dict[str, Any]]] = []
            for step in range(72):
                obs_dict = obs.model_dump()
                episode.append((step, obs_dict))
                obs = env.step(oracle_policy(obs_dict, task_id))
                if obs.done:
                    break
            grade = env.state.grade or {}
            for step, obs_dict in episode:
                rows.append(
                    {
                        "id": f"gridops_deepseek_{task_id}_seed{seed}_h{step:02d}",
                        "task_id": task_id,
                        "difficulty": DIFFICULTY[task_id],
                        "seed": seed,
                        "hour": step,
                        "observation": obs_dict,
                        "score_context": grade,
                    }
                )
                collected += 1
                if collected >= target_count:
                    break
            seed += 1
        next_seed += 1000
    return rows


def batch_prompt(batch: list[dict[str, Any]]) -> str:
    examples = []
    for row in batch:
        examples.append(
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "difficulty": row["difficulty"],
                "observation_prompt": format_observation(row["observation"]),
            }
        )
    return (
        "Return a JSON array with exactly one action object for each item.\n"
        "Each object must have keys: id, battery_dispatch, diesel_dispatch, demand_shedding.\n"
        "Action bounds: battery_dispatch [-1,1], diesel_dispatch [0,1], demand_shedding [0,1].\n"
        "No markdown, no prose, no explanations.\n\n"
        f"Items:\n{json.dumps(examples, separators=(',', ':'))}"
    )


def extract_array(text: str) -> list[dict[str, Any]]:
    stripped = (text or "").strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("missing_json_array")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("not_array")
    return [x for x in parsed if isinstance(x, dict)]


def existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row_id = row.get("id")
        if isinstance(row_id, str):
            ids.add(row_id)
    return ids


def make_trace(row: dict[str, Any], action_text: str, model: str, raw_reply: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    action = parse_action(action_text, default=None)
    valid, reason = validate_completion(action_text)
    tags = focus_tags(row["observation"], row["task_id"])
    if action is None or not valid:
        return {}, {
            "id": row["id"],
            "task_id": row["task_id"],
            "difficulty": row["difficulty"],
            "seed": row["seed"],
            "hour": row["hour"],
            "completion": action_text,
            "raw_reply": raw_reply[:4000],
            "validation": {"valid": False, "reason": reason},
        }
    completion = action_to_json(action)
    return {
        "id": row["id"],
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
            "policy": "openrouter_deepseek",
            "teacher_model": model,
            "label_policy": label_policy_name(tags),
            "focus_tags": tags,
            "score_context": row["score_context"],
            "teacher_reply": raw_reply[:4000],
            "validation": {"valid": True, "reason": "ok"},
        },
    }, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sft_traces/gridops_deepseek_v4_pro_1200.jsonl")
    parser.add_argument("--rejected-output", default="sft_traces/gridops_deepseek_v4_pro_rejected.jsonl")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro"))
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("API_KEY"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-1", type=int, default=TASK_TARGETS["task_1_normal"])
    parser.add_argument("--task-2", type=int, default=TASK_TARGETS["task_2_heatwave"])
    parser.add_argument("--task-3", type=int, default=TASK_TARGETS["task_3_crisis"])
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set OPENROUTER_API_KEY, HF_TOKEN, or API_KEY.")

    targets = {
        "task_1_normal": args.task_1,
        "task_2_heatwave": args.task_2,
        "task_3_crisis": args.task_3,
    }
    client = OpenAI(base_url=args.api_base_url, api_key=args.api_key, timeout=args.request_timeout)

    output = Path(args.output)
    rejected_output = Path(args.rejected_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rejected_output.parent.mkdir(parents=True, exist_ok=True)

    # Resume keeps successful traces and retries previous failures; provider
    # failures are often transient on frontier routes.
    seen_ids = existing_ids(output) if args.resume else set()
    candidates = [row for row in collect_observations(targets, args.seed_start) if row["id"] not in seen_ids]

    accepted = len(existing_ids(output)) if args.resume else 0
    rejected = len(existing_ids(rejected_output)) if args.resume else 0
    mode = "a" if args.resume else "w"
    with output.open(mode) as good, rejected_output.open(mode) as bad:
        for start in range(0, len(candidates), args.batch_size):
            batch = candidates[start : start + args.batch_size]
            reply = ""
            parsed: list[dict[str, Any]] = []
            api_failed = False
            for attempt in range(1, args.max_retries + 1):
                try:
                    print(
                        json.dumps(
                            {
                                "batch_start": start,
                                "batch_size": len(batch),
                                "attempt": attempt,
                                "accepted": accepted,
                                "rejected": rejected,
                            }
                        ),
                        flush=True,
                    )
                    completion = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": batch_prompt(batch)},
                        ],
                        temperature=0.2,
                        max_tokens=args.max_tokens,
                    )
                    reply = completion.choices[0].message.content or ""
                    parsed = extract_array(reply)
                    break
                except Exception as exc:
                    if attempt == args.max_retries:
                        for row in batch:
                            bad.write(
                                json.dumps(
                                    {
                                        "id": row["id"],
                                        "reason": f"api_or_parse_error:{type(exc).__name__}",
                                        "reply": reply[:4000],
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            rejected += 1
                        api_failed = True
                    else:
                        time.sleep(args.sleep * attempt)
            if api_failed:
                print(
                    json.dumps(
                        {
                            "processed": len(seen_ids) + min(start + len(batch), len(candidates)),
                            "accepted": accepted,
                            "rejected": rejected,
                            "batch_failed": True,
                        }
                    ),
                    flush=True,
                )
                time.sleep(args.sleep)
                continue
            by_id = {str(item.get("id")): item for item in parsed}
            for row in batch:
                item = by_id.get(row["id"])
                if not item:
                    bad.write(json.dumps({"id": row["id"], "reason": "missing_id", "reply": reply[:4000]}, separators=(",", ":")) + "\n")
                    rejected += 1
                    continue
                action_text = json.dumps(
                    {
                        "battery_dispatch": item.get("battery_dispatch"),
                        "diesel_dispatch": item.get("diesel_dispatch"),
                        "demand_shedding": item.get("demand_shedding"),
                    },
                    separators=(",", ":"),
                )
                trace, failure = make_trace(row, action_text, args.model, reply)
                if failure:
                    bad.write(json.dumps(failure, separators=(",", ":")) + "\n")
                    rejected += 1
                else:
                    good.write(json.dumps(trace, separators=(",", ":")) + "\n")
                    accepted += 1
            print(
                json.dumps(
                    {
                        "processed": len(seen_ids) + min(start + len(batch), len(candidates)),
                        "accepted": accepted,
                        "rejected": rejected,
                    }
                ),
                flush=True,
            )
            time.sleep(args.sleep)

    print(json.dumps({"output": str(output), "rejected_output": str(rejected_output), "accepted": accepted, "rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
