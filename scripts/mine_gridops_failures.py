"""Mine GridOps model failures against the oracle policy on holdout seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.policies import oracle_policy
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS
from scripts.evaluate_gridops_adapter import generate_action, load_model


def action_dict(action) -> dict[str, float]:
    return {
        "battery_dispatch": round(float(action.battery_dispatch), 4),
        "diesel_dispatch": round(float(action.diesel_dispatch), 4),
        "demand_shedding": round(float(action.demand_shedding), 4),
    }


def classify_failures(
    obs: dict[str, Any],
    after_obs: dict[str, Any],
    model_action,
    oracle_action,
    valid: bool,
    reason: str,
    task_id: str,
) -> list[str]:
    labels: list[str] = []
    hour = int(obs["hour"])
    mb = float(model_action.battery_dispatch)
    md = float(model_action.diesel_dispatch)
    ms = float(model_action.demand_shedding)
    ob = float(oracle_action.battery_dispatch)
    od = float(oracle_action.diesel_dispatch)
    os_ = float(oracle_action.demand_shedding)

    if not valid:
        labels.append(f"invalid:{reason}")
    if float(after_obs.get("blackout_this_step", 0.0)) > 0.01:
        labels.append("blackout_after_action")
    if abs(mb - ob) > 0.45:
        labels.append("battery_oracle_gap")
    if abs(md - od) > 0.35:
        labels.append("diesel_oracle_gap")
    if abs(ms - os_) > 0.25:
        labels.append("shedding_oracle_gap")
    if ob < -0.35 and mb > -0.10:
        labels.append("missed_charge")
    if ob > 0.35 and mb < 0.10:
        labels.append("missed_discharge")
    if od > 0.25 and md < 0.10:
        labels.append("missed_diesel")
    if md > 0.25 and od < 0.10:
        labels.append("unnecessary_diesel")
    if os_ > 0.25 and ms < 0.10:
        labels.append("missed_shedding")
    if ms > 0.25 and os_ < 0.10:
        labels.append("unnecessary_shedding")
    if task_id == "task_3_crisis" and 24 <= hour <= 29 and mb > 0.15:
        labels.append("pre_outage_discharge")
    if task_id == "task_3_crisis" and 26 <= hour <= 35 and float(obs["battery_soc"]) < 0.45:
        labels.append("low_soc_near_outage")
    if float(obs["grid_price"]) > 10.0 and task_id == "task_3_crisis" and 24 <= hour <= 29 and mb > 0.15:
        labels.append("price_greedy_before_outage")
    return labels


def run_oracle_episode(task_id: str, seed: int) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    for _ in range(72):
        obs_dict = obs.model_dump()
        obs = env.step(oracle_policy(obs_dict, task_id))
        if obs.done:
            break
    return env.state.grade or {}


def rollout_model(tokenizer, model, task_id: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    failures: list[dict[str, Any]] = []
    valid_actions = 0
    total_actions = 0

    for _ in range(72):
        obs_dict = obs.model_dump()
        oracle_action = oracle_policy(obs_dict, task_id)
        reply, model_action, valid, reason = generate_action(tokenizer, model, obs_dict, max_new_tokens)
        after_obs = env.step(model_action)
        after_dict = after_obs.model_dump()
        total_actions += 1
        valid_actions += int(valid)
        labels = classify_failures(obs_dict, after_dict, model_action, oracle_action, valid, reason, task_id)
        if labels:
            failures.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "hour": int(obs_dict["hour"]),
                    "labels": labels,
                    "obs": {
                        "demand_kw": round(float(obs_dict["demand_kw"]), 2),
                        "solar_kw": round(float(obs_dict["solar_kw"]), 2),
                        "battery_soc": round(float(obs_dict["battery_soc"]), 4),
                        "grid_price": round(float(obs_dict["grid_price"]), 2),
                        "diesel_fuel_remaining": round(float(obs_dict["diesel_fuel_remaining"]), 4),
                        "demand_forecast_4h": [round(float(v), 2) for v in obs_dict.get("demand_forecast_4h", [])],
                        "solar_forecast_4h": [round(float(v), 2) for v in obs_dict.get("solar_forecast_4h", [])],
                        "price_forecast_4h": [round(float(v), 2) for v in obs_dict.get("price_forecast_4h", [])],
                    },
                    "model_action": action_dict(model_action),
                    "oracle_action": action_dict(oracle_action),
                    "model_reply": reply[:500],
                    "after": {
                        "blackout_this_step": round(float(after_dict.get("blackout_this_step", 0.0)), 4),
                        "cost_this_step": round(float(after_dict.get("cost_this_step", 0.0)), 2),
                        "battery_soc": round(float(after_dict.get("battery_soc", 0.0)), 4),
                        "diesel_fuel_remaining": round(float(after_dict.get("diesel_fuel_remaining", 0.0)), 4),
                        "grid_kw_this_step": round(float(after_dict.get("grid_kw_this_step", 0.0)), 2),
                    },
                }
            )
        obs = after_obs
        if obs.done:
            break

    grade = env.state.grade or {}
    oracle_grade = run_oracle_episode(task_id, seed)
    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade.get("score", 0.0),
        "oracle_score": oracle_grade.get("score", 0.0),
        "score_gap": round(float(oracle_grade.get("score", 0.0)) - float(grade.get("score", 0.0)), 4),
        "valid_actions": valid_actions,
        "total_actions": total_actions,
        "valid_action_rate": valid_actions / max(total_actions, 1),
        "grade": grade,
        "oracle_grade": oracle_grade,
        "failures": failures,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    for row in rows:
        for failure in row["failures"]:
            for label in failure["labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1

    by_task = {}
    for task_id in TASKS:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = {
            "episodes": len(task_rows),
            "score": round(sum(row["score"] for row in task_rows) / max(len(task_rows), 1), 4),
            "oracle_score": round(sum(row["oracle_score"] for row in task_rows) / max(len(task_rows), 1), 4),
            "score_gap": round(sum(row["score_gap"] for row in task_rows) / max(len(task_rows), 1), 4),
            "valid_action_rate": round(
                sum(row["valid_actions"] for row in task_rows) / max(sum(row["total_actions"] for row in task_rows), 1),
                4,
            ),
            "blackout_kwh": round(
                sum((row["grade"] or {}).get("total_blackout_kwh", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
            "oracle_blackout_kwh": round(
                sum((row["oracle_grade"] or {}).get("total_blackout_kwh", 0.0) for row in task_rows) / max(len(task_rows), 1),
                2,
            ),
        }

    worst = sorted(
        (
            {
                "task_id": row["task_id"],
                "seed": row["seed"],
                "score": round(float(row["score"]), 4),
                "oracle_score": round(float(row["oracle_score"]), 4),
                "score_gap": row["score_gap"],
                "failure_count": len(row["failures"]),
                "blackout_kwh": (row["grade"] or {}).get("total_blackout_kwh", 0.0),
            }
            for row in rows
        ),
        key=lambda x: (x["score_gap"], x["failure_count"]),
        reverse=True,
    )[:20]

    return {
        "episodes": len(rows),
        "average_score": round(sum(row["score"] for row in rows) / max(len(rows), 1), 4),
        "average_oracle_score": round(sum(row["oracle_score"] for row in rows) / max(len(rows), 1), 4),
        "average_score_gap": round(sum(row["score_gap"] for row in rows) / max(len(rows), 1), 4),
        "valid_action_rate": round(
            sum(row["valid_actions"] for row in rows) / max(sum(row["total_actions"] for row in rows), 1),
            4,
        ),
        "by_task": by_task,
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True)),
        "worst_episodes": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.environ.get("GRIDOPS_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--adapter-path", default=os.environ.get("GRIDOPS_ADAPTER_PATH", "77ethers/gridops-models/sft_qwen25_3b_gridops_mixed1418_v1"))
    parser.add_argument("--seeds", default="7101,7102,7103,7104,7105")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="evals/failure_mining")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_ids = [x.strip() for x in args.tasks.split(",") if x.strip()]

    tokenizer, model = load_model(args.base_model, args.adapter_path, token, load_4bit=not args.no_4bit)
    rows = []
    for task_id in task_ids:
        for seed in seeds:
            row = rollout_model(tokenizer, model, task_id, seed, args.max_new_tokens)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "score": round(float(row["score"]), 4),
                        "oracle_score": round(float(row["oracle_score"]), 4),
                        "score_gap": row["score_gap"],
                        "failure_count": len(row["failures"]),
                    }
                ),
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "episodes.jsonl").open("w") as f:
        for row in rows:
            row_without_failures = {k: v for k, v in row.items() if k != "failures"}
            f.write(json.dumps(row_without_failures, separators=(",", ":")) + "\n")
    with (output_dir / "failure_bank.jsonl").open("w") as f:
        for row in rows:
            for failure in row["failures"]:
                f.write(json.dumps(failure, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
