"""Generate GridOps SFT curriculum traces from deterministic expert rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.policies import oracle_policy
from gridops.prompting import SYSTEM_PROMPT, action_to_json, format_observation, validate_completion
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
    """Lightweight labels for curriculum analysis and targeted sampling."""
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
    if not tags:
        tags.append("routine")
    return tags


def label_policy_name(tags: list[str]) -> str:
    """Name the targeted expert behavior represented by this trace."""
    if "outage_window" in tags:
        return "oracle_outage_guard"
    if "diesel_scarcity" in tags:
        return "oracle_diesel_rationing"
    if "low_soc" in tags:
        return "oracle_low_soc_recovery"
    if "high_price" in tags:
        return "oracle_high_price_arbitrage"
    if "rebound" in tags:
        return "oracle_rebound_avoidance"
    if "grid_cap_pressure" in tags:
        return "oracle_grid_cap_pressure"
    return "oracle_routine_dispatch"


def make_trace(task_id: str, seed: int, step: int, obs: dict[str, Any], action, grade: dict[str, Any]) -> dict[str, Any]:
    completion = action_to_json(action)
    valid, reason = validate_completion(completion)
    tags = focus_tags(obs, task_id)
    return {
        "id": f"gridops_{task_id}_seed{seed}_h{step:02d}",
        "task_id": task_id,
        "difficulty": DIFFICULTY[task_id],
        "seed": seed,
        "hour": step,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ],
        "prompt": format_observation(obs),
        "completion": completion,
        "action": json.loads(completion),
        "raw": {
            "observation": obs,
            "policy": "oracle_plus_targeted_edges",
            "label_policy": label_policy_name(tags),
            "focus_tags": tags,
            "score_context": grade,
            "validation": {"valid": valid, "reason": reason},
        },
    }


def collect_task(task_id: str, target_count: int, seed_start: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    seed = seed_start
    while len(traces) < target_count:
        env = GridOpsEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        episode_rows: list[tuple[int, dict[str, Any], Any]] = []
        for step in range(72):
            obs_dict = obs.model_dump()
            action = oracle_policy(obs_dict, task_id)
            episode_rows.append((step, obs_dict, action))
            obs = env.step(action)
            if obs.done:
                break
        grade = env.state.grade or {}
        for step, obs_dict, action in episode_rows:
            traces.append(make_trace(task_id, seed, step, obs_dict, action, grade))
            if len(traces) >= target_count:
                break
        seed += 1
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sft_traces/gridops_curriculum_1200.jsonl")
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()

    all_traces: list[dict[str, Any]] = []
    seed_start = args.seed_start
    for task_id, count in TASK_TARGETS.items():
        rows = collect_task(task_id, count, seed_start)
        all_traces.extend(rows)
        seed_start += 1000

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in all_traces:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    counts = {task: sum(1 for row in all_traces if row["task_id"] == task) for task in TASK_TARGETS}
    print(f"wrote {len(all_traces)} traces to {output}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
