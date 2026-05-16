#!/usr/bin/env python3
"""Build DPO preference pairs for GridOps strategy selection.

Each row keeps the model target at strict strategy JSON. Candidate strategies
are scored by executing them through the v7 causal controller on a copied
OpenEnv state, then rolling forward for a short horizon.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.server.environment import GridOpsEnvironment
from gridops.strategy import (
    GridOpsStrategy,
    derive_strategy,
    messages_for_strategy_observation,
    plan_strategy_action,
    strategy_dict,
    strategy_to_json,
    validate_strategy_completion,
)
from gridops.tool_agent import derive_control_context, previous_outcome_from_observation


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_values(value: str) -> list[int]:
    """Parse comma-separated seeds, allowing compact inclusive ranges."""

    seeds: list[int] = []
    for item in parse_csv(value):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending seed range: {item}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    return seeds


def parse_task_seed_map(value: str, *, tasks: list[str], seeds: list[int]) -> dict[str, list[int]]:
    """Parse task-specific seed ranges.

    Example:
        task_3_crisis=7801-7824;task_2_heatwave=7901-7908
    """

    if not value.strip():
        return {task_id: list(seeds) for task_id in tasks}
    result: dict[str, list[int]] = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Task seed map entries must use task=seeds syntax: {chunk}")
        task_id, seed_text = chunk.split("=", 1)
        task_id = task_id.strip()
        parsed = parse_seed_values(seed_text)
        if not parsed:
            raise ValueError(f"No seeds provided for task seed map entry: {chunk}")
        result[task_id] = parsed
    return result


def _env_metrics(env: GridOpsEnvironment) -> dict[str, float]:
    micro = env._micro  # noqa: SLF001 - copied-environment scoring diagnostics.
    return {
        "cost": float(micro.cumulative_cost),
        "blackout_kwh": float(micro.cumulative_blackout_kwh),
        "diesel_kwh": float(micro.cumulative_diesel_kwh),
        "battery_throughput_kwh": float(micro.cumulative_battery_throughput_kwh),
    }


def _unique(strategies: list[GridOpsStrategy]) -> list[GridOpsStrategy]:
    seen: set[str] = set()
    result: list[GridOpsStrategy] = []
    for strategy in strategies:
        key = strategy_to_json(strategy)
        if key in seen:
            continue
        seen.add(key)
        result.append(strategy)
    return result


def canonical_strategy_candidates(
    obs: dict[str, Any],
    task_id: str,
    previous_outcome: dict[str, Any] | None,
) -> list[GridOpsStrategy]:
    """Return valid strategy alternatives for preference scoring."""

    derived = derive_strategy(obs, task_id, previous_outcome)
    soc = float(obs.get("battery_soc", 0.0))
    fuel = float(obs.get("diesel_fuel_remaining", 0.0))
    hour = int(float(obs.get("hour", 0)))
    hour_of_day = (hour + 6) % 24
    base = [
        derived,
        GridOpsStrategy(mode="cost_saving", risk_level="low", battery_bias="charge" if soc < 0.8 else "neutral", diesel_policy="avoid", shedding_policy="never"),
        GridOpsStrategy(mode="peak_shaving", risk_level="medium" if soc > 0.35 else "high", battery_bias="discharge" if soc > 0.25 else "preserve", diesel_policy="avoid", shedding_policy="never"),
        GridOpsStrategy(mode="outage_prepare", risk_level="high", battery_bias="charge" if soc < 0.85 else "preserve", diesel_policy="conserve" if fuel < 0.35 else "allow_if_blackout", shedding_policy="last_resort"),
        GridOpsStrategy(mode="reliability", risk_level="critical", battery_bias="discharge" if soc > 0.25 else "preserve", diesel_policy="allow_if_blackout" if fuel > 0.08 else "conserve", shedding_policy="last_resort"),
        GridOpsStrategy(mode="recovery", risk_level="high", battery_bias="charge" if hour_of_day < 18 else "preserve", diesel_policy="allow_if_blackout" if fuel > 0.12 else "conserve", shedding_policy="last_resort"),
        GridOpsStrategy(mode="fuel_conservation", risk_level="high", battery_bias="preserve" if soc < 0.45 else "discharge", diesel_policy="conserve", shedding_policy="last_resort"),
    ]
    if task_id == "task_3_crisis":
        base.extend(
            [
                GridOpsStrategy(mode="reliability", risk_level="critical", battery_bias="preserve", diesel_policy="allow_if_blackout", shedding_policy="last_resort"),
                GridOpsStrategy(mode="outage_prepare", risk_level="critical", battery_bias="charge", diesel_policy="conserve", shedding_policy="last_resort"),
            ]
        )
    return _unique(base)


def score_strategy_candidate(
    env: GridOpsEnvironment,
    task_id: str,
    strategy: GridOpsStrategy,
    previous_outcome: dict[str, Any],
    horizon: int,
    optimizer_horizon: int,
) -> dict[str, Any]:
    """Score one strategy on a copied environment without mutating the caller."""

    sim_env = copy.deepcopy(env)
    obs = sim_env._make_observation(0.0, sim_env.state.done, "").model_dump()  # noqa: SLF001
    before = _env_metrics(sim_env)
    rewards: list[float] = []
    actions: list[dict[str, float]] = []
    strategies: list[dict[str, str]] = []
    current_previous_outcome = dict(previous_outcome)

    for step in range(max(1, int(horizon))):
        active_strategy = strategy if step == 0 else derive_strategy(obs, task_id, current_previous_outcome)
        plan = plan_strategy_action(
            sim_env,
            task_id,
            obs,
            previous_outcome=current_previous_outcome,
            strategy=active_strategy,
            optimizer_horizon=optimizer_horizon,
        )
        action = GridOpsAction(**plan["action"])
        obs_model = sim_env.step(action)
        obs = obs_model.model_dump()
        current_previous_outcome = previous_outcome_from_observation(obs)
        rewards.append(float(obs_model.reward))
        actions.append(plan["action"])
        strategies.append(strategy_dict(active_strategy))
        if obs_model.done:
            break

    after = _env_metrics(sim_env)
    delta = {
        "cost": round(after["cost"] - before["cost"], 4),
        "blackout_kwh": round(after["blackout_kwh"] - before["blackout_kwh"], 4),
        "diesel_kwh": round(after["diesel_kwh"] - before["diesel_kwh"], 4),
        "battery_throughput_kwh": round(after["battery_throughput_kwh"] - before["battery_throughput_kwh"], 4),
        "reward_sum": round(sum(rewards), 6),
    }
    preference_score = (
        float(delta["reward_sum"])
        - 3.0 * float(delta["blackout_kwh"])
        - 0.00002 * float(delta["cost"])
        - 0.001 * float(delta["diesel_kwh"])
    )
    if task_id == "task_3_crisis":
        preference_score -= 2.0 * float(delta["blackout_kwh"])
    return {
        "strategy": strategy_dict(strategy),
        "completion": strategy_to_json(strategy),
        "delta": delta,
        "actions": actions,
        "rollout_strategies": strategies,
        "preference_score": round(preference_score, 6),
    }


def difficulty_for(task_id: str, chosen: dict[str, str]) -> str:
    if task_id == "task_3_crisis" or chosen["risk_level"] in {"high", "critical"}:
        return "hard"
    if task_id == "task_2_heatwave" or chosen["mode"] in {"peak_shaving", "recovery"}:
        return "medium"
    return "easy"


def rejection_bucket(chosen: dict[str, Any], rejected: dict[str, Any], candidate_index: int, candidate_count: int) -> str:
    margin = float(chosen["preference_score"] - rejected["preference_score"])
    if candidate_index == candidate_count - 1:
        return "worst_contrast"
    if candidate_index == 1:
        return "near_miss"
    if margin >= 0.25:
        return "strong_contrast"
    return "plausible_contrast"


def select_rejections(
    candidates: list[dict[str, Any]],
    *,
    pairs_per_state: int,
    min_margin: float,
) -> list[tuple[dict[str, Any], int, str]]:
    """Pick diverse rejected strategies against the best candidate.

    DPO learns most from contrast. For each state we keep a close-but-worse
    option, a middle option, and the worst option when available, while avoiding
    duplicated JSON completions.
    """

    if len(candidates) < 2 or pairs_per_state <= 0:
        return []

    chosen = candidates[0]
    candidate_count = len(candidates)
    preferred_indexes = [1, candidate_count // 2, candidate_count - 1]
    preferred_indexes.extend(range(2, candidate_count - 1))

    selected: list[tuple[dict[str, Any], int, str]] = []
    seen: set[str] = {chosen["completion"]}
    for index in preferred_indexes:
        if index <= 0 or index >= candidate_count:
            continue
        rejected = candidates[index]
        if rejected["completion"] in seen:
            continue
        margin = float(chosen["preference_score"] - rejected["preference_score"])
        if margin < min_margin:
            continue
        bucket = rejection_bucket(chosen, rejected, index, candidate_count)
        selected.append((rejected, index, bucket))
        seen.add(rejected["completion"])
        if len(selected) >= pairs_per_state:
            break
    return selected


def build_pairs(
    *,
    tasks: list[str],
    seeds: list[int],
    task_seed_map: dict[str, list[int]] | None = None,
    stride: int,
    horizon: int,
    optimizer_horizon: int,
    min_margin: float,
    pairs_per_state: int,
    max_pairs: int | None,
    rng_seed: int,
    shuffle: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(rng_seed)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    resolved_task_seed_map = task_seed_map or {task_id: list(seeds) for task_id in tasks}
    for task_id in tasks:
        for seed in resolved_task_seed_map.get(task_id, []):
            env = GridOpsEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            previous_outcome = previous_outcome_from_observation(None)
            done = False
            while not done:
                obs_dict = obs.model_dump()
                hour = int(float(obs_dict["hour"]))
                derived = derive_strategy(obs_dict, task_id, previous_outcome)
                plan = plan_strategy_action(env, task_id, obs_dict, previous_outcome=previous_outcome, strategy=derived)
                action = GridOpsAction(**plan["action"])

                if hour % stride == 0:
                    derived_context = derive_control_context(obs_dict, task_id)
                    messages = messages_for_strategy_observation(obs_dict, derived_context, previous_outcome)
                    candidates = [
                        score_strategy_candidate(
                            env,
                            task_id,
                            strategy,
                            previous_outcome,
                            horizon=horizon,
                            optimizer_horizon=optimizer_horizon,
                        )
                        for strategy in canonical_strategy_candidates(obs_dict, task_id, previous_outcome)
                    ]
                    candidates.sort(key=lambda item: item["preference_score"], reverse=True)
                    if len(candidates) < 2:
                        skipped["too_few_candidates"] += 1
                    else:
                        chosen = candidates[0]
                        chosen_valid, chosen_reason = validate_strategy_completion(chosen["completion"])
                        if not chosen_valid:
                            failures.append(
                                {
                                    "task_id": task_id,
                                    "seed": seed,
                                    "hour": hour,
                                    "chosen_reason": chosen_reason,
                                }
                            )
                        else:
                            selected_rejections = select_rejections(
                                candidates,
                                pairs_per_state=pairs_per_state,
                                min_margin=min_margin,
                            )
                            if not selected_rejections:
                                skipped["no_rejection_after_margin"] += 1
                            for pair_index, (rejected, candidate_index, pair_kind) in enumerate(selected_rejections):
                                margin = float(chosen["preference_score"] - rejected["preference_score"])
                                rejected_valid, rejected_reason = validate_strategy_completion(rejected["completion"])
                                if not rejected_valid:
                                    failures.append(
                                        {
                                            "task_id": task_id,
                                            "seed": seed,
                                            "hour": hour,
                                            "rejected_reason": rejected_reason,
                                        }
                                    )
                                    continue
                                if chosen["completion"] == rejected["completion"]:
                                    skipped["same_completion"] += 1
                                    continue

                                chosen_strategy = chosen["strategy"]
                                row_id = f"gridops-strategy-dpo-v2-{task_id}-{seed:05d}-h{hour:03d}-p{pair_index}"
                                row = {
                                    "id": row_id,
                                    "task_id": task_id,
                                    "seed": seed,
                                    "hour": hour,
                                    "difficulty": difficulty_for(task_id, chosen_strategy),
                                    "prompt": messages[-1]["content"],
                                    "messages": messages,
                                    "chosen": chosen["completion"],
                                    "rejected": rejected["completion"],
                                    "raw": {
                                        "source": "gridops_strategy_dpo_v2_crisis_weighted",
                                        "prompt_mode": "strategy_json",
                                        "task_id": task_id,
                                        "seed": seed,
                                        "hour": hour,
                                        "observation": obs_dict,
                                        "derived_context": derived_context,
                                        "previous_outcome": previous_outcome,
                                        "chosen_strategy": chosen_strategy,
                                        "rejected_strategy": rejected["strategy"],
                                        "chosen_delta": chosen["delta"],
                                        "rejected_delta": rejected["delta"],
                                        "chosen_score": chosen["preference_score"],
                                        "rejected_score": rejected["preference_score"],
                                        "score_margin": round(margin, 6),
                                        "candidate_index": candidate_index,
                                        "candidate_count": len(candidates),
                                        "pair_kind": pair_kind,
                                        "candidate_summaries": [
                                            {
                                                "strategy": candidate["strategy"],
                                                "delta": candidate["delta"],
                                                "preference_score": candidate["preference_score"],
                                            }
                                            for candidate in candidates
                                        ],
                                        "horizon": horizon,
                                        "optimizer_horizon": optimizer_horizon,
                                    },
                                }
                                rows.append(row)
                                counts.update(
                                    [
                                        f"task:{task_id}",
                                        f"difficulty:{row['difficulty']}",
                                        f"chosen_mode:{chosen_strategy['mode']}",
                                        f"rejected_mode:{rejected['strategy']['mode']}",
                                        f"pair_kind:{pair_kind}",
                                    ]
                                )

                obs = env.step(action)
                previous_outcome = previous_outcome_from_observation(obs.model_dump())
                done = obs.done

    if shuffle:
        rng.shuffle(rows)
    if max_pairs is not None:
        rows = rows[:max_pairs]

    return rows, {
        "rows": len(rows),
        "tasks": tasks,
        "seeds": seeds,
        "task_seed_map": resolved_task_seed_map,
        "stride": stride,
        "horizon": horizon,
        "optimizer_horizon": optimizer_horizon,
        "min_margin": min_margin,
        "pairs_per_state": pairs_per_state,
        "counts": dict(counts),
        "skipped": dict(skipped),
        "validation_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="task_1_normal,task_2_heatwave,task_3_crisis")
    parser.add_argument("--seeds", default="7701,7702,7703,7704,7705,7706")
    parser.add_argument(
        "--task-seed-map",
        default="",
        help="Optional semicolon map such as task_3_crisis=7801-7824;task_2_heatwave=7901-7908.",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--optimizer-horizon", type=int, default=12)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--pairs-per-state", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=17)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--output", default="sft_traces/gridops_strategy_dpo_pairs_v1.jsonl")
    parser.add_argument("--summary", default="evals/gridops_strategy_dpo_pairs_v1_summary.json")
    args = parser.parse_args()

    rows, summary = build_pairs(
        tasks=parse_csv(args.tasks),
        seeds=parse_seed_values(args.seeds),
        task_seed_map=parse_task_seed_map(args.task_seed_map, tasks=parse_csv(args.tasks), seeds=parse_seed_values(args.seeds)),
        stride=args.stride,
        horizon=args.horizon,
        optimizer_horizon=args.optimizer_horizon,
        min_margin=args.min_margin,
        pairs_per_state=args.pairs_per_state,
        max_pairs=args.max_pairs,
        rng_seed=args.rng_seed,
        shuffle=not args.no_shuffle,
    )
    if summary["validation_failures"]:
        raise SystemExit(f"strategy DPO validation failed: {summary['validation_failures'][:3]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
