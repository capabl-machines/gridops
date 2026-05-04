"""Create SFT traces targeted at mined GridOps model failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.policies import oracle_policy
from gridops.server.environment import GridOpsEnvironment
from scripts.generate_sft_traces import make_trace


def load_targets(path: Path) -> dict[tuple[str, int], dict[int, set[str]]]:
    targets: dict[tuple[str, int], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["task_id"], int(row["seed"]))
            hour = int(row["hour"])
            for label in row.get("labels", []):
                targets[key][hour].add(label)
    return targets


def collect_episode(task_id: str, seed: int, wanted_hours: set[int]) -> tuple[list[tuple[int, dict[str, Any], Any]], dict[str, Any]]:
    env = GridOpsEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    rows: list[tuple[int, dict[str, Any], Any]] = []
    for step in range(72):
        obs_dict = obs.model_dump()
        action = oracle_policy(obs_dict, task_id)
        if step in wanted_hours:
            rows.append((step, obs_dict, action))
        obs = env.step(action)
        if obs.done:
            break
    return rows, env.state.grade or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-bank", default="evals/failure_mining_7101_7110/failure_bank.jsonl")
    parser.add_argument("--base-traces", default="sft_traces/gridops_curriculum_1200.jsonl")
    parser.add_argument("--targeted-output", default="sft_traces/gridops_failure_targeted_7101_7110.jsonl")
    parser.add_argument("--merged-output", default="sft_traces/gridops_sft_v2_failure_targeted_2055.jsonl")
    parser.add_argument("--repeat-normal", type=int, default=2)
    parser.add_argument("--repeat-heatwave", type=int, default=1)
    parser.add_argument("--repeat-crisis", type=int, default=1)
    args = parser.parse_args()

    targets = load_targets(Path(args.failure_bank))
    traces: list[dict[str, Any]] = []
    for (task_id, seed), by_hour in sorted(targets.items()):
        rows, grade = collect_episode(task_id, seed, set(by_hour))
        for step, obs_dict, action in rows:
            trace = make_trace(task_id, seed, step, obs_dict, action, grade)
            labels = sorted(by_hour[step])
            trace["id"] = f"failure_targeted_{task_id}_seed{seed}_h{step:02d}"
            trace["raw"]["policy"] = "oracle_on_failure_targeted_hours"
            trace["raw"]["failure_labels"] = labels
            trace["raw"]["focus_tags"] = sorted(set(trace["raw"].get("focus_tags", [])) | set(labels))
            traces.append(trace)

    repeat_by_task = {
        "task_1_normal": max(args.repeat_normal, 1),
        "task_2_heatwave": max(args.repeat_heatwave, 1),
        "task_3_crisis": max(args.repeat_crisis, 1),
    }
    weighted_traces: list[dict[str, Any]] = []
    for trace in traces:
        repeats = repeat_by_task.get(trace["task_id"], 1)
        for idx in range(repeats):
            item = json.loads(json.dumps(trace))
            if repeats > 1:
                item["id"] = f"{trace['id']}_rep{idx + 1}"
                item["raw"]["repeat_index"] = idx + 1
            weighted_traces.append(item)

    targeted_output = Path(args.targeted_output)
    targeted_output.parent.mkdir(parents=True, exist_ok=True)
    with targeted_output.open("w") as f:
        for trace in weighted_traces:
            f.write(json.dumps(trace, separators=(",", ":")) + "\n")

    base_rows = [json.loads(line) for line in Path(args.base_traces).open()]
    merged_output = Path(args.merged_output)
    with merged_output.open("w") as f:
        for trace in base_rows + weighted_traces:
            f.write(json.dumps(trace, separators=(",", ":")) + "\n")

    counts: dict[str, int] = {}
    weighted_counts: dict[str, int] = {}
    for trace in traces:
        counts[trace["task_id"]] = counts.get(trace["task_id"], 0) + 1
    for trace in weighted_traces:
        weighted_counts[trace["task_id"]] = weighted_counts.get(trace["task_id"], 0) + 1
    print(
        json.dumps(
            {
                "unique_targeted_traces": len(traces),
                "weighted_targeted_traces": len(weighted_traces),
                "base_traces": len(base_rows),
                "merged_traces": len(base_rows) + len(weighted_traces),
                "unique_counts": counts,
                "weighted_counts": weighted_counts,
                "targeted_output": str(targeted_output),
                "merged_output": str(merged_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
