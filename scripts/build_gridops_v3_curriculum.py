"""Build the balanced GridOps v3 SFT curriculum.

This is a deterministic merge step. It combines the known-good oracle
curriculum, mined failure replay, and only the accepted tool-augmented traces
from Gemma/DeepSeek pilots.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.prompting import format_observation, validate_completion


DEFAULT_SOURCES = [
    {
        "name": "oracle_base",
        "path": "sft_traces/gridops_curriculum_1200.jsonl",
        "repeat": 1,
    },
    {
        "name": "failure_replay",
        "path": "sft_traces/gridops_failure_targeted_7101_7110.jsonl",
        "repeat": 1,
    },
    {
        "name": "gemma_clean_json",
        "path": "sft_traces/gridops_v3_openrouter_gemma_pilot_accepted.jsonl",
        "repeat": 3,
    },
    {
        "name": "deepseek_crisis_aug",
        "path": "sft_traces/gridops_v3_openrouter_deepseek_targeted_accepted.jsonl",
        "repeat": 4,
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_trace(row: dict[str, Any]) -> tuple[bool, str]:
    valid, reason = validate_completion(row.get("completion", ""))
    if not valid:
        return False, reason
    obs = (row.get("raw") or {}).get("observation")
    if not obs:
        return False, "missing_raw_observation"
    if row.get("prompt") != format_observation(obs):
        return False, "prompt_mismatch"
    if not row.get("task_id"):
        return False, "missing_task_id"
    return True, "ok"


def clone_for_curriculum(row: dict[str, Any], source_name: str, source_path: str, repeat_index: int) -> dict[str, Any]:
    cloned = json.loads(json.dumps(row))
    base_id = str(row["id"])
    cloned["id"] = f"gridops_v3_{source_name}_{base_id}__r{repeat_index:02d}"
    raw = cloned.setdefault("raw", {})
    raw["v3_curriculum_source"] = source_name
    raw["v3_source_file"] = source_path
    raw["v3_source_id"] = base_id
    raw["v3_repeat_index"] = repeat_index
    return cloned


def build_curriculum(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    source_task_counts: dict[str, Counter[str]] = {}

    for source in sources:
        source_name = str(source["name"])
        source_path = str(source["path"])
        repeat = int(source["repeat"])
        rows = load_jsonl(Path(source_path))
        source_task_counts[source_name] = Counter()
        for row in rows:
            ok, reason = validate_trace(row)
            if not ok:
                rejected.append({"source": source_name, "id": row.get("id"), "reason": reason})
                continue
            for repeat_index in range(repeat):
                cloned = clone_for_curriculum(row, source_name, source_path, repeat_index)
                output_rows.append(cloned)
                source_counts[source_name] += 1
                source_task_counts[source_name][cloned["task_id"]] += 1

    ids = Counter(row["id"] for row in output_rows)
    duplicate_ids = [row_id for row_id, count in ids.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate output ids: {duplicate_ids[:5]}")

    task_counts = Counter(row["task_id"] for row in output_rows)
    policy_counts = Counter((row.get("raw") or {}).get("policy", "unknown") for row in output_rows)
    summary = {
        "rows": len(output_rows),
        "task_counts": dict(task_counts),
        "source_counts": dict(source_counts),
        "source_task_counts": {key: dict(value) for key, value in source_task_counts.items()},
        "policy_counts": dict(policy_counts),
        "rejected": rejected,
        "sources": sources,
    }
    return output_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sft_traces/gridops_curriculum_v3_tool_augmented.jsonl")
    parser.add_argument("--summary-output", default="evals/gridops_curriculum_v3_tool_augmented_summary.json")
    args = parser.parse_args()

    rows, summary = build_curriculum(DEFAULT_SOURCES)
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if summary["rejected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
