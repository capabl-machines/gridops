"""Validate GridOps SFT traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.prompting import SYSTEM_PROMPT, format_observation, validate_completion


def validate_file(path: Path) -> dict:
    counts: Counter[str] = Counter()
    failures: list[dict] = []
    seen_ids: set[str] = set()

    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        counts["rows"] += 1
        counts[row.get("task_id", "missing_task")] += 1

        row_id = row.get("id")
        if row_id in seen_ids:
            failures.append({"line": line_no, "id": row_id, "reason": "duplicate_id"})
        seen_ids.add(row_id)

        valid, reason = validate_completion(row.get("completion", ""))
        if not valid:
            failures.append({"line": line_no, "id": row_id, "reason": reason})

        messages = row.get("messages") or []
        if not messages or messages[0].get("content") != SYSTEM_PROMPT:
            failures.append({"line": line_no, "id": row_id, "reason": "system_prompt_mismatch"})

        obs = (row.get("raw") or {}).get("observation")
        if obs and row.get("prompt") != format_observation(obs):
            failures.append({"line": line_no, "id": row_id, "reason": "prompt_mismatch"})

    return {"counts": dict(counts), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="sft_traces/gridops_curriculum_1200.jsonl")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    report = validate_file(Path(args.path))
    print(json.dumps(report["counts"], indent=2))
    if report["failures"]:
        print(json.dumps(report["failures"][:20], indent=2))
        raise SystemExit(1 if args.fail_fast else 2)
    print("validation ok")


if __name__ == "__main__":
    main()
