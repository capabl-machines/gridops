"""Validate GridOps SFT traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.prompting import (
    REASON_ACTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_observation,
    format_reason_action_observation,
    validate_completion,
    validate_reason_action_completion,
)


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

        raw = row.get("raw") or {}
        prompt_mode = raw.get("prompt_mode", "json")
        if prompt_mode == "reason_action":
            valid, reason = validate_reason_action_completion(row.get("completion", ""))
            expected_system_prompt = REASON_ACTION_SYSTEM_PROMPT
        else:
            valid, reason = validate_completion(row.get("completion", ""))
            expected_system_prompt = SYSTEM_PROMPT
        if not valid:
            failures.append({"line": line_no, "id": row_id, "reason": reason})

        messages = row.get("messages") or []
        if not messages or messages[0].get("content") != expected_system_prompt:
            failures.append({"line": line_no, "id": row_id, "reason": "system_prompt_mismatch"})

        obs = raw.get("observation")
        if prompt_mode == "reason_action" and obs:
            expected_prompt = format_reason_action_observation(
                obs,
                raw.get("derived_context"),
                raw.get("previous_action"),
                raw.get("previous_outcome"),
            )
        elif obs:
            expected_prompt = format_observation(obs)
        else:
            expected_prompt = None
        if expected_prompt and row.get("prompt") != expected_prompt:
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
