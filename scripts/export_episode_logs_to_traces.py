"""Convert saved GridOps episode logs into validated SFT trace rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.models import GridOpsAction
from gridops.prompting import (
    action_to_json,
    format_observation,
    messages_for_observation,
    messages_for_reason_action_observation,
    validate_completion,
    validate_reason_action_completion,
)
from gridops.tool_agent import derive_control_context, previous_outcome_from_observation, tool_corrected_completion


def _iter_events(log_dir: Path):
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                row["_source_line"] = line_no
                yield row


def _action_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("event_type") == "plan":
        return (event.get("plan") or {}).get("selected_action")
    if event.get("event_type") == "step":
        return event.get("action")
    return None


def _observation_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("event_type") == "plan":
        return event.get("observation")
    if event.get("event_type") == "step":
        return event.get("observation_before") or event.get("observation")
    return None


def build_trace(event: dict[str, Any], prompt_mode: str) -> dict[str, Any] | None:
    action = _action_from_event(event)
    obs = _observation_from_event(event)
    if not action or not obs:
        return None
    try:
        selected_action = GridOpsAction(**action)
    except Exception:
        return None
    task_id = event.get("task_id", "task_1_normal")
    previous_outcome = previous_outcome_from_observation(obs)
    previous_action = event.get("previous_action")
    if prompt_mode == "reason_action":
        derived_context = derive_control_context(obs, task_id)
        plan = event.get("plan") or {
            "selected_action": selected_action.model_dump(),
            "selected_source": "logged_step",
            "selection_reason": "logged_action",
            "comparison": {},
        }
        completion = tool_corrected_completion(
            obs=obs,
            task_id=task_id,
            plan=plan,
            previous_action=previous_action,
            previous_outcome=previous_outcome,
        )
        messages = messages_for_reason_action_observation(
            obs,
            derived_context,
            previous_action,
            previous_outcome,
        )
        valid, reason = validate_reason_action_completion(completion)
    else:
        derived_context = None
        completion = action_to_json(selected_action)
        messages = messages_for_observation(obs)
        valid, reason = validate_completion(completion)
    if not valid:
        return None
    trace_id = f"episode_log_{event['episode_id']}_{event.get('event_type')}_{int(float(obs.get('hour', 0))):03d}_{event['_source_line']}"
    return {
        "id": trace_id,
        "trace_id": trace_id,
        "task_id": task_id,
        "difficulty": "logged_experience",
        "messages": messages,
        "prompt": messages[-1]["content"] if prompt_mode == "reason_action" else format_observation(obs),
        "completion": completion,
        "raw": {
            "source": "episode_log",
            "prompt_mode": prompt_mode,
            "episode_id": event["episode_id"],
            "event_type": event.get("event_type"),
            "timestamp_utc": event.get("timestamp_utc"),
            "source_file": event["_source_file"],
            "source_line": event["_source_line"],
            "observation": obs,
            "derived_context": derived_context,
            "previous_action": previous_action,
            "previous_outcome": previous_outcome,
            "action": action,
            "plan": event.get("plan"),
            "grade": event.get("grade"),
            "validation": {"valid": True, "reason": reason},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="episode_logs")
    parser.add_argument("--output", default="sft_traces/gridops_episode_logs_sft.jsonl")
    parser.add_argument("--prompt-mode", choices=["json", "reason_action"], default="reason_action")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = [row for event in _iter_events(log_dir) if (row := build_trace(event, args.prompt_mode)) is not None]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
