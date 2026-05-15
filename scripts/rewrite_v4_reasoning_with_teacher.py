"""Rewrite GridOps v4 reasoning traces with a frontier teacher model.

The teacher may rewrite only the <think> content. The simulator/oracle-approved
action is copied from the source trace and validated after every rewrite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridops.prompting import action_to_json, validate_reason_action_completion
from gridops.models import GridOpsAction


DEFAULT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_INPUT = "sft_traces/gridops_curriculum_v4_reason_action.jsonl"
DEFAULT_ACCEPTED = "sft_traces/gridops_curriculum_v4_kimi_reason_action.jsonl"
DEFAULT_REJECTED = "sft_traces/gridops_curriculum_v4_kimi_reason_action_rejected.jsonl"
DEFAULT_SUMMARY = "evals/gridops_curriculum_v4_kimi_reason_action_summary.json"

THINK_KEYS = ["time_context", "first_order", "second_order", "previous_action", "decision"]


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def source_id_for_row(row: dict[str, Any]) -> str:
    raw = row.get("raw") or {}
    return str(raw.get("teacher_source_id") or row.get("id", ""))


def teacher_source_id_for_row(row: dict[str, Any]) -> str | None:
    raw = row.get("raw") or {}
    source_id = raw.get("teacher_source_id")
    return str(source_id) if source_id else None


def action_matches(row: dict[str, Any], action: dict[str, Any]) -> bool:
    expected = row.get("action") or {}
    for key in ["battery_dispatch", "diesel_dispatch", "demand_shedding"]:
        if abs(float(expected.get(key, 0.0)) - float(action.get(key, 0.0))) > 1e-6:
            return False
    return True


def teacher_schema(batch: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["id"] for row in batch]
    return {
        "name": "gridops_kimi_teacher_reasoning",
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
                            "think": {
                                "type": "object",
                                "properties": {
                                    "time_context": {"type": "string", "maxLength": 260},
                                    "first_order": {"type": "string", "maxLength": 300},
                                    "second_order": {"type": "string", "maxLength": 340},
                                    "previous_action": {"type": "string", "maxLength": 300},
                                    "decision": {"type": "string", "maxLength": 300},
                                },
                                "required": THINK_KEYS,
                                "additionalProperties": False,
                            },
                        },
                        "required": ["id", "think"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


def compact_item(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw") or {}
    obs = raw.get("observation") or {}
    obs_keys = [
        "hour",
        "day_of_episode",
        "demand_kw",
        "solar_kw",
        "battery_soc",
        "grid_price",
        "diesel_fuel_remaining",
        "diesel_is_on",
        "demand_forecast_4h",
        "solar_forecast_4h",
        "price_forecast_4h",
        "blackout_this_step",
        "cost_this_step",
        "grid_kw_this_step",
        "narration",
    ]
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "difficulty": row.get("difficulty"),
        "hour": row.get("hour"),
        "bucket": raw.get("bucket"),
        "focus_tags": raw.get("focus_tags", []),
        "observation": {key: obs[key] for key in obs_keys if key in obs},
        "derived_context": raw.get("derived_context", {}),
        "previous_action": raw.get("previous_action", {}),
        "previous_outcome": raw.get("previous_outcome", {}),
        "approved_action": row.get("action", {}),
    }


def teacher_prompt(batch: list[dict[str, Any]]) -> str:
    items = [compact_item(row) for row in batch]
    return (
        "You are rewriting SFT teacher traces for a microgrid action model.\n"
        "For each item, write concise operator reasoning in the requested fields.\n"
        "Use only facts present in observation, derived_context, previous_action, previous_outcome, and approved_action.\n"
        "The approved_action is simulator/oracle approved. Do not critique it or invent a different action.\n"
        "Teach time-of-day context, immediate supply-demand impact, next-4-hour consequence, previous feedback, and final control choice.\n"
        "Keep units sensible: dispatch values are normalized controls; demand/solar/gaps are kW; blackout is kWh.\n\n"
        f"Items:\n{json.dumps(items, separators=(',', ':'))}"
    )


def request_batch(
    batch: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    sleep: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON matching the schema."},
            {"role": "user", "content": teacher_prompt(batch)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": teacher_schema(batch)},
        "provider": {"require_parameters": True},
        "reasoning": {"exclude": True, "enabled": False},
        "include_reasoning": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/capabl-machines/gridops",
        "X-Title": "GridOps Kimi Teacher Reasoning Factory",
    }
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"status_{response.status_code}:{response.text[:500]}")
            payload = response.json()
            content = payload["choices"][0]["message"].get("content") or ""
            parsed = json.loads(content)
            items = parsed.get("items", [])
            if not isinstance(items, list):
                raise ValueError("items_not_array")
            return {str(item["id"]): item for item in items}, payload.get("usage", {})
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:300]}"
            if attempt == retries:
                raise RuntimeError(last_error) from exc
            time.sleep(sleep * attempt)
    raise RuntimeError(last_error or "unknown_request_error")


def build_completion(row: dict[str, Any], think: dict[str, str]) -> str:
    action = GridOpsAction(**(row.get("action") or {}))
    return (
        "<think>\n"
        + "\n".join(f"{key}: {think[key].strip()}" for key in THINK_KEYS)
        + "\n</think>\n<action>\n"
        + action_to_json(action)
        + "\n</action>"
    )


def apply_teacher(row: dict[str, Any], item: dict[str, Any], model: str) -> tuple[dict[str, Any] | None, str]:
    think = item.get("think")
    if not isinstance(think, dict):
        return None, "missing_think"
    if any(not isinstance(think.get(key), str) or not think[key].strip() for key in THINK_KEYS):
        return None, "incomplete_think"
    if "action" in item and isinstance(item["action"], dict) and not action_matches(row, item["action"]):
        return None, "teacher_changed_action"

    cloned = json.loads(json.dumps(row))
    cloned["id"] = f"{row['id']}__kimi_teacher"
    cloned["completion"] = build_completion(row, think)
    valid, reason = validate_reason_action_completion(cloned["completion"])
    if not valid:
        return None, reason
    raw = cloned.setdefault("raw", {})
    raw["teacher_model"] = model
    raw["teacher_reasoning"] = think
    raw["teacher_source_id"] = row["id"]
    raw["validation"] = {"valid": True, "reason": "ok"}
    return cloned, "ok"


def selected_rows(rows: list[dict[str, Any]], limit: int, buckets: list[str], per_bucket: int | None) -> list[dict[str, Any]]:
    if not buckets and per_bucket is None:
        return rows[:limit] if limit else rows
    bucket_set = set(buckets)
    counts: Counter[str] = Counter()
    selected = []
    for row in rows:
        bucket = (row.get("raw") or {}).get("bucket", "")
        if bucket_set and bucket not in bucket_set:
            continue
        if per_bucket is not None and counts[bucket] >= per_bucket:
            continue
        selected.append(row)
        counts[bucket] += 1
        if limit and len(selected) >= limit:
            break
    return selected


def load_existing_source_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {source_id for row in load_jsonl(path) if (source_id := teacher_source_id_for_row(row))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--accepted-output", default=DEFAULT_ACCEPTED)
    parser.add_argument("--rejected-output", default=DEFAULT_REJECTED)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected rows.")
    parser.add_argument("--per-bucket", type=int, default=None)
    parser.add_argument("--bucket", action="append", default=[])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--between-batch-sleep", type=float, default=0.0)
    parser.add_argument("--append-original", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or environment.")

    source_rows = load_jsonl(Path(args.input))
    rows = selected_rows(source_rows, args.limit, args.bucket, args.per_bucket)
    existing_source_ids: set[str] = set()
    if args.resume:
        existing_source_ids = load_existing_source_ids(Path(args.accepted_output))
        rows = [row for row in rows if row["id"] not in existing_source_ids]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    usage_totals: Counter[str] = Counter()

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        batch_accepted: list[dict[str, Any]] = []
        batch_rejected: list[dict[str, Any]] = []
        try:
            by_id, usage = request_batch(
                batch,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                sleep=args.sleep,
            )
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    usage_totals[key] += value
        except Exception as exc:
            for row in batch:
                batch_rejected.append({"id": row["id"], "reason": f"api_error:{type(exc).__name__}:{str(exc)[:300]}"})
            rejected.extend(batch_rejected)
            if args.resume:
                append_jsonl(Path(args.rejected_output), batch_rejected)
            print(json.dumps({"batch": start // args.batch_size, "accepted": len(accepted), "error": str(exc)[:300]}), flush=True)
            continue

        for row in batch:
            item = by_id.get(row["id"])
            if not item:
                batch_rejected.append({"id": row["id"], "reason": "missing_teacher_item"})
                continue
            rewritten, reason = apply_teacher(row, item, args.model)
            if rewritten is None:
                batch_rejected.append({"id": row["id"], "reason": reason, "teacher_item": item})
            else:
                batch_accepted.append(rewritten)
        accepted.extend(batch_accepted)
        rejected.extend(batch_rejected)
        print(
            json.dumps(
                {
                    "batch": start // args.batch_size,
                    "processed": min(start + len(batch), len(rows)),
                    "accepted": len(accepted),
                    "rejected": len(rejected),
                }
            ),
            flush=True,
        )
        if args.resume:
            append_jsonl(Path(args.accepted_output), batch_accepted)
            if batch_rejected:
                append_jsonl(Path(args.rejected_output), batch_rejected)
        if args.between_batch_sleep > 0 and start + len(batch) < len(rows):
            time.sleep(args.between_batch_sleep)

    if args.resume:
        output_rows = load_jsonl(Path(args.accepted_output)) if Path(args.accepted_output).exists() else []
        if args.append_original:
            teacher_rows = [row for row in output_rows if teacher_source_id_for_row(row)]
            output_rows = source_rows + teacher_rows
            write_jsonl(Path(args.accepted_output), output_rows)
    else:
        output_rows = source_rows + accepted if args.append_original else accepted
        write_jsonl(Path(args.accepted_output), output_rows)
        write_jsonl(Path(args.rejected_output), rejected)

    bucket_counts = Counter((row.get("raw") or {}).get("bucket", "unknown") for row in accepted)
    task_counts = Counter(row.get("task_id", "unknown") for row in accepted)
    summary = {
        "model": args.model,
        "source_input": args.input,
        "selected_rows": len(rows) + len(existing_source_ids),
        "skipped_existing_rows": len(existing_source_ids),
        "attempted_rows_this_run": len(rows),
        "accepted_teacher_rows": len(accepted),
        "output_rows": len(output_rows),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / max(len(rows), 1), 4),
        "bucket_counts": dict(bucket_counts),
        "task_counts": dict(task_counts),
        "usage_totals": dict(usage_totals),
        "accepted_output": args.accepted_output,
        "rejected_output": args.rejected_output,
        "append_original": args.append_original,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if rejected:
        print(f"warning: {len(rejected)} rejected rows written to {args.rejected_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
