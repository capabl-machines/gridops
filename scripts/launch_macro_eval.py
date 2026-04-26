#!/usr/bin/env python3
"""Launch the 10-question CarbonAlpha macro eval on Hugging Face Jobs.

This is a thin local wrapper around scripts/hf_compare_qwen25.py. It sends the
eval cases through CARBON_ALPHA_COMPARE_CASES_JSON so the HF job uses the exact
questions saved in evals/macro_eval_10.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_ADAPTER = "grpo_qwen25_7b_adapter_phase1_100_v1"


def load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def read_cases(path: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "id" not in row or "question" not in row:
            raise ValueError(f"{path}:{line_number} needs id and question")
        cases.append({"id": str(row["id"]), "news": str(row["question"])})
    if not cases:
        raise ValueError(f"{path} contains no eval cases")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evals/macro_eval_10.jsonl"))
    parser.add_argument("--adapter-subdir", default=DEFAULT_ADAPTER)
    parser.add_argument("--flavor", default="l40sx1")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--max-new-tokens", default="420")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN is required")

    cases = read_cases(args.cases)
    api = HfApi(token=token)
    job = api.run_uv_job(
        script="scripts/hf_compare_qwen25.py",
        flavor=args.flavor,
        secrets={"HF_API_TOKEN": token},
        env={
            "CARBON_ALPHA_ADAPTER_SUBDIR": args.adapter_subdir,
            "CARBON_ALPHA_COMPARE_CASES_JSON": json.dumps(cases),
            "CARBON_ALPHA_COMPARE_MAX_NEW_TOKENS": args.max_new_tokens,
        },
        timeout=args.timeout,
        token=token,
    )
    print(job.id)
    print(job.url)
    print(f"adapter={args.adapter_subdir}")
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
