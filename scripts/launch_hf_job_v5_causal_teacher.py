"""Launch the GridOps v5 causal-teacher SFT run as a Hugging Face Job.

The job clones this branch on HF infrastructure, builds the v5 causal-teacher
dataset, runs SFT from the preserved v4 adapter, and uploads a new adapter
subfolder. It is designed to avoid browser-idle shutdowns from notebook
providers.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

from huggingface_hub import run_job


DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
DEFAULT_REPO_URL = "https://github.com/capabl-machines/gridops.git"
DEFAULT_BRANCH = "codex/gridops-sft-pipeline"
DEFAULT_RUN_LABEL = "sft_qwen25_3b_gridops_v5_causal_teacher"


def load_env_token() -> str:
    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(".env")
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"HF_API_TOKEN", "HF_TOKEN"}:
            token = value.strip().strip("\"'")
            if token:
                return token
    return ""


def build_job_command(args: argparse.Namespace) -> str:
    exports = {
        "HF_API_TOKEN": "$HF_TOKEN",
        "GRIDOPS_MODEL_REPO": args.model_repo,
        "GRIDOPS_BASE_MODEL": args.base_model,
        "GRIDOPS_INIT_ADAPTER": args.init_adapter,
        "GRIDOPS_RUN_LABEL": args.run_label,
        "GRIDOPS_SFT_STEPS": str(args.steps),
        "GRIDOPS_LEARNING_RATE": str(args.learning_rate),
        "GRIDOPS_BATCH_SIZE": str(args.batch_size),
        "GRIDOPS_GRAD_ACCUM": str(args.grad_accum),
        "GRIDOPS_MAX_LENGTH": str(args.max_length),
        "GRIDOPS_V5_SEEDS_PER_TASK": str(args.seeds_per_task),
        "GRIDOPS_V5_SEED_START": str(args.seed_start),
        "GRIDOPS_V5_HORIZON": str(args.horizon),
        "GRIDOPS_V5_BASE_SAMPLE_LIMIT": str(args.base_sample_limit),
        "GRIDOPS_UPLOAD": "1",
    }
    export_lines = []
    for key, value in exports.items():
        if value == "$HF_TOKEN":
            export_lines.append(f"export {key}=$HF_TOKEN")
        else:
            export_lines.append(f"export {key}={shlex.quote(value)}")

    return "\n".join(
        [
            "set -euo pipefail",
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update",
            "apt-get install -y --no-install-recommends git",
            "rm -rf /var/lib/apt/lists/*",
            f"git clone --branch {shlex.quote(args.branch)} --single-branch {shlex.quote(args.repo_url)} /workspace/gridops",
            "cd /workspace/gridops",
            "python -m pip install --upgrade pip",
            "python -m pip install -e .",
            "python -m pip install 'transformers>=4.45' 'trl>=0.12' 'peft>=0.13' 'datasets>=2.20' 'accelerate>=1.0' bitsandbytes 'huggingface_hub>=0.34,<2.0' scipy",
            *export_lines,
            "python scripts/build_gridops_v5_causal_teacher_traces.py "
            '--output "$GRIDOPS_TRACE_PATH" '
            '--summary-output "evals/gridops_curriculum_v5_causal_teacher_summary.json" '
            '--base-trace "$GRIDOPS_V5_BASE_TRACE" '
            '--base-sample-limit "$GRIDOPS_V5_BASE_SAMPLE_LIMIT" '
            '--seed-start "$GRIDOPS_V5_SEED_START" '
            '--seeds-per-task "$GRIDOPS_V5_SEEDS_PER_TASK" '
            '--stride "$GRIDOPS_V5_STRIDE" '
            '--horizon "$GRIDOPS_V5_HORIZON"',
            'python scripts/validate_traces.py "$GRIDOPS_TRACE_PATH"',
            "python scripts/hf_sft_gridops.py",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flavor", default=os.environ.get("GRIDOPS_HF_JOB_FLAVOR", "l4x1"))
    parser.add_argument("--timeout", default=os.environ.get("GRIDOPS_HF_JOB_TIMEOUT", "8h"))
    parser.add_argument("--namespace", default=os.environ.get("GRIDOPS_HF_JOB_NAMESPACE", ""))
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--model-repo", default="77ethers/gridops-models")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--init-adapter", default="77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4")
    parser.add_argument("--steps", type=int, default=175)
    parser.add_argument("--learning-rate", default="6e-5")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--seeds-per-task", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=16000)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--base-sample-limit", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = load_env_token()
    if not token and not args.dry_run:
        raise RuntimeError("Set HF_API_TOKEN or HF_TOKEN in the environment or .env")

    command = build_job_command(args)
    metadata: dict[str, Any] = {
        "image": args.image,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "branch": args.branch,
        "run_label": args.run_label,
        "model_repo": args.model_repo,
        "init_adapter": args.init_adapter,
    }
    if args.dry_run:
        print(json.dumps({**metadata, "command": command}, indent=2))
        return

    job = run_job(
        image=args.image,
        command=["bash", "-lc", command],
        secrets={"HF_TOKEN": token},
        env={
            "GRIDOPS_TRACE_PATH": "sft_traces/gridops_curriculum_v5_causal_teacher.jsonl",
            "GRIDOPS_V5_BASE_TRACE": "sft_traces/gridops_curriculum_v4_kimi_reason_action_500.jsonl",
            "GRIDOPS_V5_STRIDE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace or None,
        labels={"project": "gridops", "stage": "v5-causal-teacher-sft"},
        token=token,
    )
    print(json.dumps({**metadata, "job": job.model_dump()}, indent=2, default=str))


if __name__ == "__main__":
    main()
