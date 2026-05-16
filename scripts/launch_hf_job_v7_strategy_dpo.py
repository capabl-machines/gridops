#!/usr/bin/env python3
"""Launch GridOps v7.2 strategy DPO as a Hugging Face Job."""

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
DEFAULT_BRANCH = "codex/gridops-v7-strategy-sft"
DEFAULT_INIT_ADAPTER = "77ethers/gridops-models/sft_qwen25_15b_gridops_strategy_v7"
DEFAULT_RUN_LABEL = "dpo_qwen25_15b_gridops_strategy_v72"


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


def _export_lines(values: dict[str, str]) -> list[str]:
    lines = []
    for key, value in values.items():
        if value == "$HF_TOKEN":
            lines.append(f"export {key}=$HF_TOKEN")
        else:
            lines.append(f"export {key}={shlex.quote(value)}")
    return lines


def build_job_command(args: argparse.Namespace) -> str:
    exports = {
        "HF_API_TOKEN": "$HF_TOKEN",
        "GRIDOPS_MODEL_REPO": args.model_repo,
        "GRIDOPS_BASE_MODEL": args.base_model,
        "GRIDOPS_INIT_ADAPTER": args.init_adapter,
        "GRIDOPS_RUN_LABEL": args.run_label,
        "GRIDOPS_DPO_PAIR_PATH": args.pair_path,
        "GRIDOPS_DPO_STEPS": str(args.steps),
        "GRIDOPS_DPO_BETA": str(args.beta),
        "GRIDOPS_LEARNING_RATE": str(args.learning_rate),
        "GRIDOPS_BATCH_SIZE": str(args.batch_size),
        "GRIDOPS_GRAD_ACCUM": str(args.grad_accum),
        "GRIDOPS_MAX_LENGTH": str(args.max_length),
        "GRIDOPS_MAX_PROMPT_LENGTH": str(args.max_prompt_length),
        "GRIDOPS_UPLOAD": "1",
        "GRIDOPS_DPO_TASKS": args.tasks,
        "GRIDOPS_DPO_SEEDS": args.seeds,
        "GRIDOPS_DPO_STRIDE": str(args.stride),
        "GRIDOPS_DPO_HORIZON": str(args.horizon),
        "GRIDOPS_DPO_OPTIMIZER_HORIZON": str(args.optimizer_horizon),
        "GRIDOPS_DPO_MIN_MARGIN": str(args.min_margin),
        "GRIDOPS_DPO_MAX_PAIRS": str(args.max_pairs),
        "GRIDOPS_EVAL_SEEDS": args.eval_seeds,
        "GRIDOPS_RUN_EVAL": "1" if args.run_eval else "0",
    }
    max_pairs_arg = '--max-pairs "$GRIDOPS_DPO_MAX_PAIRS" ' if args.max_pairs else ""
    eval_block = r'''
if [[ "${GRIDOPS_RUN_EVAL}" == "1" ]]; then
  python scripts/evaluate_gridops_strategy_adapter.py \
    --adapter-path "${GRIDOPS_MODEL_REPO}/${GRIDOPS_RUN_LABEL}" \
    --base-model "${GRIDOPS_BASE_MODEL}" \
    --max-new-tokens 96 \
    --seeds "${GRIDOPS_EVAL_SEEDS}" \
    --output "evals/${GRIDOPS_RUN_LABEL}_holdout_strategy.json"
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo = os.environ["GRIDOPS_MODEL_REPO"]
run_label = os.environ["GRIDOPS_RUN_LABEL"]
api = HfApi(token=os.environ["HF_API_TOKEN"])
for suffix in ["", ".invalid_examples", ".valid_samples"]:
    path = f"evals/{run_label}_holdout_strategy{suffix}.json" if suffix == "" else f"evals/{run_label}_holdout_strategy{suffix}.jsonl"
    if not os.path.exists(path):
        continue
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f"{run_label}/evals/holdout/{os.path.basename(path)}",
        repo_id=repo,
        repo_type="model",
        commit_message=f"Upload GridOps v7.2 DPO eval {run_label}",
    )
PY
fi
'''
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
            "python -m pip install 'transformers>=4.56.2' 'trl>=0.22.2' 'peft>=0.17.1' 'datasets>=4.0' 'accelerate>=1.0' bitsandbytes 'huggingface_hub>=0.34,<2.0' scipy protobuf",
            *_export_lines(exports),
            "python scripts/build_gridops_strategy_dpo_pairs.py "
            '--tasks "$GRIDOPS_DPO_TASKS" '
            '--seeds "$GRIDOPS_DPO_SEEDS" '
            '--stride "$GRIDOPS_DPO_STRIDE" '
            '--horizon "$GRIDOPS_DPO_HORIZON" '
            '--optimizer-horizon "$GRIDOPS_DPO_OPTIMIZER_HORIZON" '
            '--min-margin "$GRIDOPS_DPO_MIN_MARGIN" '
            f"{max_pairs_arg}"
            '--output "$GRIDOPS_DPO_PAIR_PATH" '
            '--summary "evals/gridops_strategy_dpo_pairs_v1_summary.json"',
            "python scripts/hf_dpo_gridops_strategy.py",
            eval_block,
        ]
    )


def job_to_dict(job: Any) -> dict[str, Any]:
    if hasattr(job, "model_dump"):
        return job.model_dump()
    if hasattr(job, "_asdict"):
        return dict(job._asdict())
    result = {}
    for key in ["id", "status", "created_at", "docker_image", "flavor", "url", "labels"]:
        if hasattr(job, key):
            result[key] = getattr(job, key)
    return result


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
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--init-adapter", default=DEFAULT_INIT_ADAPTER)
    parser.add_argument("--pair-path", default="sft_traces/gridops_strategy_dpo_pairs_v1.jsonl")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--beta", default="0.1")
    parser.add_argument("--learning-rate", default="5e-6")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--tasks", default="task_1_normal,task_2_heatwave,task_3_crisis")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(7701, 7713)))
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--optimizer-horizon", type=int, default=12)
    parser.add_argument("--min-margin", default="0.05")
    parser.add_argument("--max-pairs", type=int, default=2400)
    parser.add_argument("--eval-seeds", default="7001,7002,7003")
    parser.add_argument("--run-eval", action="store_true")
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
        "base_model": args.base_model,
        "init_adapter": args.init_adapter,
        "pair_path": args.pair_path,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "max_pairs": args.max_pairs,
        "steps": args.steps,
        "run_eval": args.run_eval,
    }
    if args.dry_run:
        print(json.dumps({**metadata, "command": command}, indent=2))
        return

    job = run_job(
        image=args.image,
        command=["bash", "-lc", command],
        secrets={"HF_TOKEN": token},
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace or None,
        labels={"project": "gridops", "stage": "v7-strategy-dpo"},
        token=token,
    )
    print(json.dumps({**metadata, "job": job_to_dict(job)}, indent=2, default=str))


if __name__ == "__main__":
    main()
