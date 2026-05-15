"""Launch a small v6.1 eval-only diagnostic Hugging Face Job.

Use this after the adapter has already been uploaded. It does not retrain; it
loads the existing adapter, runs a short reason-action eval, and uploads the
report plus raw invalid completions so format failures are inspectable.
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
DEFAULT_BRANCH = "codex/gridops-v61-lp-critic-sft"
DEFAULT_RUN_LABEL = "sft_qwen3_4b_gridops_lp_critic_distilled_v1"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


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
    output_stem = f"evals/{args.run_label}_{args.eval_label}"
    command = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git
rm -rf /var/lib/apt/lists/*
git clone --branch {shlex.quote(args.branch)} --single-branch {shlex.quote(args.repo_url)} /workspace/gridops
cd /workspace/gridops
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install 'transformers>=4.56.2' 'peft>=0.17.1' 'accelerate>=1.0' bitsandbytes 'huggingface_hub>=0.34,<2.0' scipy protobuf
export HF_API_TOKEN=$HF_TOKEN
python scripts/evaluate_gridops_adapter.py \\
  --adapter-path {shlex.quote(args.model_repo + "/" + args.run_label)} \\
  --base-model {shlex.quote(args.base_model)} \\
  --prompt-mode reason_action \\
  --max-new-tokens {int(args.max_new_tokens)} \\
  --seeds {shlex.quote(args.seeds)} \\
  --tasks {shlex.quote(args.tasks)} \\
  --horizon {int(args.horizon)} \\
  --sample-limit {int(args.sample_limit)} \\
  --output {shlex.quote(output_stem + ".json")}
python - <<'PY'
import os
from huggingface_hub import HfApi

repo = {args.model_repo!r}
run_label = {args.run_label!r}
eval_label = {args.eval_label!r}
stem = f"evals/{{run_label}}_{{eval_label}}"
api = HfApi(token=os.environ["HF_API_TOKEN"])
for suffix in [".json", ".invalid_examples.jsonl", ".valid_samples.jsonl"]:
    local_path = stem + suffix
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{{run_label}}/evals/{{eval_label}}/{{run_label}}_{{eval_label}}{{suffix}}",
            repo_id=repo,
            repo_type="model",
            commit_message=f"Upload GridOps v6.1 diagnostic eval {{eval_label}}",
        )
PY
"""
    return command.strip()


def job_to_dict(job: Any) -> dict[str, Any]:
    if hasattr(job, "model_dump"):
        return job.model_dump()
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
    parser.add_argument("--timeout", default=os.environ.get("GRIDOPS_HF_JOB_TIMEOUT", "2h"))
    parser.add_argument("--namespace", default=os.environ.get("GRIDOPS_HF_JOB_NAMESPACE", "77ethers"))
    parser.add_argument("--model-repo", default="77ethers/gridops-models")
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--eval-label", default="diagnostic_tokens384_h12")
    parser.add_argument("--tasks", default="task_1_normal")
    parser.add_argument("--seeds", default="7001")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = load_env_token()
    if not token and not args.dry_run:
        raise RuntimeError("Set HF_API_TOKEN or HF_TOKEN in the environment or .env")

    command = build_job_command(args)
    metadata = {
        "image": args.image,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "branch": args.branch,
        "run_label": args.run_label,
        "eval_label": args.eval_label,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "horizon": args.horizon,
        "max_new_tokens": args.max_new_tokens,
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
        labels={"project": "gridops", "stage": "v61-eval-diagnostic"},
        token=token,
    )
    print(json.dumps({**metadata, "job": job_to_dict(job)}, indent=2, default=str))


if __name__ == "__main__":
    main()
