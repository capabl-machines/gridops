# %% [markdown]
# # GridOps v4 Kimi Reasoning-Action SFT on Kaggle
#
# Kaggle setup:
# 1. Turn on Accelerator: GPU T4 x2 or P100.
# 2. Add a Kaggle Secret named `HF_API_TOKEN`.
# 3. Run all cells.
#
# This trains a Qwen2.5-3B QLoRA adapter on:
# `sft_traces/gridops_curriculum_v4_kimi_reason_action_500.jsonl`
#
# Upload target:
# `77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4`.

# %%
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/capabl-machines/gridops.git"
BRANCH = "codex/gridops-sft-pipeline"
ROOT = Path("/kaggle/working/gridops")

if not ROOT.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(ROOT)], check=True)
else:
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=ROOT, check=True)
    subprocess.run(["git", "checkout", BRANCH], cwd=ROOT, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=ROOT, check=True)

os.chdir(ROOT)
print("Repo:", Path.cwd())

# %% [markdown]
# ## Install Runtime

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        ".",
        "pytest",
        "huggingface_hub>=0.34,<1.0",
        "transformers>=4.56.2",
        "trl>=0.22.2",
        "peft>=0.17.1",
        "datasets>=4.0",
        "accelerate>=1.0",
        "bitsandbytes",
        "protobuf",
    ],
    check=True,
)

# %% [markdown]
# ## Secrets And GPU Check

# %%
try:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_API_TOKEN")
    if token:
        os.environ["HF_API_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
except Exception as exc:
    print("Kaggle Secret lookup skipped/failed:", type(exc).__name__)

assert os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN"), "Add Kaggle Secret HF_API_TOKEN first."

subprocess.run(["nvidia-smi"], check=False)

# %% [markdown]
# ## Validate Environment And Dataset

# %%
subprocess.run([sys.executable, "scripts/oracle_test.py"], check=True)
subprocess.run(
    [
        sys.executable,
        "scripts/validate_traces.py",
        "sft_traces/gridops_curriculum_v4_kimi_reason_action_500.jsonl",
    ],
    check=True,
)

# %% [markdown]
# ## Launch SFT
#
# Defaults are conservative for Kaggle T4/P100-class GPUs. If memory is tight,
# reduce `GRIDOPS_MAX_LENGTH` to `1280`.

# %%
os.environ.setdefault("GRIDOPS_TRACE_PATH", "sft_traces/gridops_curriculum_v4_kimi_reason_action_500.jsonl")
os.environ.setdefault("GRIDOPS_RUN_LABEL", "sft_qwen25_3b_gridops_kimi_reason_action_v4")
os.environ.setdefault("GRIDOPS_MODEL_REPO", "77ethers/gridops-models")
os.environ.setdefault("GRIDOPS_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
os.environ.setdefault("GRIDOPS_SFT_STEPS", "300")
os.environ.setdefault("GRIDOPS_BATCH_SIZE", "1")
os.environ.setdefault("GRIDOPS_GRAD_ACCUM", "8")
os.environ.setdefault("GRIDOPS_MAX_LENGTH", "1536")
os.environ.setdefault("GRIDOPS_LORA_R", "16")
os.environ.setdefault("GRIDOPS_LORA_ALPHA", "32")
os.environ.setdefault("GRIDOPS_LEARNING_RATE", "2e-4")
os.environ.setdefault("GRIDOPS_GRADIENT_CHECKPOINTING", "1")
os.environ.setdefault("GRIDOPS_UPLOAD", "1")

subprocess.run(["bash", "scripts/kaggle_sft_v4_reasoning.sh"], check=True)

# %% [markdown]
# ## Holdout Eval After Upload
#
# Run this after training completes. It uses the reasoning prompt mode and parses
# only the final `<action>` block.

# %%
os.environ.setdefault("GRIDOPS_ADAPTER_PATH", "77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4")

subprocess.run(
    [
        sys.executable,
        "scripts/evaluate_gridops_adapter.py",
        "--base-model",
        os.environ["GRIDOPS_BASE_MODEL"],
        "--adapter-path",
        os.environ["GRIDOPS_ADAPTER_PATH"],
        "--prompt-mode",
        "reason_action",
        "--max-new-tokens",
        "220",
        "--seeds",
        "7001,7002,7003",
        "--output",
        "evals/gridops_sft_kimi_reason_action_v4_holdout_7001_7003.json",
    ],
    check=True,
)

# %%
import json

report_path = Path("evals/gridops_sft_kimi_reason_action_v4_holdout_7001_7003.json")
report = json.loads(report_path.read_text())
print("average_score:", report["average_score"])
print("valid_action_rate:", report["valid_action_rate"])
print(json.dumps(report["by_task"], indent=2))
