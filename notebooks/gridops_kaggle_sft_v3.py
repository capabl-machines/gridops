# %% [markdown]
# # GridOps v3 Tool-Augmented SFT on Kaggle
#
# Kaggle setup:
# 1. Turn on Accelerator: GPU T4/P100.
# 2. Add a Kaggle Secret named `HF_API_TOKEN`.
# 3. Run all cells.
#
# This trains a Qwen2.5-3B QLoRA adapter on
# `sft_traces/gridops_curriculum_v3_tool_augmented.jsonl` and uploads it to:
# `77ethers/gridops-models/sft_qwen25_3b_gridops_tool_augmented_v3`.

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
        "sft_traces/gridops_curriculum_v3_tool_augmented.jsonl",
    ],
    check=True,
)

# %% [markdown]
# ## Launch SFT
#
# Defaults are conservative for Kaggle T4/P100-class GPUs.
# If memory allows, increase `GRIDOPS_MAX_LENGTH` back toward `1536`.

# %%
os.environ.setdefault("GRIDOPS_TRACE_PATH", "sft_traces/gridops_curriculum_v3_tool_augmented.jsonl")
os.environ.setdefault("GRIDOPS_RUN_LABEL", "sft_qwen25_3b_gridops_tool_augmented_v3")
os.environ.setdefault("GRIDOPS_MODEL_REPO", "77ethers/gridops-models")
os.environ.setdefault("GRIDOPS_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
os.environ.setdefault("GRIDOPS_SFT_STEPS", "500")
os.environ.setdefault("GRIDOPS_BATCH_SIZE", "1")
os.environ.setdefault("GRIDOPS_GRAD_ACCUM", "16")
os.environ.setdefault("GRIDOPS_MAX_LENGTH", "1280")
os.environ.setdefault("GRIDOPS_LORA_R", "16")
os.environ.setdefault("GRIDOPS_LORA_ALPHA", "32")
os.environ.setdefault("GRIDOPS_LEARNING_RATE", "2e-4")
os.environ.setdefault("GRIDOPS_GRADIENT_CHECKPOINTING", "1")
os.environ.setdefault("GRIDOPS_UPLOAD", "1")

subprocess.run(["bash", "scripts/kaggle_sft_v3_gridops.sh"], check=True)

# %% [markdown]
# ## Expected Artifact
#
# Successful upload path:
# `https://huggingface.co/77ethers/gridops-models/tree/main/sft_qwen25_3b_gridops_tool_augmented_v3`
