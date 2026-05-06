#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN="${HF_API_TOKEN:-${HF_TOKEN:-}}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "Set HF_API_TOKEN as a Kaggle Secret or environment variable." >&2
  exit 1
fi
export HF_API_TOKEN="${HF_TOKEN}"

export GRIDOPS_TRACE_PATH="${GRIDOPS_TRACE_PATH:-sft_traces/gridops_curriculum_v3_tool_augmented.jsonl}"
export GRIDOPS_RUN_LABEL="${GRIDOPS_RUN_LABEL:-sft_qwen25_3b_gridops_tool_augmented_v3}"
export GRIDOPS_MODEL_REPO="${GRIDOPS_MODEL_REPO:-77ethers/gridops-models}"
export GRIDOPS_BASE_MODEL="${GRIDOPS_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# Conservative defaults for Kaggle T4/P100-class GPUs.
export GRIDOPS_SFT_STEPS="${GRIDOPS_SFT_STEPS:-500}"
export GRIDOPS_BATCH_SIZE="${GRIDOPS_BATCH_SIZE:-1}"
export GRIDOPS_GRAD_ACCUM="${GRIDOPS_GRAD_ACCUM:-16}"
export GRIDOPS_MAX_LENGTH="${GRIDOPS_MAX_LENGTH:-1280}"
export GRIDOPS_LORA_R="${GRIDOPS_LORA_R:-16}"
export GRIDOPS_LORA_ALPHA="${GRIDOPS_LORA_ALPHA:-32}"
export GRIDOPS_LEARNING_RATE="${GRIDOPS_LEARNING_RATE:-2e-4}"
export GRIDOPS_GRADIENT_CHECKPOINTING="${GRIDOPS_GRADIENT_CHECKPOINTING:-1}"
export GRIDOPS_UPLOAD="${GRIDOPS_UPLOAD:-1}"

python scripts/validate_traces.py "${GRIDOPS_TRACE_PATH}"
python scripts/hf_sft_gridops.py
