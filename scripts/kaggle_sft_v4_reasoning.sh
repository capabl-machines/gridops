#!/usr/bin/env bash
set -euo pipefail

export GRIDOPS_TRACE_PATH="${GRIDOPS_TRACE_PATH:-sft_traces/gridops_curriculum_v4_reason_action.jsonl}"
export GRIDOPS_RUN_LABEL="${GRIDOPS_RUN_LABEL:-sft_qwen25_3b_gridops_reason_action_v4}"
export GRIDOPS_MODEL_REPO="${GRIDOPS_MODEL_REPO:-77ethers/gridops-models}"
export GRIDOPS_BASE_MODEL="${GRIDOPS_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# Reasoning traces are longer than JSON-only traces, so use a slightly larger
# sequence length and keep the batch conservative for Kaggle T4-class GPUs.
export GRIDOPS_SFT_STEPS="${GRIDOPS_SFT_STEPS:-250}"
export GRIDOPS_BATCH_SIZE="${GRIDOPS_BATCH_SIZE:-1}"
export GRIDOPS_GRAD_ACCUM="${GRIDOPS_GRAD_ACCUM:-8}"
export GRIDOPS_MAX_LENGTH="${GRIDOPS_MAX_LENGTH:-1536}"
export GRIDOPS_LORA_R="${GRIDOPS_LORA_R:-16}"
export GRIDOPS_LORA_ALPHA="${GRIDOPS_LORA_ALPHA:-32}"
export GRIDOPS_LEARNING_RATE="${GRIDOPS_LEARNING_RATE:-2e-4}"
export GRIDOPS_GRADIENT_CHECKPOINTING="${GRIDOPS_GRADIENT_CHECKPOINTING:-1}"
export GRIDOPS_UPLOAD="${GRIDOPS_UPLOAD:-1}"

bash scripts/kaggle_sft_v3_gridops.sh
