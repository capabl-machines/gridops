#!/usr/bin/env bash
set -euo pipefail

export GRIDOPS_TRACE_PATH="${GRIDOPS_TRACE_PATH:-sft_traces/gridops_curriculum_v41_bound_repair_mix.jsonl}"
export GRIDOPS_RUN_LABEL="${GRIDOPS_RUN_LABEL:-sft_qwen25_3b_gridops_kimi_reason_action_v41_repair}"
export GRIDOPS_MODEL_REPO="${GRIDOPS_MODEL_REPO:-77ethers/gridops-models}"
export GRIDOPS_BASE_MODEL="${GRIDOPS_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export GRIDOPS_INIT_ADAPTER="${GRIDOPS_INIT_ADAPTER:-77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4}"

export GRIDOPS_SFT_STEPS="${GRIDOPS_SFT_STEPS:-125}"
export GRIDOPS_BATCH_SIZE="${GRIDOPS_BATCH_SIZE:-1}"
export GRIDOPS_GRAD_ACCUM="${GRIDOPS_GRAD_ACCUM:-8}"
export GRIDOPS_MAX_LENGTH="${GRIDOPS_MAX_LENGTH:-1536}"
export GRIDOPS_LORA_R="${GRIDOPS_LORA_R:-16}"
export GRIDOPS_LORA_ALPHA="${GRIDOPS_LORA_ALPHA:-32}"
export GRIDOPS_LEARNING_RATE="${GRIDOPS_LEARNING_RATE:-8e-5}"
export GRIDOPS_GRADIENT_CHECKPOINTING="${GRIDOPS_GRADIENT_CHECKPOINTING:-1}"
export GRIDOPS_UPLOAD="${GRIDOPS_UPLOAD:-1}"

python scripts/build_gridops_v41_repair_traces.py
python scripts/validate_traces.py "${GRIDOPS_TRACE_PATH}"
bash scripts/kaggle_sft_v3_gridops.sh
