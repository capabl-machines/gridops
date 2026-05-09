#!/usr/bin/env bash
set -euo pipefail

export GRIDOPS_BASE_MODEL="${GRIDOPS_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export GRIDOPS_INIT_ADAPTER="${GRIDOPS_INIT_ADAPTER:-77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4}"
export GRIDOPS_GRPO_RUN_LABEL="${GRIDOPS_GRPO_RUN_LABEL:-grpo_qwen25_3b_gridops_openenv_v4_smoke}"
export GRIDOPS_MODEL_REPO="${GRIDOPS_MODEL_REPO:-77ethers/gridops-models}"
export GRIDOPS_UPLOAD="${GRIDOPS_UPLOAD:-1}"

python scripts/hf_grpo_gridops_openenv.py \
  --mode smoke_reward_contract \
  --horizon "${GRIDOPS_GRPO_HORIZON:-4}" \
  --limit "${GRIDOPS_GRPO_PROMPT_LIMIT:-24}" \
  --output "evals/${GRIDOPS_GRPO_RUN_LABEL}_reward_contract_smoke.json"

if [[ "${GRIDOPS_RUN_TRAIN:-0}" == "1" ]]; then
  python -m pip install -q \
    -e . \
    "huggingface_hub>=0.34,<1.0" \
    "transformers>=4.56.2" \
    "trl>=0.22.2" \
    "peft>=0.17.1" \
    "datasets>=4.0" \
    "accelerate>=1.0" \
    "bitsandbytes" \
    "protobuf"

  python scripts/hf_grpo_gridops_openenv.py \
    --mode train \
    --horizon "${GRIDOPS_GRPO_TRAIN_HORIZON:-1}" \
    --prompt-limit "${GRIDOPS_GRPO_PROMPT_LIMIT:-24}" \
    --max-steps "${GRIDOPS_GRPO_STEPS:-8}" \
    --num-generations "${GRIDOPS_GRPO_NUM_GENERATIONS:-2}" \
    --batch-size "${GRIDOPS_GRPO_BATCH_SIZE:-2}" \
    --grad-accum "${GRIDOPS_GRPO_GRAD_ACCUM:-1}" \
    --learning-rate "${GRIDOPS_GRPO_LR:-2e-6}"
fi
