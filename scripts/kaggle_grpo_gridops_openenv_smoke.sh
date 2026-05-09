#!/usr/bin/env bash
set -euo pipefail

export GRIDOPS_BASE_MODEL="${GRIDOPS_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export GRIDOPS_INIT_ADAPTER="${GRIDOPS_INIT_ADAPTER:-77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4}"
export GRIDOPS_GRPO_RUN_LABEL="${GRIDOPS_GRPO_RUN_LABEL:-grpo_qwen25_3b_gridops_openenv_v4_smoke}"

python scripts/hf_grpo_gridops_openenv.py \
  --mode smoke_reward_contract \
  --horizon "${GRIDOPS_GRPO_HORIZON:-4}" \
  --limit "${GRIDOPS_GRPO_PROMPT_LIMIT:-24}" \
  --output "evals/${GRIDOPS_GRPO_RUN_LABEL}_reward_contract_smoke.json"
