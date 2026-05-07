# GridOps v3 SFT on Kaggle

Use this when RunPod is unavailable and we need a near-free GPU.

For the full teaching/process version of this workflow, see
`docs/SLM_POST_TRAINING_BOOTCAMP.md`.

For the v4 reasoning-action correction dataset and runner, see
`docs/V4_REASONING_DATASET.md`.

For the current v4 Kimi reasoning run, use:

```bash
bash scripts/kaggle_sft_v4_reasoning.sh
```

or the notebook-style script:

```text
notebooks/gridops_kaggle_sft_v4_reasoning.py
```

## Kaggle Setup

1. Create a Kaggle Notebook.
2. Enable GPU in Notebook settings.
3. Add a Kaggle Secret named `HF_API_TOKEN`.
4. Use the notebook-style script at `notebooks/gridops_kaggle_sft_v3.py`.

The notebook clones the repo, installs the training stack, validates the v3
curriculum, runs oracle smoke tests, trains Qwen2.5-3B QLoRA, and uploads a new
adapter subfolder.

## Default Run

```bash
export GRIDOPS_TRACE_PATH=sft_traces/gridops_curriculum_v3_tool_augmented.jsonl
export GRIDOPS_RUN_LABEL=sft_qwen25_3b_gridops_tool_augmented_v3
export GRIDOPS_SFT_STEPS=500
export GRIDOPS_BATCH_SIZE=1
export GRIDOPS_GRAD_ACCUM=16
export GRIDOPS_MAX_LENGTH=1280
bash scripts/kaggle_sft_v3_gridops.sh
```

Upload target:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_tool_augmented_v3
```

## If Kaggle Runs Out Of Memory

Lower these values and restart the kernel:

```bash
export GRIDOPS_MAX_LENGTH=1024
export GRIDOPS_LORA_R=8
export GRIDOPS_LORA_ALPHA=16
```

Keep `GRIDOPS_BATCH_SIZE=1` and gradient checkpointing enabled.

## Acceptance Gate

After training, evaluate against the same holdout seeds used for v1/v2:

```bash
python scripts/evaluate_gridops_adapter.py \
  --adapter-path 77ethers/gridops-models/sft_qwen25_3b_gridops_tool_augmented_v3 \
  --seeds 7001,7002,7003 \
  --output evals/gridops_sft_tool_augmented_v3_holdout_7001_7003.json
```

Promote v3 only if:

- valid action rate is at least `99.5%`;
- task 1 keeps the v2 battery improvement;
- task 3 recovers to at least the v1 crisis score;
- average holdout beats both v1 and v2.
