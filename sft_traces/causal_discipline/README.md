# Causal Discipline Continuation Dataset

This folder contains a second-stage SFT dataset for the current CarbonAlpha
Qwen2.5 model. It intentionally does **not** replace the older curriculum
datasets.

## Train On

- `causal_discipline_v1.jsonl`

## Purpose

The current model is format-stable, but it can invent fluent macro causal
bridges. This dataset teaches the model to:

- identify the actual regime/channel in the news,
- say what it is **not** assuming,
- avoid unsupported oil/energy/inflation/contagion leaps,
- use boring base-rate allocations for irrelevant news,
- use small hedges for real liquidity or credit stress.

## Schema

Rows use the same SFT schema as the existing datasets:

```json
{
  "id": "...",
  "seed_id": "...",
  "seed_year": "causal-discipline-v1",
  "seed_category": "...",
  "prompt": "...",
  "completion": "<think>...</think>\n{\"weights\": ...}",
  "raw": {}
}
```

## Generator

Run or extend:

```bash
uv run --with google-genai python sft_traces/generate_causal_discipline_traces.py \
  --per-scenario 6 \
  --batch-size 2 \
  --out sft_traces/causal_discipline/causal_discipline_v1.jsonl \
  --resume
```

`causal_discipline_v1.jsonl.failures.json` is a retry/debug artifact, not a
training file.

## DeepSeek V4 Pro Variant

DeepSeek/OpenRouter traces live in:

- `deepseek_v4_causal_discipline_v1.jsonl`

Generate or resume with:

```bash
uv run python sft_traces/generate_openrouter_deepseek_traces.py \
  --per-scenario 3 \
  --batch-size 1 \
  --model deepseek/deepseek-v4-pro \
  --out sft_traces/causal_discipline/deepseek_v4_causal_discipline_v1.jsonl \
  --resume \
  --sleep-s 12
```

The OpenRouter provider for `deepseek/deepseek-v4-pro` was rate-limited during
generation, so this file is intentionally resume-safe and may grow over
multiple passes.
