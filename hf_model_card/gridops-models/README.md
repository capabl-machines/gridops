---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
tags:
- microgrid
- energy
- openenv
- qlora
- sft
- reinforcement-learning
- decision-making
- json-actions
datasets:
- 77ethers/gridops
metrics:
- valid_action_rate
- environment_score
---

# GridOps SFT v1: A JSON-Action Model for Microgrid Dispatch

## Problem Statement

Community microgrids are becoming operationally real: rooftop solar, batteries, diesel backup, time-varying grid prices, and outage risk all interact hour by hour. A controller must decide when to charge, discharge, run diesel, or shed demand while keeping costs low and avoiding blackouts.

GridOps frames this as an OpenEnv environment for a 100-home Indian community microgrid. Each episode lasts 72 hours. At every hour, the agent observes demand, solar, battery state of charge, grid price, diesel fuel, short forecasts, cumulative cost, and blackout history, then emits one action:

```json
{"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0}
```

The hard part is not the JSON. The hard part is temporal judgment: charge before evening peaks, preserve battery before outages, ration diesel during crisis windows, and avoid demand shedding unless it is truly necessary.

## Impact

A useful small model for GridOps should do three things:

1. Produce valid bounded actions reliably enough to run inside an environment loop.
2. Improve over do-nothing/grid-only operation across normal, heatwave, and crisis tasks.
3. Show environment-visible evidence of learning, especially real battery throughput and blackout reduction rather than a formatting shortcut.

This matters because many energy-control demos stop at prose reasoning. GridOps evaluates actual actions through physics, cost, reliability, and emissions-linked diesel usage.

## Proposed Solution

This repository contains a QLoRA SFT adapter trained from `Qwen/Qwen2.5-3B-Instruct` to emit GridOps JSON actions.

Final adapter:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_mixed1418_v1
```

The adapter is stored in a subfolder of this model repo. The smoke run is also preserved separately:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_mixed1418_smoke
```

The model is intentionally SFT-only. RL/GRPO is deferred until SFT proves format stability and environment competence.

## Environment Contract

- Environment: [77ethers/gridops Space](https://huggingface.co/spaces/77ethers/gridops)
- Live demo: [77ethers-gridops.hf.space](https://77ethers-gridops.hf.space)
- Action schema: `battery_dispatch [-1, 1]`, `diesel_dispatch [0, 1]`, `demand_shedding [0, 1]`
- Tasks:
  - `task_1_normal`: normal summer arbitrage
  - `task_2_heatwave`: heatwave plus price spike
  - `task_3_crisis`: heatwave, haze, limited diesel, grid outage
- Score: 50% cost efficiency, 25% reliability, 25% green score

## Dataset

Training used a 1,418-row curriculum:

| Source | Rows |
|---|---:|
| Deterministic oracle curriculum | 1,200 |
| DeepSeek V4 Pro teacher traces via OpenRouter | 218 |
| Total | 1,418 |

The deterministic curriculum was balanced by difficulty:

| Task | Difficulty | Rows |
|---|---:|---:|
| `task_1_normal` | easy | 300 |
| `task_2_heatwave` | medium | 400 |
| `task_3_crisis` | hard | 500 |

Each trace stores the task, seed, hour, prompt messages, JSON completion, parsed action, raw observation, score context, focus tags, and validation status.

## Training

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | QLoRA SFT |
| Adapter target | LoRA on attention and MLP projection modules |
| Steps | 300 |
| Hardware | RTX 5090, 4-bit training |
| Upload path | `sft_qwen25_3b_gridops_mixed1418_v1` |

Training curve:

![GridOps SFT training curve](evals/plots/gridops_sft_training_curve.png)

The logged loss dropped from `1.53` to `0.1478`; final mean token accuracy was `0.9486`.

## Evaluation

Held-out seeds: `7001,7002,7003`.

| Policy | Avg score | Valid JSON | Task 1 | Task 2 | Task 3 |
|---|---:|---:|---:|---:|---:|
| Do-nothing | 0.5133 | 100.00% | 0.5820 | 0.5057 | 0.4522 |
| GridOps SFT v1 | 0.6854 | 99.85% | 0.6615 | 0.7300 | 0.6648 |
| Oracle | 0.7688 | 100.00% | 0.7932 | 0.8087 | 0.7046 |

![GridOps holdout scores](evals/plots/gridops_holdout_scores.png)

## Did It Really Learn Battery Usage?

Yes. The key anti-hack check is battery throughput and blackout reduction.

| Task | SFT battery throughput | Do-nothing | Oracle |
|---|---:|---:|---:|
| Normal | 577.97 kWh | 0.00 kWh | 970.62 kWh |
| Heatwave | 1,721.05 kWh | 0.00 kWh | 2,075.75 kWh |
| Crisis | 2,898.10 kWh | 0.00 kWh | 3,170.60 kWh |

![Battery throughput](evals/plots/gridops_battery_throughput.png)

Blackout reduction versus do-nothing:

| Task | SFT blackout | Do-nothing blackout | Oracle blackout |
|---|---:|---:|---:|
| Normal | 177.57 kWh | 298.85 kWh | 15.24 kWh |
| Heatwave | 258.30 kWh | 895.00 kWh | 41.25 kWh |
| Crisis | 978.99 kWh | 2,425.76 kWh | 699.56 kWh |

![Blackout reduction](evals/plots/gridops_blackout_kwh.png)

This is not a do-nothing shortcut: the model uses the battery heavily, especially in heatwave and crisis regimes, and cuts blackout energy substantially.

## SFT Gate Verdict

| Gate | Target | SFT v1 | Pass |
|---|---:|---:|---|
| Valid JSON action rate | >= 98% | 99.85% | yes |
| Average holdout score | >= 0.65 | 0.6854 | yes |
| No task below do-nothing | required | all above | yes |
| Task 3 crisis score | >= 0.55 | 0.6648 | yes |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

base_model = "Qwen/Qwen2.5-3B-Instruct"
adapter = "77ethers/gridops-models"
subfolder = "sft_qwen25_3b_gridops_mixed1418_v1"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(adapter, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant, device_map="auto")
model = PeftModel.from_pretrained(model, adapter, subfolder=subfolder)
```

The model should be prompted with the same GridOps prompt contract used by the training harness in GitHub:

- `gridops/prompting.py`
- `scripts/evaluate_gridops_adapter.py`

GitHub branch with training/eval code:

```text
https://github.com/capabl-machines/gridops/tree/codex/gridops-sft-pipeline
```

## Artifacts

- Final adapter: [`sft_qwen25_3b_gridops_mixed1418_v1`](sft_qwen25_3b_gridops_mixed1418_v1)
- Smoke adapter: [`sft_qwen25_3b_gridops_mixed1418_smoke`](sft_qwen25_3b_gridops_mixed1418_smoke)
- Holdout summary: [`evals/plots/gridops_holdout_summary.json`](evals/plots/gridops_holdout_summary.json)
- Parsed training metrics: [`evals/plots/gridops_sft_training_metrics.json`](evals/plots/gridops_sft_training_metrics.json)

## Limitations

- This is a compact SFT policy model, not a natural-language reasoning assistant.
- The model is below oracle on all tasks; it is strongest on heatwave and crisis, weaker on normal-day precision timing.
- One invalid JSON was observed in 648 generated holdout actions.
- The teacher policy is heuristic, not a mathematical optimum.
- The model is intended for benchmarking and research inside GridOps, not deployment to real energy infrastructure.

## Next Steps

The sensible next phase is either targeted SFT v2 for normal-day timing and late-crisis robustness, or a tiny RL/GRPO smoke run now that format stability is proven.
