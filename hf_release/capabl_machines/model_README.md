---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
- energy
- microgrid
- openenv
- strategy
- dpo
- peft
- qlora
- decision-making
- india
metrics:
- environment_score
- valid_strategy_rate
- lp_ceiling_capture
---

# GridOps Strategy Selector v7

GridOps Strategy Selector v7 is a small learned policy for community microgrid
operation. It does not directly output dispatch floats. Instead, it reads a
GridOps/OpenEnv observation and emits strict strategy JSON. A deterministic
causal optimizer converts that strategy into the final bounded GridOps action.

This is the central release lesson:

```text
LLM -> high-level operating strategy -> causal optimizer -> GridOpsAction -> OpenEnv
```

That split keeps the language model focused on contextual judgment while
leaving constrained numerical dispatch to an optimizer.

## Problem

Indian apartments, housing societies, campuses, and community microgrids are
increasingly operating with rooftop solar, batteries, diesel backup, grid price
variation, demand spikes, and outage risk. The operator must decide every hour
when to charge, preserve, discharge, run diesel, or tolerate limited demand
response.

Directly asking a small model to output exact battery/diesel/shedding floats
proved brittle. The model could learn JSON, but the optimization burden was too
large for a small SFT policy. GridOps v7 turns the model into a strategy
selector and lets a causal LP/MPC controller execute the details.

## Output Schema

The model emits only strict JSON:

```json
{
  "mode": "cost_saving",
  "risk_level": "low",
  "battery_bias": "charge",
  "diesel_policy": "avoid",
  "shedding_policy": "never"
}
```

Allowed values:

```text
mode:            cost_saving | peak_shaving | outage_prepare | reliability | recovery | fuel_conservation
risk_level:      low | medium | high | critical
battery_bias:    charge | preserve | discharge | neutral
diesel_policy:   avoid | allow_if_blackout | prewarm | conserve
shedding_policy: never | last_resort
```

## Model Lineage

```text
Base:        Qwen/Qwen2.5-1.5B-Instruct
SFT:         77ethers/gridops-models/sft_qwen25_15b_gridops_strategy_v7
DPO v7.2:    77ethers/gridops-models/dpo_qwen25_15b_gridops_strategy_v72
DPO v7.3:    77ethers/gridops-models/dpo_qwen25_15b_gridops_strategy_v73_crisis
Release:     capabl-machines/gridops-strategy-selector-v7
```

The released adapter is the v7.3 crisis-weighted DPO checkpoint. v7.3 remained
stable and matched v7.2, but did not beat the deterministic controller. The
recommended production policy is therefore the strategy-controller harness, with
this model as the learned strategy selector.

## Evaluation

Holdout seeds: `7001,7002,7003`.

![GridOps holdout task scores](assets/gridops_v7_task_scores.png)

| System | Avg score | Valid strategy/action | Task 1 normal | Task 2 heatwave | Task 3 crisis | LP capture |
|---|---:|---:|---:|---:|---:|---:|
| v5.1 direct action model | 0.7354 | 0.9969 action | 0.7896 | 0.7681 | 0.6484 | - |
| v7 deterministic strategy-controller | **0.7907** | 1.0000 action | **0.7995** | **0.8224** | **0.7503** | **96.04%** |
| v7.1 SFT strategy selector | 0.7880 | 1.0000 strategy | 0.7994 | 0.8224 | 0.7421 | 95.71% |
| v7.2 DPO strategy selector | 0.7888 | 1.0000 strategy | 0.7993 | 0.8223 | 0.7449 | 95.81% |
| v7.3 DPO strategy selector | 0.7888 | 1.0000 strategy | 0.7993 | 0.8223 | 0.7449 | 95.81% |
| Full-episode LP ceiling | 0.8233 | - | 0.8372 | 0.8416 | 0.7912 | 100.00% |

![GridOps LP ceiling capture](assets/gridops_v7_lp_capture.png)

## Operational Footprint

The crisis task is the real stress test: haze reduces solar, demand rises,
diesel is limited, and the grid outage forces islanded operation. The learned
selector stays close to the deterministic controller, but the remaining gap to
LP is mostly crisis blackout and cost.

![GridOps crisis operational footprint](assets/gridops_v7_crisis_footprint.png)

## Why This Is Useful

The learned model is small, stable, and schema-reliable. The controller is the
stronger deployable policy. Together they show a practical pattern for domain
AI systems:

```text
Do not force the model to be the whole controller.
Teach it the decision language.
Use tools for physics, constraints, validation, and scoring.
```

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = "capabl-machines/gridops-strategy-selector-v7"

tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

The model output should be parsed as `GridOpsStrategy`, then passed to the
GridOps controller. The final OpenEnv action remains:

```json
{"battery_dispatch":0.0,"diesel_dispatch":0.0,"demand_shedding":0.0}
```

## Intended Use

- Research and demos for strategy-conditioned microgrid operation.
- OpenEnv-style environment evaluation.
- Tool-assisted energy dispatch workflows where a validator/controller handles
  the final physical action.

## Limitations

- This adapter is not a standalone power-system controller.
- It should not be used for real grid operation without hardware validation,
  safety review, and local regulatory checks.
- It was evaluated in the GridOps simulated 72-hour environment, not on live
  metered deployments.
- The deterministic strategy-controller remains the recommended runtime
  baseline until a learned selector beats it.

## Links

- Demo Space: [capabl-machines/gridops-demo](https://huggingface.co/spaces/capabl-machines/gridops-demo)
- Source repo: [capabl-machines/gridops](https://github.com/capabl-machines/gridops)
- Earlier model archive: [77ethers/gridops-models](https://huggingface.co/77ethers/gridops-models)
