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

India does not only need more solar panels. It needs intelligence between the
panel, the battery, the grid, the diesel backup, and the people depending on
power.

GridOps is our attempt to build that intelligence layer for community
microgrids: a small learned strategy selector, a deterministic optimizer, and a
simulated OpenEnv world where every decision is scored by cost, reliability,
and diesel use.

GridOps Strategy Selector v7 is the learned part of that system. It does not
pretend to be the whole grid engineer. It reads a GridOps/OpenEnv observation
and emits strict strategy JSON. A causal optimizer then converts that strategy
into the final bounded dispatch action.

This is the central release lesson:

```text
microgrid state
  -> small AI chooses operating strategy
  -> optimizer converts strategy into safe dispatch
  -> OpenEnv scores cost, reliability, diesel, blackout
```

That split keeps the language model focused on contextual judgment while
leaving constrained numerical dispatch to an optimizer.

## Why This Matters

Distributed solar is scaling quickly across Indian apartments, societies,
campuses, factories, and local energy systems. That creates a second-order
problem: who operates the system once it exists?

A local operator now has to answer questions like:

- Should the battery charge now because grid power is cheap?
- Should it preserve charge for the evening peak?
- Should diesel be allowed during a crisis, or conserved for an outage?
- Should demand response be used, knowing it rebounds later?
- How do we keep the lights on while still reducing cost and diesel?

Bad control turns clean infrastructure into higher bills, battery misuse,
blackouts, or unnecessary diesel. Good control makes the same infrastructure
more useful.

GridOps explores a practical pattern for sustainable infrastructure AI:

```text
model for judgment
tools for physics
environment for truth
metrics for accountability
```

## What We Built

The environment is a 72-hour community microgrid simulation with three regimes:

| Task | Situation | What it tests |
|---|---|---|
| Task 1 normal | Normal summer demand and solar | basic battery arbitrage |
| Task 2 heatwave | demand spike plus price stress | forecast-aware peak planning |
| Task 3 crisis | haze, heatwave, limited diesel, grid outage | islanding and reliability |

The action space remains the real OpenEnv dispatch contract:

```json
{"battery_dispatch":0.0,"diesel_dispatch":0.0,"demand_shedding":0.0}
```

But the model is not asked to emit those floats directly. That was the wrong
abstraction.

Directly asking a small model to output exact battery/diesel/shedding floats
proved brittle. The model could learn JSON, but the optimization burden was too
large for a small SFT policy. GridOps v7 turns the model into a strategy
selector and lets a causal LP/MPC controller execute the details.

This is the mature pattern we arrived at:

```text
LLM: choose operating intent
Optimizer: satisfy constraints and choose dispatch
OpenEnv: score the result
```

## Key Results

```text
100% valid strategy outputs from the learned selector
96.04% LP ceiling capture from the strategy-controller system
3 tested operating regimes: normal, heatwave, crisis/outage
```

The strongest deployable system today is the deterministic v7
strategy-controller. The learned v7.3 selector nearly matches it while producing
perfectly valid strategy JSON on the holdout set.

| Release highlight | Value |
|---|---:|
| v7 deterministic controller average score | 0.7907 |
| v7.3 learned selector average score | 0.7888 |
| v7.3 valid strategy rate | 100.00% |
| v7.3 LP ceiling capture | 95.81% |
| v5.1 direct-action baseline average score | 0.7354 |

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

## Engineering Journey

The most important result was not one checkpoint. It was the discovery of the
right interface.

```text
v4:    direct action SFT from reasoning traces
v5:    causal LP teacher imitation
v5.1:  crisis repair continuation
v6:    tool-corrected action SFT, not promoted
v6.1:  clean LP-critic action SFT, not promoted
v7:    strategy-first harness
v7.1:  SFT strategy selector
v7.2:  DPO preference tuning
v7.3:  crisis-weighted DPO release checkpoint
```

The lesson:

> The model does not need to become the entire operator. It needs to learn the
> operating language that lets deterministic tools act safely.

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

That pattern is bigger than microgrids. The same structure can apply to:

- apartment and society energy systems;
- water pump scheduling;
- cold chains;
- EV charging depots;
- factory energy optimization;
- farm irrigation and storage;
- disaster-resilient local infrastructure.

GridOps is one case study in a broader Capabl Machines thesis: useful AI for
physical systems should be trained and evaluated inside the world it claims to
operate.

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
