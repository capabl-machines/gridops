---
title: GridOps
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
tags:
- openenv
- microgrid
- energy
- india
- optimization
- reinforcement-learning
- decision-making
models:
- capabl-machines/gridops-strategy-selector-v7
---

# GridOps: Strategy-First Microgrid Operator

GridOps is an OpenEnv microgrid environment and demo for Indian community energy
operation. A strategy selector chooses high-level operating intent, and a causal
optimizer converts that intent into safe bounded dispatch actions.

```text
Observation -> strategy -> optimizer -> GridOpsAction -> OpenEnv
```

## Release Metrics

| System | Avg score | Task 1 | Task 2 | Task 3 | Validity | LP capture |
|---|---:|---:|---:|---:|---:|---:|
| v7 deterministic strategy-controller | **0.7907** | **0.7995** | **0.8224** | **0.7503** | 1.0000 action | **96.04%** |
| v7.3 learned strategy selector | 0.7888 | 0.7993 | 0.8223 | 0.7449 | 1.0000 strategy | 95.81% |
| Full LP ceiling | 0.8233 | 0.8372 | 0.8416 | 0.7912 | - | 100.00% |

## What The Demo Shows

- normal summer, heatwave, and crisis/outage tasks;
- live OpenEnv reset/step loop;
- optimizer-backed `/api/plan`;
- strategy-mediated planning;
- battery, diesel, cost, blackout, and reward dynamics.

## Model

The learned selector is published at:

[capabl-machines/gridops-strategy-selector-v7](https://huggingface.co/capabl-machines/gridops-strategy-selector-v7)

It emits strict strategy JSON. The demo keeps final dispatch guarded by the
causal optimizer.
