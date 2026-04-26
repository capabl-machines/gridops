---
title: CarbonAlpha
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
hardware: t4-small
---

# CarbonAlpha — climate-aware portfolio reasoner

SFT-warmed → GRPO-tuned Qwen2.5-7B-Instruct, trained on the CarbonAlpha OpenEnv
(Reasoning-Under-Constraints Hackathon, Round 2).

Given today's news, the model commits ONE 12-quarter allocation across
[TECH, OIL, GREEN, REAL_ESTATE, BONDS] subject to a 25 kg carbon cap.

This Space uses a custom FastAPI + HTML walkthrough UI. It exposes:

- `/` custom dashboard
- `/health`
- `/metadata`
- `/api/start`
- `/api/step`

The dashboard frames three things:

1. **Long-horizon commitment** — a "Locked Path" chart projects CarbonAlpha's allocation across all 12 quarters and overlays three counterfactual strategies (equal-weight, oil-heavy, green-heavy) replayed on the *same shock schedule*. You can see, at the moment of locking, what the next 3 years would look like under each choice.
2. **Carbon prominence** — a vertical thermometer in the score rail tracks cumulative kg vs the 25 kg cap, color-shifting as it fills. Beside it, a stacked-area chart attributes carbon to the asset that emitted it (OIL dominates when picked).
3. **Financial + environmental verdict** — a Pareto scatter plots all four strategies on (real return %, carbon kg). CarbonAlpha lands in the lower-right "good" quadrant: high return AND low carbon. A one-line verdict above the charts spells it out: *"+X% return, Y kg / 25 kg carbon, dominates Z/3 alt strategies."*

Backing model: [`77ethers/CarbonAlpha`](https://huggingface.co/77ethers/CarbonAlpha) — subfolder `grpo_qwen25_7b_adapter_phase1_100_v1` (GRPO Phase 1, 100 steps; 5/5 holdout beats baseline, mean regret 0.106). Override at boot via the `MODEL_SUBFOLDER` env var to A/B against the SFT adapter.

Eval: 5/5 valid, 5/5 closed `<think></think>`, mean holdout regret +0.02796.
