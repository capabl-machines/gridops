# GridOps SFT Pipeline Model Card

## Summary

This card documents the planned GridOps SFT milestone: a compact model that reads one hourly microgrid observation and emits a valid JSON `GridOpsAction`.

The environment itself remains the original GridOps OpenEnv:

- action: `battery_dispatch`, `diesel_dispatch`, `demand_shedding`;
- tasks: normal summer, heatwave + price spike, crisis + grid outage;
- score: 50% cost efficiency, 25% reliability, 25% green score.

## Current Status

No trained GridOps SFT adapter is promoted as final yet. This branch adds the reproducible training pipeline, generated curriculum traces, validation tests, and guarded SFT launch script.

## Planned Model Lineage

| Stage | Default |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Fallback base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Training method | QLoRA SFT |
| Dataset | `sft_traces/gridops_curriculum_1200.jsonl` |
| Output contract | JSON-only `GridOpsAction` |
| Upload target | `77ethers/gridops-models/sft_qwen25_3b_gridops_curriculum1200_v1` |

## Dataset

The initial curriculum contains 1,200 deterministic expert traces:

| Task | Difficulty | Rows |
|---|---:|---:|
| `task_1_normal` | easy | 300 |
| `task_2_heatwave` | medium | 400 |
| `task_3_crisis` | hard | 500 |

Each row stores:

- task, seed, hour, difficulty;
- system/user messages;
- JSON-only completion;
- parsed action;
- raw observation;
- focus tags such as `low_soc`, `high_price`, `grid_cap_pressure`, `outage_window`, `rebound`, and `diesel_scarcity`;
- score context from the expert rollout;
- validation result.

Targeted label strategies in the generated dataset:

| Label strategy | Rows |
|---|---:|
| `oracle_high_price_arbitrage` | 445 |
| `oracle_routine_dispatch` | 425 |
| `oracle_diesel_rationing` | 117 |
| `oracle_grid_cap_pressure` | 80 |
| `oracle_outage_guard` | 70 |
| `oracle_low_soc_recovery` | 60 |
| `oracle_rebound_avoidance` | 3 |

## Reward / Objective Connection

The SFT model is trained to imitate actions that perform well under the existing environment grader. RL is intentionally deferred until SFT proves JSON reliability and basic task competence.

SFT acceptance gates before any RL:

- valid action rate >= 98%;
- average holdout score >= 0.65;
- no task below do-nothing baseline;
- Task 3 score >= 0.55;
- deterministic holdout evaluation stable across fixed seeds.

## Evaluation Plan

The evaluation harness reports:

- valid action rate;
- per-task average score;
- total blackout kWh;
- diesel kWh;
- actual cost;
- comparison against do-nothing, oracle, and adversarial policies.

Adversarial policies include always charge, always discharge, always diesel, shed-farmer, diesel-chatter, blackout-acceptor, price-greedy, and grid-only.

## Known Limitations

- The initial model optimizes valid JSON actions, not explanatory reasoning.
- The expert policy is a strong heuristic, not a mathematical optimum.
- Trace labels are generated from deterministic rollouts and should be expanded with LP/MPC or human-reviewed traces before a high-stakes deployment.
- RL/GRPO should only be attempted after the SFT acceptance gates pass.

## Reproducibility

Generate and validate traces:

```bash
python scripts/generate_sft_traces.py
python scripts/validate_traces.py sft_traces/gridops_curriculum_1200.jsonl
```

Evaluate baselines:

```bash
python scripts/oracle_test.py
python scripts/evaluate_gridops_model.py --policy oracle
python scripts/evaluate_gridops_model.py --policy do_nothing
```

Evaluate an API-hosted trained model:

```bash
export HF_API_TOKEN=...
python scripts/evaluate_gridops_model.py --model-name "$MODEL_NAME"
```

Launch SFT only when ready:

```bash
export HF_API_TOKEN=...
python scripts/hf_sft_gridops.py
```
