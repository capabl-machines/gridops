# GridOps SFT Pipeline Model Card

## Summary

This card documents the GridOps SFT v1 milestone: a compact model that reads one hourly microgrid observation and emits a valid JSON `GridOpsAction`.

The environment itself remains the original GridOps OpenEnv:

- action: `battery_dispatch`, `diesel_dispatch`, `demand_shedding`;
- tasks: normal summer, heatwave + price spike, crisis + grid outage;
- score: 50% cost efficiency, 25% reliability, 25% green score.

## Current Status

SFT v1 is trained, uploaded, and evaluated on held-out seeds `7001,7002,7003`.

Adapter:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_mixed1418_v1
```

SFT v1 passes the SFT gates and should be treated as the current small-model baseline. RL/GRPO remains a follow-up phase, not part of this result.

## Model Lineage

| Stage | Default |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Training method | QLoRA SFT |
| Dataset | `sft_traces/gridops_mixed_oracle_deepseek_1418.jsonl` |
| Output contract | JSON-only `GridOpsAction` |
| Upload target | `77ethers/gridops-models/sft_qwen25_3b_gridops_mixed1418_v1` |
| Training steps | 300 |
| Hardware | RTX 5090, QLoRA 4-bit |

## Dataset

The deterministic curriculum contains 1,200 expert traces:

| Task | Difficulty | Rows |
|---|---:|---:|
| `task_1_normal` | easy | 300 |
| `task_2_heatwave` | medium | 400 |
| `task_3_crisis` | hard | 500 |

The training run used a mixed 1,418-row dataset:

| Source | Rows |
|---|---:|
| Deterministic oracle curriculum | 1,200 |
| DeepSeek V4 Pro teacher traces via OpenRouter | 218 |
| Total | 1,418 |

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

## Evaluation Results

Held-out seeds: `7001,7002,7003`.

| Policy | Avg score | Valid JSON | Task 1 | Task 2 | Task 3 |
|---|---:|---:|---:|---:|---:|
| Do-nothing | 0.5133 | 100.00% | 0.5820 | 0.5057 | 0.4522 |
| GridOps SFT v1 | 0.6854 | 99.85% | 0.6615 | 0.7300 | 0.6648 |
| Oracle | 0.7688 | 100.00% | 0.7932 | 0.8087 | 0.7046 |

Gate verdict:

| Gate | Target | SFT v1 | Pass |
|---|---:|---:|---|
| Valid JSON action rate | >= 98% | 99.85% | yes |
| Average holdout score | >= 0.65 | 0.6854 | yes |
| No task below do-nothing | required | all above | yes |
| Task 3 crisis score | >= 0.55 | 0.6648 | yes |

Behavioral anti-hack check:

| Task | SFT battery throughput | Do-nothing | Oracle |
|---|---:|---:|---:|
| Normal | 577.97 kWh | 0.00 kWh | 970.62 kWh |
| Heatwave | 1,721.05 kWh | 0.00 kWh | 2,075.75 kWh |
| Crisis | 2,898.10 kWh | 0.00 kWh | 3,170.60 kWh |

Blackout reduction:

| Task | SFT blackout | Do-nothing blackout | Oracle blackout |
|---|---:|---:|---:|
| Normal | 177.57 kWh | 298.85 kWh | 15.24 kWh |
| Heatwave | 258.30 kWh | 895.00 kWh | 41.25 kWh |
| Crisis | 978.99 kWh | 2,425.76 kWh | 699.56 kWh |

The model learned meaningful battery dispatch rather than a do-nothing shortcut. It still lags the oracle, especially on normal-day timing and crisis blackout minimization.

Training evidence:

- [SFT training curve](evals/plots/gridops_sft_training_curve.png)
- [Holdout score plot](evals/plots/gridops_holdout_scores.png)
- [Battery throughput plot](evals/plots/gridops_battery_throughput.png)
- [Blackout reduction plot](evals/plots/gridops_blackout_kwh.png)
- [Holdout summary JSON](evals/plots/gridops_holdout_summary.json)
- [Raw SFT holdout JSON](evals/gridops_sft_mixed1418_v1_holdout_7001_7003.json)
- [Parsed training metrics](evals/plots/gridops_sft_training_metrics.json)

The evaluation harness reports:

- valid action rate;
- per-task average score;
- total blackout kWh;
- diesel kWh;
- actual cost;
- comparison against do-nothing, oracle, and adversarial policies.

Adversarial policies include always charge, always discharge, always diesel, shed-farmer, diesel-chatter, blackout-acceptor, price-greedy, and grid-only.

## Known Limitations

- The model optimizes valid JSON actions, not explanatory reasoning.
- The expert policy is a strong heuristic, not a mathematical optimum.
- Trace labels are generated from deterministic rollouts and should be expanded with LP/MPC or human-reviewed traces before a high-stakes deployment.
- One invalid JSON was observed in 648 generated actions: `{"battery_dispatch":-0.8,"diesel_dispatch",0.0,"demand_shedding":0.0}`.
- RL/GRPO should be attempted only as a small smoke run because SFT now passes format and score gates.

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

Evaluate the local adapter:

```bash
export HF_API_TOKEN=...
python scripts/evaluate_gridops_adapter.py \
  --adapter-path outputs/sft_qwen25_3b_gridops_mixed1418_v1 \
  --seeds 7001,7002,7003 \
  --output evals/gridops_sft_mixed1418_v1_holdout_7001_7003.json
```

Generate plots:

```bash
python scripts/plot_gridops_evals.py
```

Launch SFT only when ready:

```bash
export HF_API_TOKEN=...
python scripts/hf_sft_gridops.py
```
