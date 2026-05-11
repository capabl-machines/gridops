# GridOps Training Learnings

This is the running engineering notebook for GridOps model-building decisions.

## Current Best Model

Best SFT baseline:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4
```

Holdout:

```text
average_score:      0.7076
valid_action_rate:  0.9738
task_1_normal:      0.7891
task_2_heatwave:    0.7082
task_3_crisis:      0.6255
crisis diesel_kwh:  123.05
```

v4 is the current model to preserve.

## v4.1 Repair Lesson

The v4.1 bound-repair continuation was useful but not promoted.

It improved crisis diesel behavior on the smoke seed, but hurt normal and
heatwave operation. That taught us:

- narrow SFT repair can oversteer the policy;
- validity repair alone is not enough;
- we should avoid hand-writing increasingly specific trace patches.

## GRPO Lesson

The first OpenEnv-backed GRPO smoke proved the RL loop works:

- v4 adapter can be initialized and updated;
- completions can be parsed into `GridOpsAction`;
- OpenEnv can score completions inside the reward function;
- artifacts can be uploaded to Hugging Face.

The first horizon-1 reward damaged heatwave behavior. Horizon-4 reward largely
fixed that, but it still did not clearly beat v4 on the smoke seed.

Core learning:

```text
OpenEnv-backed GRPO is operational, but reward engineering must respect
multi-hour heatwave/rebound and SOC preservation, not only one-step cost.
```

## Why 0.90 Needs A Ceiling Test

The user target is `0.90` on all tasks.

Before training toward `0.90`, we must prove that `0.90` is reachable under the
current environment physics and scoring. If the best non-LLM controller cannot
approach it, then more model training cannot reliably solve it.

The current grader is:

```text
score = 0.50 * cost_efficiency + 0.25 * reliability + 0.25 * green_score
```

That means even a perfect no-blackout, no-diesel episode needs:

```text
0.90 = 0.50 * cost_efficiency + 0.25 * 1.00 + 0.25 * 1.00
cost_efficiency = 0.80
actual_cost <= 20% of do-nothing baseline cost
```

So `0.90` is not merely "use the battery correctly". It requires extremely low
cost while also keeping reliability and diesel use near perfect.

This is why we added:

```text
scripts/evaluate_gridops_mpc_planner.py
scripts/evaluate_gridops_lp_oracle.py
```

The MPC planner is not a model. It is a ceiling-finding controller:

```text
state -> sample candidate action sequences -> simulate in copied OpenEnv -> execute best first action
```

The LP oracle is an even stronger ceiling test. It solves a full-episode relaxed
linear program using the known demand, solar, price, outage, battery, grid,
diesel, shedding, rebound, and blackout constraints, then replays the resulting
actions through the actual OpenEnv grader.

If this search planner reaches high scores, it can become the next teacher:

```text
MPC traces -> v5 SFT -> OpenEnv GRPO
```

If it does not, then the bottleneck is likely one of:

- environment physics;
- action space;
- score weights;
- available controllable resources;
- planning horizon/search quality.

## LP Ceiling Result

First full-episode LP oracle run:

```text
script: scripts/evaluate_gridops_lp_oracle.py
seeds:  7001,7002,7003
output: evals/gridops_lp_oracle_holdout_7001_7003.json

average_score: 0.8233

task_1_normal:   0.8372
  blackout_kwh:  0.00
  diesel_kwh:    0.00
  avg_cost:      28,698.45

task_2_heatwave: 0.8416
  blackout_kwh:  0.00
  diesel_kwh:    101.50
  avg_cost:      61,356.04

task_3_crisis:   0.7912
  blackout_kwh:  62.02
  diesel_kwh:    792.00
  avg_cost:      183,761.90
```

Interpretation:

- the v4 SFT model is not near the true ceiling yet, so there is real room to
  improve;
- `0.90` on all three tasks is not supported by this first ceiling run;
- crisis is especially constrained by outage/fuel/backup physics;
- the next high-leverage work is to turn the LP/MPC controller into a stronger
  teacher and then use OpenEnv-backed GRPO against that controller/environment.

This does not prove `0.90` is impossible, because the LP is a relaxed
approximation and the MPC planner is still basic. It does prove that `0.90`
should be treated as a research target, not an ordinary next-training-run gate.

## Next Decision Gate

Run MPC on holdout seeds:

```bash
python scripts/evaluate_gridops_mpc_planner.py \
  --seeds 7001,7002,7003 \
  --horizon 6 \
  --sequence-count 96 \
  --max-actions 48 \
  --output evals/gridops_mpc_planner_h6_seq96_holdout_7001_7003.json
```

Run the LP oracle on holdout seeds:

```bash
python scripts/evaluate_gridops_lp_oracle.py \
  --seeds 7001,7002,7003 \
  --output evals/gridops_lp_oracle_holdout_7001_7003.json
```

Interpretation:

```text
LP avg >= 0.90: 0.90 is physically reachable; distill LP/MPC traces.
LP avg 0.80-0.90: model can probably improve, but 0.90 all-task may be hard.
LP avg <= v4: current scoring/physics or LP approximation needs debugging.
LP task ceiling < 0.90: 0.90 likely requires environment/action/scoring changes.
```

## Artifact Rule

Do not overwrite v4.

Every new model or planner run gets a new subfolder or eval path.
