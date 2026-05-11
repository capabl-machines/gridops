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

## v5 Causal Teacher Plan

The next model should be trained from a causal teacher, not from the full
72-hour LP oracle. The full LP is useful as a ceiling, but it has hindsight.
The model only sees the current observation, 4-hour forecasts, task context,
and previous action feedback.

New implementation:

```text
scripts/build_gridops_v5_causal_teacher_traces.py
scripts/kaggle_sft_v5_causal_teacher.sh
tests/test_v5_causal_teacher_traces.py
```

The v5 teacher is a rolling LP controller:

```text
current observation + previous outcome
-> build short-horizon LP using current + forecast demand/solar/price
-> include SOC, rebound, fuel, outage, grid cap, diesel, shedding, blackout constraints
-> execute only the first action
-> repeat each hour
```

This makes it a trainable teacher rather than a hindsight oracle.

Default v5 dataset design:

```text
base rows:    1,800 sampled v4 reason-action traces
teacher rows: 12 seeds/task * 72 hours = 2,592 causal LP traces
total:        about 4,392 traces
horizon:      12
init model:   77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4
run label:    sft_qwen25_3b_gridops_v5_causal_teacher
```

Why mix v4 base rows:

- v4.1 taught us that narrow repair can oversteer;
- teacher rows improve control behavior;
- v4 rows preserve stable formatting and general operating behavior.

Smoke check:

```text
command:
python scripts/build_gridops_v5_causal_teacher_traces.py \
  --output /tmp/gridops_v5_full_seed_smoke.jsonl \
  --summary-output /tmp/gridops_v5_full_seed_smoke_summary.json \
  --base-sample-limit 0 \
  --seed-start 17600 \
  --seeds-per-task 1 \
  --stride 1 \
  --horizon 12

rows: 216
validation: ok

teacher smoke scores:
task_1_normal:   0.8312
task_2_heatwave: 0.8187
task_3_crisis:   0.7407
```

Interpretation:

- the teacher is close to the LP ceiling on normal and heatwave;
- crisis remains below the full LP ceiling, but is a better supervision target
  than the current v4 model;
- v5 SFT should be judged by ceiling capture and task-wise improvement, not by
  an absolute `0.90` target.

Kaggle launch command:

```bash
GRIDOPS_SFT_STEPS=175 \
GRIDOPS_LEARNING_RATE=6e-5 \
GRIDOPS_RUN_LABEL=sft_qwen25_3b_gridops_v5_causal_teacher \
bash scripts/kaggle_sft_v5_causal_teacher.sh
```

Hugging Face Jobs launch command:

```bash
python scripts/launch_hf_job_v5_causal_teacher.py
```

Default HF Job settings:

```text
image:   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
flavor:  l4x1
timeout: 8h
secret:  HF_TOKEN from local HF_API_TOKEN/HF_TOKEN
branch:  codex/gridops-sft-pipeline
```

Use HF Jobs when notebook providers disconnect mid-run. The job clones the repo,
builds v5 traces, validates them, trains from the v4 adapter, and uploads the
new adapter to `77ethers/gridops-models/sft_qwen25_3b_gridops_v5_causal_teacher`.
HF Jobs require prepaid compute credits; HF Pro alone may not be sufficient.

Promotion gate:

```text
valid_action_rate >= 99%
task_1_normal >= v4 task_1 or no material regression
task_2_heatwave > v4 task_2
task_3_crisis > v4 task_3
average_score > 0.72
crisis blackout and diesel reported explicitly
```
