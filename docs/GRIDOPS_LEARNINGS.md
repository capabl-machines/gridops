# GridOps Training Learnings

This is the running engineering notebook for GridOps model-building decisions.
It is also the walkthrough of how we moved from "train a small model to emit
dispatch floats" toward a more capable system: OpenEnv as the world, LP/MPC as
the critic/controller, and the LLM as a strategy selector.

## Research Arc

The core problem was not only format following. GridOps is a sequential control
environment: each hourly action changes battery SOC, diesel fuel, rebound load,
blackout risk, and future cost. A model that emits three floats directly has to
learn both language formatting and control optimization inside its weights.

The journey so far:

```text
v4:    action SFT from Kimi/OpenRouter reasoning traces
v5:    causal LP teacher imitation
v5.1:  crisis repair continuation
v6:    tool-corrected action SFT, not promoted
v6.1:  clean LP-critic action SFT with Qwen3, not promoted
v7:    strategy-first harness, deterministic strategy -> causal optimizer
v7.1:  1.5B model learns strict strategy JSON, controller executes action
```

The important architectural shift:

```text
Before:
LLM -> exact battery/diesel/shedding floats -> OpenEnv

Now:
OpenEnv observation -> LLM or deterministic strategy -> causal LP/MPC
-> bounded GridOpsAction -> OpenEnv
```

This keeps the model useful where language models are strongest: selecting
intent under context. It keeps numerical dispatch where deterministic tools are
strongest: constrained optimization, validation, and safety.

## Current Best Systems

Best deployed/controller baseline:

```text
v7 deterministic strategy-controller
average_score:      0.7907
valid_action_rate:  1.0000
task_1_normal:      0.7995
task_2_heatwave:    0.8224
task_3_crisis:      0.7503
LP ceiling capture: 96.04%
```

Best model-only checkpoint before strategy harness:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_v51_crisis_repair
average_score:      0.7354
valid_action_rate:  0.9969
task_1_normal:      0.7896
task_2_heatwave:    0.7681
task_3_crisis:      0.6484
```

Latest strategy-selector experiment:

```text
77ethers/gridops-models/sft_qwen25_15b_gridops_strategy_v7
base_model: Qwen/Qwen2.5-1.5B-Instruct
output: strict GridOpsStrategy JSON
average_score:      0.7880
valid_strategy_rate: 1.0000
task_1_normal:      0.7994
task_2_heatwave:    0.8224
task_3_crisis:      0.7421
LP ceiling capture: 95.71%
```

Historical model to preserve:

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

v4 was the first useful baseline. It is no longer the best system, but it
remains a preserved comparison point.

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

## v5 Holdout Result

HF Jobs:

```text
training job: 6a022d0faff1cd33e8f33e56
eval job:     6a023c67317220dbbd1a7c2e
model:        77ethers/gridops-models/sft_qwen25_3b_gridops_v5_causal_teacher
eval file:    sft_qwen25_3b_gridops_v5_causal_teacher/evals/holdout_7001_7003_summary.json
```

Holdout on seeds `7001,7002,7003`:

```text
average_score:      0.7282
valid_action_rate:  0.9907

task_1_normal:      0.7923
task_2_heatwave:    0.7553
task_3_crisis:      0.6370
```

Compared with v4:

```text
v4 average:      0.7076 -> v5 average:      0.7282  (+0.0206)
v4 task_1:       0.7891 -> v5 task_1:       0.7923  (+0.0032)
v4 task_2:       0.7082 -> v5 task_2:       0.7553  (+0.0471)
v4 task_3:       0.6255 -> v5 task_3:       0.6370  (+0.0115)
v4 valid rate:   0.9738 -> v5 valid rate:   0.9907  (+0.0169)
```

Ceiling capture against the relaxed LP oracle:

```text
task_1: 0.7923 / 0.8372 = 94.6%
task_2: 0.7553 / 0.8416 = 89.8%
task_3: 0.6370 / 0.7912 = 80.5%
```

Interpretation:

- v5 passes the promotion gate and beats v4 on every task;
- heatwave improved substantially, confirming the causal-teacher approach;
- crisis remains the bottleneck: blackout is still high and diesel use is
  inconsistent across seeds;
- the next repair should be crisis-focused, but it must preserve the v5
  heatwave gain and valid-output stability.

## v5.1 Crisis Repair Scaffold

Prepared but not launched as paid training:

```text
scripts/build_gridops_v51_crisis_repair_traces.py
scripts/kaggle_sft_v51_crisis_repair.sh
tests/test_v51_crisis_repair_traces.py
```

Purpose:

```text
Continue from v5, not v4:
77ethers/gridops-models/sft_qwen25_3b_gridops_v5_causal_teacher

Small repair:
70 SFT steps
learning_rate: 3e-5
run label: sft_qwen25_3b_gridops_v51_crisis_repair
```

Dataset shape:

- crisis rows from the causal LP teacher around pre-outage, active outage,
  previous-blackout correction, post-outage recovery, and late crisis evening;
- normal/heatwave stability anchors to reduce forgetting;
- optional sampled v4 reason-action rows for format stability.

Smoke check:

```text
rows: 51
task_1_normal: 8
task_2_heatwave: 4
task_3_crisis: 39
diesel_positive: 11
shedding_positive: 3
validation: ok
```

Recommended next run only if we accept v5 as baseline and want one targeted
repair:

```bash
GRIDOPS_SFT_STEPS=70 \
GRIDOPS_LEARNING_RATE=3e-5 \
GRIDOPS_RUN_LABEL=sft_qwen25_3b_gridops_v51_crisis_repair \
bash scripts/kaggle_sft_v51_crisis_repair.sh
```

Promotion gate for v5.1:

```text
valid_action_rate >= 99%
task_3_crisis > 0.66
task_2_heatwave >= 0.74
task_1_normal >= 0.78
average_score > v5 average_score 0.7282
```

## v5.1 Holdout Result

HF Jobs:

```text
training job: 6a02c06e317220dbbd1a7ece
eval job:     6a02c6b8317220dbbd1a7eeb
model:        77ethers/gridops-models/sft_qwen25_3b_gridops_v51_crisis_repair
eval file:    sft_qwen25_3b_gridops_v51_crisis_repair/evals/holdout_7001_7003_summary.json
```

Holdout on seeds `7001,7002,7003`:

```text
average_score:      0.7354
valid_action_rate:  0.9969

task_1_normal:      0.7896
task_2_heatwave:    0.7681
task_3_crisis:      0.6484
```

Compared with v5:

```text
v5 average:      0.7282 -> v5.1 average:      0.7354  (+0.0072)
v5 task_1:       0.7923 -> v5.1 task_1:       0.7896  (-0.0027)
v5 task_2:       0.7553 -> v5.1 task_2:       0.7681  (+0.0128)
v5 task_3:       0.6370 -> v5.1 task_3:       0.6484  (+0.0114)
v5 valid rate:   0.9907 -> v5.1 valid rate:   0.9969  (+0.0062)
```

Ceiling capture against the relaxed LP oracle:

```text
task_1: 0.7896 / 0.8372 = 94.3%
task_2: 0.7681 / 0.8416 = 91.3%
task_3: 0.6484 / 0.7912 = 82.0%
avg:    0.7354 / 0.8233 = 89.3%
```

Decision:

- v5.1 passes the promotion gate;
- valid action rate is the best so far;
- heatwave and crisis improve over v5;
- task 1 has a tiny regression versus v5 but remains above v4 and above the
  v5.1 gate.

Current best checkpoint:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_v51_crisis_repair
```

## Hybrid Tool-Agent v1

The next improvement is not another checkpoint first. It is a runtime harness:
the model proposes or explains an action, while a causal optimizer, validator,
and short simulator decide whether that action is safe to execute.

Implemented tools:

```text
optimizer:  short-horizon causal LP using current observation + forecasts
validator:  GridOpsAction JSON/schema/bounds validation
sim guard:  compare model vs optimizer action on copied environment state
logger:     reset/plan/step JSONL episode capture for future SFT/RL data
exporter:   episode_logs/*.jsonl -> validated SFT trace JSONL
```

Important design decision:

- the full 72-hour LP oracle remains an offline ceiling because it sees the
  whole future;
- the deployed tool uses the causal LP only;
- in crisis, the LLM may override only when the rollout predicts lower
  blackout, not merely lower short-term cost.

Holdout smoke on seeds `7001,7002,7003`:

```text
optimizer-only average: 0.7882
hybrid-guard average:   0.7946
valid_action_rate:      1.0000

hybrid task_1_normal:   0.8182
hybrid task_2_heatwave: 0.8226
hybrid task_3_crisis:   0.7428
```

Compared with v5.1 SFT:

```text
v5.1 average: 0.7354 -> hybrid guard: 0.7946 (+0.0592)
v5.1 task_1:  0.7896 -> hybrid task_1: 0.8182 (+0.0286)
v5.1 task_2:  0.7681 -> hybrid task_2: 0.8226 (+0.0545)
v5.1 task_3:  0.6484 -> hybrid task_3: 0.7428 (+0.0944)
```

This confirms the main lesson: for GridOps, a small model plus executable
optimizer/validator tools is more reliable than trying to make the model
memorize every control rule through SFT alone.

## Tool-Corrected SFT Loop

The right v6 path is tool distillation before more GRPO. The loop is:

```text
model/candidate proposes -> optimizer proposes -> validator/simulator selects
-> save correction -> SFT on selected action -> evaluate model-only and hybrid
```

Implemented dataset tools:

```text
scripts/build_gridops_tool_corrected_sft.py
  Generates deterministic rollout traces where the guard chooses the label.

scripts/export_episode_logs_to_traces.py
  Converts real demo/API episode_logs/*.jsonl into SFT-ready rows.
```

Smoke command:

```bash
.venv/bin/python scripts/build_gridops_tool_corrected_sft.py \
  --tasks task_1_normal,task_2_heatwave,task_3_crisis \
  --seeds 7301,7302,7303 \
  --stride 1 \
  --candidate-policy price_greedy \
  --output sft_traces/gridops_tool_corrected_sft_v1.jsonl \
  --summary-output evals/gridops_tool_corrected_sft_v1_summary.json
```

Convert real usage logs:

```bash
.venv/bin/python scripts/export_episode_logs_to_traces.py \
  --log-dir episode_logs \
  --prompt-mode reason_action \
  --output sft_traces/gridops_episode_logs_reason_action.jsonl
```

Train v6 as a continuation of v5.1:

```bash
GRIDOPS_TRACE_PATH=sft_traces/gridops_tool_corrected_sft_v1.jsonl \
GRIDOPS_INIT_ADAPTER=77ethers/gridops-models/sft_qwen25_3b_gridops_v51_crisis_repair \
GRIDOPS_RUN_LABEL=sft_qwen25_3b_gridops_v6_tool_corrected \
GRIDOPS_SFT_STEPS=100 \
GRIDOPS_LEARNING_RATE=3e-5 \
python scripts/hf_sft_gridops.py
```

Launch the same flow on Hugging Face Jobs:

```bash
.venv/bin/python scripts/launch_hf_job_v6_tool_corrected_sft.py --run-eval
```

Default HF settings:

```text
image:        pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
flavor:       l4x1
timeout:      10h
branch:       codex/gridops-sft-pipeline
init_adapter: 77ethers/gridops-models/sft_qwen25_3b_gridops_v51_crisis_repair
run_label:    sft_qwen25_3b_gridops_v6_tool_corrected
trace rows:   3 tasks x 6 seeds x 72 hours = 1296 rows
```

Why this matters:

- SFT now teaches the model the action that survived validation and rollout,
  not just the action a generator guessed;
- correction rows explicitly capture where the model/candidate was rejected;
- once the model imitates the tool reliably, DPO/GRPO can become a polishing
  stage rather than a rescue attempt.

## v6 Result And v6.1 Pivot

The first v6 tool-corrected run was useful, but not promotable. It trained on
tool-selected labels, yet the completion target included too much diagnostic
tool language. The model learned to emit verbose internal traces instead of
stable `<think>...</think><action>{...}</action>` completions, so holdout
validity collapsed.

Decision:

- keep `sft_qwen25_3b_gridops_v51_crisis_repair` as the promoted model-only
  baseline;
- keep the hybrid tool-agent as the best deployable system;
- mark `sft_qwen25_3b_gridops_v6_tool_corrected` as failed/not promoted;
- rebuild v6.1 as clean LP-critic distillation.

v6.1 architecture:

```text
OpenEnv state -> weak candidate action -> causal LP critic on copied state
-> chosen action -> clean operator reasoning -> final bounded JSON action
```

The full 72-hour LP remains an offline ceiling. The training critic uses the
causal LP controller only: current observation, short forecasts, task rules,
SOC/fuel/rebound, and previous feedback. Critic details stay in `raw`; the SFT
completion reads like an operator decision, not a tool transcript.

Implemented files:

```text
gridops/critics/lp_critic.py
scripts/build_gridops_lp_critic_distilled_sft.py
scripts/launch_hf_job_v61_lp_critic_distilled_sft.py
tests/test_lp_critic_distilled_sft.py
```

Smoke dataset:

```bash
.venv/bin/python scripts/build_gridops_lp_critic_distilled_sft.py \
  --tasks task_1_normal,task_2_heatwave,task_3_crisis \
  --seeds 7401,7402 \
  --stride 6 \
  --output /tmp/gridops_lp_critic_distilled_smoke.jsonl \
  --summary /tmp/gridops_lp_critic_distilled_smoke_summary.json

.venv/bin/python scripts/validate_traces.py \
  /tmp/gridops_lp_critic_distilled_smoke.jsonl \
  --strict-clean-reasoning
```

HF dry run:

```bash
.venv/bin/python scripts/launch_hf_job_v61_lp_critic_distilled_sft.py --dry-run
```

HF launch, after reviewing the dry run:

```bash
.venv/bin/python scripts/launch_hf_job_v61_lp_critic_distilled_sft.py --run-eval
```

Default v6.1 settings:

```text
base_model: Qwen/Qwen3-4B-Instruct-2507
run_label:  sft_qwen3_4b_gridops_lp_critic_distilled_v1
trace_path: sft_traces/gridops_lp_critic_distilled_sft_v1.jsonl
max_rows:   3600
steps:      160
```

Promotion gate:

- valid action rate >= 0.99;
- average score > v5.1 model-only `0.7354`;
- task 3 crisis > v5.1 `0.6484`, target >= `0.68`;
- no task regresses below v5.1 by more than `0.01`.

## v7 Strategy Harness

The v6.1 Qwen3 experiment clarified a larger architecture lesson: forcing the
model to emit exact dispatch floats is brittle. GridOps is a control problem,
so the model should select operating intent and the deterministic controller
should turn that intent into bounded action.

v7 changes the model target from action JSON to strategy JSON:

```json
{
  "mode": "cost_saving | peak_shaving | outage_prepare | reliability | recovery | fuel_conservation",
  "risk_level": "low | medium | high | critical",
  "battery_bias": "charge | preserve | discharge | neutral",
  "diesel_policy": "avoid | allow_if_blackout | prewarm | conserve",
  "shedding_policy": "never | last_resort"
}
```

Runtime flow:

```text
OpenEnv observation
-> deterministic or model-selected GridOpsStrategy
-> strategy_to_optimizer_config(...)
-> causal LP/MPC optimizer
-> final GridOpsAction
```

The OpenEnv action and observation contracts remain unchanged. `/api/reset`,
`/api/step`, `/api/state`, `/ws`, and the dashboard still operate on
`GridOpsAction`. `/api/plan` now optionally accepts a `strategy` and returns
the strategy candidate, optimizer config, selected action, and diagnostics.

Implemented files:

```text
gridops/strategy.py
scripts/build_gridops_strategy_dataset.py
scripts/evaluate_gridops_strategy_controller.py
tests/test_strategy_harness.py
```

Validation and smoke commands:

```bash
.venv/bin/python scripts/build_gridops_strategy_dataset.py \
  --tasks task_1_normal,task_2_heatwave,task_3_crisis \
  --seeds 7601 \
  --stride 12 \
  --output /tmp/gridops_strategy_v7_smoke.jsonl \
  --summary /tmp/gridops_strategy_v7_smoke_summary.json

.venv/bin/python scripts/validate_traces.py \
  /tmp/gridops_strategy_v7_smoke.jsonl \
  --fail-fast

.venv/bin/python scripts/evaluate_gridops_strategy_controller.py \
  --mode strategy \
  --tasks task_1_normal,task_2_heatwave,task_3_crisis \
  --seeds 7001,7002,7003 \
  --output evals/gridops_v7_strategy_controller_holdout_7001_7003.json
```

Holdout result on seeds `7001,7002,7003`:

```text
strategy-controller average: 0.7907
valid_action_rate:           1.0000
LP ceiling capture:          96.04%

task_1_normal:               0.7995
task_2_heatwave:             0.8224
task_3_crisis:               0.7503
```

Compared with v5.1 model-only:

```text
v5.1 average: 0.7354 -> v7 strategy: 0.7907 (+0.0553)
v5.1 task_1:  0.7896 -> v7 task_1:   0.7995 (+0.0099)
v5.1 task_2:  0.7681 -> v7 task_2:   0.8224 (+0.0543)
v5.1 task_3:  0.6484 -> v7 task_3:   0.7503 (+0.1019)
```

Compared with the previous hybrid guard:

```text
hybrid average: 0.7946 -> v7 strategy: 0.7907 (-0.0039)
hybrid task_1:  0.8182 -> v7 task_1:   0.7995 (-0.0187)
hybrid task_2:  0.8226 -> v7 task_2:   0.8224 (-0.0002)
hybrid task_3:  0.7428 -> v7 task_3:   0.7503 (+0.0075)
```

Extended check on seeds `7201-7210`:

```text
average_score:      0.7921
valid_action_rate:  1.0000
LP ceiling capture: 96.21%

task_1_normal:      0.8049
task_2_heatwave:    0.8257
task_3_crisis:      0.7457
```

Decision:

- v7 passes the harness milestone with no paid model training;
- strategy JSON is a much easier future SFT target than raw dispatch floats;
- deterministic strategy-controller beats v5.1 model-only and improves crisis
  versus the previous hybrid guard;
- the next training step should be a strategy-selector model, not another
  action-float model.

## v7.1 Strategy-Selector SFT Prep

The next checkpoint should learn strategy selection, not direct dispatch. The
model output stays small and categorical, while the controller keeps numerical
optimization, safety bounds, and final `GridOpsAction` generation.

Implemented files:

```text
scripts/evaluate_gridops_strategy_adapter.py
scripts/launch_hf_job_v7_strategy_sft.py
```

Default training target:

```text
base_model:  Qwen/Qwen2.5-1.5B-Instruct
run_label:   sft_qwen25_15b_gridops_strategy_v7
trace_path:  sft_traces/gridops_strategy_v7_sft.jsonl
rows:        up to 3600 deterministic strategy rows
steps:       120
output:      strict strategy JSON only
```

Why `1.5B` is acceptable here:

- the model no longer has to calculate exact dispatch floats;
- it only chooses among six modes and four small enum fields;
- the causal LP/MPC controller remains responsible for numerical dispatch;
- smaller inference is easier to deploy and fine-tune repeatedly.

Smoke data command:

```bash
.venv/bin/python scripts/build_gridops_strategy_dataset.py \
  --tasks task_1_normal,task_2_heatwave,task_3_crisis \
  --seeds 7601,7602 \
  --stride 6 \
  --max-rows 80 \
  --output /tmp/gridops_strategy_v7_sft_smoke.jsonl \
  --summary /tmp/gridops_strategy_v7_sft_smoke_summary.json

.venv/bin/python scripts/validate_traces.py \
  /tmp/gridops_strategy_v7_sft_smoke.jsonl \
  --fail-fast
```

HF dry run:

```bash
.venv/bin/python scripts/launch_hf_job_v7_strategy_sft.py \
  --dry-run \
  --run-eval
```

Launch when ready:

```bash
.venv/bin/python scripts/launch_hf_job_v7_strategy_sft.py \
  --run-eval
```

Promotion gate:

```text
valid_strategy_rate >= 0.995
average_score >= deterministic v7 strategy-controller average 0.7907 - 0.005
task_3_crisis >= 0.74
no task below v5.1 model-only baseline
```

If the strategy model fails, keep deterministic v7 strategy-controller as the
deployable system and repair the strategy dataset before considering DPO/GRPO.

## v7.1 HF Training Run

HF Job:

```text
job_id:    6a075ed33308d79117b907fd
url:       https://huggingface.co/jobs/77ethers/6a075ed33308d79117b907fd
model:     77ethers/gridops-models/sft_qwen25_15b_gridops_strategy_v7
base:      Qwen/Qwen2.5-1.5B-Instruct
dataset:   3600 strategy rows
steps:     120
max_len:   1024
lr:        8e-5
LoRA:      r=16, alpha=32
```

Training completed and uploaded the adapter successfully:

```text
uploaded_to: 77ethers/gridops-models/sft_qwen25_15b_gridops_strategy_v7
adapter_model.safetensors: 37.0 MB
```

Final holdout on seeds `7001,7002,7003`:

```text
task_1_normal seed 7001: 0.7936, valid_strategy_rate 1.0000
task_1_normal seed 7002: 0.8100, valid_strategy_rate 1.0000
task_1_normal seed 7003: 0.7947, valid_strategy_rate 1.0000

task_2_heatwave seed 7001: 0.8208, valid_strategy_rate 1.0000
task_2_heatwave seed 7002: 0.8257, valid_strategy_rate 1.0000
task_2_heatwave seed 7003: 0.8206, valid_strategy_rate 1.0000

task_3_crisis seed 7001: 0.7374, valid_strategy_rate 1.0000
task_3_crisis seed 7002: 0.7354, valid_strategy_rate 1.0000
task_3_crisis seed 7003: 0.7535, valid_strategy_rate 1.0000
```

Aggregate:

```text
average_score:       0.7880
valid_strategy_rate: 1.0000
LP ceiling capture:  95.71%

task_1_normal:       0.7994
task_2_heatwave:     0.8224
task_3_crisis:       0.7421
```

Uploaded eval artifacts:

```text
sft_qwen25_15b_gridops_strategy_v7/evals/holdout/sft_qwen25_15b_gridops_strategy_v7_holdout_strategy.json
sft_qwen25_15b_gridops_strategy_v7/evals/holdout/sft_qwen25_15b_gridops_strategy_v7_holdout_strategy.valid_samples.jsonl
sft_qwen25_15b_gridops_strategy_v7/evals/holdout/sft_qwen25_15b_gridops_strategy_v7_holdout_strategy.invalid_examples.jsonl
```

Interpretation:

- the strategy interface is solved: no invalid strategy JSON in observed logs;
- normal and heatwave match the deterministic v7 strategy-controller almost
  exactly;
- crisis is valid but slightly softer than deterministic v7, likely because
  the tiny model learned the schema and common strategy choices but not all
  crisis-specific switching points;
- even the softer crisis result is far above v5.1 model-only crisis
  `0.6484`, confirming the harness shift is doing real work.

Promotion decision:

- promote v7.1 as the best model-facing strategy selector;
- keep deterministic v7 strategy-controller as the safest deployed fallback;
- do not return to direct action-float SFT as the main path;
- next improvement should be a small crisis-weighted strategy repair or a
  DPO-style preference set over strategy choices, not a bigger action model.

## Controller And Evaluator Readiness For DPO

Before DPO/GRPO, the scoring loop itself must be trusted. The DPO target will
be strategy preferences, so the evaluator must do three things consistently:

```text
model reply -> strict GridOpsStrategy parse
strategy -> causal LP/MPC controller -> GridOpsAction
GridOpsAction -> copied or live OpenEnv rollout -> score/regret
```

Checks now covered locally:

- strict strategy JSON parsing succeeds for valid replies;
- invalid strategy replies fall back cleanly instead of crashing;
- controller-generated actions remain bounded by `GridOpsAction`;
- strategy planning does not mutate the environment when used as a planner;
- evaluator summaries include v5.1, deterministic v7, and LP-ceiling baselines;
- valid-strategy rate is tracked separately from environment score.

This gives us a stable base for DPO pair generation:

```text
prompt:   observation + derived control context + previous outcome
chosen:   strategy JSON with better copied-rollout score
rejected: valid but worse strategy JSON
reward:   OpenEnv outcome after controller execution, not textual preference
```

Next implementation target:

```text
scripts/build_gridops_strategy_dpo_pairs.py
```

The builder should sample strategy candidates per state, score each through the
same controller/evaluator path, keep high-margin pairs, and write standard
DPO rows without ever changing the OpenEnv action/observation contract.

## v7.2 DPO Plan

DPO is the next safer learning step. In GridOps terms:

```text
DPO = senior operator preference review
For the same observation, show two strategy cards.
Chosen is the card whose controller rollout performed better.
Rejected is a valid but worse card.
The model learns which operating intent to prefer.
```

This is different from GRPO:

```text
GRPO = online bonus system
The model proposes several strategies, OpenEnv scores them, and reward pushes
the model toward higher-scoring samples.
```

Because v7.1 already has perfect strategy-format validity, DPO lets us teach
better crisis timing without exposing the model to noisy online reward updates
first.

Implemented DPO files:

```text
scripts/build_gridops_strategy_dpo_pairs.py
scripts/hf_dpo_gridops_strategy.py
scripts/launch_hf_job_v7_strategy_dpo.py
tests/test_strategy_dpo_pairs.py
```

DPO pair builder:

```text
OpenEnv state
-> generate multiple valid GridOpsStrategy candidates
-> execute each through the causal strategy controller on a copied env
-> roll forward short horizon
-> choose best strategy JSON
-> reject worst useful strategy JSON
```

Smoke command:

```bash
.venv/bin/python scripts/build_gridops_strategy_dpo_pairs.py \
  --tasks task_1_normal,task_2_heatwave,task_3_crisis \
  --seeds 7701 \
  --stride 12 \
  --horizon 4 \
  --optimizer-horizon 6 \
  --min-margin 0.0 \
  --output /tmp/gridops_strategy_dpo_pairs_smoke.jsonl \
  --summary /tmp/gridops_strategy_dpo_pairs_smoke_summary.json
```

Smoke result:

```text
rows: 18
validation_failures: []
task_1_normal: 6
task_2_heatwave: 6
task_3_crisis: 6
```

HF DPO dry run:

```bash
.venv/bin/python scripts/launch_hf_job_v7_strategy_dpo.py \
  --dry-run \
  --run-eval
```

Default DPO run:

```text
init_adapter: 77ethers/gridops-models/sft_qwen25_15b_gridops_strategy_v7
run_label:    dpo_qwen25_15b_gridops_strategy_v72
base_model:   Qwen/Qwen2.5-1.5B-Instruct
pairs:        up to 2400
steps:        80
beta:         0.1
lr:           5e-6
```

Promotion gate:

```text
valid_strategy_rate >= 0.995
average_score > v7.1 0.7880
task_3_crisis > v7.1 0.7421
task_2_heatwave >= 0.8220
no task below v5.1 model-only baseline
```

If DPO improves crisis without hurting validity, continue with a larger
crisis-weighted pair set. If DPO regresses format or heatwave, stop and keep
v7.1 plus deterministic v7 fallback.
