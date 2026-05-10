# GridOps OpenEnv-Backed GRPO Plan

## Decision

Lock v4 as the best SFT baseline:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v4
```

v4 metrics:

```text
average_score:      0.7076
valid_action_rate:  0.9738
task_1_normal:      0.7891
task_2_heatwave:    0.7082
task_3_crisis:      0.6255
crisis diesel_kwh:  123.05
```

The v4.1 repair attempt is useful evidence, but it should not replace v4 unless
the full holdout unexpectedly reverses the smoke result. Smoke showed better
crisis diesel use but worse normal and heatwave behavior.

## Why Stop Trace Micromanagement

SFT is behavior cloning. It teaches the model to imitate traces:

```text
observation -> expert-looking action
```

That is useful for format and basic operation, but it makes us keep hand-writing
cases such as "heatwave bound repair" or "crisis diesel repair." We end up
debugging labels instead of letting the environment teach the strategy.

OpenEnv lets us move to actual reinforcement:

```text
model completion -> parse action -> OpenEnv step/rollout -> reward -> update
```

This is the right way to make the model learn:

- when to charge before a shortage;
- when to discharge into an evening peak;
- when diesel is worth its cost;
- when diesel hurts the green/cost score;
- when demand shedding creates rebound pain;
- how to preserve SOC before outage hours.

## Core Principle

Use SFT as the driver's license. Use OpenEnv-backed GRPO as the driving school.

SFT should teach:

- valid `<think>` and `<action>`;
- valid JSON;
- bounded action values;
- microgrid vocabulary.

GRPO should teach:

- environment score;
- regret vs baseline;
- blackout prevention;
- cost and diesel tradeoffs;
- short-horizon SOC preservation.

## Training Architecture

For each training prompt:

1. Select a deterministic GridOps state from train seeds.
2. Render the same v4 reasoning-action prompt used in eval.
3. Generate `num_generations` completions from the model.
4. Parse each completion into `GridOpsAction`.
5. Score each action with OpenEnv.
6. Return a scalar reward per completion.
7. GRPO compares completions for the same prompt and updates toward the better
   actions.

## State Sampler

Do not use holdout seeds `7001,7002,7003`.

Use train seeds such as:

```text
task_1_normal:   15000-15080
task_2_heatwave: 16000-16080
task_3_crisis:   17000-17080
```

Oversample states that create strategic learning:

- evening ramp;
- pre-evening charge window;
- outage-near;
- active outage;
- low SOC;
- high price;
- heatwave/rebound;
- previous blackout.

## Reward Function

The reward must be strict about invalid actions. Invalid completions should not
fall back to a neutral action and accidentally receive a usable environment
score.

Suggested reward:

```text
reward =
  format_reward
  + env_step_reward
  + short_horizon_reward
  + regret_reward
  + baseline_advantage_reward
  + heatwave_rebound_reward
  + soc_preservation_reward
  + diesel_context_reward
  + brevity_reward
```

Components:

```text
format_reward:
  +1.0 valid <think>/<action> and Pydantic action
  -25.0 missing action block, bad JSON, or out-of-bound action

env_step_reward:
  scaled OpenEnv one-step reward after executing the action

short_horizon_reward:
  rollout 3-4 more hours using oracle policy after the model's first action
  this teaches SOC/fuel consequences without doing a full 72h rollout

regret_reward:
  candidate_horizon_reward - oracle_first_action_horizon_reward
  small reward if near oracle; larger reward if better

baseline_advantage_reward:
  candidate_horizon_reward - do_nothing_horizon_reward
  prevents the policy from learning passive no-action behavior

blackout_penalty:
  task-weighted blackout penalty, heavier for heatwave and crisis

heatwave_rebound_reward:
  extra penalty for heatwave blackout and shedding/rebound
  bonus for no blackout and no shedding across the horizon

soc_preservation_reward:
  bonus/penalty for ending risky evening/outage-near windows with enough SOC

diesel_context_reward:
  normal/heatwave diesel when no high gap: penalty
  crisis/outage with high gap and bounded diesel: bonus

brevity_reward:
  small penalty if completion is too long
```

## Why Short Horizon First

A full 72-hour rollout inside every reward call is too slow.

Start with:

```text
horizon = 1 for smoke
horizon = 4 for first real run
horizon = 72 only for eval
```

This gives the model immediate feedback while still letting it learn the
important second-order behavior: preserve battery/fuel when the next few hours
are dangerous.

## GRPO Smoke Gate

Start tiny:

```text
max_steps: 5-10
num_generations: 2-4
prompt_count: 24-48
adapter init: v4
```

Continue only if:

```text
completion mean length > 50
valid action rate >= 90% during smoke
reward std nonzero
grad_norm nonzero/non-NaN
no 1-token collapse
task_3 sampled completions include bounded nonzero diesel
task_1 sampled completions do not spam diesel
```

Then scale:

```text
phase_1: 20-30 steps, horizon 4, lr 1e-6
phase_2: 50-100 steps, horizon 4, only if phase 1 improves heatwave
phase_3: eval only, full 72-hour holdout
```

## Promotion Gate

GRPO model must beat v4 without losing action reliability:

```text
average_score > 0.7076
valid_action_rate >= 0.97
task_1_normal >= 0.78
task_2_heatwave >= 0.71
task_3_crisis >= 0.63
crisis diesel_kwh > 0
normal diesel_kwh near 0
```

If GRPO improves crisis but damages normal/heatwave like v4.1 did, do not
promote it.

## Artifact Naming

Never overwrite v4.

Suggested names:

```text
77ethers/gridops-models/grpo_qwen25_3b_gridops_openenv_v4_smoke
77ethers/gridops-models/grpo_qwen25_3b_gridops_openenv_v4_phase1
77ethers/gridops-models/grpo_qwen25_3b_gridops_openenv_v4_phase2
```

## Implementation Files

Planned:

```text
scripts/hf_grpo_gridops_openenv.py
scripts/kaggle_grpo_gridops_openenv.sh
evals/gridops_grpo_openenv_smoke_summary.json
```

The GRPO script should reuse:

- `gridops.prompting` for prompt/action parsing;
- `GridOpsEnvironment` for environment execution;
- `oracle_policy` for short-horizon continuation and regret baseline;
- `scripts/evaluate_gridops_adapter.py` for final holdout eval.

## Kaggle Commands

Reward-contract smoke only:

```bash
bash scripts/kaggle_grpo_gridops_openenv_smoke.sh
```

Actual tiny GRPO smoke:

```bash
GRIDOPS_RUN_TRAIN=1 \
GRIDOPS_GRPO_STEPS=20 \
GRIDOPS_GRPO_TRAIN_HORIZON=4 \
GRIDOPS_GRPO_PROMPT_LIMIT=24 \
GRIDOPS_GRPO_NUM_GENERATIONS=2 \
GRIDOPS_GRPO_LR=1e-6 \
bash scripts/kaggle_grpo_gridops_openenv_smoke.sh
```

The smoke uploads to:

```text
77ethers/gridops-models/grpo_qwen25_3b_gridops_openenv_v4_smoke
```

After training, evaluate before promotion:

```bash
GRIDOPS_ADAPTER_PATH=77ethers/gridops-models/grpo_qwen25_3b_gridops_openenv_v4_smoke \
GRIDOPS_RUN_LABEL=grpo_qwen25_3b_gridops_openenv_v4_smoke \
python scripts/kaggle_overnight_eval_v4.py --skip-long-decode
```
