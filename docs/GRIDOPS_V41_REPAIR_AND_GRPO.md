# GridOps v4.1 Repair And GRPO Plan

## Why v4.1

v4 is the first strong reasoning-action model:

- average holdout score: `0.7076`
- valid action rate: `0.9738`
- task 3 crisis score: `0.6255`
- task 3 diesel use: `123.05 kWh`

It fixed the v3 shortcut where the model emitted valid JSON but avoided diesel
entirely. The remaining weakness is not policy intelligence. It is action-bound
reliability in heatwave and crisis states.

The overnight eval showed:

- `max_new_tokens=160` is too short and causes missing `<action>` blocks.
- `max_new_tokens=220` and `320` produce identical full-holdout metrics.
- Remaining invalids are Pydantic `ValidationError`s, so longer decoding is not
  the fix.

## v4.1 SFT Repair

Build a small continuation dataset:

```bash
python scripts/build_gridops_v41_repair_traces.py
```

Outputs:

```text
sft_traces/gridops_curriculum_v41_bound_repair_mix.jsonl
sft_traces/gridops_curriculum_v41_bound_repair_only.jsonl
evals/gridops_curriculum_v41_bound_repair_summary.json
```

The mix includes the v4 Kimi dataset plus focused repair rows for:

- heatwave evening-ramp bound repair;
- crisis diesel bound repair;
- previous-blackout correction;
- charge/discharge bound anchors;
- concise format anchors.

Continuation SFT on Kaggle:

```bash
bash scripts/kaggle_sft_v41_repair.sh
```

Default artifact:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_kimi_reason_action_v41_repair
```

Default training:

```text
init_adapter: sft_qwen25_3b_gridops_kimi_reason_action_v4
steps: 125
learning_rate: 8e-5
max_length: 1536
```

Promotion gate:

```text
average_score >= 0.70
valid_action_rate >= 0.99
task_3_crisis score >= 0.62
task_3_crisis diesel_kwh > 0
task_1_normal diesel_kwh remains near 0
```

## GRPO Position

Yes, GRPO is worth trying after v4.1, but it should not be the immediate next
step before repair.

Reason:

- RL rewards can improve score, regret, blackout, diesel timing, and cost.
- But if the model still produces invalid actions, GRPO can spend capacity
  relearning format instead of optimizing the environment.
- Worse, an invalid/default parser fallback can accidentally reward bad
  completions if the reward wrapper is not strict.

So the sequence is:

1. v4.1 SFT repair for validity.
2. v4.1 holdout eval.
3. Tiny GRPO smoke only if v4.1 clears the parser/action gate.

## GRPO Reward Shape

Each completion should be parsed into `GridOpsAction` and scored in the
environment. Reward components:

```text
format_reward:
  +1.0 valid <think>/<action> and Pydantic action
  -2.0 missing action block, bad JSON, or out-of-bound action

step_reward:
  one-step score delta from environment feedback

oracle_regret_reward:
  action should be close to or better than oracle one-step outcome

blackout_penalty:
  strong penalty for blackout kWh

diesel_policy_reward:
  penalty for unnecessary diesel in normal/heatwave
  bonus for bounded diesel in crisis when blackout risk is high

brevity_reward:
  small penalty for overly long reasoning
```

GRPO smoke gate:

```text
5-10 steps only
2-4 generations per prompt
completion mean length > 50
valid action rate >= 95% in smoke
reward std nonzero
grad_norm nonzero/non-NaN
task_3 sampled completions include nonzero diesel
```

If any collapse appears, stop and keep v4.1 SFT as the shippable model.
