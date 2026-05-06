# GridOps v4 Reasoning-Action Dataset

v4 is a targeted correction after the v3 result:

```text
v3 learned valid JSON perfectly, but regressed into zero diesel usage.
```

The v4 dataset teaches a compact operator reasoning loop before the final
action. The action schema remains unchanged.

## Format

Assistant completion:

```text
<think>
time_context: ...
1st_order: ...
2nd_order: ...
previous_action: ...
decision: ...
</think>
<action>
{"battery_dispatch": ..., "diesel_dispatch": ..., "demand_shedding": ...}
</action>
```

The evaluator parses only the JSON inside `<action>`.

## Dataset Builder

```bash
python scripts/build_gridops_v4_reasoning_traces.py
python scripts/validate_traces.py sft_traces/gridops_curriculum_v4_reason_action.jsonl
```

Outputs:

```text
sft_traces/gridops_curriculum_v4_reason_action.jsonl
evals/gridops_curriculum_v4_reason_action_summary.json
```

Current generated dataset:

```text
rows: 4000
task_1_normal: 833
task_2_heatwave: 1629
task_3_crisis: 1538
```

Bucket balance:

```text
crisis_diesel_positive: 900
normal_no_diesel: 700
heatwave_rebound: 600
previous_action_correction: 500
low_resource_edges: 400
time_context_mix: 600
format_anchors: 300
```

Action balance:

```text
diesel_positive: 1029
diesel_zero: 2971
battery_charge: 1444
battery_discharge: 563
shedding_positive: 592
```

## How Rows Are Built

The factory creates traces through three routes:

- oracle rollouts across normal, heatwave, and crisis tasks;
- failure-bank corrections where a prior model action caused blackout or missed diesel;
- stressed crisis rollouts that deliberately recreate the v3 zero-diesel failure mode,
  then label the recovery action with the oracle.

Every row includes:

- current observation;
- derived time and forecast context;
- previous action;
- previous outcome;
- oracle action;
- short structured reasoning;
- final `<action>` JSON.

## Training

Kaggle runner:

```bash
bash scripts/kaggle_sft_v4_reasoning.sh
```

Default adapter target:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_reason_action_v4
```

Default settings:

```text
base: Qwen/Qwen2.5-3B-Instruct
steps: 250
max_length: 1536
batch_size: 1
grad_accum: 8
LoRA r: 16
```

## Evaluation

Use the reasoning prompt mode:

```bash
python scripts/evaluate_gridops_adapter.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path 77ethers/gridops-models/sft_qwen25_3b_gridops_reason_action_v4 \
  --prompt-mode reason_action \
  --max-new-tokens 220 \
  --seeds 7001,7002,7003 \
  --output evals/gridops_sft_reason_action_v4_holdout_7001_7003.json
```

Promotion gate:

```text
valid action rate >= 99.5%
average score > 0.6894
task_3_crisis score > 0.6201
task_3_crisis diesel_kwh > 0
task_3_crisis blackout_kwh materially below v2
task_1_normal diesel remains near 0
```
