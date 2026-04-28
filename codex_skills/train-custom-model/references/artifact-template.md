# Artifact Template

## Artifact Ledger

| Role | Path / Repo | Status | Notes |
|---|---|---|---|
| Safe baseline | `<repo>/<subfolder>` | preserved | Never overwrite |
| SFT warm-start | `<repo>/<sft_subfolder>` | candidate | Dataset, steps, LoRA config |
| RL experiment | `<repo>/<rl_subfolder>` | candidate/final | Reward setup, steps, smoke result |
| Dataset | `<dataset_repo>/<file>` | versioned | Count and curriculum split |
| Logs | `<repo>/training_logs/...` | evidence | Raw job logs |
| Plots | `<repo>/assets/...` | evidence | Loss/reward/length plots |
| Notebook | `<repo>/notebooks/...ipynb` | rerunnable | Default read-only |
| Demo | `<space_url>` | public | Runnable UI/API |

## Model Card Sections

- Summary
- Intended use
- Model lineage
- Dataset and curriculum
- Training procedure
- Reward functions
- Evaluation
- Training evidence
- Known weaknesses
- Reproducibility
- Limitations and safety
- Links

## README Submission Checklist

- Problem motivation
- How the environment/harness works
- How to run locally
- Public demo link
- Colab/notebook link
- Model repo link
- Dataset link or dataset description
- Loss/reward plots
- Results table
- Blog/video link
- No large embedded videos

## Standard Smoke Gate

```text
parse_rate >= target
mean_completion_length > task_minimum
min/max length not collapsed
grad_norm finite and nonzero
at least one reward component has nonzero std
holdout metric does not regress badly vs SFT
manual examples reveal no catastrophic format drift
```

## Naming Pattern

```text
<task>_<base_model>_<stage>_<dataset_or_curriculum>_<steps_or_smoke>_v<version>
```

Examples:

```text
sft_qwen25_7b_curriculum400_v1
grpo_qwen25_7b_adapter_phase1_100_v1
grpo_qwen3_4b_base_smoke_v2
```
