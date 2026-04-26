# CarbonAlpha — progress snapshot

Round 2 submission for the Meta PyTorch × Scaler OpenEnv Hackathon
(Bangalore, Apr 25-26, 2026). One-page status doc — for the architectural
narrative see [MASTER_UNDERSTANDING.md](MASTER_UNDERSTANDING.md), for the
ticket-by-ticket plan see [HACKATHON_PLAN.md](HACKATHON_PLAN.md), for the
gnarly debugging story see [TRAINING_ERRORS.md](TRAINING_ERRORS.md).

Last updated: 2026-04-25 (Day 1 evening).

---

## What's shipped

### Model (the deliverable)
- **`77ethers/CarbonAlpha/v6_sft_only_v2`** — Qwen3-4B-Instruct + LoRA r=16, merged & saved as full bf16 weights (8.06 GB).
- Trained via SFT on `merged_v6_aligned.jsonl` (200 traces — Gemini-generated + your manual edits).
- 150 SFT steps on L40S, ~4 min runtime via HF Jobs (l40sx1).
- **Hold-out eval (5 unseen seeds):** 5/5 valid format, mean regret **+0.034** vs equal-weighted baseline, beats baseline on **3/5** seeds.

### Demo
- **`77ethers/CarbonAlpha-demo`** — Gradio Space on T4 small ($0.40/hr).
- Pulls the private model at boot, exposes a Q&A textbox: paste news, get `<think>` reasoning + JSON allocation.
- 4 example shocks pre-loaded (hurricane, stagflation, rare-earth, routine).

### Environment / compliance
- **`portfolio_env/`** — full OpenEnv `Environment` subclass: reset / step / state / get_metadata, plus FastAPI server via `openenv.core.create_app`. All standard routes registered (`/reset`, `/step`, `/state`, `/health`, `/metadata`, `/schema`, `/ws`).
- **17-shock pool** across 3 difficulty tiers (easy / ambiguous / hard). 3 placeholders remaining; 14 concrete.
- **5-component reward stack** (format, regret-vs-baseline, sharpe, carbon, drawdown) with phase-weighted carbon. Adversarial-tested across 8 policies — none beat equal-weighted baseline beyond noise threshold.
- **Path-dependent state**: transaction costs, spent-as-you-go carbon, 4Q infra-commit lockups, put-hedge premium bleed, regime-shift inflation accumulator.
- **Inflation regimes**: normal / stagflationary / deflationary, switchable mid-episode by hard-tier shocks.
- **Dockerfile + openenv.yaml** at repo root, ready for HF Space env deploy.

### Training pipeline
- [`notebooks/grpo_training.py`](notebooks/grpo_training.py) — single-script SFT warm-start + 3-phase GRPO curriculum, parameterised via CLI (`--phase`, `--sft-traces`, `--sft-steps`).
- [`scripts/hf_train.py`](scripts/hf_train.py) — UV-script launcher for HF Jobs with PEP 723 inline deps; pulls code from `77ethers/CarbonAlpha-train` dataset, runs training, uploads adapter to `77ethers/CarbonAlpha`.
- [`sft_traces/generate_traces.py`](sft_traces/generate_traces.py) — Gemini 3.1 Pro pipeline for generating SFT examples from sampled news shocks.

### Compliance assets
- [`scripts/plot_training.py`](scripts/plot_training.py) — emits committed PNG plots (loss curve + reward curve + holdout eval) from training logs.
- [`scripts/dump_episode.py`](scripts/dump_episode.py) — emits a 12-quarter episode JSON for the demo UI.
- [`scripts/deploy_to_hf.sh`](scripts/deploy_to_hf.sh) — one-command deploy helper.
- `tests/test_adversarial.py` — 8 adversarial policies tested; passes.

---

## SFT model lineage (eval comparison)

| Model | Traces | Recipe | Hold-out (5 seeds) | Beats baseline |
|---|---|---|---|---|
| v1 (broken) | 120, old prompt | 4-bit, 60 steps | 0/5 valid (format wrong) | 0/5 |
| v2 SFT (4-bit) | 120 v2 | 4-bit, alpha=16 | 5/5 valid, regret -0.09 | 0/5 |
| v2 SFT (16-bit) | 120 v2 | 16-bit, alpha=16, gc=False | 5/5 valid, regret +0.014 | 3/5 |
| **v6 SFT v2 (current best)** | **200 v6** | **16-bit, alpha=16, gc=False** | **5/5 valid, regret +0.034** | **3/5** |
| v2 + GRPO ("phase all") | 120 v2 | 16-bit, alpha=32, gc=unsloth | 5/5 valid, regret -0.25 | 0/5 ← GRPO degraded the model |
| v2 + GRPO (alpha reverted) | 120 v2 | 16-bit, alpha=16, gc=False | 5/5 valid, regret -0.16 | 2/5 ← still degraded |

**Read:** SFT works, more diverse traces (v6) > fewer (v2), the right LoRA recipe matters (alpha=16 not 32 on small datasets). GRPO is currently broken — see [TRAINING_ERRORS.md §4](TRAINING_ERRORS.md).

---

## Hackathon validation checklist

| Requirement | Status | Notes |
|---|---|---|
| Public HF Space | 🟡 deploying | `77ethers/CarbonAlpha-demo` (built once, hit Python 3.13/audioop bug, rebuild in flight) |
| Valid OpenEnv structure | ✅ | All routes registered, env subclass clean, schema introspectable |
| Training plot PNGs in repo | ✅ | `assets/loss_curve.png`, `assets/reward_curve.png` (placeholder mode confirmed; real data after run) |
| Runnable Colab notebook | ⏳ | Notebook adapts grpo_training.py for Colab; needs final test |
| README with inline plots + links | 🟡 | Skeleton in place; needs final v6 numbers + Space link |
| Reward stack adversarial-tested | ✅ | 8 policies, no exploit beats baseline |
| Episode JSON dumper for demo | ✅ | `dump_episode.py` works for both LLM and scripted policies |
| Demo UI (Greenberg Terminal) | ⏳ | Brother's task |

---

## What's NOT working (limitations to document in README)

1. **GRPO collapses to 1-token rollouts** under Unsloth + vLLM 0.15.1 stack. Three full GRPO runs all show `completions/mean_length: 1.0` throughout — model emits EOS as first token, no reward variance, no learning. Hypothesis: chat-template stop-token interaction. Documented in [TRAINING_ERRORS.md §4](TRAINING_ERRORS.md).
2. **Pivot path for GRPO not yet attempted:** drop Unsloth entirely, use vanilla `transformers + peft + trl`. Bigger rewrite (~30-45 min) but standard, well-tested path.
3. **3 placeholder shocks** still in `portfolio_env/shocks.py` (`easy_PLACEHOLDER_6`, `ambig_PLACEHOLDER_6`, `ambig_PLACEHOLDER_7`).

---

## Cost so far (HF Jobs only — RunPod was free credits)

- ~6 HF Jobs runs on l40sx1 ($1.80/hr × 0.05-0.20 hr each) ≈ **$2-3**
- HF Space on t4-small (~$0.40/hr) — auto-pauses when idle
- Total: under $5 burned

---

## Next 12 hours

1. **Get Space RUNNING** — fix any further Python/Gradio issues; share URL with brother
2. **Brother tests v6 model interactively** — feedback informs whether trace set needs more iteration
3. **Decision: attempt vanilla TRL GRPO** OR **ship SFT-only as final** — depends on time vs. confidence
4. **Finalise README** with inline plots, eval table, Space link, model link
5. **Brother:** Greenberg Terminal UI, 3 placeholder shocks
