# Reasoning-Under-Constraints OpenEnv

**Meta PyTorch × Scaler OpenEnv Hackathon · April 25–26, 2026 · Bangalore**

An OpenEnv environment that trains LLMs to reason about **competing constraints under ambiguous signals and path-dependent decisions**. We flatten a 12-quarter portfolio-manager MDP into a single-turn prompt-completion task, then apply GRPO (via TRL + Unsloth) on Qwen3-4B-Instruct to teach the model to connect news → causal reasoning → portfolio action.

**Team:** Ekansh + brother
**Themes:** #3.1 World Modeling · #2 Long-Horizon · #5 Wild Card

---

## Hackathon deliverables (compliance check)

| # | Required | Where |
|---|---|---|
| 1 | Public, cloneable HF Space | **TBD — `huggingface.co/spaces/<our-org>/portfolio-env` will be linked here at submission** |
| 2 | OpenEnv `Environment` base class + `openenv.yaml` | [portfolio_env/env.py](portfolio_env/env.py) (PortfolioEnv inherits from `openenv.core.env_server.interfaces.Environment`) · [openenv.yaml](openenv.yaml) |
| 3 | Loss curve + reward curve as committed PNGs | [assets/loss_curve.png](assets/loss_curve.png) · [assets/reward_curve.png](assets/reward_curve.png) |
| 4 | Runnable training script (Colab preferred) | [notebooks/grpo_training.ipynb](notebooks/grpo_training.ipynb) (Colab) · [notebooks/grpo_training.py](notebooks/grpo_training.py) (Python) |
| 5 | README with inline plots + every-deliverable links | this file |

### Loss curve

![SFT + GRPO loss curve](assets/loss_curve.png)

### Reward curve

![5-component composite reward over training](assets/reward_curve.png)

---

## What we built in one paragraph

A 12-quarter (3-year bull-bear cycle) portfolio environment where each quarter the LLM reads a macro news headline with conflicting 1st/2nd/3rd-order causal hooks, emits `<think>` reasoning + a JSON action containing 5 portfolio weights and 4 optional interventions (infra_commit lockup, carbon_offset_buy, put_hedge, tech_bet thesis). Path-dependent physics (transaction costs, locked capital, accumulated carbon, inflation regime) tie Q1 decisions to Q8 outcomes. Episode reward is a composite of 5 verifiable functions: format compliance, regret-vs-equal-weighted-baseline on inflation-adjusted real returns, Sharpe, non-linear carbon penalty above cap, and max drawdown. The agent is trained via SFT warm-start (120 Gemini-generated traces) → GRPO with DAPO loss in 3 curriculum phases. Adversarial pre-training stress-test repaired 4 reward exploits before any compute was spent. Hold-out seeds reserved for clean generalization measurement.

---

## Repo map

| Path | What it is |
|---|---|
| **[MASTER_UNDERSTANDING.md](MASTER_UNDERSTANDING.md)** | **Read this first.** Single canonical narrative — what we're building in OpenEnv terms + every design decision with its rationale |
| [portfolio_env/](portfolio_env/) | The OpenEnv package |
| └── [env.py](portfolio_env/env.py) | `PortfolioEnv(Environment)` — reset/step/state/get_metadata |
| └── [models.py](portfolio_env/models.py) | `PortfolioAction(Action)`, `PortfolioObs(Observation)`, `PortfolioState(State)` |
| └── [shocks.py](portfolio_env/shocks.py) | 17-shock pool with 3-tier difficulty taxonomy |
| └── [rewards.py](portfolio_env/rewards.py) | 5 composite reward functions for GRPO |
| └── [inflation.py](portfolio_env/inflation.py) | Regime dynamics + real-return math |
| └── [sampling.py](portfolio_env/sampling.py) | Hold-out seed isolation |
| └── [server/app.py](portfolio_env/server/app.py) | FastAPI app via `openenv.core create_app` |
| [openenv.yaml](openenv.yaml) | HF Space deployment spec |
| [Dockerfile](Dockerfile) | Container build for HF Spaces |
| [tests/test_adversarial.py](tests/test_adversarial.py) | Pre-training reward stress-test (8 adversarial policies) |
| [tests/test_env_smoke.py](tests/test_env_smoke.py) | End-to-end sanity check across 3 phases |
| [tests/test_holdout.py](tests/test_holdout.py) | Verifies training sampler never leaks holdout seeds |
| [notebooks/grpo_training.ipynb](notebooks/grpo_training.ipynb) | Colab-ready training notebook (the deliverable) |
| [notebooks/grpo_training.py](notebooks/grpo_training.py) | Same as above as a runnable Python script |
| [scripts/dump_episode.py](scripts/dump_episode.py) | Episode → JSON state for the Greenberg Terminal UI |
| [scripts/plot_training.py](scripts/plot_training.py) | Reads training logs → emits committed PNG plots |
| [sft_traces/traces.jsonl](sft_traces/traces.jsonl) | 120 expert `<think>` traces for SFT warm-start |
| [sft_traces/generate_traces.py](sft_traces/generate_traces.py) | Gemini 3.1 Pro pipeline that produced the traces |
| [ui/](ui/) | Greenberg Terminal (brother's React deliverable) |
| [portfolio_env_design.md](portfolio_env_design.md) | Full design spec (v0.7) |
| [HACKATHON_PLAN.md](HACKATHON_PLAN.md) | Live status + risk register + per-phase checklist |
| [BROTHER_BRIEF.md](BROTHER_BRIEF.md) | Self-contained brief for brother's parallel work |
| [gemini_deep_research_output.md](gemini_deep_research_output.md) | Google-grounded research transcript (caught the MDP-bandit mismatch) |
| [round_1/](round_1/) | Round 1 GridOps submission (archived for reference) |

---

## The stack (locked April 23, empirically validated)

| Layer | Choice | Reason |
|---|---|---|
| Base model | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | Hackathon §59.1 recommends Advanced Qwen3 4B GRPO recipe; Instruct chosen over Thinking after empirical test (Thinking variant generates 2000+ tokens, never closes `</think>`) |
| Training | `trl.GRPOTrainer` with `loss_type="dapo"` (TRL v1.0 default), `beta=0.0` | DAPO token-level loss is TRL's stable default; KL-free per Open-Reasoner-Zero |
| Efficiency | Unsloth 4-bit QLoRA, LoRA r=16 on 7 attn+MLP layers | 33M trainable / 4B base (0.81% trained); 3.6 GB VRAM at runtime |
| Architecture | Flatten 12-quarter MDP to single-turn prompt-completion | Hackathon §59.6 explicitly notes multi-turn GRPO not yet mature in Unsloth — flattening is the accepted state-of-art |
| Warm-start | SFT on 120 Gemini-generated chat-template-formatted traces, 150 steps | Empirically: cold Qwen3 emits 0% valid format; SFT pushes to 60% (3/5 holdout) — GRPO bootstraps from there |
| Compute | RunPod RTX 5090 32GB (Blackwell) for prep · HF Spaces credits onsite | Measured throughput: 80 tok/s batched on long-context rollouts → ~31hr training budget fits 48hr window |

---

## How to run locally

```bash
git clone <this repo>
cd gridops
pip install -e .

# Smoke test
python -m tests.test_env_smoke

# Adversarial reward stress-test (must pass before any training)
python tests/test_adversarial.py

# Boot the OpenEnv FastAPI server locally
uvicorn portfolio_env.server.app:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs (interactive API)
# → http://localhost:8000/metadata (env description)
# → http://localhost:8000/ws (WebSocket for OpenEnv clients)
```

## How to train

**Colab (recommended):** open [notebooks/grpo_training.ipynb](notebooks/grpo_training.ipynb), Runtime → Change runtime type → T4 GPU, **Run all**.

**Local / pod:**
```bash
python notebooks/grpo_training.py --phase sft-only        # SFT warm-start (~5 min on T4)
python notebooks/grpo_training.py --phase 1               # SFT + Phase 1 GRPO (~2 hr on T4)
python notebooks/grpo_training.py --phase all             # full curriculum (~12 hr on T4)
```

After training, generate plots:
```bash
python scripts/plot_training.py --sft-log <log> --grpo-log <log>
```

---

## Discoveries that shaped the design (in order found)

### 1. Gemini grounded research caught the CRITICAL MDP-bandit mismatch
Before writing any training code, we ran a one-shot deep research call to Gemini 3.1 Pro with Google grounding ([gemini_deep_research.py](gemini_deep_research.py)). It surfaced that **TRL's `GRPOTrainer` is fundamentally a contextual bandit**, not a multi-step MDP trainer. Our 12-quarter MDP must be flattened to single-turn for GRPO to work. Hackathon docs §59.6 confirms multi-turn GRPO with stepwise rewards is not yet a mature first-class recipe in Unsloth. Without this finding we'd have burned hours debugging.

### 2. Adversarial reward stress-test caught 4 reward bugs before training
Per FAQ #57 ("don't optimize a reward you haven't tried to break yourself first") we ran 8 adversarial policies before kicking off GRPO. Found:
- `all_oil` beat baseline +0.58 (CARBON_CAP=120 too lax) → fixed at 25
- `infra_max` beat baseline +0.47 (unlock formula double-counted principal) → fixed
- `put_hedge_farmer` exploit (1% TECH + max hedge) → fixed trigger to portfolio NAV
- `infra` had zero downside → added -8% per physical-risk shock during lockup

After fixes, no degenerate policy beats the equal-weighted baseline. Concentration policies (`all_tech`, +0.08) marginally beat baseline because TECH has highest base return — this is the **target** for the trained agent, not a bug.

### 3. Empirical model selection on the Blackwell pod
Tested Qwen3-4B-Thinking-2507 vs Qwen3-4B-Instruct-2507 on RTX 5090. Thinking variant generated 2000+ tokens of reasoning before ever closing `</think>` — token budget overshoots, JSON never emitted. Instruct variant responds to explicit `<think>...</think>` prompting and is bounded. Locked Instruct.

### 4. SFT format mismatch caused 0/5 holdout valid on first try
Initial SFT on plain `prompt + '\n' + completion` text → 0/5 valid completions on holdout. Root cause: training-eval format mismatch — eval used `tokenizer.apply_chat_template([{role: user, ...}])` which produces `<|im_start|>user ... <|im_end|><|im_start|>assistant`, but training never saw that structure. Fixed by pre-applying chat template to text field. SFT v3: 3.94 → 1.46 loss, **3/5 holdout valid with mean regret +0.020**.

---

## Demo arc (silent + captions, 2 min)

1. **0:00–0:20** *"LLMs pattern-match when signals are clear. They fail when objectives conflict and shocks are ambiguous. We trained past that."*
2. **0:20–0:45** Untrained Qwen3-4B-Instruct on a 12-quarter episode. Q3 hurricane → dumps OIL (wrong). Q6 rare-earth → buys GREEN (wrong). Q7 stagflation → piles into BONDS (real return -2.5%/yr). Final NAV: -12%.
3. **0:45–1:15** GRPO-trained model on identical seed. `<think>` streams. Q3 keeps OIL citing supply chain. Q6 sees rare-earth → GREEN supply collapse before buying. Q7 stagflation rotates into OIL + REAL_ESTATE. Final NAV: +18%.
4. **1:15–1:40** *"Real returns matter. The trained model read 'PCE core 5.8%' and rotated. That's economic reasoning, not pattern matching."*
5. **1:40–2:00** All 5 reward curves rising over training. Carbon respected. Hold-out eval: trained beats baseline. *"48 hours. Single GPU. Open-source env."*

---

## Acknowledgments

- **Unsloth team** — Advanced Qwen3 4B GRPO recipe (§59.1)
- **Hugging Face TRL v1.0** — stable GRPO with DAPO default
- **DeepSeek-R1** — the CoT+GRPO recipe we build on
- **DAPO paper** (arXiv 2503.14476) — overlong reward shaping
- **Gemini 3.1 Pro** with Google grounding — caught the MDP-bandit mismatch before we burned compute on it
