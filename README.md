# Reasoning-Under-Constraints OpenEnv — Round 2 Submission

**Meta PyTorch × Scaler OpenEnv Hackathon · April 25–26, 2026 · Bangalore**

An OpenEnv environment that trains LLMs to reason about competing constraints — return vs carbon vs risk — under ambiguous macro news and path-dependent decisions. We flatten a 12-quarter portfolio-manager MDP into a single-turn prompt-completion task, then apply GRPO (via TRL + Unsloth) on Qwen3-4B-Instruct to teach the model to connect news text → causal reasoning → portfolio action.

**Team:** Ekansh (RL/env engineer) + brother (trading platform / UI)
**Themes:** #3.1 World Modeling · #2 Long-Horizon · #5 Wild Card

---

## What's in this repo

| Path | What it is |
|---|---|
| [portfolio_env/](portfolio_env/) | The OpenEnv package — Pydantic models, shocks, rewards, path-dependent env |
| [tests/](tests/) | Smoke test + adversarial reward stress-test |
| [notebooks/](notebooks/) | GRPO training (WIP — tonight/Apr 24) |
| [sft_traces/](sft_traces/) | Expert `<think>` traces for SFT warm-start (generated Apr 24) |
| [ui/](ui/) | Greenberg Terminal React UI (brother's deliverable) |
| [portfolio_env_design.md](portfolio_env_design.md) | Full design spec (v0.7) |
| [HACKATHON_PLAN.md](HACKATHON_PLAN.md) | Execution checklist + risk register |
| [gemini_deep_research_output.md](gemini_deep_research_output.md) | Google-grounded research pass, surfaced MDP issue |
| [round_1/](round_1/) | Round 1 GridOps submission (archived) |

---

## The stack (locked after empirical validation April 23)

| Layer | Choice | Reason |
|---|---|---|
| Base model | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | Hackathon §59.1 recommends advanced Qwen3 recipe; Instruct chosen over Thinking (Thinking variant overshoots token budget) |
| Training | `trl.GRPOTrainer` with `loss_type="dapo"` default | TRL v1.0, DAPO is the default token-level loss |
| Efficiency | Unsloth 4-bit QLoRA | 3.6 GB VRAM, 80 tok/s batched on RTX 5090 |
| Architecture | **Flatten** 12-quarter MDP to single-turn | Hackathon §59.6 explicitly confirms multi-turn GRPO not mature |
| Warm-start | SFT on 150+ expert traces | Baseline Qwen3-4B emits 0% valid format cold — SFT required, not optional |
| Compute | RunPod RTX 5090 32GB (pre-hackathon), HF credits onsite | ~$22 for full rehearsal; de-risks pipeline |
| Expected budget | ~31 hr training + 17 hr buffer | Measured, not estimated |

---

## What we've discovered so far

### From Gemini's grounded deep research (April 23)
- **TRL GRPOTrainer is a contextual bandit, not a multi-step MDP trainer** — our 12-quarter env must be flattened. Hackathon docs confirm this is the accepted path.
- **T4/5090 throughput is ~40–80 tok/s, not 300** — original budget math was 4× too optimistic, but still fits.
- **Qwen3 Thinking variant generates unbounded reasoning** — Instruct variant is the right pick with explicit `<think>` prompting + SFT enforcement.
- **Gemma 4 tokenizer was a wrong claim** — we tested and Qwen3 Instruct chat template works fine.

### From adversarial reward stress-test (April 23)
Before any training, we ran 8 adversarial policies to stress-test our composite reward. Caught **4 real exploits** that GRPO would have found:

| Exploit | Root cause | Fix |
|---|---|---|
| `all_oil` beat baseline by +0.58 | CARBON_CAP=120 too lax (all-OIL emits ~30 kg) | Tightened to 25 — now overshoot triggers quadratic penalty |
| `infra_max` beat baseline by +0.47 | Unlock formula added principal + return, but principal was never subtracted → double-count bug | Add only the return at unlock |
| `put_hedge_farmer` exploit (1% TECH + max hedge) | Trigger was single-asset-drop < -15% | Trigger on portfolio-NAV-drop < -15% |
| Infra had no downside risk | No counter-penalty for physical-risk shocks during lockup | -8% per physical-risk shock (matches transition-risk gain magnitude) |

After fixes: **all 6 genuine exploits below baseline**. Only single-asset concentration policies (`all_tech`) marginally beat baseline — that's a *benchmark*, not a bug. The trained agent must surpass it.

---

## Quick start — running the env locally

```bash
git clone <this repo>
cd gridops
pip install -e .

# Smoke test (deterministic, runs 3 phases)
python -m tests.test_env_smoke

# Adversarial reward stress-test (no policy should beat baseline)
python tests/test_adversarial.py
```

Expected output on adversarial test:
```
✅ No adversarial policy beats equal-weighted. Reward stack is robust.
```

---

## Working on the RunPod RTX 5090 (our pre-hackathon compute)

```bash
# SSH in (key must be added to your RunPod account)
ssh root@74.2.96.43 -p 10168 -i ~/.ssh/id_ed25519

# Activate venv (persisted on /workspace network volume)
source /workspace/venv/bin/activate

# Pull latest code from local:
#   rsync -avz -e "ssh -i ~/.ssh/id_ed25519 -p 10168" \
#     --exclude='.git' --exclude='round_1' --exclude='__pycache__' \
#     /Users/ekansh/gridops/ root@74.2.96.43:/workspace/gridops/

# On the pod:
cd /workspace/gridops
pip install -e . --quiet
python tests/test_adversarial.py
```

The venv persists across pod restarts as long as the Network Volume stays attached.

---

## The demo narrative (locked April 23)

**Silent 2-minute video** (brother's domain):

1. **0:00–0:20** — *"LLMs pattern-match when signals are clear. They fail when objectives conflict and shocks are ambiguous. We trained past that."*
2. **0:20–0:45** — **Baseline Qwen3-4B-Instruct** plays the 12-quarter episode. Q3 hurricane → dumps OIL (wrong, refinery supply shock makes it go up). Q6 rare-earth → buys GREEN (wrong, supply chain collapse). Q7 stagflation → piles into BONDS (real return −2.5%/yr). Final NAV: −12%.
3. **0:45–1:15** — **GRPO-trained Qwen3-4B-Instruct**, same seed. `<think>` streams on-screen. Q3: keeps OIL, cites supply chain. Q6: sees rare-earth → GREEN collapse *before* buying. Q7 stagflation: rotates into OIL + REAL_ESTATE. Final NAV: +18%.
4. **1:15–1:40** — The inflation moment: real returns > nominal matters. Regret is vs real equal-weighted baseline.
5. **1:40–2:00** — Training curves. 5 independent reward components rising. *"48 hours. Single GPU. Open-source env."*

---

## Current status

See [HACKATHON_PLAN.md](HACKATHON_PLAN.md) for live status dashboard, risk register, and per-phase checklists.

**Next on the critical path:**
1. Write SFT warm-start pipeline (generate 150+ expert traces via frontier LLM API)
2. Fork Unsloth Advanced Qwen3 4B GRPO notebook as our training driver
3. Add hold-out test seeds + per-component reward logging
4. v0.7 patch to design doc reflecting empirical findings
5. Brother: fill 11 placeholder shocks + Greenberg Terminal UI scaffold

---

## Acknowledgments

- Unsloth team — advanced Qwen3 4B GRPO recipe (§59.1)
- Hugging Face TRL v1.0 — stable GRPO with DAPO default
- DeepSeek-R1 — the CoT+GRPO recipe we build on
- DAPO paper (arXiv 2503.14476) — overlong reward shaping
- Gemini 3.1 Pro with Google grounding — caught the MDP-bandit mismatch before we burned compute on it
