# Hackathon Execution Plan & Checklist

**Project:** Reasoning-Under-Constraints OpenEnv (Climate-Stressed Portfolio Manager)
**Event:** Meta PyTorch × Scaler OpenEnv Hackathon, Bangalore, April 25–26, 2026
**Team:** Ekansh (RL/env) + brother (trading/UI)
**Design doc:** [portfolio_env_design.md](portfolio_env_design.md) (currently v0.6, patching to v0.7)

---

## Status dashboard (update every sync)

| Field | Value |
|---|---|
| Today | 2026-04-23 (Thursday late evening) |
| Phase | Tonight — pre-training validation complete; SFT pipeline next |
| Current blocker | None |
| Critical path next | Write SFT warm-start trace generation pipeline |
| Training hours used | 0 / 48 |
| Model locked | **Qwen3-4B-Instruct-2507** (Thinking variant overshoots token budget — tested and rejected) |
| Compute target | **RunPod RTX 5090 32GB (active)**; HF credits onsite Apr 25 |
| Measured stack throughput | 80 tok/s batched on 5090 Blackwell (Unsloth 2026.4.7 + TRL 0.24 + torch 2.10) |

---

## 1. Open decisions — resolve before build

| # | Decision | Status | Notes |
|---|---|---|---|
| 1 | Multi-step MDP: flatten or custom loop? | **DECIDED: flatten** | Hackathon §59.6 confirms multi-turn GRPO not mature |
| 2 | Base model | **DECIDED: Qwen3 4B-Instruct** | Advanced Unsloth recipe §59.1, native `<think>`, reward-shaping features |
| 3 | RLVE procedural shocks vs static pool | **DECIDED: static for hackathon** | Mention Option B (procedural) as v2 in README |
| 4 | SFT warm-start trace count | **DECIDED: 150+** | 50 steps on 15 overfits (Gemini finding) |
| 5 | UI stack (brother's call) | **PENDING** | React + recharts likely; brother confirms Apr 24 |
| 6 | Demo narration style | **DECIDED: silent + captions** | Brother's call, v0.6 sign-off |
| 7 | Hackathon compute tier (T4 vs A100?) | **UNKNOWN — ask organizers Apr 25 morning** | Affects model choice optionality |
| 8 | Qwen3 `<think>` token format verification | **PENDING** | Task on tonight's list — load tokenizer, inspect chat template |

---

## 2. Risk register

| Risk | Severity | Status | Mitigation |
|---|---|---|---|
| Multi-turn MDP incompatibility with GRPOTrainer | was CRITICAL | **Resolved** | Flatten to single-turn prompt |
| Gemma-4 `<think>` tag mismatch | was HIGH | **Avoided** | Switched to Qwen3 (native `<think>`) |
| T4 throughput 40 tok/s (not 300) | HIGH | **Partial mitigation** | Qwen3 4B + reduced iters; pending actual measurement Apr 24 |
| `put_hedge` single-asset-drop exploit | HIGH | **Fix planned tonight** | Trigger on portfolio NAV drop, not asset drop |
| 15 SFT traces causes overfitting | MEDIUM | **Plan scaled to 150+** | Apr 24 pipeline builds them |
| Conflicting reward components (regret vs sharpe vs carbon) | MEDIUM | **Monitoring plan** | Per-component logging in training loop |
| Unsloth KV-cache bug in Gemma 4 | MEDIUM | **Avoided (moved off Gemma)** | N/A after Qwen3 pivot |
| `infra_commit` dominance under `green_leaps` | LOW-MED | **Fix planned v0.7** | Counter-penalty on physical-risk shocks |
| Notebook/version drift breaking GRPOTrainer | MEDIUM | **Pin versions** | Test small run on Apr 24 |
| Reward hacking on our composite stack | HIGH | **Fix tonight** | Adversarial stress-test policies before training |

---

## 3. Done log (oldest → newest)

- [x] Round 1 GridOps env built, scored top in Round 1 (52K devs → top 2K)
- [x] Round 2 strategy brainstorm with Gemini (agent arena / hierarchical / multi-agent)
- [x] Format strategy notebooks drafted (`format1/2/3_*.ipynb` — now in `round_1/` as prep material)
- [x] Climate Portfolio env design v0.1 → v0.6 iterations complete
- [x] Gemini v0.1 review → v0.2 reasoning-under-constraints reframing
- [x] v0.3 → curriculum + inference budget + fallback plan
- [x] v0.4 → 12-quarter episode (brother's full-cycle call)
- [x] v0.5 → inflation regimes + real returns + `tech_bet='inflationary'`
- [x] v0.6 → `fragmentation` tech_bet + silent demo + brother's realism sign-off
- [x] Repo reorg: moved Round 1 into `round_1/`, created `round-2` branch
- [x] Three Round 2 commits landed on branch
- [x] `portfolio_env/` scaffold built: constants, inflation, models, shocks (6 concrete + 11 placeholders), rewards (5), env with path-dep state
- [x] `pyproject.toml` + smoke test passing end-to-end
- [x] Deep research pass with Gemini + Google grounding → surfaced CRITICAL MDP + HIGH tokenizer + HIGH throughput + HIGH put-hedge issues
- [x] Hackathon FAQ + §59 Unsloth recipe guide digested → locked model (Qwen3 4B) + flatten approach + adversarial testing mandate
- [x] Provisioned RunPod RTX 5090 32GB pod (Blackwell) with persistent Network Volume
- [x] Full stack installed + verified on pod: Unsloth 2026.4.7, TRL 0.24, transformers 5.5, torch 2.10+cu128, bitsandbytes 0.49
- [x] Smoke test passing on pod (deterministic, identical to local)
- [x] Qwen3-4B-Thinking-2507 tested → rejected (unbounded reasoning overshoots token budget)
- [x] Qwen3-4B-Instruct-2507 locked — confirmed native chat template + explicit `<think>` prompting works
- [x] Measured real throughput: 80 tok/s batched long-context, 217 tok/s short-context on 5090
- [x] **Adversarial reward stress-test** run: caught 4 real exploits (all_oil, infra double-count, put_hedge farmer, infra no-downside) — all fixed
- [x] v0.7 reward patches committed (commit `8d63d4e`) — all exploits below baseline

---

## 4. Phase 0 — Tonight (April 23, 8 PM — midnight)

**Goal: stabilize the design, break our rewards ourselves, lock the model stack.**

### Must-do (blocking further work)

- [x] ~~**Adversarial reward stress-test**~~ — **DONE**. Caught 4 exploits, all fixed. See commit `8d63d4e`.
- [x] ~~**Fix `put_hedge` exploit**~~ — **DONE**. Now triggers on portfolio NAV drop.
- [x] ~~**Verify Qwen3 `<think>` tag format**~~ — **DONE**. Thinking variant overshoots; Instruct variant with explicit prompting is the path.
- [ ] **Add hold-out test seeds** — reserve `[100, 200, 300, 400, 500]`; `env.reset(seed)` must reject these during training. — *~15 min*
- [ ] **Write SFT warm-start pipeline** — generate 150+ expert `<think>` + JSON traces via Gemini/Claude API, filter by reward threshold. Tonight's critical path. — *~1.5 hrs*
- [ ] **v0.7 patch to design doc** — incorporate all empirical findings from pod validation + adversarial test. — *~30 min*
- [ ] **Commit adversarial test findings to Git** — summarize reward bugs + fixes in commit message for submission history. — DONE in commit `8d63d4e`.

### Should-do (shapes Apr 24 work)

- [ ] **Review Unsloth Advanced Qwen3 4B GRPO notebook** — clone, skim, identify what we port and what we replace — *Ekansh, ~30 min*
- [ ] **Review OpenEnv tutorial examples** — `meta-pytorch/OpenEnv/tree/main/tutorial/examples`, find closest-reference env — *Ekansh, ~20 min*
- [ ] **Share v0.7 design doc with brother** — triggers his Apr 24 work on shocks + UI

### Nice-to-have

- [ ] Background-play the "Mega Lecture" YouTube (Jew4lhAiqnw) while coding — *Ekansh*

---

## 5. Phase 1 — April 24 (Friday) — prep & dry runs

**Goal: ship working SFT + GRPO pipeline with one tiny run before onsite.**

### Ekansh (RL/env/training)

- [ ] **Fork Advanced Qwen3 4B GRPO notebook** → `notebooks/grpo_training.ipynb`; swap model name, imports, reward functions — *2 hrs*
- [ ] **Format-reward regex** — production-quality parser for `<think>...</think>` + JSON block; unit-tested — *30 min*
- [ ] **Implement overlong-think reward shaping** (DAPO technique #4) as 6th reward function — *30 min*
- [ ] **Flatten prompt builder** — generates the single-turn prompt that lists all 12 quarters of macro news + expects a 12-quarter action plan in JSON — *1 hr*
- [ ] **SFT trace generation pipeline** — script calls Gemini 1.5 Pro via grounding to produce 150+ `<think>` traces; filter by reward threshold — *1.5 hrs*
- [ ] **Per-component reward logging** — trainer callback emits all 6 rewards per iter to console + CSV — *30 min*
- [ ] **10-iter Phase-1 smoke run on T4** — confirms training loop, measures actual tok/s, catches bugs — *~1 hr (plus training time)*
- [ ] **Deploy skeleton env to HF Spaces** — minimal OpenEnv server wrapper; confirms Docker + Space build works — *1.5 hrs*

### Brother (trading/UI/market realism)

- [ ] **Fill 11 placeholder shocks** in `portfolio_env/shocks.py` (2 easy, 4 ambiguous, 2 hard, 3 across any tier where he has strong instincts) — *2 hrs*
- [ ] **Pressure-test v0.7 design §5.1 inflation magnitudes** — answer Q10/Q11 from design doc — *30 min*
- [ ] **Write 5 expert `<think>` traces** for hard-tier shocks — anchors for the automated SFT pipeline — *1.5 hrs*
- [ ] **Greenberg Terminal UI scaffold** (3 panels, dummy JSON state, dark theme) — *4 hrs*
- [ ] **Lock UI stack choice** (React+recharts / Next.js / Streamlit) and report — *5 min decision + start*

### Together (Apr 24 evening sync)

- [ ] Verify 10-iter smoke run shows rising format reward at minimum
- [ ] Lock JSON schema contract between env backend and UI frontend
- [ ] Rehearse 2-minute demo narrative cold once
- [ ] Checkpoint: all Phase 1 deliverables green → sleep. Any red → debug before bed.

---

## 6. Phase 2 — April 25 (Saturday, Day 1 onsite)

**Goal: execute curriculum Phases 1 and 2 of training. Start Phase 3 by end of day.**

### Morning (arrival → noon)

- [ ] Confirm HF Spaces credits + compute tier — pivot model if A100 available
- [ ] Connect Colab / HF Spaces runner
- [ ] Run SFT warm-start (50 steps on 150 traces) — *~30 min*
- [ ] Kick off Phase 1 training (easy shocks only, 4Q episodes, 50 iters) — *~2 hrs*
- [ ] Sample 10 rollouts manually → inspect for reward hacking
- [ ] **Checkpoint 1 (noon):** Phase 1 regret > 0 on 50% of rollouts? → proceed / debug / Tier 3 fallback

### Afternoon (noon → 6 PM)

- [ ] Kick off Phase 2 training (ambiguous shocks, 8Q episodes, 100 iters) — *~10–12 hrs*
- [ ] Iterate on reward weights based on per-component curves
- [ ] Brother: wire UI to live backend; stream first episode through the Greenberg Terminal
- [ ] Sample 10 rollouts again → inspect
- [ ] **Checkpoint 2 (evening):** Phase 2 regret median > 0.05? → Phase 3 / Tier 2 fallback / Tier 3 fallback

### Evening (6 PM → midnight)

- [ ] Start Phase 3 training (all shocks, 12Q episodes, 80 iters)
- [ ] First half of Phase 3 should run overnight (~6 hrs)
- [ ] Brother: demo video B-roll, record baseline-model episodes

---

## 7. Phase 3 — April 26 (Sunday, Day 2 onsite)

**Goal: finalize training, record demo, submit.**

### Morning

- [ ] **Checkpoint 3 (9 AM):** is Phase 3 reward climbing? → continue / invoke Tier 2 fallback (demote hard shocks, reframe as capability probe)
- [ ] Complete Phase 3 training runs (~4 more hrs)
- [ ] Pick best checkpoint by hold-out regret score (not training reward)
- [ ] Freeze the winner. Merge LoRA. Push to HF Hub.

### Afternoon (noon → 4 PM)

- [ ] Record final "before vs after" episode on identical seed — baseline Qwen3 vs trained Qwen3
- [ ] Render 5 reward-curve panels → export to PNG
- [ ] Record 2-minute demo video (silent, captions synced)
- [ ] Write mini-blog (<2 min read) for HF
- [ ] Push env Docker image to HF Space
- [ ] Verify HF Space responds to `/reset` and `/step` from external client

### Evening (4 PM → submission)

- [ ] Submit per hackathon portal
- [ ] Tweet / post soft launch link if pre-submission permits
- [ ] Pitch rehearsal (2 min cold, no slides)
- [ ] In-person presentation to judges

---

## 8. Minimum-requirement submission checklist (hackathon guide)

- [ ] Uses OpenEnv (latest release) — *verify version pin*
- [ ] Minimal training script in Colab using HF TRL + Unsloth — `notebooks/grpo_training.ipynb`
- [ ] Mini-blog on HF OR video on YouTube, <2 minutes
- [ ] OpenEnv-compliant environment hosted on HF Spaces
- [ ] README with problem, env, agent behavior, reward curves, before/after

---

## 9. Artifacts inventory (current)

| Path | Purpose | Status |
|---|---|---|
| `portfolio_env_design.md` | Design spec | v0.6 current, v0.7 patch tonight |
| `portfolio_env/` | OpenEnv package | Scaffold complete, 11 placeholder shocks remain |
| `tests/test_env_smoke.py` | Smoke test | Passing; adversarial tests to add tonight |
| `pyproject.toml` | Package config | Complete |
| `notebooks/grpo_training.ipynb` | Training driver | **Not yet written** — Apr 24 priority |
| `sft_traces/*.jsonl` | SFT warm-start data | **Not yet generated** — Apr 24 priority |
| `ui/` | Greenberg Terminal | **Not yet built** — brother's Apr 24 task |
| `gemini_round2.py`, `gemini_deep_research.py` | Gemini advisors | Done; deep research output saved to `gemini_deep_research_output.md` |
| `round_1/` | Round 1 artifacts (archived) | Done — preserved with git history |

---

## 10. References (most important)

### Hackathon artifacts
- Official guide (FAQ 1–58, Unsloth recipes 59)
- Official themes (we target Theme #2 Long-Horizon + #3.1 World Modeling + #5 Wild Card)

### Design predecessors
- [portfolio_env_design.md](portfolio_env_design.md)
- [gemini_deep_research_output.md](gemini_deep_research_output.md) — the grounded research that caught the MDP issue

### Upstream docs
- TRL GRPOTrainer: https://huggingface.co/docs/trl/main/en/grpo_trainer
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- Advanced Qwen3 4B GRPO recipe: check Unsloth notebooks repo (https://github.com/unslothai/notebooks)
- OpenEnv tutorials: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial

### Papers we're drawing from
- DeepSeekMath / GRPO: arXiv 2402.03300
- DAPO: arXiv 2503.14476
- DeepSeek-R1 (CoT + GRPO at scale): arXiv 2501.12948
- RLVE adaptive environments: see FAQ #22
- Reward hacking / specification gaming: DeepMind + Lilian Weng

---

## 11. Sync protocol

**During hackathon:** update "Status dashboard" (§0) every 2 hours. Move tasks between In Progress → Done in real time. Any blocker → immediately update §2 risk register with severity.

**Before each checkpoint (§6, §7):** re-read decision tree in design doc §18.4 and make the go/no-go call explicitly. No rabbit-holing.

**End of day:** `git commit -am "checkpoint: <day> — <summary>"` so we have a rollback trail.

---

**This file is the source of truth for what's next. Edit freely — don't let it rot.**
