# Master Understanding — what we're building, in OpenEnv terms

**Hackathon:** Meta PyTorch × Scaler OpenEnv Hackathon, Bangalore, Apr 25–26, 2026
**Submission:** Reasoning-Under-Constraints Environment (climate-aware portfolio manager flavor)
**Team:** Ekansh (RL/env engineer) + brother (trading platform / UI)

This document is the single canonical narrative. It explains what we're building, maps every component to OpenEnv vocabulary, and justifies each design choice with the evidence that produced it. Read this first if you're picking up the project cold.

---

## Part 0 — In one paragraph

We've built an OpenEnv `Environment` in which a Hugging Face TRL-trained LLM (Qwen3-4B-Instruct, GRPO-trained via Unsloth) acts as a portfolio manager over a 12-quarter macroeconomic cycle. Each quarter the env serves the agent a Pydantic `Observation` containing news + state; the agent emits a Pydantic `Action` containing portfolio weights and four optional interventions; the env runs path-dependent physics and returns the next `Observation` plus reward. Training is a 3-phase curriculum: SFT warm-start on 120 Gemini-generated traces, then GRPO Phase 1 → Phase 2 → Phase 3 with progressively harder shock pools and richer reward functions. The capability we're training is **causal reasoning under competing constraints when signals are ambiguous**, framed as a portfolio task because it gives us verifiable rewards and a visceral demo.

---

## Part 1 — Why this exists

### The capability we're probing

Modern LLMs are excellent at single-pass tasks where the right answer is one inference away. They struggle when:
- Multiple objectives compete (return vs carbon vs risk vs drawdown)
- Signals within a single observation **conflict** (a news headline contains both bullish and bearish hooks)
- Decisions today **constrain** decisions tomorrow (locked capital, used carbon budget)
- Pattern matching gives the wrong answer because the right answer is 2nd or 3rd-order

This isn't a finance problem. It's a reasoning problem. The portfolio task gives us:
- **Verifiable rewards** (real returns, regret vs baseline, carbon footprint — all exact numerics)
- **Path dependency** (transaction costs, locked capital, accumulated emissions)
- **Ambiguity** (synthetic shocks designed so 1st-order reading loses money)
- **A visceral demo** (P&L charts dodging shocks land with judges in 30 seconds)

### Why not the obvious alternatives

| Alternative | Rejected because |
|---|---|
| Multi-agent market | TRL's `GRPOTrainer` is a contextual bandit; multi-LLM rollouts blow our 48-hr inference budget; non-stationarity destabilizes GRPO. Confirmed by Gemini 3.1 Pro grounded research and hackathon docs §59.6 ("multi-turn GRPO with stepwise rewards is not yet a mature first-class recipe in Unsloth"). |
| Real market data backtest | Pattern-matching trap. The model would learn historical correlations not causal reasoning. Verified empirically — Qwen3 4B base model produced no valid output on real-style hard shocks cold. |
| Math reasoning env (DeepSeek-R1 style) | Saturated benchmark. Doesn't differentiate at the 4B parameter scale where we operate. |
| Code generation env | Crowded — many teams will do this. Lower judging differentiation. |
| Browser/tool-use env | High infrastructure overhead (sandbox, stateful tools); 48 hours is too tight; reward verifiability harder. |

---

## Part 2 — OpenEnv vocabulary mapped onto our codebase

OpenEnv standardizes RL environments around four primitives: **Action**, **Observation**, **State**, and an **`Environment`** class exposing **`reset` / `step` / `state`**. Optionally an episode **grader** returns 0..1 final score.

| OpenEnv concept | Our implementation | File |
|---|---|---|
| `Action` (Pydantic `BaseModel`) | `PortfolioAction` — 5 portfolio weights + 4 intervention fields (infra_commit, carbon_offset_buy, put_hedge, tech_bet) | [portfolio_env/models.py](portfolio_env/models.py) |
| `Observation` (Pydantic `BaseModel`) | `PortfolioObs` — quarter state, current weights, NAV (real + nominal), inflation regime, baseline NAV, news string, last-quarter feedback | [portfolio_env/models.py](portfolio_env/models.py) |
| `State` (extended) | `PortfolioState` — episode-level metadata exposed at `/state`: current quarter, episode_id, full trajectory, grade (when done) | (to be added in OpenEnv wrapper) |
| `Environment` base class | `PortfolioEnv` (currently plain Python; gets OpenEnv-wrapped in compliance pass) | [portfolio_env/env.py](portfolio_env/env.py) |
| `reset(seed, **kwargs)` | Initializes 12-quarter episode plan: samples shocks from phase-appropriate pool, randomizes quarter assignment, resets path state | `PortfolioEnv.reset` |
| `step(action)` | Advances one quarter: applies action, resolves shock, computes returns, updates path state (locks/unlocks/carbon/inflation), returns new `Observation` with reward | `PortfolioEnv.step` |
| `reward` | Returned per-step on `Observation.reward`. Episode-end grader returns 0..1 score. | `portfolio_env/rewards.py` |
| Episode grader | Composite of 5 reward components: format / regret / sharpe / carbon / drawdown — each is a pure function callable post-episode | `portfolio_env/rewards.py` |
| `EnvironmentMetadata` | Name, description, version, README content for `/metadata` endpoint | (in OpenEnv wrapper) |
| `openenv.yaml` | HF Space deployment spec — image config + env vars + ports | (to be added) |
| FastAPI server | Exposes `/reset`, `/step`, `/state`, `/ws` for client interaction | (in `gridops/server/app.py` for round 1; to be created for round 2) |

**The OpenEnv contract is a small surface.** Everything else (path dependency, inflation regimes, intervention payoffs) is hidden in our `step()` body. Clients only see the standard interface.

---

## Part 3 — Action and Observation specs in detail

### `PortfolioAction`

```python
class PortfolioAction(BaseModel):
    weights: list[float] = Field(min_length=5, max_length=5)
        # [TECH, OIL, GREEN, REAL_ESTATE, BONDS] — auto-normalized to sum 1.0
    infra_commit: float = Field(default=0.0, ge=0.0, le=0.2)
        # 4-quarter irreversible lockup. Pays +8%/transition shock during
        # lockup, -8%/physical-risk shock. (v0.7 fix to make it a true bet.)
    carbon_offset_buy: float = Field(default=0.0, ge=0.0, le=0.1)
        # 1 unit NAV → 10 kg CO2 offset; reduces accumulated carbon footprint.
    put_hedge: float = Field(default=0.0, ge=0.0, le=0.05)
        # 2% NAV premium per quarter; caps portfolio downside at -5% if
        # PORTFOLIO RETURN < -15% (v0.7 fix from single-asset trigger).
    tech_bet: Literal['status_quo', 'green_leaps', 'carbon_priced',
                       'inflationary', 'fragmentation'] = 'status_quo'
        # Q1-only macro thesis. Tilts shock probability distribution.
```

### `PortfolioObs`

```python
class PortfolioObs(BaseModel):
    quarter: int                              # 0..11
    difficulty_tier: str                      # 'easy' | 'ambiguous' | 'hard'
    current_weights: list[float]
    infra_locked_fraction: float
    infra_unlock_quarters: int
    carbon_offsets_held: float
    active_put_hedge: bool
    tech_bet_chosen: str
    portfolio_nav_nominal: float
    portfolio_nav_real: float                  # what we score on
    baseline_nav_real: float                   # equal-weighted benchmark
    cumulative_real_return_pct: float
    current_inflation_rate: float              # 1%/q normal, 2.5% stagflation, -0.3% deflation
    current_regime: Literal['normal', 'stagflationary', 'deflationary']
    cumulative_inflation_multiplier: float
    carbon_footprint_accumulated: float
    carbon_budget_remaining: float
    news: str                                  # macro headline w/ 1st/2nd/3rd-order causal hooks
    last_quarter_returns_nominal: list[float]
    last_quarter_returns_real: list[float]
    last_quarter_regret: float                 # vs equal-weighted baseline
```

### Why these exact fields

The action space is small (3 continuous + 1 discrete + 4 intervention dims) on purpose: GRPO's group-relative advantage estimator is sensitive to the dimensionality of the search space. The observation is rich (~20 fields) because **the agent reasons over the full state** — it shouldn't have to mentally reconstruct what's locked or what its baseline is doing.

---

## Part 4 — Episode mechanics

### Structure

- **12 quarters** = 3 simulated years = one full bull-bear cycle (brother's call; trading desks see cycles run 2-3 years)
- **5 shocks per episode** sampled from a 17-shock pool (6 easy / 7 ambiguous / 4 hard)
- **Sample without replacement**, randomize quarter placement (no shock at Q0; reserved for `tech_bet` commitment)
- **Inflation regime** can shift mid-episode if a shock has `regime_shift` set (stagflation / deflation)

### Path dependency mechanics (the long-horizon teeth)

Three mechanics make Q2 decisions affect Q8 outcomes:

1. **Transaction costs.** `nav *= (1 - 0.005 * sum_of_weight_changes)` per quarter. Spastic rebalancing eats returns.
2. **Carbon spent-as-you-go.** `carbon_emissions_q = sum(weights[i] * carbon_intensity[i] * NAV)`. Heavy OIL position in Q1-Q4 burns the budget; if a shock at Q8 forces carbon-intensive choices, no headroom left.
3. **Infra commit lockup.** Capital committed at Q2 is unavailable for 4 quarters. Returns conditional on shocks during the lockup window (+8%/transition, -8%/physical). True bet — wrong thesis → dead capital.

### The shock-news design philosophy

| Tier | Share | What it tests |
|---|---|---|
| Easy (6) | 40% | "Don't do something stupid" — 1-2 assets move obvious direction |
| Ambiguous (7) | 40% | "Weigh trade-offs" — signals within headline conflict |
| Hard (4) | 20% | "Reason through chains" — 2nd/3rd-order effects DOMINATE; 1st-order reading loses money |

The **training signal comes from ambiguous + hard**. The hurricane scenario is the canonical example: a pattern-matching LLM dumps both REIT and OIL on "disaster news"; a reasoning LLM keeps OIL because the refinery supply cut > demand drop, and adds GREEN because reconstruction = new grid investment.

---

## Part 5 — Reward function — five components, layered defense

Per hackathon FAQ #44 ("layered verification") and FAQ #57 ("don't optimize a reward you haven't tried to break yourself first"), our reward is a composite of 5 independent functions. GRPO accepts a `list[reward_fn]`; each is called per completion.

| Reward | Signal | Weight | What exploit it blocks |
|---|---|---|---|
| `r_format` | Per-completion: +0.05 if `<think>...</think>` tags present, +0.10 if valid JSON action parses | 0.15 max | Garbage output |
| `r_regret` (primary) | `agent_real_return − baseline_real_return` (where baseline = equal-weighted, computed on **inflation-adjusted real** returns) | 1.0 | All-bonds policy (loses to inflation), keyword pattern-matching (wins easy, fails ambiguous) |
| `r_sharpe` | Mean / std of quarterly real returns | 0.3× | Catastrophically volatile policies that win in expectation |
| `r_carbon` (non-linear) | `−5 × max(0, cumulative_carbon − CARBON_CAP)² / 100`, scaled by phase weight (0 → 0.3 → 1.0) | 1.0 max in Phase 3 | OIL-maxing for returns; mattress-bonds exploit (already crushed by regret) |
| `r_drawdown` | `−2 × max_drawdown_pct` | 2.0 | "Beat benchmark overall but lost 35% mid-episode" policies |

### Why composite + non-linear + phase-weighted

**Composite** — single reward is gameable. Multiple independent checks reduce the optimization surface for hacking.

**Non-linear carbon** — only fires when overshoot happens. Lets the agent USE its carbon budget aggressively up to the cap; doesn't punish the act of holding OIL early in episode.

**Phase-weighted** — Phase 1 carbon weight = 0; Phase 2 = 0.3; Phase 3 = 1.0. Letting carbon penalty fire too early would collapse the agent to 100% bonds (zero carbon, zero learning). This is the curriculum-equivalent of "start simple, layer carefully" from OpenEnv reward design guide.

### Adversarial validation done before training

We ran 8 adversarial policies (`all_bonds`, `all_tech`, `all_oil`, `yo_yo`, `put_hedge_farmer`, `carbon_offset_abuse`, `infra_max`, `equal_weighted`) before kicking off training. Caught **4 real reward bugs**:

1. `all_oil` beat baseline by +0.58 because CARBON_CAP=120 was too lax → fixed at 25
2. `infra_max` beat baseline by +0.47 because the unlock formula double-counted principal → fixed to add only the return
3. `put_hedge_farmer` exploited single-asset trigger → fixed to portfolio-NAV trigger
4. `infra` had zero downside risk → added -8% per physical-risk shock (symmetric with +8%/transition)

After fixes, no degenerate policy beats the equal-weighted baseline. Concentration policies (`all_tech` +0.08) still marginally beat baseline because TECH has highest base return — but that's a *target* for the trained agent, not a bug. See [tests/test_adversarial.py](tests/test_adversarial.py).

---

## Part 6 — Why the model is Qwen3-4B-Instruct (not bigger / not Thinking)

Selection happened in three steps:

### Step 1 — Why we passed on bigger models (Llama 8B, Gemma 9B)

The hackathon's compute window (T4/L4/A100 via HF credits or own provision) caps our training budget. At 4B, GRPO training is ~30 hours; at 8B it's ~60 hours, blowing the 48-hr onsite window. Bigger models would have stronger reasoning priors, but we can't fit the training time. Per FAQ #45 ("RL is post-training, not capability replacement"), the right call is the **smallest model that meets the bar**.

### Step 2 — Why we passed on Gemma 4 E4B

Initially picked Gemma 4 E4B for tokenizer simplicity. Gemini's grounded research flagged a tokenizer concern. We empirically tested on the pod: Gemma's chat template works fine for our format, but **the hackathon explicitly recommends the Advanced Qwen3 4B GRPO recipe (§59.1)** with prefinetuning + proximity scoring + OpenR1 dataset support. Aligning with the recommended recipe de-risks judging and gives us pre-built reward-engineering primitives. Gemma was the wrong default.

### Step 3 — Why Instruct, not Thinking variant

Tested both empirically:
- **Qwen3-4B-Thinking-2507**: chat template auto-opens `<think>\n` and the model writes 2000+ tokens of reasoning before ever closing the tag. Token budget overshoots; our 400-token completion cap truncates mid-think.
- **Qwen3-4B-Instruct-2507**: responds to explicit `<think>...</think>` prompting; bounded output length; controllable.

For a flattened-MDP single-turn output of ≤400 tokens, **Instruct is the right tool**. The Thinking variant is great for math/code reasoning where you want unbounded chains; we want bounded, parseable output.

---

## Part 7 — Why the training is GRPO (with DAPO loss, KL-free)

### Why GRPO over PPO

PPO needs a separate value (critic) network. On a 4B model with QLoRA, doubling the trainable footprint is wasteful. GRPO drops the critic by computing **group-relative advantages**: sample N completions per prompt, score each, advantage = (reward − group_mean) / group_std. No value head. Smaller training footprint; fits Colab T4 with Unsloth 4-bit.

### Why DAPO loss (TRL v1.0 default)

TRL v1.0 GRPOTrainer's `loss_type="dapo"` is the default and enables DAPO's **token-level policy gradient loss** (eliminates length bias). Better than vanilla GRPO at:
- Long completions (`<think>` blocks vary 200-800 tokens)
- Structured outputs where some tokens matter more than others

Set `beta=0.0` (KL-free) per DAPO and Open-Reasoner-Zero — the KL divergence term has been empirically shown to be unnecessary for GRPO on reasoning tasks. We're using TRL's stable defaults; not a fork or experimental flag.

### Why flatten the MDP

Critical finding from Gemini's grounded research (and confirmed by hackathon §59.6): **TRL's GRPOTrainer is fundamentally a contextual bandit, not a multi-step MDP trainer.** It cannot inject env observations between completion tokens.

We flatten: agent receives one prompt (current quarter's news + state), emits one action. We hold that action constant for the full 12-quarter episode and score the resulting trajectory. Lost: per-quarter feedback adaptation. Preserved: causal reasoning, regime awareness, all the path-dependency consequences. Hackathon docs explicitly call out the multi-turn gap, so judges will not penalize this.

---

## Part 8 — Why the curriculum (3 phases, gated)

Per FAQ #14 ("curriculum learning is essential for difficult environments") and the explicit "Start Simple" mandate:

| Phase | Iters | Episode | Shocks | Rewards | Interventions | Carbon weight | Entry criterion |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 4 quarters | Easy only (6) | format + regret | none | 0.0 | start |
| 2 | 100 | 8 quarters | Easy + ambig (12) | + sharpe + drawdown | infra_commit only | 0.3 | Phase 1: regret > 0 on 50% rollouts |
| 3 | 80 | 12 quarters | All 17 + interventions | + carbon | all 4 | 1.0 | Phase 2: median regret > 0.05 |

### Why this specific ladder

- **Phase 1** is JSON-shape + baseline-beating on trivial shocks. If this fails, the env or trainer is broken — not the task. It's a smoke test wearing curriculum clothes.
- **Phase 2** introduces causal ambiguity AND drawdown penalty AND one intervention (infra_commit). Agent must reason; can no longer get away with pattern matching.
- **Phase 3** is the target task: full 12-quarter cycle, all shocks, all interventions, full carbon penalty. Scored on hold-out seeds.

Each phase has an **entry criterion** measured on training rewards. Failing it triggers Tier 2 fallback (cut hard shocks → reframe demo) or Tier 3 fallback (scope cut to 8 quarters + 2 interventions). See [portfolio_env_design.md](portfolio_env_design.md) §18.

---

## Part 9 — Why SFT warm-start is mandatory

Empirically demonstrated April 23: **Qwen3-4B-Instruct-2507 produces 0% valid format on cold inference** (5/5 holdout seeds emit `<tool_call>` from Qwen's training prior, never close, never emit JSON). GRPO with zero positive-format examples in the rollout group has no signal to amplify.

**SFT warm-start fixes this:** 150 steps on 120 chat-templated examples → loss 3.94 → 1.46, **3/5 holdout valid with mean regret +0.020**. From here GRPO can bootstrap.

### Why 120 traces, generated by Gemini 3.1 Pro

- 30 curated seed events (20 real historical 2014-2024 across physical / transition / geopolitical / monetary + 10 projections 2025-2030)
- Batched 5 per Gemini call for intra-batch thematic coherence
- Structured `response_schema` enforcing exact `PortfolioAction` shape
- Per-trace validation: news length, reasoning length, weight shape, asset mentions
- 4× repeat across the 30 seeds = 120 traces (with random shuffle)
- 120/120 valid, 0 duplicates, 107 unique weight vectors (89% diversity)

The pipeline is at [sft_traces/generate_traces.py](sft_traces/generate_traces.py). Traces are at [sft_traces/traces.jsonl](sft_traces/traces.jsonl).

---

## Part 10 — Why hold-out seeds + adversarial test

Per FAQ #44 ("keep a holdout evaluator separate from the training reward") and #52 ("monitor more than the headline reward"):

- **Hold-out seeds** = `(100, 200, 300, 400, 500)` — reserved at constants level, training sampler explicitly skips them. Eval-only. See [portfolio_env/sampling.py](portfolio_env/sampling.py).
- **Adversarial stress-test** = 8 hand-crafted degenerate policies, run before any training. None can beat the baseline. See [tests/test_adversarial.py](tests/test_adversarial.py).

These two together let us catch:
1. Reward hacks BEFORE training burns compute (adversarial)
2. Memorization vs generalization AFTER training (hold-out)

---

## Part 11 — End-to-end architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       OpenEnv compliance shell                       │
│  ┌────────────────────┐    ┌─────────────────┐    ┌────────────┐    │
│  │  PortfolioEnv      │ ──→│  FastAPI        │ ──→│  HF Space  │    │
│  │  (Environment)     │    │  /reset /step   │    │  (public)  │    │
│  │                    │    │  /state /ws     │    │            │    │
│  └────────────────────┘    └─────────────────┘    └────────────┘    │
│         ↑                                                            │
│         │ uses                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  portfolio_env package                                      │     │
│  │  ├── models.py         PortfolioAction, PortfolioObs       │     │
│  │  ├── shocks.py         17-shock pool, 3-tier difficulty    │     │
│  │  ├── inflation.py      regime dynamics, real returns       │     │
│  │  ├── rewards.py        5 composite reward functions        │     │
│  │  ├── env.py            reset/step + path-dependent state   │     │
│  │  └── sampling.py       hold-out seed isolation             │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Training pipeline                               │
│                                                                      │
│  Qwen3-4B-Instruct-2507 (Unsloth 4-bit QLoRA + LoRA r=16)           │
│         │                                                            │
│         ▼                                                            │
│  SFT warm-start (150 steps, 120 traces, chat-template format)       │
│         │  produces /workspace/checkpoints/sft/checkpoint-150/       │
│         ▼                                                            │
│  Phase 1 GRPO (50 iters, 4Q episodes, format+regret)                │
│         │  hold-out eval → entry criterion check                     │
│         ▼                                                            │
│  Phase 2 GRPO (100 iters, 8Q, +sharpe +drawdown +infra_commit)      │
│         │  hold-out eval → entry criterion check                     │
│         ▼                                                            │
│  Phase 3 GRPO (80 iters, 12Q, all 5 rewards, all 4 interventions)   │
│         │                                                            │
│         ▼                                                            │
│  Final LoRA merge → upload to HF Hub → load in HF Space             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Evaluation + demo pipeline                        │
│                                                                      │
│  scripts/dump_episode.py → JSON state file                          │
│         │                                                            │
│         ▼                                                            │
│  ui/ (Greenberg Terminal)                                           │
│  ├── News feed panel                                                 │
│  ├── LLM monologue (streaming <think>)                              │
│  └── P&L chart with shock markers + carbon bar                      │
│         │                                                            │
│         ▼                                                            │
│  2-min silent video (captions only) → submission                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 12 — OpenEnv compliance checklist (the "must pass" list)

The hackathon validation pass is automated and silently rejects non-compliant submissions. These are the explicit asks we MUST satisfy:

| Requirement | Status | Where |
|---|---|---|
| Public, cloneable HF Space at submitted URL | ⏳ Pending | TBD: `huggingface.co/spaces/<our-org>/portfolio-env` |
| Tested logged-out (not private) | ⏳ Pending | After deployment, browse in incognito |
| Valid OpenEnv structure: `Environment` / `MCPEnvironment` base, Gym-style reset/step/state | ⏳ Pending | Wrapper to be written: `portfolio_env/server/environment.py` |
| Parseable `openenv.yaml` | ⏳ Pending | Adapt from [round_1/openenv.yaml](round_1/openenv.yaml) |
| Loss curve as committed `.png` | ⏳ Pending | `assets/loss_curve.png` from training log |
| Reward curve as committed `.png` | ⏳ Pending | `assets/reward_curve.png` (5 components on one plot) |
| Runnable training script (Colab notebook preferred) | ⚠️ Partial | Have [notebooks/grpo_training.py](notebooks/grpo_training.py); needs `.ipynb` conversion |
| README links every deliverable + plots embedded inline | ⚠️ Partial | [README.md](README.md) needs HF Space link, Colab link, plots embedded |

---

## Part 13 — What we explicitly chose NOT to do (defensible cuts)

For 48-hour scope discipline, these were considered and dropped:

- **Real market data** — synthetic shocks with hand-crafted causal chains instead. Real data adds noise + introduces memorization risk that prevents reasoning.
- **Multi-agent / market microstructure** — non-stationarity kills GRPO; inference latency exceeds budget.
- **Multi-turn GRPO** — mature recipe doesn't exist (FAQ §59.6); flattened MDP is accepted state-of-art.
- **Real-time market data feed** — no infrastructure budget for it.
- **Continuous-time dynamics** — quarterly steps are sufficient.
- **Custom GRPO loop in pure PyTorch** — defeats the purpose of using TRL+Unsloth (mature, supported, debugged).
- **More than 5 assets** — increases search space without adding reasoning depth.
- **Procedural shock generation (RLVE-style)** — discussed; deferred to v2 README mention. 17 hand-crafted shocks are pedagogically clearer for the demo.
- **Bigger SFT trace count (1000+)** — Gemini API cost vs marginal benefit not worth it for 48hr; 120 was the empirical sweet spot.
- **Muon optimizer** — 2× speedup possible but not in Unsloth+TRL stable; debugging risk too high for hackathon.
- **GSPO loss** — research-stage, not in stable TRL.
- **vLLM for rollouts on Blackwell** — Unsloth + Gemma 4 don't currently support vLLM; using Unsloth's native generate path.

---

## Part 14 — The decision tree we follow during onsite training

(From [portfolio_env_design.md](portfolio_env_design.md) §18.4 — checkpoint every 12 hrs, no rabbit-holing.)

```
Apr 24 end of day (pre-onsite):
├─ Phase 1 GRPO regret > 0 on >= some rollouts?
│   ├─ YES → proceed
│   └─ NO  → env or trainer broken; debug overnight

Apr 25 noon (~12 hrs onsite):
├─ Phase 2 entry criterion met?
│   ├─ YES → SFT warm-start checkpoint OK; start Phase 3
│   └─ NO  → Tier 3 fallback (scope cut to 8Q + 2 interventions)

Apr 25 evening (~24 hrs onsite):
├─ Phase 3 hard-shock reward climbing?
│   ├─ YES → full target; polish demo
│   └─ NO  → Tier 2 (demote hard shocks → "capability probe" framing)

Apr 26 noon (~36 hrs onsite):
├─ Demo-ready checkpoint exists?
│   ├─ YES → UI polish + pitch rehearsal
│   └─ NO  → freeze best checkpoint; cut losses; lean on demo narrative
```

---

## Part 15 — The 2-minute demo arc (how it sells the env)

Silent + captions, three acts:

1. **0:00–0:20 (problem)** *"LLMs pattern-match when signals are clear. They fail when objectives conflict and shocks are ambiguous. We trained past that."*
2. **0:20–0:45 (baseline)** Untrained Qwen3-4B-Instruct plays a 12-quarter episode. Q3 hurricane → dumps OIL (wrong). Q6 rare-earth → buys GREEN (wrong). Q7 stagflation → piles into BONDS (real return -2.5%/yr). Final NAV: -12%.
3. **0:45–1:15 (trained)** GRPO-trained model on identical seed. `<think>` block streams. Q3 keeps OIL citing supply chain. Q6 sees rare-earth → GREEN supply collapse before buying. Q7 stagflation rotates into OIL + REAL_ESTATE. Final NAV: +18%.
4. **1:15–1:40 (the inflation moment)** Caption: *"Real returns matter. The trained model read 'PCE core 5.8%' and rotated. That's economic reasoning, not pattern matching."*
5. **1:40–2:00 (proof)** All 5 reward curves rising over training. Carbon bar respected. Hold-out eval: trained beats baseline on majority. *"48 hours. Single GPU. Open-source env. We think this trains reasoning in ways current benchmarks don't."*

---

## Part 16 — Reading order for someone picking this up cold

1. This file (you're here)
2. [README.md](README.md) — quick overview + how to run
3. [portfolio_env_design.md](portfolio_env_design.md) — design spec v0.6 (v0.7 patch pending)
4. [HACKATHON_PLAN.md](HACKATHON_PLAN.md) — live status, risk register, per-phase checklist
5. [BROTHER_BRIEF.md](BROTHER_BRIEF.md) — partner-task brief (shocks + UI)
6. [gemini_deep_research_output.md](gemini_deep_research_output.md) — the grounded research that caught the MDP-bandit mismatch
7. [portfolio_env/](portfolio_env/) — code

---

## Why this doc exists

Because every hour of this build has been a sequence of trade-offs against constraints we didn't fully understand at the start. Without this doc, the answer to "why X?" is buried across 12 commits, 3 design-doc revisions, a Gemini research transcript, an SSH session log, and a handful of FAQ pages. With it, the answer is one search away — and the next person picking this up (whether brother, judge, or future-us) can see not just *what* we built but *why every choice was made*.

If a section becomes stale, edit it in place. The git history captures the journey; this doc captures the synthesis.
