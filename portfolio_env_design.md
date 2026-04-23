# Reasoning-Under-Constraints Environment
## OpenEnv Hackathon Round 2 — Environment Design Doc

**Version:** v0.6 — brother sign-off on realism calibration + `fragmentation` tech_bet added (5 categories) + demo format locked (silent chart + captions). **Build-ready.**
**Team:** Ekansh (RL/env engineer) + brother (trading platform / UI)
**Hackathon:** Meta PyTorch × Scaler OpenEnv Hackathon, onsite Bangalore, April 25–26, 2026
**Timeline:** ~24 hours prep (April 24) → 48 hours onsite (April 25–26)
**Target themes:** #3.1 World Modeling (primary) + #2 Long-Horizon Planning (secondary) + #5 Wild Card

---

## 0. Version history

| Version | Changed |
|---|---|
| v0.1 | Initial draft — climate-stressed portfolio manager |
| v0.2 | Reframed as reasoning-under-constraints. Added 3-tier shock taxonomy, path dependency (transaction costs / spent-as-you-go carbon / 4-quarter lockups), interventions (infra_commit / carbon_offset_buy / put_hedge / tech_bet), regret-vs-baseline as primary reward |
| v0.3 | Added explicit 3-phase curriculum (§13.2), inference budget math (§13.3), carbon weight ramp-in (§9.2), and 3-tier fallback plan + decision tree (§18) |
| v0.4 | **Episode length 10 → 12 quarters (3-year cycle).** Brother's input: bull/bear cycles typically run ~3 years; a full cycle must fit in one episode for the model to learn regime transitions. Updates cascade through §4, §6.3, §9.2, §13.2, §13.3, §18.3 |
| v0.5 | **Inflation / money depreciation added as silent constraint.** Brother's call. Regret now computed on **real** (inflation-adjusted) returns — doubly kills the bonds exploit. Three inflation regimes (normal / stagflationary / deflationary) enter through news cues. `tech_bet` gains `inflationary` option. +2 shocks to pool (stagflation + deflation pulse). ESG S+G deferred to v2. |
| **v0.6** | **Brother sign-off on v0.5 realism calibration** (hurricane/rare-earth directions, put premium 2%, infra +8%/shock, carbon offset 10 kg/$ = $100/tonne confirmed as premium-end realistic). **Added `fragmentation` as 5th tech_bet option** (deglobalization / trade-war / supply-chain nationalism world). **Demo format locked: silent chart with captions, no voiceover.** Brother confirmed he'll write 5 of 15 SFT warm-start traces. §15 pruned to 2 remaining open questions (inflation magnitudes + regime observability). |

---

## 1. What we are building (reframed from v0.1)

**This is not a trading simulator.** This is an environment that teaches an LLM to reason under competing constraints when the signals are ambiguous and decisions are path-dependent.

The agent is presented as a climate-aware portfolio manager because that framing gives us clean, verifiable rewards and a visually compelling demo. But the **capability** we are training is: *read a partial-information scenario with conflicting signals, weigh trade-offs across multiple objectives, commit to positions that constrain future options, and recover from suboptimal early decisions.*

If an LLM can do this in a portfolio setting with hand-crafted shocks, it can transfer to any resource-allocation problem under uncertainty. That's the pitch.

Each quarter the agent:
1. Reads a macro news headline with **conflicting first/second/third-order impacts**
2. Emits a `<think>` block reasoning about trade-offs
3. Outputs portfolio weights **plus intervention decisions** as JSON
4. Receives next quarter's returns, path-dependent feedback, and a reward

Over 12 quarters (a full 3-year bull-bear cycle) it must beat an equal-weighted baseline while respecting a carbon budget and surviving rare-but-severe shocks — some of which were *predictable from Q1* if it committed early to the right thesis.

---

## 2. Why this env (not a market, not multi-agent)

We considered Carbon Credit Market (multi-agent) and standard portfolio backtesting. We rejected both:

- **Multi-agent market:** GRPO is ill-suited to non-stationary environments. Inference latency of coordinated multi-LLM rollouts exceeds our 48-hour window.
- **Simple portfolio backtest:** Pattern matching, not reasoning. Risk: we ship what judges call "finance sim with climate twist" (Innovation = 40% of rubric).

v0.3 sidesteps both failure modes by pushing the task toward **causal reasoning + world-shaping interventions + genuine long-horizon credit assignment**, with a disciplined curriculum and fallback plan.

---

## 3. Judging fit (v0.3)

| Criterion | Weight | Alignment |
|---|---|---|
| Environment Innovation | 40% | Causal reasoning under ambiguity + agent-as-world-shaper via interventions. Not a thing that exists in OpenEnv today. |
| Storytelling | 30% | "Greenberg Terminal" demo. Ambiguous hurricane scene: baseline LLM reflexively dumps OIL (wrong — supply shock makes it *go up*); trained LLM reasons through supply vs insurance vs reconstruction chains. |
| Reward Improvement | 20% | 5 independent reward curves (format, regret, Sharpe, carbon, drawdown) rising together across 3 training phases. |
| Reward + Pipeline | 10% | GRPO + Unsloth 4-bit + TRL, with explicit curriculum + inference budget engineering. |

---

## 4. Episode structure

```
Q1 → news → think → weights + interventions → end-of-quarter returns → path state updates → Q2 → ...
```

- **Episode length:** **12 quarters = 3 years (full market cycle)**
- **Rationale (brother's call):** bull/bear cycles typically run 2–3 years. A 10-quarter episode truncates a full cycle mid-phase, so the agent never learns regime transitions. 12 quarters fits one full cycle with room for a regime reversal mid-episode.
- **Steps per episode:** 12 LLM inferences
- **Randomization at reset:**
  - Sample shocks from tier-appropriate pool; randomize quarter assignment
  - Randomize order of headline clauses (prevents token-position memorization)
  - ±2% Gaussian noise on base returns per asset per quarter
  - **Optional regime marker:** some episodes start in "bull" base-return regime, others in "bear" (returns halved) — agent observes regime indirectly via market NAV drift
- **Q1 special:** agent commits to a `tech_bet` (one-time) that tilts future shock probabilities
- **Quarterly re-commit window (new in v0.4):** `infra_commit` locks for 4 quarters → can make up to 3 sequential commits over 12 quarters (Q1 unlocks Q5, Q5 unlocks Q9, Q9 unlocks at end). Tests serial conviction.
- **Inflation regime (new in v0.5):** each quarter has an inflation rate that reduces *real* returns. Agent observes the current rate in obs, can predict changes from news cues. Stagflation / deflation regimes enter via specific shocks (see §5.1).

---

## 5. The 5 assets

| Asset | Base quarterly return | Volatility | Carbon intensity (kg CO₂ / $) | Climate sensitivity |
|---|---|---|---|---|
| `TECH` | +3.0% | high | 0.05 (low) | neutral |
| `OIL` | +2.0% | medium | 2.50 (very high) | transition-risk ↑ |
| `GREEN` | +1.5% | medium | 0.01 (≈0) | transition-benefits |
| `REAL_ESTATE` | +1.0% | low | 0.10 (low) | physical-risk ↑ |
| `BONDS` | +0.5% | very low | 0.00 | refuge |

**Synthetic, not real market data.** Brother's calibration welcome — placeholders flagged.

### 5.1 Inflation regime (new in v0.5)

Each quarter has an inflation rate that silently reduces *real* returns. Agent observes current rate; must reason about future changes from news cues.

**Three regimes:**

| Regime | Quarterly rate | Annualized | Entry trigger |
|---|---|---|---|
| Normal | 1.0% | ~4% | Default |
| Stagflationary | 2.5% | ~10% | Specific shock (e.g., "Fed minutes show PCE core at 5.8%") |
| Deflationary | −0.3% | ~−1.2% | Specific shock (e.g., "China manufacturing contracts 8% YoY") |

**Asset behavior under inflation** (applied as a *multiplier* on nominal returns, not a flat subtraction — so real-asset dynamics behave realistically):

| Asset | Normal regime | Stagflationary | Deflationary |
|---|---|---|---|
| `TECH` | nominal | **−2% hit** (long-duration crushed by real rates) | +1% (deflation helps duration) |
| `OIL` | nominal | **+3% rally** (commodity inflation hedge) | −2% (demand destruction) |
| `GREEN` | nominal | **−3% hit** (long-duration, policy uncertainty) | +0.5% |
| `REAL_ESTATE` | nominal | +0.5% (pace inflation) | −1% (asset deflation) |
| `BONDS` | **−1% real** (always loses to inflation) | **−2.5% real** (gets crushed) | +0.3% (deflation friend) |

**Key insight:** BONDS is structurally punished in *any* non-deflationary regime. In the default (normal) regime, a 100% BONDS policy has real return of −0.5%/quarter — it loses real wealth even when "nothing happens." This kills the mattress exploit harder than regret alone.

### 5.2 Computing real returns

```python
real_return[asset][q] = nominal_return[asset][q] - inflation_rate[q] + regime_adjust[asset][regime]
cumulative_real_nav = product(1 + real_return[q] for q in range(Q))
```

The `r_regret` reward (§9) is computed on **real** returns, not nominal. This is the main mechanical change in v0.5.

---

## 6. Shock design — three difficulty tiers

### 6.1 Taxonomy

| Tier | Share of full pool | Description | LLM must |
|---|---|---|---|
| **Easy** | 6 of 17 | 1–2 assets move in obvious direction; rest stable | Not pattern-match a single keyword |
| **Ambiguous** | 7 of 17 | Signals within headline conflict; reasonable actions diverge 10–20% reward | Weigh trade-offs explicitly |
| **Hard** | 4 of 17 | 2nd/3rd-order effects *dominate*; naïve reading loses money | Reason through chains |

Pool expanded to 17 (v0.5 added 2 inflation-regime shocks — 1 ambiguous stagflation, 1 hard deflation pulse). Sample **5 shocks per episode**. Shock density ~0.4/quarter.

The training signal comes from **ambiguous + hard** — the LLM that stops at first-order impacts gets punished; the one that reasons through causal chains gets rewarded.

### 6.1.1 Regime coherence (new in v0.4)

With 12 quarters we can stage a **full bull-to-bear-to-recovery** arc in one episode:

- Q1–4: bull phase (1–2 easy shocks, mild positive drift)
- Q5–8: transition + crisis (1 ambiguous + 1 hard shock cluster)
- Q9–12: recovery or prolonged bear (1–2 shocks depending on earlier regime)

Implementation: shocks are drawn from tier-appropriate pool, but timing is constrained to preserve macro coherence (not just random scatter). This makes the `infra_commit` / `tech_bet` decisions matter — agent must predict *regime*, not just pattern-match.

### 6.2 Worked examples

**Easy (Q2):**
```python
Shock(quarter=2,
  news="Routine earnings season. Tech majors beat estimates 2.8% on average. "
       "Bond yields steady. No macro surprises.",
  impacts={'TECH': +0.04, 'OIL': 0.0, 'GREEN': 0.0,
           'REAL_ESTATE': +0.01, 'BONDS': 0.0})
```
Trivial. Reward for not doing anything stupid.

**Ambiguous (Q3):**
```python
Shock(quarter=3,
  news="Category 5 hurricane forecast for US Gulf Coast. Insurers downgrade "
       "REIT exposure. Gulf refineries at risk. FEMA preparing $80B "
       "reconstruction package. Fed hints at emergency rate cut.",
  impacts={
      'REAL_ESTATE': -0.25,   # direct physical damage (first-order, obvious)
      'OIL':         +0.08,   # refinery outage = supply cut > demand hit (second-order — counterintuitive)
      'GREEN':       +0.12,   # reconstruction favors new grid/renewable (third-order)
      'TECH':        -0.03,   # mild risk-off sentiment
      'BONDS':       +0.08,   # rate-cut expectation + flight to safety
  })
```
Pattern-matching LLM dumps both OIL and REIT. Reasoning LLM dumps REIT, **keeps or adds OIL** (refinery supply shock wins), rotates into GREEN, adds BONDS.

**Hard (Q6):**
```python
Shock(quarter=6,
  news="China announces 80% reduction in rare-earth exports over 18 months "
       "citing domestic demand. US semiconductor export controls tighten. "
       "Renewable manufacturers warn of 3-quarter supply chain disruption. "
       "Oil majors announce record buybacks on sector rotation inflows.",
  impacts={
      'TECH':        -0.18,   # rare-earth + export controls hit hard
      'GREEN':       -0.22,   # renewable supply chain depends on rare-earth magnets → surprise hit
      'OIL':         +0.14,   # capital rotation INTO oil (counterintuitive — "green bad, oil good" crosswind)
      'REAL_ESTATE': -0.02,
      'BONDS':       +0.05,
  })
```
First-order reading: *"China export cuts = protectionism = sell everything."* Wrong — OIL rallies on sector rotation. Base LLM that bought GREEN because "green = climate = safe" gets mauled.

**Ambiguous stagflation (new v0.5 shock, Q7):**
```python
Shock(quarter=7,
  news="Fed minutes leaked: PCE core unexpectedly at 5.8%, committee signals "
       "sustained tightening into 2027. 10-year yields climb 80bp. Dollar "
       "rallies against EM. Oil services announce capacity expansion.",
  regime_shift='stagflationary',       # <-- triggers 2.5% inflation next quarters
  impacts={
      'TECH':        -0.10,   # rate-sensitive growth crushed
      'OIL':         +0.11,   # commodity hedge + supply response
      'GREEN':       -0.08,   # rate-sensitive + policy uncertainty
      'REAL_ESTATE': +0.02,   # paces inflation
      'BONDS':       -0.09,   # duration hit on top of real-return loss
  })
```
Triggers a **regime shift** that persists for subsequent quarters. A naive LLM sees "Fed tightening" and runs to BONDS. A reasoning LLM reads *stagflation* and rotates into OIL + REAL_ESTATE. The BONDS move is doubly punished (nominal duration loss + structural real-return bleed — see §5.1).

**Hard deflation pulse (new v0.5 shock, Q9):**
```python
Shock(quarter=9,
  news="China manufacturing PMI crashes to 41; export prices fall 12% YoY. "
       "Global supply gluts detected across semiconductors, oil, real estate. "
       "Treasury yields plunge on safe-haven bid. Bank of Japan intervenes.",
  regime_shift='deflationary',         # <-- −0.3% inflation next quarters
  impacts={
      'TECH':        -0.12,   # demand destruction but duration benefit mutes
      'OIL':         -0.14,   # supply glut + demand collapse
      'GREEN':       -0.05,
      'REAL_ESTATE': -0.08,   # asset deflation
      'BONDS':       +0.06,   # ONLY regime where BONDS is the right call
  })
```
First-order: "crash = everything falls." Wrong — BONDS actively rally here (deflation + flight to quality + duration gain). Agents who've learned "BONDS always bad" from stagflation training now must learn the exception.

---

## 7. Action space — weights + interventions

```python
from typing import Literal
from pydantic import BaseModel, Field

class PortfolioAction(BaseModel):
    # Liquid allocation — rebalanced every quarter
    weights: list[float] = Field(..., min_length=5, max_length=5)

    # Intervention 1: irreversible commit, 4-quarter lockup
    infra_commit: float = Field(default=0.0, ge=0.0, le=0.2)

    # Intervention 2: sustainability lever — spend now to offset carbon
    carbon_offset_buy: float = Field(default=0.0, ge=0.0, le=0.1)

    # Intervention 3: insurance — caps downside but costs premium each quarter
    put_hedge: float = Field(default=0.0, ge=0.0, le=0.05)

    # Intervention 4: Q1-only one-time thesis bet
    tech_bet: Literal['status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'] = 'status_quo'
```

### 7.1 Intervention semantics

| Intervention | Cost | Payoff | Lock-up | When |
|---|---|---|---|---|
| `infra_commit` | Fraction of NAV | +8%/quarter conditional on transition shocks hitting later | **4 quarters irreversible** | Any quarter |
| `carbon_offset_buy` | 1 unit capital → offsets 10 kg CO₂ | Reduces cumulative carbon footprint | Immediate | Any quarter |
| `put_hedge` | 2% NAV premium per quarter | Caps portfolio downside at −5% if quarter's worst-asset return < −15% | 1-quarter protection | Any quarter |
| `tech_bet` | Free | Tilts shock-pool sampling for remainder of episode | N/A | **Q1 only** |

**`tech_bet` option semantics (5 worlds):**
- `status_quo`: default shock distribution
- `green_leaps`: transition-risk shocks 2× more likely, physical-risk 0.5× (accelerating energy transition)
- `carbon_priced`: both transition + physical-risk 1.5× (stringent climate policy world)
- `inflationary` (new v0.5): stagflation shock 3× more likely, deflation 0.3× (sticky inflation world)
- `fragmentation` (new v0.6): rare-earth / supply-chain / trade-war shocks 2.5× more likely; TECH and GREEN base-return penalized −0.5%/quarter during episode (persistent disruption); OIL base-return +0.3%/quarter (national-security premium). Deglobalization / supply-chain nationalism thesis.

Curriculum phase gates which interventions are available — see §13.2.

---

## 8. Observation space

```python
class PortfolioObs(BaseModel):
    # Time
    quarter: int                             # 0..11
    difficulty_tier: str                     # 'easy', 'ambiguous', 'hard' — for curriculum

    # Current state
    current_weights: list[float]
    infra_locked_fraction: float
    infra_unlock_quarters: int
    carbon_offsets_held: float
    active_put_hedge: bool
    tech_bet_chosen: str

    # Financials (all REAL, inflation-adjusted)
    portfolio_nav_nominal: float
    portfolio_nav_real: float                 # what agent's performance is actually graded on
    baseline_nav_real: float                  # equal-weighted benchmark, real terms
    cumulative_real_return_pct: float

    # Inflation state (new v0.5)
    current_inflation_rate: float             # quarterly, e.g. 0.010 for 1%
    current_regime: str                       # 'normal' | 'stagflationary' | 'deflationary'
    cumulative_inflation_multiplier: float    # compound price level since Q1

    # Sustainability
    carbon_footprint_accumulated: float
    carbon_budget_remaining: float

    # The reasoning signal
    news: str                                 # macro headline with 1st/2nd/3rd-order effects

    # Feedback
    last_quarter_returns_nominal: list[float]
    last_quarter_returns_real: list[float]
    last_quarter_regret: float                # computed on REAL returns
```

---

## 8.5. Path dependency (three mechanics)

### 8.5.1 Transaction costs
```python
turnover = sum(abs(new_weights[i] - old_weights[i]) for i in range(5))
tc = 0.005 * turnover
nav *= (1 - tc)
```
Discourages spastic rebalancing.

### 8.5.2 Carbon spent-as-you-go
```python
carbon_this_quarter = sum(weights[i] * CARBON_INTENSITY[i] * nav for i in range(5))
carbon_footprint_accumulated += carbon_this_quarter
carbon_footprint_accumulated -= carbon_offsets_used_this_quarter
```
Early heavy OIL position eats budget → constrains Q8 options. Q2 ↔ Q8 linkage.

### 8.5.3 Lock-ups
```python
if unlock_quarter == current_quarter:
    transition_shocks_hit = count_transition_shocks_in(past_4_quarters)
    infra_return = 0.08 * transition_shocks_hit
    nav += infra_locked_fraction * nav * (1 + infra_return)
    infra_locked_fraction = 0
```
Q2 commitment locks capital until Q6. Return depends on what shocks actually arrived. "Recover from early mistakes" mechanism.

---

## 9. Reward function — five components

### 9.1 Definitions

```python
# ── 1. FORMAT (per-completion, immediate) ─────────────────────────
def r_format(completion: str) -> float:
    has_think = '<think>' in completion and '</think>' in completion
    action = parse_json_action(completion)
    valid = action is not None
    return (0.05 if has_think else 0.0) + (0.10 if valid else 0.0)

# ── 2. REGRET vs EQUAL-WEIGHTED BASELINE (primary, REAL returns) ───
def r_regret(trajectory) -> float:
    # v0.5: computed on inflation-adjusted REAL returns, not nominal
    agent_return = trajectory.nav_real_series[-1] / trajectory.nav_real_series[0] - 1
    baseline_return = trajectory.baseline_nav_real_series[-1] / trajectory.baseline_nav_real_series[0] - 1
    return float(agent_return - baseline_return)

# ── 3. SHARPE (secondary) ──────────────────────────────────────────
def r_sharpe(trajectory) -> float:
    q_returns = trajectory.quarterly_returns
    if len(q_returns) < 2: return 0.0
    sharpe = float(np.mean(q_returns) / (np.std(q_returns) + 1e-6))
    return 0.3 * sharpe

# ── 4. CARBON PENALTY (non-linear, ramp-in via curriculum) ────────
CARBON_CAP = 120.0  # scaled to 12-quarter episode (v0.4)
def r_carbon(trajectory, phase_weight: float) -> float:
    net_carbon = trajectory.carbon_footprint_accumulated
    overshoot = max(0.0, net_carbon - CARBON_CAP)
    return -phase_weight * 5.0 * (overshoot ** 2) / 100.0

# ── 5. MAX DRAWDOWN PENALTY ────────────────────────────────────────
def r_drawdown(trajectory) -> float:
    peak, max_dd = 0.0, 0.0
    for v in trajectory.nav_series:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak if peak > 0 else 0)
    return -2.0 * max_dd
```

### 9.2 Carbon weight ramp-in (new in v0.3)

If carbon penalty fires at full strength on Phase 1, the agent collapses to 100% BONDS (safe + zero carbon) and never learns regret-beating. Ramp in by phase:

| Phase | `carbon_weight` | Reasoning |
|---|---|---|
| Phase 1 | 0.0 | Ignore carbon entirely. Learn format + baseline beating. |
| Phase 2 | 0.3 | Soft nudge. Agent starts trading a bit of return for sustainability. |
| Phase 3 | 1.0 | Full non-linear penalty above cap. |

This also doubles as a demo ablation: *"here's what happens when we turn on carbon constraints mid-training."*

### 9.3 Why this set blocks exploits

| Component | Blocks |
|---|---|
| `r_format` | Garbage output |
| `r_regret` (on real returns) | All-bonds (loses to equal-weighted AND loses to inflation), keyword pattern-matching (wins easy, loses ambiguous), high-nominal-return-but-inflation-bleed policies |
| `r_sharpe` (low weight) | Super-high-return catastrophically volatile policies |
| `r_carbon` (non-linear, ramped) | OIL-maxing; also protects against mattress-exploit via ramp-in |
| `r_drawdown` | "I beat benchmark overall but lost 35% mid-episode" policies |

---

## 10. Why GRPO specifically

- No value network → smaller training footprint → Unsloth 4-bit + T4 Colab fits
- Samples N completions per prompt, scores each, uses **within-group relative advantages**
- Needs **verifiable rewards** (not learned) — all 5 of ours are exact calculations
- Works beautifully with `<think>` CoT output (this is how DeepSeek-R1 was trained)

---

## 11. Demo arc — "Greenberg Terminal"

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│  MACRO NEWS FEED                      │  LLM INNER MONOLOGUE (streaming)      │
│  Q1: EARLY COMMIT phase…              │  <think>                              │
│      Tech bet: green_leaps            │  Transition risk rising. Bet          │
│      Infra commit: 15%                │  green_leaps, lock 15% infra for 4Q.  │
│  Q2: Routine earnings                 │  Put hedge = 0 (no imminent shock).   │
│  Q3: HURRICANE + FEMA + RATE CUT      │                                       │
│      (ambiguous: OIL goes UP)         │  Q3 reasoning:                        │
│  Q4: China rare earths ...            │  Hurricane hits REIT directly.        │
│                                       │  BUT refinery supply cut = OIL UP.    │
│                                       │  Reconstruction = GREEN UP.           │
│                                       │  Rate cut = BONDS UP.                 │
│                                       │  → sell REIT, hold OIL, add GREEN     │
│                                       │  Buy put hedge ahead of known shock.  │
│                                       │  </think>                             │
│                                       │  {"weights":[0.25,0.20,0.25,0,0.30],  │
│                                       │   "put_hedge":0.03, ...}              │
├───────────────────────────────────────┴───────────────────────────────────────┤
│  PORTFOLIO P&L (agent vs equal-weight baseline)                               │
│    ─── agent (RL-trained) ──────▲  +23.4%                                    │
│    ─── baseline untrained LLM ─── −7.1%                                       │
│    ─── equal-weighted passive ──  +8.9%                                       │
│    [shock markers: Q3 hurricane, Q6 rare-earth, Q8 IRA 2.0]                  │
│    [carbon budget bar: 87/100 kg CO₂; 15 kg offsets bought Q5]               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 11.1 The 2-minute demo — silent chart + captions format (locked v0.6)

**No voice-over.** Captions on-screen synchronized to chart animation + streaming `<think>` tokens. Judges read the captions while watching the terminal + P&L chart update live. Tight, deliberate, no narration fatigue.

| Time | Caption (appears on screen) | Visual |
|---|---|---|
| 0:00–0:20 | *"LLMs pattern-match when signals are clear. They fail when objectives conflict and shocks are ambiguous. We trained past that."* | Problem statement over split-screen of two blank terminals |
| 0:20–0:45 | *"Baseline Llama 8B. Same 12 quarters. Same shocks."* | Baseline plays. Q3: caption *"→ hurricane → dumped OIL — wrong (refinery supply cut = OIL up)"*. Q6: *"→ rare-earth news → bought GREEN — wrong (supply chain collapse)"*. Q7: *"→ stagflation → piled into BONDS — real return −2.5%/yr"*. P&L ends at −12%. |
| 0:45–1:15 | *"GRPO-trained Llama 8B. Identical seed."* | Trained agent plays. `<think>` streams on-screen. Q3: *"Supply chain > direct damage → hold OIL"*. Q6: *"Rare-earth → GREEN supply shock → sell"*. Q7: *"PCE at 5.8% → stagflation → OIL, REAL_ESTATE; bonds bleed real"*. P&L ends at +18%. |
| 1:15–1:40 | *"The difference isn't finance. It's reasoning under competing constraints + inflation + sustainability. Three different causal chains. One trained agent."* | Split-screen P&L: baseline red line, trained green line, shock markers on both |
| 1:40–2:00 | *"48 hours on a single T4. 5 independent reward curves rising through 3 curriculum phases. Env open-source on HF Hub. This trains what current benchmarks don't."* | Cut to 5-panel reward curves animating through training |

**Production note:** record the terminal + chart interactions at real speed, then time-scale in post. Captions in Inter or SF Pro, dark background, cyan-accented — match the Greenberg Terminal aesthetic.

---

## 12. Division of labor

### Ekansh (RL / env engineer)
- OpenEnv scaffold with v0.3 Pydantic models
- Path-dependent mechanics (transaction costs, carbon accrual, lockup accounting)
- 13-shock pool with tier tags
- Intervention mechanics
- 5 reward functions + unit tests with adversarial policies (bonds-only, all-TECH)
- Colab notebook: Unsloth 4-bit Llama 3.1 8B + TRL GRPOTrainer
- **Format-reward regex + action parser** (day-1 priority)
- **3-phase curriculum runner** (entry-criterion checks)
- HF Spaces deployment

### Brother (trading / UI / market realism)
- ✅ **Shock realism pressure-tested** (v0.6 sign-off: hurricane/rare-earth directions confirmed)
- ✅ **Intervention realism confirmed** (put 2%, infra +8%/shock, offset 10kg/$ all good)
- ✅ **Tech-bet taxonomy settled** (5 categories with `fragmentation` added)
- Build "Greenberg Terminal" React UI (news feed, LLM monologue, P&L chart with shock markers, regime indicator, carbon bar)
- **Write 5 of 15 SFT warm-start `<think>` traces** (brother confirmed, his domain reasoning on hard shocks > frontier model cold)
- Demo captions + chart animation (silent format, no voice-over)

### Together (April 24)
- Agree on exact shock list
- Lock JSON schema contract (backend ↔ UI)
- Generate ~15 expert `<think>` traces for SFT warm-start (see §18 fallback Tier 1)
- Rehearse 2-minute pitch once

### Together onsite (April 25–26)
- **Checkpoint every 12 hrs** (see §18 decision tree)
- Iterate reward weights if any component dominates
- **Sample generations every 50 iterations** (reward-hacking check)
- Record final before/after demo on identical seed

---

## 13. Prep + execution plans

### 13.1 24-hour prep plan (April 24)

| Hours | Ekansh | Brother |
|---|---|---|
| 0–2 | Install Unsloth, TRL; test GRPOTrainer on toy text task | Scaffold React app with 3 panels + dummy JSON |
| 2–6 | OpenEnv scaffold, v0.3 Pydantic models, path-dep state | Style three panels — news feed, monologue, P&L |
| 6–10 | Shock pool draft (5 easy + 5 ambiguous + 3 hard); reward fns unit-tested | Review shocks with Ekansh; correct directional calls |
| 10–14 | Integrate shocks + interventions; random-agent episode | Wire dummy JSON → UI; see full episode render |
| 14–18 | Colab: load Llama 3.1 8B Unsloth; single-step rollout + format reward | Polish shock-candle animations |
| 18–22 | Generate ~15 SFT traces (call frontier model); SFT for 50 steps | Test UI across multiple episodes |
| 22–24 | Minimal 10-iter Phase-1 GRPO to verify loop | Pitch rehearsal |

### 13.2 Curriculum schedule (new in v0.3)

The "Start Simple" mandate. Three phases, each with clear gates.

| Phase | Target iters | Episode length | Shock pool | Rewards active | Interventions available | Entry criterion | `carbon_weight` |
|---|---|---|---|---|---|---|---|
| **1. Format + regret** | 0–50 | 4 quarters | **Easy only** (6 shocks, sample 2/ep) | `r_format` + `r_regret` | — | Start | 0.0 |
| **2. Ambiguity** | 50–150 | 8 quarters (half cycle) | Easy + Ambiguous (12, sample 3/ep) | + `r_sharpe` + `r_drawdown` | `infra_commit` only | Phase 1: `r_regret > 0` on 50% of rollouts | 0.3 |
| **3. Full task** | 150–230 | **12 quarters (full cycle)** | All 15 + full interventions (sample 5/ep) | + `r_carbon` | All 4 interventions | Phase 2: median `r_regret > 0.05` | 1.0 |

**Why:**
- Phase 1 teaches JSON shape and regret-beating on trivial shocks. If this fails, the env or trainer is broken — not the task.
- Phase 2 adds causal ambiguity + drawdown penalty + one intervention. Agent must reason.
- Phase 3 is the target task: full horizon, all shocks, all interventions, full carbon penalty.

If a phase fails its entry criterion within ~20% over target iters, trigger fallback (§18).

### 13.3 Inference budget (new in v0.3)

Rollouts dominate. The math:

```
Prompt:       ~400 tokens (news + obs state + shot)
<think>:      ~150 tokens (CAPPED via stop tokens)
JSON action:  ~80 tokens
Per rollout:  ~630 tokens

Updated for 12-quarter episodes (v0.4):
Full Phase-3 iter = batch(6) × N(6) × steps(12) = 432 rollouts = ~272K tokens
Unsloth 4-bit Llama 8B on T4 ≈ 300 tok/s generous
  → ~15 min/iter (Phase 3 at reduced batch)
```

Budget planning (v0.4, 12-quarter Phase 3):

| Phase | batch | N | steps | Rollouts | Tokens | Sec/iter | Target iters | Hours |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 4 | 4 | 64 | ~40K | ~135 | 50 | **~2 hrs** |
| 2 | 6 | 6 | 8 | 288 | ~181K | ~605 | 100 | **~17 hrs** |
| 3 | 6 | 6 | 12 | 432 | ~272K | ~910 | 80 | **~20 hrs** |

Three levers to stay under budget:

1. **Cap `<think>` at 150 tokens** via stop tokens — prompt: *"Think in 2–3 sentences. Then output JSON."* Untrimmed LLMs ramble to 500+ tokens.
2. **Phase 3 reduced target**: 80 iters instead of 150. Diminishing returns after ~80 iters.
3. **Batch/N held at 6/6**: bumping to 8/8 would push Phase 3 to ~30 hrs.

Realistic final budget (v0.4):
- Phase 1: 2 hrs
- Phase 2: 17 hrs (up from 13 due to longer episodes)
- Phase 3: 20 hrs (up from 17 due to 12Q not 10Q)
- **Total: ~39 hrs training.** Fits 48-hour onsite with ~9 hrs margin — tighter than v0.3. If we need more buffer, drop Phase 2 target iters to 80 (saves ~3 hrs).

---

## 14. Known failure modes

| Failure | Trigger | Mitigation |
|---|---|---|
| Agent outputs wrong action shape | No format reward yet | Format reward is **day-1 priority**, weight 0.15 |
| Mattress exploit (100% bonds) | Regret with weak baseline | Equal-weighted makes ~9%; bonds get −8% regret |
| Sharpe-only low-var policy | Sharpe weighted too heavy | Kept at 0.3× |
| Reward flat 500 steps | Task too hard, zero success baseline | Curriculum §13.5 + SFT warm-start §18 Tier 1 |
| Model memorizes shock quarters | Fixed timing | Randomize shock-to-quarter at reset |
| Demo looks fake | Synthetic data obvious | Brother's calibration; realistic headlines |
| Agent never commits `infra` | Expected return too low / lockup scary | Tune infra conditional return; validate after Phase 2 |
| Agent always max `put_hedge` | Premium too cheap | Keep 2%/quarter |
| Judges say "just stocks" | Sustainability thread weak | Lean on carbon-offset Q5 demo moment |
| LLM ignores news, uses numerics | Observation too numeric-heavy | `news` string FIRST in prompt template |
| Phase 3 hard shocks don't crack | GRPO can't find hard-shock success | Fallback §18 Tier 2 (demote hard → capability probe) |
| Whole training stalls | Env / trainer bug | Fallback §18 Tier 3 (scope cut) |

---

## 15. Open questions — v0.6 status

### ✅ Resolved (brother's v0.6 sign-off)

1. Hurricane → OIL up (supply cut) — **confirmed**
2. Rare-earth → OIL rallies (sector rotation) — **confirmed**
3. Put hedge 2% quarterly premium — **confirmed, right order**
4. Infra lockup +8%/transition shock — **confirmed for green-infra fund**
5. Tech-bet taxonomy — **5 categories locked** (added `fragmentation`)
6. Carbon offset 10 kg/$ = $100/tonne — **confirmed, premium-end realistic for 2026 compliance market**
8. Demo format — **silent chart + captions**
9. SFT traces — **brother writing 5 of 15**

### Still open (need brother's answer or our call)

7. **UI stack.** React + recharts? Next.js? Streamlit? Brother's call — optimize for what *he* ships fastest.
10. **Inflation magnitudes.** The stagflation asset adjustments in §5.1 (TECH −2%, OIL +3%, GREEN −3%, REAL_ESTATE +0.5%, BONDS −1% additional real bleed). Right magnitudes? Especially OIL — is +3% additional return during stagflation too generous, about right, or conservative?
11. **Inflation observability.** Agent sees `current_inflation_rate` (scalar) + `current_regime` (string label) in obs. Is seeing the regime label directly too much of a hint? More realistic alternative: show only the numeric rate + news, let LLM *infer* the regime. Training-difficulty tradeoff.

---

## 16. What this does NOT do (intentional cuts)

- No real market data — synthetic shocks with hand-crafted causal chains
- No multi-asset derivatives beyond the put hedge abstraction
- No multi-agent
- No real-time market data feed
- No continuous-time dynamics — quarterly steps only
- No interpretability tooling beyond `<think>` parsing

Each defensible under 48 hours.

---

## 17. References

- OpenEnv Hackathon Round 2 official guide — received April 23, 2026
- TRL GRPOTrainer docs: https://huggingface.co/docs/trl
- Unsloth 4-bit GRPO template: https://github.com/unslothai/unsloth
- DeepSeek-R1 paper (CoT + GRPO): https://arxiv.org/abs/2501.12948
- Muse Spark announcement (Meta's direction): https://ai.meta.com/blog/introducing-muse-spark-msl/
- Gemini 3.1 Pro strategy review + v0.1/v0.2 critiques

---

## 18. Fallback plan (new in v0.3)

**If training stalls, what do we ship?** Three graceful degradations + decision tree.

### 18.1 Tier 1 fallback — SFT warm-start

Before Phase 3 GRPO, run supervised fine-tuning on ~15 expert `<think>` traces. Generate these by:
1. Prompt GPT-4 / Claude / Gemini with shock news + obs state
2. Ask for high-quality `<think>` + JSON action
3. Validate action beats baseline regret on the hidden impact dict
4. Keep only traces that pass

SFT for ~50 steps on these traces before Phase 3 GRPO. **DeepSeek-R1 recipe.** Cost: ~30 min to generate traces + ~10 min SFT. Low-effort, high-upside.

**This is the primary fallback and should be applied even if training is going well** — it boosts Phase 3 success probability at near-zero cost.

### 18.2 Tier 2 fallback — demote hard shocks to capability probe

If Phase 3 reward curve stays flat on hard shocks after ~50 iters (even with SFT warm-start):

- Cut hard shocks from training. Train on easy + ambiguous (10 shocks).
- Reach a strong trained model on those.
- In the demo, show **one** hard-shock episode where the trained model partially handles it (better than baseline, not perfect).
- Caption it: *"Hard-tier shocks with 3rd-order causal chains remain an open research question — here's where current training reaches."*

Judges respect *honesty about limits* more than overselling. Demo narrative (trained > baseline) preserved.

### 18.3 Tier 3 fallback — scope cut to 8 quarters + 2 interventions

If Phase 2 stalls (regret never climbs):

- 8-quarter episodes instead of 12 (half cycle instead of full)
- Drop `put_hedge` and `carbon_offset_buy`; keep only `infra_commit` + `tech_bet`
- Use only easy + ambiguous shocks
- Keep full reward structure + curriculum

**Preserves:** path dependency, causal-ambiguity demo moment, intervention innovation story, 5 reward curves, regime-transition demo (8Q = one full bull-bear shift).
**Loses:** some Theme #5 wild-card novelty. But still a working submission.

### 18.4 Decision tree

```
Apr 24, end of day (before onsite):
├─ Phase 1 GRPO shows learning (regret > 0)?
│   ├─ YES → proceed as planned
│   └─ NO  → env or trainer broken. Debug overnight.

Apr 25, noon (~12 hrs onsite):
├─ Phase 2 entry criterion met (Phase 1: r_regret > 0 on 50% rollouts)?
│   ├─ YES → SFT warm-start, then start Phase 3
│   └─ NO  → invoke Tier 3 scope cut. Ship working v0.3 at reduced spec.

Apr 25, evening (~24 hrs onsite):
├─ Phase 3 hard-shock reward climbing (median regret improving)?
│   ├─ YES → full v0.3 target. Polish demo next 24 hrs.
│   └─ NO  → invoke Tier 2. Reframe demo — hard shocks as capability probe.

Apr 26, noon (~36 hrs onsite):
├─ Demo-ready checkpoint exists?
│   ├─ YES → spend afternoon on UI polish, pitch rehearsal, video recording
│   └─ NO  → freeze best checkpoint, cut losses, focus on demo narrative quality
```

**Rule: no rabbit-holing. Make the go/no-go call at each checkpoint. Most teams fail by chasing training that isn't working.**

---

## 19. Decision point — v0.6 build-ready

**v0.6 is the stable spec.** Brother has signed off on realism calibration (shocks, interventions, tech-bet taxonomy, carbon offset pricing, demo format). Two open questions remain (Q7 UI stack, Q10 inflation magnitudes, Q11 regime observability) — none block the build.

### Go / no-go

- **Ekansh kicks off tonight (April 23, evening):**
  - OpenEnv scaffold with v0.6 Pydantic models
  - Path-dependent state (transaction costs, carbon accrual, lockup accounting)
  - 17-shock pool skeleton with tier tags
  - 5 reward functions with unit tests
  - Colab notebook: Unsloth 4-bit Llama 3.1 8B + TRL GRPOTrainer + format-reward regex

- **Brother parallel (April 24):**
  - React UI scaffold (his stack choice)
  - 5 SFT warm-start traces for hard/ambiguous shocks
  - Answers to Q7, Q10, Q11

- **First checkpoint (end of April 24):** Phase 1 GRPO showing `r_regret > 0` on a subset of rollouts. If yes, we're on track. If no, debug overnight per §18 decision tree.

**Iteration from here should be targeted patches (v0.7, v0.8) not full rewrites. The design is locked.**
