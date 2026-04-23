# Brother's brief — April 24

Everything you need to pick up your two deliverables for the Meta PyTorch hackathon. Read this cold; it's self-contained.

**Your two tasks:**
1. Fill 11 placeholder shocks in `portfolio_env/shocks.py` (~2 hrs)
2. Build the "Greenberg Terminal" UI that renders episodes for the demo (~4 hrs)

The first unblocks training quality. The second is the demo. Do them in that order.

---

## Context in 5 bullets

- **What we're building**: an OpenEnv environment where an LLM manages a climate-aware portfolio over 12 quarters (3-year cycle). News events (shocks) hit at specific quarters. LLM outputs `<think>` + JSON allocation.
- **Why**: trains reasoning under competing constraints (return vs carbon vs risk) with ambiguous signals — a capability probe, not a finance sim.
- **Our model**: Qwen3-4B-Instruct-2507, trained with GRPO via Unsloth on RunPod RTX 5090. Already running SFT warm-start on 120 Gemini-generated traces.
- **5 assets**: TECH, OIL, GREEN (renewables), REAL_ESTATE, BONDS. Action = `weights[5]` + 4 interventions (infra_commit, carbon_offset_buy, put_hedge, tech_bet).
- **The spec**: [portfolio_env_design.md](portfolio_env_design.md) v0.6 (to be patched v0.7). [README.md](README.md) has the overview.

---

## Task 1 — Fill 11 placeholder shocks

**File:** `portfolio_env/shocks.py`

**Current pool:** 17 shocks total, 6 concrete + 11 placeholders (search "PLACEHOLDER"). By difficulty:
- Easy (6 slots): 4 concrete + 2 placeholders
- Ambiguous (7 slots): 3 concrete + 4 placeholders
- Hard (4 slots): 2 concrete + 2 placeholders

**Why 3-tier difficulty matters:**
- **Easy** (40% of pool): 1–2 assets move obvious direction; LLM just needs to not do something stupid
- **Ambiguous** (40%): conflicting signals within headline — reasoning matters, pattern-matching fails
- **Hard** (20%): 2nd/3rd-order effects dominate; naive first-order reading *loses money*

The training signal comes from ambiguous + hard. Your job is to write shocks where a pattern-matching LLM gets hurt and a reasoning LLM benefits.

### The `Shock` dataclass

```python
@dataclass
class Shock:
    id: str                            # unique, snake_case
    tier: Literal['easy', 'ambiguous', 'hard']
    news: str                          # 2-4 sentence market-wire style
    impacts: dict[str, float]          # additive return adjustment per asset
    regime_shift: Regime | None = None # optional: triggers stagflation / deflation
    tags: list[str] = []               # 'transition_risk', 'physical_risk',
                                        # 'supply_chain', 'fragmentation',
                                        # 'inflation', 'deflation'
```

### Example — an ambiguous shock done right (hurricane)

```python
Shock(
    id='ambig_hurricane_gulf',
    tier='ambiguous',
    news='Category 5 hurricane forecast for US Gulf Coast. Insurers downgrade '
         'REIT exposure. Gulf refineries at risk. FEMA preparing $80B '
         'reconstruction package. Fed hints at emergency rate cut.',
    impacts={
        'TECH':        -0.03,   # mild risk-off (1st-order obvious)
        'OIL':         +0.08,   # refinery supply cut > demand hit (2nd-order COUNTERINTUITIVE)
        'GREEN':       +0.12,   # reconstruction → new grid/renewables (3rd-order)
        'REAL_ESTATE': -0.25,   # direct physical damage (1st-order obvious)
        'BONDS':       +0.08,   # rate-cut expectation + flight to safety
    },
    tags=['physical_risk'],
),
```

Notice: **4 of 5 assets move in non-obvious directions.** OIL goes *up* despite the disaster (supply shock > demand drop). That's the ambiguity.

### What makes a good placeholder fill — checklist

- **First-order impact is identifiable** — at least one asset moves in the direction a naive LLM would expect
- **Second-order effect contradicts or amplifies** — one asset's move is *because* of a chain the LLM has to reason about
- **Impact magnitudes in the range** `[-0.30, +0.20]` — anything bigger dominates the episode unrealistically
- **News is 2–4 sentences** — enough detail to reason from, short enough to fit prompt budget
- **At least one tag** (enables tech_bet skewing later)
- **Not duplicative** — check the 6 concrete shocks; don't repeat themes

### 11 placeholder slot suggestions (pick/modify/replace freely)

#### Easy (2 slots)

| Slot | Suggested theme | Why it's easy |
|---|---|---|
| `easy_PLACEHOLDER_5` | Auto sector Q3 numbers (EV penetration crosses 20% in Europe) | Direct TECH+GREEN boost, no conflicts |
| `easy_PLACEHOLDER_6` | Corporate bond issuance resumes at record volume | Direct BONDS boost, mild TECH/RE signal |

#### Ambiguous (4 slots)

| Slot | Suggested theme | The 2nd-order the LLM must find |
|---|---|---|
| `ambig_PLACEHOLDER_4` | Insurance industry pulls out of Florida/California | 1st: RE -15%. 2nd: regulator intervention means TAXPAYER bailout → long-term moral hazard priced into BONDS and RE recovery. Fed rate cut probability rises. |
| `ambig_PLACEHOLDER_5` | ChatGPT-style AI breakthrough: next-gen reasoning model halves inference costs | 1st: TECH +10. 2nd: GREEN *loses* because data-center energy demand forecasts get revised down; utility REITs hit. |
| `ambig_PLACEHOLDER_6` | German nuclear phaseout reversal; reopening 3 reactors | 1st: GREEN ambiguous (is nuclear "green"?). 2nd: OIL/GAS -5 because baseload supply returns. Fragmentation narrative weakens. |
| `ambig_PLACEHOLDER_7` | Major pension funds announce divestment from fossil in 5-year window | 1st: OIL -8 long-horizon. 2nd: forced-selling pressure creates *short-term* OIL buying opportunity (historical pattern). |

#### Hard (2 slots)

| Slot | Suggested theme | The 3rd-order that kills pattern matchers |
|---|---|---|
| `hard_PLACEHOLDER_3` | Chip fab water shortage in Taiwan + TSMC output cut 30% | 1st: TECH -15. 2nd: GREEN -10 (solar cell supply). 3rd: OIL +12 because Taiwan/China tension narrative forces sector-rotation rotation into commodity defensive. BONDS +5 on flight to quality. |
| `hard_PLACEHOLDER_4` | Carbon credit market fraud exposed — 40% of offsets declared invalid | 1st: "carbon companies down." 2nd: real compliance demand SPIKES (offsets no longer cheap alternative), OIL/industrials crater. 3rd: GREEN +20 (actual abatement now required). |

**Do not lock these in blindly** — your read of 2026 markets is sharper than mine. Replace any scenario you don't find plausible.

### Validation

After editing, run:
```bash
cd /Users/ekansh/gridops
python tests/test_adversarial.py
```

Should still say `✅ No adversarial policy beats equal-weighted.` If your new shocks create a new exploit (some concentration policy now wins big), tune impact magnitudes down.

---

## Task 2 — Greenberg Terminal UI

**Directory:** `ui/` (currently empty)

**What it shows** (3 panels, 2-minute demo-ready):

```
┌─────────────────────────────┬─────────────────────────────┐
│  MACRO NEWS FEED            │  LLM MONOLOGUE (streaming)  │
│  Q1: tech_bet=green_leaps   │  <think>                    │
│  Q2: ...                    │  Hurricane hits REIT...     │
│  Q3: HURRICANE (ambiguous)  │  Refinery cut = OIL UP...   │
│  Q4: ...                    │  </think>                   │
│                             │  {"weights":[...]}          │
├─────────────────────────────┴─────────────────────────────┤
│  PORTFOLIO P&L (agent vs equal-weighted baseline)        │
│  ─── trained agent ──▲ +18.3%                            │
│  ─── baseline LLM ──▼  -7.1%                             │
│  ─── equal-weighted ▲  +8.9%                             │
│  [shock markers] [carbon bar: 87/120 kg]                 │
└───────────────────────────────────────────────────────────┘
```

**Input:** a JSON state file (below) that the Python side writes per-quarter. UI reads it, renders panels.

**Stack:** your choice — React + recharts, Next.js, Streamlit. Optimize for *what you can ship fastest in 4 hours*. I'll wire the Python-side JSON emitter to whatever format you design.

### JSON state schema (draft — adjust to what's easiest for your UI)

```json
{
  "episode_id": "abc123",
  "current_quarter": 5,
  "total_quarters": 12,
  "agent_model": "qwen3-4b-trained",
  "tech_bet": "green_leaps",

  "news_feed": [
    {"quarter": 0, "news": "Tech bet: green_leaps, Infra commit: 15%", "type": "action"},
    {"quarter": 1, "news": "Routine earnings season...", "type": "easy"},
    {"quarter": 2, "news": "Hurricane forecast...", "type": "ambiguous"},
    ...
  ],

  "think_stream": [
    {"quarter": 1, "text": "Standard earnings, maintain diversification..."},
    {"quarter": 2, "text": "Hurricane hits REIT directly. But refinery supply cut = OIL up..."}
  ],

  "weights_history": [
    [0.25, 0.20, 0.25, 0.05, 0.25],   // Q0 weights
    [0.25, 0.20, 0.25, 0.05, 0.25],   // Q1
    ...
  ],

  "nav_series": {
    "agent_real":     [1.0, 1.02, 1.04, 0.98, ...],
    "baseline_real":  [1.0, 1.01, 1.02, 0.95, ...],
    "agent_nominal":  [1.0, 1.03, 1.06, 1.02, ...]
  },

  "carbon": {
    "accumulated": 87.3,
    "cap": 120.0,
    "offsets_held": 15.0
  },

  "shock_markers": [
    {"quarter": 2, "id": "ambig_hurricane_gulf", "tier": "ambiguous"},
    {"quarter": 5, "id": "hard_rare_earth_rotation", "tier": "hard"}
  ],

  "interventions_used": [
    {"quarter": 0, "type": "tech_bet", "value": "green_leaps"},
    {"quarter": 0, "type": "infra_commit", "value": 0.15},
    {"quarter": 3, "type": "put_hedge", "value": 0.03}
  ]
}
```

### Visual priorities

1. **P&L chart** is the hero — make it beautiful, fluid animation as quarters advance, clear shock markers on the x-axis
2. **LLM monologue** streams token-by-token feel (we'll feed it per-quarter from training logs — you can fake the streaming by typing it out char-by-char in the UI)
3. **Carbon bar** visible at all times — judges need to see the sustainability constraint, not just returns

### Demo mode vs live mode

- **Demo mode**: read a pre-recorded JSON from file, play it back with animations. This is what we show judges.
- **Live mode** (optional stretch goal): websocket to the training backend. Not needed for submission.

Stick to demo mode — simpler + bulletproof.

### What to skip

- Controls / interactivity (no one clicks during demo)
- Real-time market data
- Backtesting features
- Everything except the 3 core panels + carbon bar

### What I'll provide on the backend

Once training produces checkpoints, I'll write a `dump_episode_to_json.py` that takes a model + seed and emits the state JSON. You render it. We meet at the JSON schema.

---

## Timing

| When | What |
|---|---|
| Apr 24 morning | Read this brief, clone repo, run `python tests/test_adversarial.py` to verify local setup works |
| Apr 24 late morning | Fill 11 placeholder shocks (~2 hrs), re-run adversarial test, commit |
| Apr 24 afternoon | UI scaffold — 3 panels with dummy JSON, no backend wiring yet (~4 hrs) |
| Apr 25 (onsite Day 1) | Wire UI to real training JSON, polish animations |
| Apr 26 morning (Day 2) | Record 2-minute silent demo video |
| Apr 26 afternoon | Submit |

---

## Questions to me (tag them in messages)

- Q10 from v0.6 doc: stagflation asset impacts (TECH -2%, OIL +3%, GREEN -3%, RE +0.5%, BONDS -1% additional real bleed) — still need your call on realism
- Q11: show `current_regime` label in obs, or make agent *infer* from news + inflation rate?
- UI stack — what ships fastest for you?

---

## Repo state at brief-writing time

- Branch: `round-2`
- 8 commits since branch creation, latest `44ebc91`
- All tests pass: smoke, adversarial (7/7 exploits killed), holdout
- 120 SFT traces generated and committed
- SFT warm-start currently training on RunPod RTX 5090
- Next critical path: verify SFT produces valid `<think>` + JSON format → Phase 1 GRPO → Phase 2 → Phase 3

The train will run mostly autonomously; your parallel work determines whether the demo (30% of judging) sells the env (40% of judging). Both tasks matter.

Let's cook.
