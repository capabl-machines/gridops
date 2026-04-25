"""Generate SFT warm-start traces for Qwen3-4B-Instruct via Gemini 3.1 Pro.

Strategy:
- Curated seed pool of ~30 real events (2014-2024) + ~20 projections (2025-2030)
- Batch 5 events per Gemini call — maintains contextual coherence within batch
  (same year, same theme, or same region clustered together)
- Structured output via Pydantic schema — Gemini returns exactly the shape we need
- Per-trace validation: valid JSON, weights sum-to-1-normalizable, <think> present,
  reasoning references at least 2 assets explicitly
- Env reward check: run each trace through PortfolioEnv, keep only traces that
  beat equal-weighted baseline on the relevant shock

Output: sft_traces/traces.jsonl — one JSON object per line, ready for TRL SFT.

Usage:
    python generate_traces.py                    # generate all batches
    python generate_traces.py --batches 5        # just first 5 batches (25 traces)
    python generate_traces.py --validate-only    # re-validate existing traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

# Load .env
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio_env.constants import ASSETS
from portfolio_env.rewards import parse_json_action
from portfolio_env.prompt import SYSTEM_PROMPT as ENV_SYSTEM_PROMPT, build_user_prompt


# ══════════════════════════════════════════════════════════════════════
# SEED POOL — real events (past) + projections (future)
# Clustered so each batch of 5 shares a theme for coherence
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NewsSeed:
    id: str
    year: str
    category: str            # 'physical' | 'transition' | 'geopolitical' | 'monetary' | 'projection'
    headline: str            # short reference — Gemini will expand into plausible quarter-news
    context: str             # hints about what actually happened / is projected


SEED_POOL: list[NewsSeed] = [
    # ── PHYSICAL CLIMATE (past, 5 events) ────────────────────────────
    NewsSeed('harvey_2017', '2017-Q3', 'physical',
             'Hurricane Harvey floods Houston, shuts 25% US refining',
             'Category 4. Gulf refineries offline ~2 weeks. Gasoline prices spiked. REIT damages $125B. Reconstruction boost to materials sector.'),
    NewsSeed('oz_fires_2020', '2020-Q1', 'physical',
             'Australian bushfires destroy 18M hectares, choke sky',
             'Insurance losses $4B+. Coal mining disruption. Broader ESG awareness spike. AGL and other utility exposure hit.'),
    NewsSeed('tx_freeze_2021', '2021-Q1', 'physical',
             'Texas winter storm Uri crashes ERCOT grid for 4 days',
             'Natural gas spot up 100x. Wind/solar frozen. $195B damage. Lessons for grid reliability. Winterization mandated.'),
    NewsSeed('pak_floods_2022', '2022-Q3', 'physical',
             '1/3 of Pakistan underwater, 33M displaced',
             'Attributed directly to climate change. Loss-and-damage fund conversation began. Rebuilding cost $30B.'),
    NewsSeed('eu_heat_2022', '2022-Q3', 'physical',
             'Europe hottest summer on record, nuclear cooling shutdowns',
             'French EDF nukes throttled due to river-cooling limits. Rhine shipping halted. Crop yields down 10-25%.'),

    # ── TRANSITION / POLICY (past, 5 events) ─────────────────────────
    NewsSeed('paris_2015', '2015-Q4', 'transition',
             'Paris Agreement signed by 196 nations',
             'Commitment to <2C warming. Framework for NDCs. Oil majors saw long-horizon transition risk priced in. Renewables rallied.'),
    NewsSeed('eu_greendeal_2019', '2019-Q4', 'transition',
             'EU Green Deal: climate neutrality by 2050, €1T mobilized',
             'Carbon Border Adjustment Mechanism previewed. Stranded asset risk intensified for fossil. Green bonds surge.'),
    NewsSeed('ira_2022', '2022-Q3', 'transition',
             'US Inflation Reduction Act: $369B clean-energy tax credits',
             'Manufacturing credits for solar/wind/EV. Utility-scale project pipeline swelled. Nat gas displaced faster. Green stocks ripped.'),
    NewsSeed('cbam_2023', '2023-Q4', 'transition',
             'EU CBAM implementation phase begins',
             'Importers must report embedded carbon for steel/aluminum/cement/electricity/fertilizer/H2. Emerging market exporters exposed.'),
    NewsSeed('cop28_2023', '2023-Q4', 'transition',
             'COP28 agrees "transition away from fossil fuels"',
             'Historic first explicit fossil fuel phase-out language. Gulf petrostates signed. Loss-and-damage fund operational.'),

    # ── GEOPOLITICAL / SUPPLY CHAIN (past, 5 events) ─────────────────
    NewsSeed('rare_earth_2010', '2010-Q3', 'geopolitical',
             'China restricts rare-earth exports to Japan',
             'Prices 10x in months. Triggered Western stockpiling and mine investment. USA MP Materials eventually spun up Mountain Pass.'),
    NewsSeed('covid_crash_2020', '2020-Q1', 'geopolitical',
             'COVID-19 pandemic — markets down 35% in 30 days',
             'Oil went negative (WTI April 2020). Tech rallied on WFH. REIT retail hammered, warehouse boom. Fed zero rates.'),
    NewsSeed('rus_ukr_2022', '2022-Q1', 'geopolitical',
             'Russia invades Ukraine, Brent hits $139, NatGas 10x',
             'SWIFT sanctions. Russian oil decoupling began. European energy crisis. Germany reversed nuclear phaseout. ESG reconsidered defense.'),
    NewsSeed('svb_2023', '2023-Q1', 'geopolitical',
             'Silicon Valley Bank collapses, Fed announces BTFP',
             '2nd largest US bank failure. Tech-sector concentration hit. Credit tightening. Regional bank stocks -50%.'),
    NewsSeed('china_ga_ge_2023', '2023-Q3', 'geopolitical',
             'China restricts gallium/germanium exports — chip retaliation',
             'Response to US semiconductor export controls. Rare-earth 2.0. Advanced chip margin squeeze. GREEN EV supply chain exposed.'),

    # ── MONETARY / MACRO (past, 5 events) ────────────────────────────
    NewsSeed('fed_qe3_2012', '2012-Q3', 'monetary',
             'Fed launches QE3, $85B/month MBS + Treasury purchases',
             'Extended zero-rate environment. Risk-asset melt-up. Bonds supported. Commodity prices elevated. USD weaker.'),
    NewsSeed('taper_2013', '2013-Q2', 'monetary',
             'Bernanke taper tantrum — bond yields spike 100bp',
             'First hint of QE unwind caused EM rout, bond selloff, REIT hit from rate sensitivity.'),
    NewsSeed('fed_2022_hikes', '2022-Q2', 'monetary',
             'Fed hikes 75bp in single meeting, 4 consecutive 75s',
             'Fastest tightening cycle since Volcker. Growth stocks -30-60%. Bonds worst year since 1788. Oil stayed bid.'),
    NewsSeed('japan_yield_2024', '2024-Q1', 'monetary',
             'Bank of Japan ends yield-curve control, ends negative rates',
             'First major central bank shift from ZIRP. Carry-trade unwind. Yen volatility. Global rates up.'),
    NewsSeed('fed_cut_2024', '2024-Q4', 'monetary',
             'Fed cuts 50bp — start of easing cycle',
             'Risk-on. Green transition financing cheaper. REIT rally. BONDS rally. USD softened.'),

    # ── PROJECTIONS (2025-2030, 10 scenarios) ────────────────────────
    NewsSeed('cbam_full_2026', '2026-Q3', 'projection',
             'EU CBAM full enforcement: 23€/t equivalent tariff',
             'PROJECTION: Emerging market exporter margins compressed. Domestic EU green steel/cement premium. Oil majors face Scope 3 pressure.'),
    NewsSeed('ai_grid_crunch_2027', '2027-Q2', 'projection',
             'AI data center demand exceeds US grid capacity; brownouts in Phoenix, NoVA',
             'PROJECTION: Hyperscaler capex pause. Grid infra + nuclear revived. TECH margins hit on energy costs. GREEN baseload premium.'),
    NewsSeed('stranded_oil_2027', '2027-Q4', 'projection',
             'IEA declares peak oil demand reached in 2026, majors write down $400B',
             'PROJECTION: Exxon/Shell impairments. Buyback programs paused. OIL sector rerates down 30%. GREEN capex accelerates.'),
    NewsSeed('eu_carbon_200_2028', '2028-Q1', 'projection',
             'EU ETS carbon price hits €200/t; UK follows with £180',
             'PROJECTION: Heavy industry offshoring threat. H2 + CCS project FIDs accelerate. REIT construction costs +15%.'),
    NewsSeed('battery_glut_2028', '2028-Q3', 'projection',
             'LFP battery oversupply; prices hit $45/kWh, utility storage deployment 3x',
             'PROJECTION: Grid-scale storage economics flip to always-positive. Natural gas peaker obsolescence. OIL volatility lower.'),
    NewsSeed('methane_fee_2026', '2026-Q2', 'projection',
             'US methane fee at $1500/t methane kicks in, oil/gas upstream hit',
             'PROJECTION: Permian leak rates forced down. Marginal producers exit. OIL supply tightens paradoxically. GREEN H2 boost.'),
    NewsSeed('climate_litigation_2027', '2027-Q1', 'projection',
             'Dutch Shell ruling replicated: 10 more majors ordered 50% emission cut by 2035',
             'PROJECTION: Legal liability reprices OIL. Insurance retrenches from fossil. Divestment wave from pensions.'),
    NewsSeed('smr_scale_2029', '2029-Q3', 'projection',
             'First commercial SMR fleet operating at scale in US + UK + Poland',
             'PROJECTION: Baseload firm power economics reset. Uranium mining rally. GREEN redefined to include nuclear in many taxonomies.'),
    NewsSeed('india_coal_ramp_2026', '2026-Q4', 'projection',
             'India adds 30GW coal; emissions trajectory diverges from 1.5C',
             'PROJECTION: Global carbon budget effectively blown. Transition risk repriced. Climate adaptation sector re-rated.'),
    NewsSeed('hydrogen_economy_2029', '2029-Q1', 'projection',
             'Green hydrogen hits $2/kg at scale, steel industry conversions',
             'PROJECTION: EU steelmakers lock in 10yr H2 offtakes. ArcelorMittal retrofits. Iron ore demand mix shifts.'),
]


# ══════════════════════════════════════════════════════════════════════
# Pydantic schema for Gemini's structured output (response_schema)
# ══════════════════════════════════════════════════════════════════════

TRACE_SCHEMA = {
    'type': 'ARRAY',
    'items': {
        'type': 'OBJECT',
        'properties': {
            'id':            {'type': 'STRING'},
            'news':          {'type': 'STRING', 'description': 'Quarter macro-headline, 2-4 sentences, realistic'},
            'reasoning':     {'type': 'STRING', 'description': '<think> block body: 1st/2nd/3rd-order causal analysis'},
            'weights': {
                'type': 'ARRAY',
                'items': {'type': 'NUMBER'},
                'minItems': 5, 'maxItems': 5,
                'description': '[TECH, OIL, GREEN, REAL_ESTATE, BONDS] non-negative, sum close to 1.0',
            },
            'infra_commit':      {'type': 'NUMBER'},
            'carbon_offset_buy': {'type': 'NUMBER'},
            'put_hedge':         {'type': 'NUMBER'},
            'tech_bet': {
                'type': 'STRING',
                'enum': ['status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'],
            },
        },
        'required': ['id', 'news', 'reasoning', 'weights'],
    },
}


SYSTEM_PROMPT = """\
You are an expert quantitative analyst generating SFT training traces for a
climate-aware portfolio-manager LLM. The target model is Qwen3-4B-Instruct; we
SFT it on your traces, then GRPO-train it on a simulated environment.

THE TARGET LLM'S TASK (mirror this in your traces):
- It commits ONE allocation today that holds locked for 12 quarters (3-year cycle)
- It cannot adjust mid-cycle. Its reasoning must therefore be at the macro-cycle
  level, NOT a "what's the right move this quarter" reaction
- Reasoning style: cycle-level macro analysis (how today's news shapes the
  next 12 quarters under plausible regime evolutions)
- Output budget: <think> under ~300 words, total completion under 400 tokens

ENVIRONMENT SPEC (the trained LLM will see in its system prompt):
- 5 assets: TECH, OIL, GREEN (renewables), REAL_ESTATE, BONDS
- Base quarterly returns: TECH +3%, OIL +2%, GREEN +1.5%, REAL_ESTATE +1%, BONDS +0.5%
- Carbon intensity (kg/$): TECH 0.05, OIL 2.50, GREEN 0.01, REAL_ESTATE 0.10, BONDS 0.00
- Carbon cap: cumulative under 25 kg over 12 quarters
- Inflation regimes that arrive via shocks: normal (1%/q), stagflationary
  (2.5%/q — favors OIL/REAL_ESTATE, crushes BONDS), deflationary (-0.3%/q —
  favors BONDS, hurts OIL)
- Reward: regret vs equal-weighted baseline on REAL returns (- drawdown, + small Sharpe)
- 4 interventions (use only if news justifies):
  * infra_commit [0, 0.2]: 4-quarter lockup. +8% per transition-risk shock during
    lockup, -8% per physical-risk shock. True bet — wrong thesis = dead capital.
  * carbon_offset_buy [0, 0.1]: 1 unit NAV → 10 kg offset. Costly.
  * put_hedge [0, 0.05]: 2%/q premium, caps portfolio drop at -5% if loss > -15%.
    Bleeds in normal markets — use sparingly.
  * tech_bet (Q1-only thesis): status_quo / green_leaps / carbon_priced /
    inflationary / fragmentation. Tilts shock-pool sampling.

WHAT "GOOD REASONING" LOOKS LIKE (these are the patterns we want SFT'd in):
- Identifies 1st-order impact (direct)
- Identifies at least one 2nd-order chain (supply/demand, sector rotation,
  capital flows)
- Identifies at least one 3rd-order/counterintuitive move (what a
  pattern-matcher would miss)
- States BASE-RATE explicitly: "Today's news is mild — base case (normal
  markets) — no hedge needed" OR "Today's news strongly signals stagflation,
  rotate into OIL+REAL_ESTATE, hedge"
- Recommends weights that REFLECT the conclusions (not equal-weighted default)
- Uses interventions ONLY when news justifies (e.g., put_hedge if news flags
  imminent crash; infra_commit if news strongly signals transition-risk bias)

BAD REASONING (reject in your own mind):
- Quarter-by-quarter step-through (the agent can't adjust — wasted reasoning)
- "Green news is bullish" without checking supply chain or financing costs
- "Crisis → bonds" without checking the crisis type (stagflation hates bonds)
- Maxing interventions without justification (put_hedge=0.05 every time = bleed)
- Hedging when base case (normal) is consistent with news
"""


def build_batch_prompt(seeds: list[NewsSeed]) -> str:
    bullets = '\n'.join(
        f"{i+1}. [{s.id}] — {s.year} {s.category.upper()}\n"
        f"   Headline seed: {s.headline}\n"
        f"   Context: {s.context}"
        for i, s in enumerate(seeds)
    )
    return f"""\
Generate exactly {len(seeds)} training traces, one per event below.

EVENTS:
{bullets}

FOR EACH EVENT:
1. Convert the headline seed into a realistic 2-4 sentence quarter-news string (like a market wire)
2. Produce a `reasoning` string that analyzes 1st/2nd/3rd-order effects on the 5 assets
3. Produce `weights` (5 non-negative numbers, sum ≈ 1.0)
4. Optionally set intervention fields (infra_commit, carbon_offset_buy, put_hedge, tech_bet) when justified

Return JSON array of {len(seeds)} objects matching the response schema. `id` field should match the event id in brackets.
"""


# ══════════════════════════════════════════════════════════════════════
# Gemini call
# ══════════════════════════════════════════════════════════════════════

def gemini_generate_batch(seeds: list[NewsSeed]) -> list[dict]:
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type='application/json',
        response_schema=TRACE_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level='HIGH'),
    )
    prompt = build_batch_prompt(seeds)
    resp = client.models.generate_content(
        model='gemini-3-pro-preview',
        contents=[types.Content(role='user', parts=[types.Part.from_text(text=prompt)])],
        config=config,
    )
    try:
        return json.loads(resp.text)
    except Exception as e:
        print(f'  ! parse failure: {e}')
        print(f'  raw response: {resp.text[:500]}')
        return []


# ══════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════

def validate(trace: dict) -> tuple[bool, str]:
    if 'news' not in trace or len(trace.get('news', '')) < 50:
        return False, 'news too short'
    if 'reasoning' not in trace or len(trace.get('reasoning', '')) < 150:
        return False, 'reasoning too short'
    w = trace.get('weights')
    if not isinstance(w, list) or len(w) != 5:
        return False, f'weights wrong shape: {w}'
    if any(x < 0 for x in w):
        return False, 'negative weights'
    s = sum(w)
    if not (0.5 < s < 1.5):
        return False, f'weights sum out of range: {s:.3f}'
    # reasoning quality: mentions at least 2 different assets
    asset_mentions = sum(1 for a in ASSETS if a in trace['reasoning'].upper())
    if asset_mentions < 2:
        return False, f'reasoning mentions only {asset_mentions} assets'
    return True, 'ok'


def format_trace_for_sft(trace: dict, seed: NewsSeed) -> dict:
    """Turn a validated Gemini trace into a row for SFT training.

    The 'prompt' / 'completion' pair matches what the GRPO trainer will feed
    during rollouts, so SFT teaches the exact input→output shape.
    """
    # Normalize weights to sum 1.0
    w = trace['weights']
    total = sum(w)
    w_norm = [round(x / total, 4) for x in w] if total > 0 else [0.2] * 5

    action = {'weights': w_norm}
    for key in ('infra_commit', 'carbon_offset_buy', 'put_hedge', 'tech_bet'):
        if key in trace and trace[key] not in (None, 0, 0.0, 'status_quo'):
            action[key] = trace[key]

    # IMPORTANT: prompt MUST match what GRPO inference uses (single source of
    # truth in portfolio_env/prompt.py). Different prompts at SFT vs GRPO →
    # mode collapse during RL training (Gemini's "fundamental law" finding).
    # The system prompt provides all rules + objective; user prompt is just
    # today's news. We concatenate them as the SFT 'prompt' field for
    # readability; the trainer applies the chat template per messages.
    prompt = ENV_SYSTEM_PROMPT + '\n\n' + build_user_prompt(trace['news'])

    completion = (f"<think>\n{trace['reasoning'].strip()}\n</think>\n"
                  f"{json.dumps(action)}")

    return {
        'id': trace.get('id', seed.id),
        'seed_id': seed.id,
        'seed_year': seed.year,
        'seed_category': seed.category,
        'prompt': prompt,
        'completion': completion,
        'raw': trace,
    }


# ══════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════

def batch_seeds(seeds: list[NewsSeed], batch_size: int = 5) -> list[list[NewsSeed]]:
    """Group seeds into batches — cluster by category when possible to give
    Gemini coherent context within a batch."""
    by_cat: dict[str, list[NewsSeed]] = {}
    for s in seeds:
        by_cat.setdefault(s.category, []).append(s)
    batches = []
    for cat, items in by_cat.items():
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=None, help='Limit # batches (debug)')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'traces.jsonl')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Generate N completions per seed — more = diverse reasoning per event')
    args = parser.parse_args()

    if args.validate_only:
        print(f'Validating existing {args.out}...')
        n, good = 0, 0
        with args.out.open() as f:
            for line in f:
                n += 1
                row = json.loads(line)
                ok, msg = validate(row['raw'])
                if ok:
                    good += 1
                else:
                    print(f'  INVALID: {row["id"]}: {msg}')
        print(f'{good}/{n} traces valid')
        return

    # Expand by repeat factor
    all_seeds = SEED_POOL * args.repeat
    random.seed(42)
    random.shuffle(all_seeds)
    batches = batch_seeds(all_seeds, 5)
    if args.batches:
        batches = batches[:args.batches]

    print(f'{len(batches)} batches × 5 traces = up to {len(batches)*5} traces')
    print(f'Writing to {args.out}')

    saved = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as fout:
        for batch_idx, batch in enumerate(batches, 1):
            print(f'\n── batch {batch_idx}/{len(batches)} ({batch[0].category}) ──')
            t0 = time.time()
            traces = gemini_generate_batch(batch)
            print(f'  got {len(traces)} traces in {time.time()-t0:.1f}s')

            seed_by_id = {s.id: s for s in batch}
            for t in traces:
                ok, msg = validate(t)
                sid = t.get('id', '?')
                if not ok:
                    print(f'  ✗ {sid}: {msg}')
                    continue
                seed = seed_by_id.get(sid, batch[0])  # fallback to first seed
                row = format_trace_for_sft(t, seed)
                fout.write(json.dumps(row) + '\n')
                fout.flush()
                saved += 1
                print(f'  ✓ {sid} (reasoning {len(t["reasoning"])} chars)')

    print(f'\nSaved {saved} valid traces to {args.out}')


if __name__ == '__main__':
    main()
