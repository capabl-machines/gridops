"""Generate causal-discipline SFT traces through OpenRouter DeepSeek V4 Pro.

This is a sibling to the Gemini generator. It keeps all older datasets intact
and writes a new JSONL compatible with the existing Qwen2.5 SFT loader:

    {"id", "seed_id", "seed_year", "seed_category", "prompt", "completion", "raw"}

Required env:
    OPENROUTER_API_KEY, HFTOKEN, HF2_TOKEN, HF_API_TOKEN, or HF_TOKEN
    API_BASE_URL, API2_BASE_URL, or OPENROUTER_BASE_URL

Usage:
    uv run python sft_traces/generate_openrouter_deepseek_traces.py \
        --per-scenario 6 \
        --batch-size 2 \
        --out sft_traces/causal_discipline/deepseek_v4_causal_discipline_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent.parent
if (ROOT / '.env').exists():
    for line in (ROOT / '.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

sys.path.insert(0, str(ROOT))

from portfolio_env.constants import ASSETS
from portfolio_env.prompt import SYSTEM_PROMPT as ENV_SYSTEM_PROMPT, build_user_prompt


DEFAULT_MODEL = 'deepseek/deepseek-v4-pro'


@dataclass(frozen=True)
class Scenario:
    name: str
    regime: str
    brief: str
    preferred_behavior: str
    forbidden_claims: tuple[str, ...] = ()
    preferred_tilts: tuple[str, ...] = ()


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name='yen_carry_unwind',
        regime='fx_carry_unwind',
        brief='Yen carry trade unwind, rapid yen appreciation or disorderly FX volatility, forced deleveraging, global risk-off.',
        preferred_behavior='Treat primarily as liquidity/risk-off. Favor BONDS and a small put hedge; cut TECH/GREEN/REAL_ESTATE risk. Do not overweight OIL unless energy supply or inflation is explicitly present.',
        forbidden_claims=('oil demand surges', 'energy demand surges', 'stagflation unless inflation is explicitly stated'),
        preferred_tilts=('BONDS up', 'put_hedge small', 'TECH down', 'GREEN down', 'REAL_ESTATE down'),
    ),
    Scenario(
        name='bitcoin_legalization',
        regime='crypto_risk_on',
        brief='Major nations legalize or regulate Bitcoin/crypto access, institutional custody expands, but no broad macro shock.',
        preferred_behavior='Mild risk-on/TECH support. Avoid pretending crypto legalization is a climate or inflation regime shift. Keep diversified allocation.',
        forbidden_claims=('systemic inflation shock', 'oil spike', 'green transition shock'),
        preferred_tilts=('TECH modest up', 'BONDS normal', 'no hedge'),
    ),
    Scenario(
        name='political_scandal_noise',
        regime='political_noise',
        brief='A single politician or celebrity-style scandal with no clear policy, trade, fiscal, monetary, or energy channel.',
        preferred_behavior='Explicitly call it non-systemic and stay near base-rate diversified allocation.',
        forbidden_claims=('global contagion', 'regime shift', 'commodity shock'),
        preferred_tilts=('near base-rate', 'no hedge', 'status_quo'),
    ),
    Scenario(
        name='sovereign_capital_controls',
        regime='sovereign_instability',
        brief='Large emerging-market government imposes capital controls, banking holidays, or abrupt foreign-investor restrictions.',
        preferred_behavior='Distinguish regional sovereign stress from global regime shift. Reduce risk assets, use BONDS/hedge; consider fragmentation if supply chains/trade are involved.',
        forbidden_claims=('all global real estate collapses', 'green transition suddenly accelerates without policy'),
        preferred_tilts=('BONDS up', 'put_hedge small', 'TECH modest down'),
    ),
    Scenario(
        name='bank_credit_stress',
        regime='credit_stress',
        brief='Regional banks, private credit funds, or commercial real estate lenders report sudden losses and funding stress.',
        preferred_behavior='Treat as credit/liquidity stress. Avoid REAL_ESTATE, trim TECH/GREEN beta, favor BONDS and small hedge.',
        forbidden_claims=('oil demand surge', 'green subsidy boom'),
        preferred_tilts=('REAL_ESTATE down', 'BONDS up', 'put_hedge small'),
    ),
    Scenario(
        name='energy_supply_shock',
        regime='geopolitical_energy_supply',
        brief='Explicit oil/gas supply disruption from war, shipping closure, sanctions, or refinery outages with inflation pressure.',
        preferred_behavior='This is one of the few cases where OIL overweight and inflationary thesis are justified. Keep carbon cap in mind and avoid excessive OIL.',
        forbidden_claims=('deflationary bond rally as primary effect',),
        preferred_tilts=('OIL up but carbon-aware', 'BONDS down if inflation', 'put_hedge optional'),
    ),
    Scenario(
        name='ai_capex_efficiency',
        regime='base_rate',
        brief='AI model efficiency improves sharply; hyperscalers reduce data-center capex and power-demand forecasts.',
        preferred_behavior='Separate software TECH winners from data-center real estate and power/green buildout losers.',
        forbidden_claims=('all tech is bad', 'green transition accelerates because AI improved'),
        preferred_tilts=('TECH up', 'GREEN down modest', 'REAL_ESTATE down modest'),
    ),
    Scenario(
        name='carbon_policy_real',
        regime='transition_policy',
        brief='Concrete carbon price, CBAM enforcement, clean-energy subsidy, or offset-fraud enforcement changes economics.',
        preferred_behavior='Favor GREEN and carbon_priced thesis when policy is concrete. Avoid carbon offsets if offset integrity is questioned.',
        forbidden_claims=('oil benefits from carbon enforcement',),
        preferred_tilts=('GREEN up', 'OIL down', 'carbon_priced'),
    ),
    Scenario(
        name='physical_insurance_retreat',
        regime='physical_risk',
        brief='Insurers exit climate-exposed property markets, mortgage availability freezes, coastal property values reprice.',
        preferred_behavior='Avoid REAL_ESTATE and infra_commit; BONDS may benefit from flight to quality; GREEN only modest adaptation tailwind.',
        forbidden_claims=('infra is safe during physical damage', 'all green assets immediately moon'),
        preferred_tilts=('REAL_ESTATE zero/low', 'BONDS up', 'infra_commit zero'),
    ),
    Scenario(
        name='deflation_global_slump',
        regime='deflation',
        brief='Manufacturing PMI collapse, commodity glut, export prices fall, yields plunge, global demand weakens.',
        preferred_behavior='Deflation is the exception where BONDS are strongly favored. Avoid OIL/REAL_ESTATE; keep only modest TECH.',
        forbidden_claims=('stagflation', 'oil inflation hedge'),
        preferred_tilts=('BONDS high', 'OIL low', 'REAL_ESTATE low'),
    ),
    Scenario(
        name='boring_micro_news',
        regime='base_rate',
        brief='Company-level product launch, minor court ruling, celebrity dispute, local event, or PR headline with no portfolio-wide channel.',
        preferred_behavior='Do not manufacture macro causality. State no portfolio-wide channel and stay diversified.',
        forbidden_claims=('massive global rotation', 'regime shift', 'systemic shock'),
        preferred_tilts=('base-rate diversified', 'no intervention'),
    ),
    Scenario(
        name='stagflation_real',
        regime='stagflation',
        brief='Explicit inflation surprise, hawkish central bank, yields up, supply constraints, real growth pressure.',
        preferred_behavior='Favor OIL/REAL_ESTATE selectively, reduce BONDS and long-duration TECH/GREEN. Use inflationary thesis.',
        forbidden_claims=('flight-to-safety bonds dominate despite explicit inflation shock',),
        preferred_tilts=('OIL up', 'BONDS down', 'inflationary'),
    ),
)


SYSTEM = ENV_SYSTEM_PROMPT + """

------
You are generating second-stage SFT examples for CarbonAlpha.

The goal is causal discipline, not dramatic storytelling. The current model
sometimes invents unsupported 2nd/3rd-order effects. Your traces must teach it
to be explicit, bounded, and humble.

Return ONLY a JSON array. No markdown, no prose outside JSON.

Each JSON object must contain:
- id: string
- scenario: string
- news: fresh plausible quarter-news, 2-4 sentences
- regime: one of the requested target regimes
- direct_channel: grounded only in the news
- second_order_channel: only if justified, otherwise say "limited"
- not_assuming: a tempting unsupported causal leap you are explicitly rejecting
- reasoning: under 300 words, with exact labels:
  "Base-rate/regime:", "Direct channel:", "Second-order:", "Not assuming:", "Allocation:"
- weights: [TECH, OIL, GREEN, REAL_ESTATE, BONDS], non-negative, sum exactly 1.0
- infra_commit: 0.0 to 0.2
- carbon_offset_buy: 0.0 to 0.1
- put_hedge: 0.0 to 0.05
- tech_bet: status_quo | green_leaps | carbon_priced | inflationary | fragmentation

Rules:
- If the news has no clear portfolio-wide channel, say so and stay near
  base-rate; do not manufacture drama.
- Use put_hedge sparingly, but do use a small hedge for genuine liquidity,
  credit, or crash-risk shocks.
- Do not use infra_commit during physical-risk or generic liquidity shocks.
- Keep OIL high only for explicit energy supply/inflation shocks or explicit
  sector-rotation examples. Do not claim oil demand surges during generic
  risk-off unless the news states it.
"""


def build_specs(per_scenario: int, seed: int) -> list[tuple[Scenario, int]]:
    specs = [(scenario, i + 1) for scenario in SCENARIOS for i in range(per_scenario)]
    rng = random.Random(seed)
    rng.shuffle(specs)
    return specs


def chunked(items: list[tuple[Scenario, int]], size: int) -> list[list[tuple[Scenario, int]]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_batch_prompt(batch: list[tuple[Scenario, int]]) -> str:
    lines = []
    for idx, (scenario, variant) in enumerate(batch, 1):
        forbidden = '; '.join(scenario.forbidden_claims) or 'none'
        tilts = '; '.join(scenario.preferred_tilts) or 'balanced'
        lines.append(
            f'{idx}. id={scenario.name}_deepseek_v{variant:03d}\n'
            f'   scenario={scenario.name}\n'
            f'   target_regime={scenario.regime}\n'
            f'   brief={scenario.brief}\n'
            f'   preferred_behavior={scenario.preferred_behavior}\n'
            f'   forbidden_claims={forbidden}\n'
            f'   preferred_tilts={tilts}'
        )
    return (
        f'Generate exactly {len(batch)} CarbonAlpha causal-discipline SFT traces.\n\n'
        'Preserve every id and scenario exactly. Return a JSON array only.\n\n'
        'REQUESTED ITEMS:\n' + '\n\n'.join(lines)
    )


def openrouter_key() -> str:
    key = (
        os.environ.get('OPENROUTER_API_KEY')
        or os.environ.get('OPENROUTER_KEY')
        or os.environ.get('HFTOKEN')
        or os.environ.get('HF_TOKEN')
        or os.environ.get('HF_API_TOKEN')
        or os.environ.get('HF2_TOKEN')
    )
    if not key:
        raise RuntimeError('OpenRouter key required: set OPENROUTER_API_KEY, HFTOKEN, HF2_TOKEN, HF_API_TOKEN, or HF_TOKEN')
    return key


def openrouter_url() -> str:
    base = (
        os.environ.get('OPENROUTER_BASE_URL')
        or os.environ.get('API_BASE_URL')
        or os.environ.get('API2_BASE_URL')
        or 'https://openrouter.ai/api/v1'
    ).rstrip('/')
    if base.endswith('/chat/completions'):
        return base
    return f'{base}/chat/completions'


def extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:].strip()
    start = text.find('[')
    end = text.rfind(']')
    if start < 0 or end <= start:
        raise ValueError(f'no JSON array found: {text[:300]}')
    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError('response JSON is not a list')
    return data


def openrouter_generate_batch(
    batch: list[tuple[Scenario, int]],
    *,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': build_batch_prompt(batch)},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'response_format': {'type': 'json_object'},
    }
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        openrouter_url(),
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/77ethers/CarbonAlpha',
            'X-Title': 'CarbonAlpha Trace Generator',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'OpenRouter HTTP {exc.code}: {detail}') from exc
    data = json.loads(raw)
    content = data['choices'][0]['message']['content']
    parsed = json.loads(content)
    if isinstance(parsed, dict):
        if 'id' in parsed:
            return [parsed]
        for key in ('traces', 'items', 'data', 'examples'):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        raise ValueError(f'JSON object did not contain trace list or trace id: {content[:300]}')
    if isinstance(parsed, list):
        return parsed
    return extract_json_array(content)


def validate_trace(trace: dict[str, Any], scenario_by_id: dict[str, Scenario]) -> tuple[bool, str]:
    trace_id = str(trace.get('id'))
    scenario = scenario_by_id.get(trace_id)
    if scenario is None:
        return False, f'unexpected id {trace_id}'
    if trace.get('scenario') != scenario.name:
        return False, f'scenario mismatch {trace.get("scenario")} != {scenario.name}'
    if trace.get('regime') != scenario.regime:
        return False, f'regime mismatch {trace.get("regime")} != {scenario.regime}'
    if len(str(trace.get('news', ''))) < 60:
        return False, 'news too short'
    reasoning = str(trace.get('reasoning', ''))
    if len(reasoning) < 220:
        return False, 'reasoning too short'
    required = ['Base-rate/regime:', 'Direct channel:', 'Second-order:', 'Not assuming:', 'Allocation:']
    missing = [phrase for phrase in required if phrase not in reasoning]
    if missing:
        return False, f'missing reasoning labels: {missing}'

    lower_reasoning = reasoning.lower()
    not_assuming_text = str(trace.get('not_assuming', '')).lower()
    claim_text = lower_reasoning
    for line in reasoning.splitlines():
        if line.strip().lower().startswith('not assuming:'):
            claim_text = claim_text.replace(line.lower(), '')
    for forbidden in scenario.forbidden_claims:
        forbidden_lower = forbidden.lower()
        if forbidden and forbidden_lower in claim_text and forbidden_lower not in not_assuming_text:
            return False, f'contains forbidden claim: {forbidden}'

    weights = trace.get('weights')
    if not isinstance(weights, list) or len(weights) != 5:
        return False, f'bad weights shape: {weights}'
    try:
        weights = [float(x) for x in weights]
    except Exception:
        return False, f'non-numeric weights: {weights}'
    if any(x < 0 for x in weights):
        return False, 'negative weights'
    if not (0.98 <= sum(weights) <= 1.02):
        return False, f'weights sum {sum(weights):.3f}'
    asset_mentions = sum(1 for asset in ASSETS if asset in reasoning.upper())
    min_asset_mentions = 2 if scenario.regime in {'base_rate', 'political_noise', 'crypto_risk_on'} else 3
    if asset_mentions < min_asset_mentions:
        return False, f'reasoning mentions only {asset_mentions} assets'
    for key, lo, hi in (
        ('infra_commit', 0.0, 0.2),
        ('carbon_offset_buy', 0.0, 0.1),
        ('put_hedge', 0.0, 0.05),
    ):
        try:
            value = float(trace.get(key, 0.0) or 0.0)
        except Exception:
            return False, f'{key} not numeric'
        if not (lo <= value <= hi):
            return False, f'{key}={value} out of range'
    if trace.get('tech_bet') not in ('status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'):
        return False, f'bad tech_bet: {trace.get("tech_bet")}'
    return True, 'ok'


def assemble_sft_row(trace: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    weights = [float(x) for x in trace['weights']]
    total = sum(weights)
    norm = [round(x / total, 4) for x in weights] if total > 0 else [0.2] * 5
    action = {
        'weights': norm,
        'infra_commit': round(float(trace.get('infra_commit') or 0.0), 4),
        'carbon_offset_buy': round(float(trace.get('carbon_offset_buy') or 0.0), 4),
        'put_hedge': round(float(trace.get('put_hedge') or 0.0), 4),
        'tech_bet': trace.get('tech_bet', 'status_quo'),
    }
    prompt = ENV_SYSTEM_PROMPT + '\n\n' + build_user_prompt(str(trace['news']))
    completion = f"<think>\n{str(trace['reasoning']).strip()}\n</think>\n{json.dumps(action)}"
    return {
        'id': trace['id'],
        'seed_id': scenario.name,
        'seed_year': 'deepseek-v4-causal-discipline-v1',
        'seed_category': scenario.regime,
        'prompt': prompt,
        'completion': completion,
        'raw': {
            'id': trace['id'],
            'scenario': scenario.name,
            'news': trace['news'],
            'regime': trace['regime'],
            'direct_channel': trace['direct_channel'],
            'second_order_channel': trace['second_order_channel'],
            'not_assuming': trace['not_assuming'],
            'reasoning': str(trace['reasoning']).strip(),
            'weights': norm,
            'infra_commit': action['infra_commit'],
            'carbon_offset_buy': action['carbon_offset_buy'],
            'put_hedge': action['put_hedge'],
            'tech_bet': action['tech_bet'],
            'generator_model': DEFAULT_MODEL,
            'preferred_behavior': scenario.preferred_behavior,
            'forbidden_claims': list(scenario.forbidden_claims),
            'preferred_tilts': list(scenario.preferred_tilts),
        },
    }


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        try:
            done.add(json.loads(line)['id'])
        except Exception:
            continue
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-scenario', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--out', default=str(ROOT / 'sft_traces/causal_discipline/deepseek_v4_causal_discipline_v1.jsonl'))
    parser.add_argument('--model', default=os.environ.get('OPENROUTER_MODEL') or os.environ.get('MODEL2_NAME') or DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sleep-s', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.35)
    parser.add_argument('--max-tokens', type=int, default=4000)
    parser.add_argument('--timeout-s', type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_specs(args.per_scenario, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        done = load_done_ids(out_path)
        specs = [(s, v) for s, v in specs if f'{s.name}_deepseek_v{v:03d}' not in done]
        print(f'Resume: {len(done)} existing ids, {len(specs)} remaining.', flush=True)

    print(
        f'Generating {len(specs)} DeepSeek/OpenRouter traces '
        f'({len(SCENARIOS)} scenarios x {args.per_scenario}); model={args.model}',
        flush=True,
    )
    print(f'Output: {out_path}', flush=True)

    if args.dry_run:
        for scenario, variant in specs:
            print(f'DRY {scenario.name}_deepseek_v{variant:03d} -> {scenario.regime}')
        return

    api_key = openrouter_key()
    mode = 'a' if args.resume and out_path.exists() else 'w'
    saved = 0
    failed: list[tuple[str, str]] = []
    with out_path.open(mode) as fh:
        for batch_idx, batch in enumerate(chunked(specs, args.batch_size), 1):
            print(f'\nBatch {batch_idx}: {len(batch)} traces', flush=True)
            scenario_by_id = {f'{scenario.name}_deepseek_v{variant:03d}': scenario for scenario, variant in batch}
            try:
                generated = None
                last_exc: Exception | None = None
                for attempt in range(4):
                    try:
                        generated = openrouter_generate_batch(
                            batch,
                            model=args.model,
                            api_key=api_key,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            timeout_s=args.timeout_s,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        if 'HTTP 429' not in str(exc) or attempt == 3:
                            raise
                        wait_s = 3 * (attempt + 1)
                        print(f'  rate limited; retrying in {wait_s}s', flush=True)
                        time.sleep(wait_s)
                if generated is None:
                    raise last_exc or RuntimeError('OpenRouter returned no traces')
            except Exception as exc:
                print(f'  FAIL batch call: {type(exc).__name__}: {exc}', flush=True)
                for scenario, variant in batch:
                    failed.append((f'{scenario.name}_deepseek_v{variant:03d}', f'{type(exc).__name__}: {exc}'))
                continue

            for trace in generated:
                trace_id = str(trace.get('id'))
                scenario = scenario_by_id.get(trace_id)
                ok, msg = validate_trace(trace, scenario_by_id)
                if not ok or scenario is None:
                    failed.append((trace_id, msg))
                    print(f'  FAIL {trace_id}: {msg}', flush=True)
                    continue
                fh.write(json.dumps(assemble_sft_row(trace, scenario)) + '\n')
                fh.flush()
                saved += 1
                print(f'  OK {trace_id}', flush=True)

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    print(f'\nSaved {saved}/{len(specs)} traces -> {out_path}', flush=True)
    if failed:
        fail_path = out_path.with_suffix(out_path.suffix + '.failures.json')
        fail_path.write_text(json.dumps(failed, indent=2))
        print(f'Failures: {len(failed)} -> {fail_path}', flush=True)


if __name__ == '__main__':
    main()
