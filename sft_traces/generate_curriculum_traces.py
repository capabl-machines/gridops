"""Generate curriculum-balanced SFT traces with Gemini, 10 traces per API call.

Difficulty source of truth:
  easy       -> portfolio_env.shocks.Shock.tier == "easy"       -> Phase 1
  medium     -> Shock.tier == "ambiguous"                       -> Phase 2
  hard       -> Shock.tier == "hard"                             -> Phase 3

The output row schema intentionally matches `merged_v6_aligned.jsonl`:
`id`, `seed_id`, `seed_year`, `seed_category`, `prompt`, `completion`, `raw`.
Curriculum labels are stored under `raw` so existing SFT loaders continue to
work unchanged.

Usage:
    python sft_traces/generate_curriculum_traces.py \
        --easy 100 --medium 250 --hard 250 \
        --out sft_traces/curriculum_600.jsonl

Smoke:
    python sft_traces/generate_curriculum_traces.py \
        --easy 3 --medium 4 --hard 3 --out /tmp/curriculum_smoke.jsonl
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


ROOT = Path(__file__).parent.parent
if (ROOT / '.env').exists():
    for line in (ROOT / '.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(ROOT))

from portfolio_env.constants import ASSETS
from portfolio_env.prompt import SYSTEM_PROMPT as ENV_SYSTEM_PROMPT, build_user_prompt
from portfolio_env.shocks import AMBIGUOUS_SHOCKS, EASY_SHOCKS, HARD_SHOCKS, Shock


DIFFICULTY_TO_TIER = {
    'easy': 'easy',
    'medium': 'ambiguous',
    'hard': 'hard',
}

TIER_TO_PHASE = {
    'easy': 1,
    'ambiguous': 2,
    'hard': 3,
}

TIER_TO_DIFFICULTY = {
    'easy': 'easy',
    'ambiguous': 'medium',
    'hard': 'hard',
}

SHOCKS_BY_DIFFICULTY = {
    'easy': [s for s in EASY_SHOCKS if 'PLACEHOLDER' not in s.id],
    'medium': [s for s in AMBIGUOUS_SHOCKS if 'PLACEHOLDER' not in s.id],
    'hard': [s for s in HARD_SHOCKS if 'PLACEHOLDER' not in s.id],
}


@dataclass(frozen=True)
class TraceSpec:
    id: str
    difficulty: str
    variant: int
    shock: Shock


TRACE_SCHEMA = {
    'type': 'ARRAY',
    'items': {
        'type': 'OBJECT',
        'properties': {
            'id': {'type': 'STRING'},
            'source_shock_id': {'type': 'STRING'},
            'difficulty_tier': {
                'type': 'STRING',
                'enum': ['easy', 'medium', 'hard'],
            },
            'news': {
                'type': 'STRING',
                'description': 'Quarter macro-news, 2-4 sentences.',
            },
            'reasoning': {
                'type': 'STRING',
                'description': 'Macro-cycle reasoning; 1st/2nd/3rd-order effects. Under 300 words.',
            },
            'weights': {
                'type': 'ARRAY',
                'items': {'type': 'NUMBER'},
                'minItems': 5,
                'maxItems': 5,
                'description': '[TECH, OIL, GREEN, REAL_ESTATE, BONDS], non-negative, sum near 1.',
            },
            'infra_commit': {'type': 'NUMBER'},
            'carbon_offset_buy': {'type': 'NUMBER'},
            'put_hedge': {'type': 'NUMBER'},
            'tech_bet': {
                'type': 'STRING',
                'enum': ['status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'],
            },
        },
        'required': [
            'id',
            'source_shock_id',
            'difficulty_tier',
            'news',
            'reasoning',
            'weights',
            'infra_commit',
            'carbon_offset_buy',
            'put_hedge',
            'tech_bet',
        ],
    },
}


GEN_SYSTEM_PROMPT = ENV_SYSTEM_PROMPT + """

------
You are generating SFT training examples for CarbonAlpha.

Return a JSON array matching the schema. Each requested item has a difficulty:
- easy: direct 1-2 asset move; base-rate awareness matters more than drama.
- medium: use the requested difficulty_tier value "medium"; this corresponds
  to the environment tier "ambiguous", where signals conflict.
- hard: 2nd/3rd-order effects dominate; explicitly call out the trap a
  pattern-matching model would fall into.

For every trace:
- Keep reasoning under 300 words.
- Mention at least 3 of TECH, OIL, GREEN, REAL_ESTATE, BONDS.
- State the base-rate/regime assumption.
- Use interventions only when news justifies them.
- Make the allocation reflect the reasoning.
"""


def build_specs(easy: int, medium: int, hard: int, seed: int) -> list[TraceSpec]:
    rng = random.Random(seed)
    specs: list[TraceSpec] = []
    for difficulty, count in [('easy', easy), ('medium', medium), ('hard', hard)]:
        pool = SHOCKS_BY_DIFFICULTY[difficulty]
        if not pool:
            raise RuntimeError(f'No concrete shocks available for {difficulty}')
        for i in range(count):
            shock = pool[i % len(pool)]
            specs.append(
                TraceSpec(
                    id=f'{difficulty}_{shock.id}_v{i // len(pool) + 1:03d}',
                    difficulty=difficulty,
                    variant=i // len(pool) + 1,
                    shock=shock,
                )
            )
    rng.shuffle(specs)
    return specs


def chunked(items: list[TraceSpec], size: int) -> list[list[TraceSpec]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def spec_line(spec: TraceSpec, idx: int) -> str:
    shock = spec.shock
    impacts = ', '.join(f'{asset}={value:+.2f}' for asset, value in shock.impacts.items())
    tags = ', '.join(shock.tags) if shock.tags else 'none'
    return (
        f'{idx}. id={spec.id}\n'
        f'   source_shock_id={shock.id}\n'
        f'   difficulty_tier={spec.difficulty}\n'
        f'   environment_tier={shock.tier}\n'
        f'   variant={spec.variant}\n'
        f'   base_news={shock.news}\n'
        f'   impacts={impacts}\n'
        f'   regime_shift={shock.regime_shift or "none"}\n'
        f'   tags={tags}'
    )


def build_batch_prompt(specs: list[TraceSpec]) -> str:
    body = '\n\n'.join(spec_line(spec, i + 1) for i, spec in enumerate(specs))
    return f"""Generate exactly {len(specs)} CarbonAlpha SFT traces.

For each requested item:
1. Preserve `id`, `source_shock_id`, and `difficulty_tier` exactly.
2. Rewrite `base_news` into a fresh plausible quarter-news variant.
3. Keep difficulty consistent with the requested `difficulty_tier`.
4. Produce reasoning and action fields.

REQUESTED ITEMS:
{body}
"""


def gemini_api_key_slots() -> list[tuple[str, str]]:
    """Return configured Gemini API keys as named slots without exposing values."""
    names = ['GEMINI_API_KEY']
    names.extend(f'GEMINI_API_KEY{i}' for i in range(2, 11))
    names.extend(f'GEMINI_API_KEY_{i}' for i in range(2, 11))
    for name in sorted(os.environ):
        if name.startswith('GEMINI_API_KEY') and name not in names:
            names.append(name)

    slots: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    for name in names:
        value = os.environ.get(name)
        if not value or value in seen_values:
            continue
        slots.append((name, value))
        seen_values.add(value)
    return slots


def client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def gemini_generate_batch(specs: list[TraceSpec], model: str, api_key: str) -> list[dict]:
    gemini = client(api_key)
    config = types.GenerateContentConfig(
        system_instruction=GEN_SYSTEM_PROMPT,
        response_mime_type='application/json',
        response_schema=TRACE_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level='HIGH'),
    )
    resp = gemini.models.generate_content(
        model=model,
        contents=[types.Content(role='user', parts=[types.Part.from_text(text=build_batch_prompt(specs))])],
        config=config,
    )
    return json.loads(resp.text)


def is_quota_error(exc: Exception) -> bool:
    text = f'{type(exc).__name__}: {exc}'
    return 'RESOURCE_EXHAUSTED' in text or 'quota' in text.lower() or '429' in text


def validate_trace(trace: dict, expected: TraceSpec) -> tuple[bool, str]:
    if trace.get('id') != expected.id:
        return False, f'id mismatch: {trace.get("id")} != {expected.id}'
    if trace.get('source_shock_id') != expected.shock.id:
        return False, 'source_shock_id mismatch'
    if trace.get('difficulty_tier') != expected.difficulty:
        return False, f'difficulty mismatch: {trace.get("difficulty_tier")} != {expected.difficulty}'
    if len(trace.get('news', '')) < 50:
        return False, 'news too short'
    if len(trace.get('reasoning', '')) < 150:
        return False, 'reasoning too short'
    weights = trace.get('weights')
    if not isinstance(weights, list) or len(weights) != 5:
        return False, f'bad weights shape: {weights}'
    try:
        weights = [float(x) for x in weights]
    except Exception:
        return False, f'non-numeric weights: {weights}'
    if any(x < 0 for x in weights):
        return False, 'negative weights'
    total = sum(weights)
    if not (0.5 < total < 1.5):
        return False, f'weights sum {total:.3f}'
    asset_mentions = sum(1 for asset in ASSETS if asset in trace['reasoning'].upper())
    if asset_mentions < 3:
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


def assemble_sft_row(trace: dict, spec: TraceSpec) -> dict:
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
    prompt = ENV_SYSTEM_PROMPT + '\n\n' + build_user_prompt(trace['news'])
    completion = f"<think>\n{trace['reasoning'].strip()}\n</think>\n{json.dumps(action)}"
    return {
        'id': trace['id'],
        'seed_id': spec.shock.id,
        'seed_year': f'curriculum-phase-{TIER_TO_PHASE[spec.shock.tier]}',
        'seed_category': trace['difficulty_tier'],
        'prompt': prompt,
        'completion': completion,
        'raw': {
            'id': trace['id'],
            'news': trace['news'],
            'reasoning': trace['reasoning'].strip(),
            'weights': norm,
            'infra_commit': action['infra_commit'],
            'carbon_offset_buy': action['carbon_offset_buy'],
            'put_hedge': action['put_hedge'],
            'tech_bet': action['tech_bet'],
            'difficulty_tier': trace['difficulty_tier'],
            'environment_tier': spec.shock.tier,
            'curriculum_phase': TIER_TO_PHASE[spec.shock.tier],
            'source_shock_id': spec.shock.id,
            'source_impacts': spec.shock.impacts,
            'source_tags': spec.shock.tags,
            'source_regime_shift': spec.shock.regime_shift,
        },
    }


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            done.add(json.loads(line)['id'])
        except Exception:
            continue
    return done


def generate(args: argparse.Namespace) -> None:
    total = args.easy + args.medium + args.hard
    if total <= 0:
        raise SystemExit('Choose at least one trace: --easy N --medium N --hard N')
    if args.batch_size <= 0:
        raise SystemExit('--batch-size must be positive')
    if args.batch_size != 10:
        print(f'Note: batch size is {args.batch_size}; default/recommended is 10 traces per API call.')

    specs = build_specs(args.easy, args.medium, args.hard, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        done = load_done_ids(out_path)
        specs = [spec for spec in specs if spec.id not in done]
        print(f'Resume: {len(done)} existing ids, {len(specs)} remaining.')

    print(
        f'Generating {len(specs)} traces: easy={args.easy} medium={args.medium} hard={args.hard}; '
        f'batch_size={args.batch_size}; model={args.model}'
    )
    print(f'Output: {out_path}')

    key_slots = gemini_api_key_slots()
    if not args.dry_run and not key_slots:
        raise RuntimeError('No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY2 in .env.')
    if key_slots:
        print(f'Gemini key slots available: {", ".join(name for name, _ in key_slots)}')

    saved = 0
    failed: list[tuple[str, str]] = []
    key_cursor = 0
    mode = 'a' if args.resume and out_path.exists() else 'w'
    with out_path.open(mode) as fh:
        for batch_idx, batch in enumerate(chunked(specs, args.batch_size), 1):
            print(f'\nBatch {batch_idx}: {len(batch)} traces')
            if args.dry_run:
                for spec in batch:
                    print(f'  DRY {spec.id} ({spec.difficulty}, source={spec.shock.id})')
                continue

            by_id = {spec.id: spec for spec in batch}
            generated = None
            last_exc: Exception | None = None
            for offset in range(max(1, len(key_slots))):
                key_name, api_key = key_slots[(key_cursor + offset) % len(key_slots)]
                try:
                    generated = gemini_generate_batch(batch, args.model, api_key)
                    key_cursor = (key_cursor + offset) % len(key_slots)
                    break
                except Exception as exc:
                    last_exc = exc
                    if len(key_slots) > 1 and is_quota_error(exc):
                        print(f'  key slot {key_name} quota-limited; trying next key')
                        continue
                    break
            if generated is None:
                exc = last_exc or RuntimeError('Gemini call failed')
                for spec in batch:
                    failed.append((spec.id, f'{type(exc).__name__}: {exc}'))
                print(f'  FAIL batch call: {type(exc).__name__}: {exc}')
                continue

            for trace in generated:
                spec = by_id.get(trace.get('id'))
                if spec is None:
                    failed.append((str(trace.get('id')), 'unexpected id from Gemini'))
                    continue
                ok, msg = validate_trace(trace, spec)
                if not ok:
                    failed.append((spec.id, msg))
                    print(f'  FAIL {spec.id}: {msg}')
                    continue
                fh.write(json.dumps(assemble_sft_row(trace, spec)) + '\n')
                fh.flush()
                saved += 1
                print(f'  OK {spec.id}')

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    print(f'\nSaved {saved}/{len(specs)} traces -> {out_path}')
    if failed:
        fail_path = out_path.with_suffix(out_path.suffix + '.failures.json')
        fail_path.write_text(json.dumps(failed, indent=2))
        print(f'Failures: {len(failed)} -> {fail_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--easy', type=int, default=0)
    parser.add_argument('--medium', type=int, default=0,
                        help='Medium traces map to environment tier "ambiguous".')
    parser.add_argument('--hard', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--out', default=str(Path(__file__).parent / 'curriculum_traces.jsonl'))
    parser.add_argument('--model', default=os.environ.get('GEMINI_MODEL', 'gemini-3-pro-preview'))
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sleep-s', type=float, default=0.0,
                        help='Optional pause between API calls for rate-limit friendliness.')
    return parser.parse_args()


if __name__ == '__main__':
    generate(parse_args())
