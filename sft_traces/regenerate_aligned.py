"""Re-run Gemini 3 Pro on every trace in merged-*.json using the NEW prompt
defined in portfolio_env/prompt.py, and write a SFT-ready JSONL whose
prompt+completion exactly match what GRPO inference will see.

Why: the merged dataset was built against the OLD prompt format
("News this quarter:\n... Format: <think>...</think>{json}"). The trained
model now uses the richer SYSTEM_PROMPT that documents constraints, the
intervention menu, and the strict output format. Mismatched prompts at
SFT vs. GRPO -> mode collapse.

Usage:
    python regenerate_aligned.py                       # all 200 traces
    python regenerate_aligned.py --limit 5             # smoke test
    python regenerate_aligned.py --workers 10          # concurrency
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types

_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio_env.constants import ASSETS
from portfolio_env.prompt import SYSTEM_PROMPT as ENV_SYSTEM_PROMPT, build_user_prompt


TRACE_SCHEMA = {
    'type': 'OBJECT',
    'properties': {
        'reasoning': {'type': 'STRING', 'description': '<think> body: macro-cycle analysis, 1st/2nd/3rd-order. Under 300 words.'},
        'weights': {
            'type': 'ARRAY',
            'items': {'type': 'NUMBER'},
            'minItems': 5, 'maxItems': 5,
            'description': '[TECH, OIL, GREEN, REAL_ESTATE, BONDS] non-negative, sum to 1.0',
        },
        'infra_commit':      {'type': 'NUMBER', 'description': '0 to 0.2'},
        'carbon_offset_buy': {'type': 'NUMBER', 'description': '0 to 0.1'},
        'put_hedge':         {'type': 'NUMBER', 'description': '0 to 0.05'},
        'tech_bet': {
            'type': 'STRING',
            'enum': ['status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'],
        },
    },
    'required': ['reasoning', 'weights', 'infra_commit', 'carbon_offset_buy', 'put_hedge', 'tech_bet'],
}


GEN_SYSTEM_PROMPT = ENV_SYSTEM_PROMPT + """

------
You are generating one SFT training example. Produce reasoning that follows
the rules above (macro-cycle, 1st/2nd/3rd-order, base-rate explicit, only
hedge/intervene when news justifies it). Return STRUCTURED JSON matching
the response schema. The reasoning string will be wrapped in <think>...</think>
and the action fields will be serialized as the JSON action by the caller.
"""


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ['GEMINI_API_KEY'])


def gemini_one(news: str) -> dict:
    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=GEN_SYSTEM_PROMPT,
        response_mime_type='application/json',
        response_schema=TRACE_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level='HIGH'),
    )
    user = build_user_prompt(news)
    resp = client.models.generate_content(
        model='gemini-3-pro-preview',
        contents=[types.Content(role='user', parts=[types.Part.from_text(text=user)])],
        config=config,
    )
    return json.loads(resp.text)


def validate(out: dict) -> tuple[bool, str]:
    if len(out.get('reasoning', '')) < 150:
        return False, 'reasoning too short'
    w = out.get('weights')
    if not isinstance(w, list) or len(w) != 5 or any(x < 0 for x in w):
        return False, f'bad weights: {w}'
    s = sum(w)
    if not (0.5 < s < 1.5):
        return False, f'weights sum {s:.3f}'
    asset_mentions = sum(1 for a in ASSETS if a in out['reasoning'].upper())
    if asset_mentions < 2:
        return False, f'mentions only {asset_mentions} assets'
    for k, lo, hi in (('infra_commit', 0, 0.2),
                      ('carbon_offset_buy', 0, 0.1),
                      ('put_hedge', 0, 0.05)):
        v = out.get(k, 0) or 0
        if not (lo <= v <= hi):
            return False, f'{k}={v} out of [{lo},{hi}]'
    if out.get('tech_bet') not in ('status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'):
        return False, f'bad tech_bet: {out.get("tech_bet")}'
    return True, 'ok'


def assemble_row(trace: dict, gen: dict) -> dict:
    """Build the SFT row in the EXACT format that GRPO training will use."""
    w = gen['weights']
    total = sum(w)
    w_norm = [round(x / total, 4) for x in w] if total > 0 else [0.2] * 5

    action = {
        'weights': w_norm,
        'infra_commit': round(float(gen.get('infra_commit') or 0), 4),
        'carbon_offset_buy': round(float(gen.get('carbon_offset_buy') or 0), 4),
        'put_hedge': round(float(gen.get('put_hedge') or 0), 4),
        'tech_bet': gen.get('tech_bet', 'status_quo'),
    }

    news = trace['raw']['news']
    prompt = ENV_SYSTEM_PROMPT + '\n\n' + build_user_prompt(news)
    completion = (f"<think>\n{gen['reasoning'].strip()}\n</think>\n"
                  f"{json.dumps(action)}")

    return {
        'id': trace.get('id'),
        'seed_id': trace.get('seed_id'),
        'seed_year': trace.get('seed_year'),
        'seed_category': trace.get('seed_category'),
        'prompt': prompt,
        'completion': completion,
        'raw': {
            'id': trace.get('id'),
            'news': news,
            'reasoning': gen['reasoning'].strip(),
            'weights': w_norm,
            'infra_commit': action['infra_commit'],
            'carbon_offset_buy': action['carbon_offset_buy'],
            'put_hedge': action['put_hedge'],
            'tech_bet': action['tech_bet'],
        },
    }


def process_one(trace: dict, max_retries: int = 2) -> tuple[dict | None, str]:
    last_err = ''
    for attempt in range(max_retries + 1):
        try:
            gen = gemini_one(trace['raw']['news'])
            ok, msg = validate(gen)
            if ok:
                return assemble_row(trace, gen), 'ok'
            last_err = msg
        except Exception as e:
            last_err = f'{type(e).__name__}: {e}'
        time.sleep(2 ** attempt)
    return None, last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp',
                    default=str(Path(__file__).parent / 'merged-1777105798432.json'))
    ap.add_argument('--out',
                    default=str(Path(__file__).parent / 'merged_v6_aligned.jsonl'))
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process first N (smoke test)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--resume', action='store_true',
                    help='Skip IDs already present in --out and append')
    ap.add_argument('--rpm', type=int, default=0,
                    help='Throttle to N requests/minute total (0 = no throttle)')
    args = ap.parse_args()

    traces = json.loads(Path(args.inp).read_text())
    if args.limit:
        traces = traces[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.open():
            try:
                done_ids.add(json.loads(line)['id'])
            except Exception:
                pass
        traces = [t for t in traces if t.get('id') not in done_ids]
        print(f'Resume: {len(done_ids)} already done, {len(traces)} remaining')

    print(f'Processing {len(traces)} traces, {args.workers} workers, '
          f'rpm={args.rpm or "unlimited"}')
    print(f'Output: {args.out}')

    min_interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_submit = [0.0]
    submit_lock = __import__('threading').Lock()

    def throttled_process(t):
        if min_interval > 0:
            with submit_lock:
                wait = (last_submit[0] + min_interval) - time.time()
                if wait > 0:
                    time.sleep(wait)
                last_submit[0] = time.time()
        return process_one(t)

    saved = 0
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    open_mode = 'a' if (args.resume and out_path.exists()) else 'w'
    with out_path.open(open_mode) as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(throttled_process, t): t for t in traces}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            row, msg = fut.result()
            if row is None:
                failed.append((t.get('id', '?'), msg))
                print(f'  [{i}/{len(traces)}] FAIL {t.get("id")}: {msg}')
                continue
            fout.write(json.dumps(row) + '\n')
            fout.flush()
            saved += 1
            if i % 10 == 0 or i == len(traces):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(traces) - i) / rate if rate > 0 else 0
                print(f'  [{i}/{len(traces)}] saved={saved} fail={len(failed)} '
                      f'elapsed={elapsed:.0f}s eta={eta:.0f}s')

    print(f'\n{saved}/{len(traces)} saved -> {args.out}')
    if failed:
        print(f'{len(failed)} failed:')
        for sid, msg in failed[:20]:
            print(f'  - {sid}: {msg}')


if __name__ == '__main__':
    main()
