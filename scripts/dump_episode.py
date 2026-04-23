"""Dump one episode's state trajectory to JSON for the Greenberg Terminal UI.

Usage:
    # Dump with untrained baseline Qwen3 (no checkpoint)
    python scripts/dump_episode.py --out ui/demo_baseline.json

    # Dump with trained LoRA checkpoint
    python scripts/dump_episode.py --checkpoint /workspace/checkpoints/final_merged \\
        --out ui/demo_trained.json --seed 100

    # Dump a "scripted" run (no LLM, uses equal-weighted + random interventions)
    # — useful for brother to smoke-test the UI before any training exists
    python scripts/dump_episode.py --policy scripted --out ui/demo_stub.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--policy', choices=['llm', 'scripted', 'equal_weighted'], default='scripted')
    p.add_argument('--model-name', default='unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit')
    p.add_argument('--checkpoint', type=Path, default=None, help='LoRA adapter path')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--phase', type=int, default=3)
    p.add_argument('--max-new-tokens', type=int, default=400)
    return p.parse_args()


def load_llm(model_name: str, checkpoint: Path | None):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=4096, load_in_4bit=True, dtype=None,
    )
    if checkpoint is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(checkpoint))
        print(f'Loaded LoRA from {checkpoint}')
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_episode(args):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from portfolio_env import (
        PortfolioEnv, PortfolioAction, parse_json_action, extract_think,
    )
    from portfolio_env.constants import ASSETS, EPISODE_LENGTH, BASELINE_WEIGHTS

    env = PortfolioEnv(phase=args.phase, seed=args.seed)
    obs = env.reset(seed=args.seed)

    # Policy
    if args.policy == 'llm':
        model, tokenizer = load_llm(args.model_name, args.checkpoint)
    else:
        model, tokenizer = None, None

    rng = np.random.default_rng(args.seed)

    # Per-quarter capture
    news_feed = []
    think_stream = []
    weights_history = []
    interventions_used = []
    shock_markers = []
    nav_agent_real = [obs.portfolio_nav_real]
    nav_agent_nominal = [obs.portfolio_nav_nominal]
    nav_baseline = [obs.baseline_nav_real]
    regime_series = [obs.current_regime]
    carbon_series = [obs.carbon_footprint_accumulated]

    for q in range(EPISODE_LENGTH):
        # capture the shock if one hits this quarter
        shock_id = None
        if env._plan and q in env._plan.shocks_by_quarter:
            sh = env._plan.shocks_by_quarter[q]
            shock_id = sh.id
            shock_markers.append({'quarter': q, 'id': sh.id, 'tier': sh.tier})

        news_feed.append({
            'quarter': q,
            'news': obs.news,
            'tier': obs.difficulty_tier if shock_id else 'routine',
            'has_shock': shock_id is not None,
        })

        # Pick action
        action, thought = get_action(args, obs, model, tokenizer, rng, q)
        think_stream.append({'quarter': q, 'text': thought})
        weights_history.append([round(w, 4) for w in action.weights])

        # Record interventions
        if q == 0 and action.tech_bet != 'status_quo':
            interventions_used.append({'quarter': 0, 'type': 'tech_bet', 'value': action.tech_bet})
        if action.infra_commit > 0:
            interventions_used.append({'quarter': q, 'type': 'infra_commit', 'value': float(action.infra_commit)})
        if action.put_hedge > 0:
            interventions_used.append({'quarter': q, 'type': 'put_hedge', 'value': float(action.put_hedge)})
        if action.carbon_offset_buy > 0:
            interventions_used.append({'quarter': q, 'type': 'carbon_offset_buy', 'value': float(action.carbon_offset_buy)})

        dummy_completion = f'<think>{thought}</think>' + json.dumps({'weights': action.weights})
        obs = env.step(action, completion=dummy_completion)[0]

        nav_agent_real.append(float(obs.portfolio_nav_real))
        nav_agent_nominal.append(float(obs.portfolio_nav_nominal))
        nav_baseline.append(float(obs.baseline_nav_real))
        regime_series.append(obs.current_regime)
        carbon_series.append(float(obs.carbon_footprint_accumulated))

    state = {
        'episode_id': f'{args.policy}_seed{args.seed}_phase{args.phase}',
        'policy_label': args.policy if args.checkpoint is None else f'{args.policy}+lora',
        'current_quarter': EPISODE_LENGTH,
        'total_quarters': EPISODE_LENGTH,
        'tech_bet': weights_history[0] and next(
            (iu['value'] for iu in interventions_used if iu['type'] == 'tech_bet'),
            'status_quo',
        ),
        'news_feed': news_feed,
        'think_stream': think_stream,
        'weights_history': weights_history,
        'interventions_used': interventions_used,
        'shock_markers': shock_markers,
        'nav_series': {
            'agent_real':    [round(x, 5) for x in nav_agent_real],
            'agent_nominal': [round(x, 5) for x in nav_agent_nominal],
            'baseline_real': [round(x, 5) for x in nav_baseline],
        },
        'regime_series': list(regime_series),
        'carbon': {
            'accumulated': round(carbon_series[-1], 2),
            'cap': 25.0,                           # from constants.py CARBON_CAP
            'series': [round(x, 2) for x in carbon_series],
            'offsets_held': float(obs.carbon_offsets_held),
        },
        'final_regret_real': round(nav_agent_real[-1] - nav_baseline[-1], 5),
        'timestamp': int(time.time()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2))
    print(f'Wrote {args.out} ({len(news_feed)} quarters, final regret {state["final_regret_real"]:+.3f})')


def get_action(args, obs, model, tokenizer, rng, q):
    """Produce an action + thought for this quarter."""
    from portfolio_env import PortfolioAction, parse_json_action, extract_think
    from portfolio_env.constants import BASELINE_WEIGHTS

    if args.policy == 'equal_weighted':
        return PortfolioAction(weights=list(BASELINE_WEIGHTS)), \
               f'Q{q}: equal-weighted baseline — no reasoning.'

    if args.policy == 'scripted':
        # Simple sensible hand-coded policy for UI smoke-test:
        # equal-weight with tilt based on current regime
        w = list(BASELINE_WEIGHTS)
        thought = f'Q{q}: scripted policy. Regime={obs.current_regime}.'
        if obs.current_regime == 'stagflationary':
            w = [0.1, 0.4, 0.1, 0.2, 0.2]
            thought += ' Tilting heavily to OIL, REAL_ESTATE as inflation hedge.'
        elif obs.current_regime == 'deflationary':
            w = [0.15, 0.05, 0.10, 0.10, 0.60]
            thought += ' Tilting to BONDS as deflation benefits duration.'
        else:
            thought += ' Holding equal-weighted.'
        # Q0 commit
        infra = 0.15 if q == 0 else 0.0
        tech_bet = 'green_leaps' if q == 0 else 'status_quo'
        return PortfolioAction(weights=w, infra_commit=infra, tech_bet=tech_bet), thought

    # --- LLM policy ---
    import torch
    prompt = build_prompt(obs)
    msg_text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(msg_text, return_tensors='pt').to('cuda')
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.9,
    )
    completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    thought = extract_think(completion) or completion[:500]
    raw = parse_json_action(completion) or {}
    w = raw.get('weights') or list(BASELINE_WEIGHTS)
    if not isinstance(w, list) or len(w) != 5:
        w = list(BASELINE_WEIGHTS)
    total = sum(max(0.0, x) for x in w) or 1.0
    w = [max(0.0, float(x)) / total for x in w]
    try:
        action = PortfolioAction(
            weights=w,
            infra_commit=float(raw.get('infra_commit', 0.0) or 0.0),
            carbon_offset_buy=float(raw.get('carbon_offset_buy', 0.0) or 0.0),
            put_hedge=float(raw.get('put_hedge', 0.0) or 0.0),
            tech_bet=raw.get('tech_bet', 'status_quo'),
        )
    except Exception:
        action = PortfolioAction(weights=list(BASELINE_WEIGHTS))
    return action, thought[:800]


def build_prompt(obs):
    return (
        f"You are a climate-aware portfolio manager. News this quarter:\n"
        f"{obs.news}\n\n"
        f"Current state: quarter {obs.quarter}, regime {obs.current_regime}, "
        f"NAV (real) {obs.portfolio_nav_real:.3f}, carbon used {obs.carbon_footprint_accumulated:.1f}/25 kg.\n\n"
        f"Think step by step about 1st/2nd/3rd-order impacts on TECH, OIL, "
        f"GREEN, REAL_ESTATE, BONDS. Then output your allocation.\n\n"
        f"Format: <think>reasoning</think>"
        f'{{"weights": [TECH, OIL, GREEN, REAL_ESTATE, BONDS]}}'
    )


if __name__ == '__main__':
    args = parse_args()
    run_episode(args)
