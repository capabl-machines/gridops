"""GRPO training driver — Qwen3-4B-Instruct + Unsloth + TRL.

Forked from Unsloth's Advanced Qwen3 4B GRPO recipe (§59.1 of hackathon guide).
Structured as a .py so we can run on the pod via `python` and later convert to
.ipynb via jupytext if needed.

Three-phase curriculum (see portfolio_env_design.md §13.2):
  Phase 1: easy shocks only, 4Q episodes, format + regret rewards, 50 iters
  Phase 2: easy+ambiguous, 8Q episodes, + sharpe + drawdown, 100 iters
  Phase 3: full 15+ shock pool, 12Q episodes, all 5 rewards, 80 iters

SFT warm-start runs BEFORE Phase 1 on the traces from sft_traces/traces.jsonl.

Usage:
    python notebooks/grpo_training.py --phase 1         # Phase 1 only (smoke test)
    python notebooks/grpo_training.py --phase all       # full curriculum
    python notebooks/grpo_training.py --sft-only        # just SFT warm-start
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Unsloth MUST be imported before transformers / trl ──
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import numpy as np
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio_env import (
    PortfolioEnv, PortfolioAction, Trajectory,
    r_format, r_regret, r_sharpe, r_carbon, r_drawdown,
    parse_json_action, extract_think,
    training_seeds,
)
from portfolio_env.constants import (
    ASSETS, EPISODE_LENGTH, N_ASSETS, BASELINE_WEIGHTS,
)


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

MODEL_NAME = 'unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit'
MAX_SEQ_LEN = 4096
OUTPUT_DIR = Path('/workspace/checkpoints')

PHASE_CONFIG = {
    1: dict(
        episode_length=4, phase=1, shock_cap=2,
        max_iters=50, batch_size=4, num_generations=4,
        rewards=['format', 'regret'],
        carbon_weight=0.0,
    ),
    2: dict(
        episode_length=8, phase=2, shock_cap=3,
        max_iters=100, batch_size=6, num_generations=6,
        rewards=['format', 'regret', 'sharpe', 'drawdown'],
        carbon_weight=0.3,
    ),
    3: dict(
        episode_length=12, phase=3, shock_cap=5,
        max_iters=80, batch_size=6, num_generations=6,
        rewards=['format', 'regret', 'sharpe', 'drawdown', 'carbon'],
        carbon_weight=1.0,
    ),
}


# ══════════════════════════════════════════════════════════════════════
# Reward functions — adapted to GRPO's signature
#   GRPOTrainer expects: reward_fn(prompts, completions, **kwargs) -> list[float]
# Each of our env rewards needs the trajectory AND the completion.
# Strategy: reconstruct a synthetic trajectory from the single-turn action.
# ══════════════════════════════════════════════════════════════════════

def _simulate_episode_from_action(action: PortfolioAction, seed: int, phase: int) -> Trajectory:
    """Given a single PortfolioAction (the LLM's one-shot plan), run the env for
    EPISODE_LENGTH quarters with that same action each quarter, collect trajectory.

    Flatten-MDP interpretation: the LLM's one allocation is held throughout the
    episode. Reward functions score the resulting trajectory.
    """
    env = PortfolioEnv(phase=phase, seed=seed)
    env.reset(seed=seed)
    dummy_completion = ''   # Don't need per-step completions for env logic
    for _ in range(PHASE_CONFIG[phase]['episode_length']):
        env.step(action, completion=dummy_completion)
    return env.trajectory


def _action_from_completion(completion: str) -> PortfolioAction | None:
    """Parse the LLM's completion into a PortfolioAction. Returns None on failure."""
    raw = parse_json_action(completion)
    if raw is None or not isinstance(raw, dict):
        return None
    w = raw.get('weights')
    if not isinstance(w, list) or len(w) != 5:
        return None
    try:
        return PortfolioAction(
            weights=[max(0.0, float(x)) for x in w],
            infra_commit=float(raw.get('infra_commit', 0.0)),
            carbon_offset_buy=float(raw.get('carbon_offset_buy', 0.0)),
            put_hedge=float(raw.get('put_hedge', 0.0)),
            tech_bet=raw.get('tech_bet', 'status_quo'),
        )
    except Exception:
        return None


def make_reward_fn(component: str, phase: int, carbon_weight: float = 1.0):
    """Build a GRPO-compatible reward function for a single component."""

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        seeds_for_batch = kwargs.get('seed', [42] * len(completions))
        if isinstance(seeds_for_batch, int):
            seeds_for_batch = [seeds_for_batch] * len(completions)
        out = []
        for comp, seed in zip(completions, seeds_for_batch):
            # completions may come as str or list[dict]; normalize to str
            if isinstance(comp, list):
                text = comp[0].get('content', '') if comp else ''
            else:
                text = comp
            if component == 'format':
                out.append(r_format(text))
                continue
            action = _action_from_completion(text)
            if action is None:
                out.append(-0.5 if component == 'regret' else 0.0)
                continue
            traj = _simulate_episode_from_action(action, seed=seed, phase=phase)
            if component == 'regret':
                out.append(r_regret(traj))
            elif component == 'sharpe':
                out.append(r_sharpe(traj))
            elif component == 'carbon':
                out.append(r_carbon(traj, phase_weight=carbon_weight))
            elif component == 'drawdown':
                out.append(r_drawdown(traj))
            else:
                out.append(0.0)
        return out

    reward_fn.__name__ = f'r_{component}_phase{phase}'
    return reward_fn


# ══════════════════════════════════════════════════════════════════════
# Dataset construction — single-turn prompts for flattened-MDP GRPO
# ══════════════════════════════════════════════════════════════════════

def build_prompt(news_preview: str) -> str:
    """Single prompt for flattened MDP. Imports from portfolio_env.prompt
    so SFT and GRPO use *exactly* the same context (Gemini's RLHF rule)."""
    from portfolio_env.prompt import SYSTEM_PROMPT, build_user_prompt
    return SYSTEM_PROMPT + '\n\n' + build_user_prompt(news_preview)


def build_training_dataset(n_prompts: int, phase: int, rng: np.random.Generator) -> Dataset:
    """Build a training dataset of prompts. Each prompt maps to a different seed.

    GRPO will sample N completions per prompt; our reward fn re-simulates the
    env with the sampled action on that seed to score each completion."""
    from portfolio_env.shocks import shocks_available
    pool = shocks_available(phase)
    seeds = training_seeds(rng, n_prompts)
    rows = []
    for i, seed in enumerate(seeds):
        # Pick a specific shock from the phase-appropriate pool for this prompt's news
        shock = pool[rng.integers(0, len(pool))]
        rows.append({
            'prompt': build_prompt(shock.news),
            'seed': seed,
        })
    return Dataset.from_list(rows)


# ══════════════════════════════════════════════════════════════════════
# SFT warm-start
# ══════════════════════════════════════════════════════════════════════

def run_sft_warmstart(model, tokenizer, sft_path: Path, max_steps: int = 150):
    print(f'\n══ SFT warm-start — {sft_path} ══')
    if not sft_path.exists():
        print(f'  ! {sft_path} does not exist. Skipping SFT.')
        return model
    # Pre-format as plain `text` using the tokenizer's chat template. This
    # avoids Unsloth's `formatting_func` requirement while still ensuring the
    # model trains on the same `<|im_start|>user ... <|im_end|><|im_start|>assistant ...`
    # structure that eval produces.
    rows = []
    with sft_path.open() as f:
        for line in f:
            t = json.loads(line)
            text = tokenizer.apply_chat_template(
                [
                    {'role': 'user',      'content': t['prompt']},
                    {'role': 'assistant', 'content': t['completion']},
                ],
                tokenize=False,
            )
            rows.append({'text': text})
    print(f'  {len(rows)} SFT examples loaded (chat format in `text`)')
    ds = Dataset.from_list(rows)

    FastLanguageModel.for_training(model)
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR / 'sft'),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,           # bumped 2e-5 → 5e-5; format learning needs it
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy='steps',
        save_steps=max_steps,         # one save at end so we can inspect adapters
        save_total_limit=1,
        bf16=is_bfloat16_supported(),
        dataset_text_field='text',
        max_length=MAX_SEQ_LEN,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_config,
    )
    t0 = time.time()
    trainer.train()
    print(f'  SFT done in {(time.time()-t0)/60:.1f} min')
    return model


# ══════════════════════════════════════════════════════════════════════
# GRPO phase runner
# ══════════════════════════════════════════════════════════════════════

def run_grpo_phase(model, tokenizer, phase: int):
    cfg = PHASE_CONFIG[phase]
    print(f'\n══ GRPO Phase {phase}: {cfg["episode_length"]}Q episodes, '
          f'{cfg["max_iters"]} iters, rewards={cfg["rewards"]} ══')

    rng = np.random.default_rng(42 + phase)
    n_prompts = cfg['batch_size'] * cfg['max_iters']
    dataset = build_training_dataset(n_prompts, phase=cfg['phase'], rng=rng)

    reward_fns = [make_reward_fn(r, cfg['phase'], cfg['carbon_weight']) for r in cfg['rewards']]

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR / f'phase{phase}'),
        max_steps=cfg['max_iters'],
        per_device_train_batch_size=cfg['batch_size'],
        num_generations=cfg['num_generations'],
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy='steps',
        save_steps=cfg['max_iters'] // 4,
        max_prompt_length=1024,
        max_completion_length=400,     # strict cap per DAPO overlong-shaping spirit
        temperature=0.9,
        top_p=0.95,
        loss_type='dapo',              # v1.0 default but explicit for clarity
        beta=0.0,                      # KL-free (DAPO / R1-Zero)
        bf16=is_bfloat16_supported(),
    )

    FastLanguageModel.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        train_dataset=dataset,
        args=grpo_config,
    )
    t0 = time.time()
    trainer.train()
    print(f'  Phase {phase} done in {(time.time()-t0)/60:.1f} min')
    return model


# ══════════════════════════════════════════════════════════════════════
# Hold-out eval
# ══════════════════════════════════════════════════════════════════════

def evaluate_holdout(model, tokenizer, phase: int = 3, verbose_samples: int = 1) -> dict:
    """Eval on reserved holdout seeds. Prints raw completion for first N seeds
    to help diagnose format/structure issues (e.g. 0/5 valid)."""
    from portfolio_env import holdout_seeds
    FastLanguageModel.for_inference(model)
    results = {}
    for i, seed in enumerate(holdout_seeds()):
        from portfolio_env.shocks import shocks_available
        pool = shocks_available(phase)
        rng = np.random.default_rng(seed)
        shock = pool[rng.integers(0, len(pool))]
        prompt = build_prompt(shock.news)
        msg_text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(msg_text, return_tensors='pt').to('cuda')
        out = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        action = _action_from_completion(completion)

        if i < verbose_samples:
            print(f'\n  [diagnostic] seed={seed} raw completion (first 500 chars):')
            print('  ' + completion[:500].replace('\n', '\n  '))
            print(f'  [parse_action result]: {action}')

        if action is None:
            results[seed] = {'valid': False, 'regret': None}
            continue
        traj = _simulate_episode_from_action(action, seed=seed, phase=phase)
        regret = r_regret(traj)
        results[seed] = {'valid': True, 'regret': regret, 'final_nav_real': traj.nav_real_series[-1]}

    valid_regrets = [r['regret'] for r in results.values() if r['valid']]
    print(f'\n── Hold-out eval ({len(valid_regrets)}/{len(results)} valid) ──')
    print(f'  mean regret: {np.mean(valid_regrets):+.4f}' if valid_regrets else '  no valid completions')
    print(f'  beat baseline: {sum(1 for r in valid_regrets if r > 0)}/{len(valid_regrets)}')
    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all', help="1 | 2 | 3 | 'all' | 'sft-only'")
    parser.add_argument('--sft-traces', type=Path, default=Path(__file__).parent.parent / 'sft_traces' / 'traces.jsonl')
    parser.add_argument('--sft-steps', type=int, default=60)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading {MODEL_NAME}...')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    # Add LoRA adapters for efficient training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.0,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=42,
    )
    print(f'VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')

    # SFT warm-start
    model = run_sft_warmstart(model, tokenizer, args.sft_traces, max_steps=args.sft_steps)

    # Hold-out eval before GRPO (baseline)
    print('\n══ Pre-GRPO hold-out eval (SFT-only) ══')
    evaluate_holdout(model, tokenizer, phase=3)

    # GRPO phases
    if args.phase == 'sft-only':
        print('SFT-only mode. Done.')
        return
    if args.phase == 'all':
        phases = [1, 2, 3]
    else:
        phases = [int(args.phase)]

    for p in phases:
        model = run_grpo_phase(model, tokenizer, p)
        # Quick eval checkpoint
        evaluate_holdout(model, tokenizer, phase=p)

    # Save final
    final_path = OUTPUT_DIR / 'final_merged'
    model.save_pretrained_merged(str(final_path), tokenizer, save_method='lora')
    print(f'\nSaved LoRA adapters to {final_path}')


if __name__ == '__main__':
    main()
