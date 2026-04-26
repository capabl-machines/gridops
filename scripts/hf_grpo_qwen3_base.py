# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "huggingface_hub>=0.34",
#   "openenv-core>=0.2",
#   "fastapi",
#   "pydantic",
#   "uvicorn",
#   "vllm==0.15.1",
#   "transformers==4.56.2",
#   "trl==0.22.2",
#   "unsloth",
#   "torchvision",
#   "bitsandbytes",
#   "xformers",
#   "peft",
#   "datasets",
#   "accelerate",
#   "numpy",
#   "pillow",
#   "matplotlib",
# ]
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# ///
"""HF Jobs entrypoint for the Qwen3-4B-Base GRPO rescue run.

This script intentionally does not reuse `notebooks/grpo_training.py`.
The current Instruct/v6 lineage is the safe deliverable; this file is a
separate experiment that follows Unsloth's official Qwen3 4B GRPO notebook:
Base model, custom chat template, SFT warm-start, vLLM sampling params.

Recommended launch:
    hf jobs uv run --flavor l40sx1 --secrets HF_API_TOKEN \\
        scripts/hf_grpo_qwen3_base.py

Useful smoke launch:
    hf jobs uv run --flavor l40sx1 --secrets HF_API_TOKEN \\
        --env CARBON_ALPHA_GRPO_MAX_STEPS=10 \\
        --env CARBON_ALPHA_GRPO_RUN_LABEL=grpo_qwen3_4b_base_smoke_v1 \\
        scripts/hf_grpo_qwen3_base.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


WORK = Path(os.environ.get('CARBON_ALPHA_WORK_DIR', '/tmp/CarbonAlphaBaseGRPO'))
CODE_REPO = os.environ.get('CARBON_ALPHA_CODE_REPO', '77ethers/CarbonAlpha-train')
MODEL_REPO = os.environ.get('CARBON_ALPHA_MODEL_REPO', '77ethers/CarbonAlpha')
MODEL_NAME = os.environ.get('CARBON_ALPHA_BASE_MODEL', 'unsloth/Qwen3-4B-Base')
TRACES = os.environ.get('CARBON_ALPHA_TRACES', 'sft_traces/merged_v6_aligned.jsonl')
RUN_LABEL = os.environ.get('CARBON_ALPHA_GRPO_RUN_LABEL', 'grpo_qwen3_4b_base_smoke_v1')
OUTPUT_DIR = Path(os.environ.get('CARBON_ALPHA_OUTPUT_DIR', str(WORK / 'checkpoints')))

MAX_SEQ_LEN = int(os.environ.get('CARBON_ALPHA_MAX_SEQ_LEN', '2048'))
LORA_RANK = int(os.environ.get('CARBON_ALPHA_LORA_RANK', '32'))
SFT_STEPS = int(os.environ.get('CARBON_ALPHA_SFT_STEPS', '150'))
GRPO_MAX_STEPS = int(os.environ.get('CARBON_ALPHA_GRPO_MAX_STEPS', '10'))
GRPO_NUM_GENERATIONS = int(os.environ.get('CARBON_ALPHA_GRPO_NUM_GENERATIONS', '4'))
GRPO_BATCH_SIZE = int(os.environ.get('CARBON_ALPHA_GRPO_BATCH_SIZE', str(GRPO_NUM_GENERATIONS)))
GRPO_PROMPTS = int(os.environ.get('CARBON_ALPHA_GRPO_PROMPTS', '40'))
SEED = int(os.environ.get('CARBON_ALPHA_SEED', '3407'))

THINK_OPEN = '<think>\n'
THINK_RE = re.compile(r'<think>\s*(.*?)\s*</think>', re.DOTALL)
TECH_BETS = {'status_quo', 'green_leaps', 'carbon_priced', 'inflationary', 'fragmentation'}
ACTION_KEYS = {'weights', 'infra_commit', 'carbon_offset_buy', 'put_hedge', 'tech_bet'}


def load_dotenv_for_local() -> None:
    """Best-effort local convenience. HF Jobs should receive secrets directly."""
    env_path = Path.cwd() / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


def select_hf_token() -> str:
    """Use HF_API_TOKEN first; it is the valid token for this project."""
    token = os.environ.get('HF_API_TOKEN')
    if not token:
        token = os.environ.get('HF_TOKEN')
        if token:
            print('! HF_API_TOKEN missing; falling back to HF_TOKEN', flush=True)
    if not token:
        raise RuntimeError('HF_API_TOKEN is required for CarbonAlpha private repos')

    # huggingface_hub and Unsloth often prefer HF_TOKEN implicitly. Override the
    # invalid local HF_TOKEN with the known-good project token for this process.
    os.environ['HF_TOKEN'] = token
    os.environ['HUGGINGFACE_HUB_TOKEN'] = token
    return token


def check_hf_access(token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami(token=token)
    print(f"HF auth OK: {who.get('name')}", flush=True)
    for repo_id, repo_type in ((MODEL_REPO, 'model'), (CODE_REPO, 'dataset')):
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
        print(
            f'HF access OK: {repo_type}:{repo_id} '
            f'private={getattr(info, "private", None)} files={len(files)}',
            flush=True,
        )


def download_code_bundle(token: str) -> Path:
    from huggingface_hub import snapshot_download

    WORK.mkdir(parents=True, exist_ok=True)
    code_dir = snapshot_download(
        repo_id=CODE_REPO,
        repo_type='dataset',
        token=token,
        local_dir=str(WORK / 'code'),
    )
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)
    print(f'Code bundle: {code_dir}', flush=True)
    return Path(code_dir)


def install_base_chat_template(tokenizer) -> None:
    """Official Unsloth-style template, adapted to open with `<think>`."""
    from portfolio_env.prompt import SYSTEM_PROMPT

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ system_prompt + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ think_open }}"
        "{% endif %}"
    )
    tokenizer.chat_template = (
        chat_template
        .replace('system_prompt', repr(SYSTEM_PROMPT))
        .replace('think_open', repr(THINK_OPEN))
    )


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[0].get('content', '') if completion else ''
    return str(completion)


def with_prompt_think(completion: str) -> str:
    """GRPO completions omit the opening tag because the template prompts it."""
    if completion.lstrip().startswith('<think>'):
        return completion
    return THINK_OPEN + completion


def json_payload_from_completion(completion: str) -> dict[str, Any] | None:
    from portfolio_env import parse_json_action

    raw = parse_json_action(with_prompt_think(completion))
    return raw if isinstance(raw, dict) else None


def parse_action_from_base_completion(completion: str):
    from portfolio_env import PortfolioAction

    raw = json_payload_from_completion(completion)
    if raw is None:
        return None
    weights = raw.get('weights')
    if not isinstance(weights, list) or len(weights) != 5:
        return None
    try:
        return PortfolioAction(
            weights=[max(0.0, float(x)) for x in weights],
            infra_commit=float(raw.get('infra_commit', 0.0) or 0.0),
            carbon_offset_buy=float(raw.get('carbon_offset_buy', 0.0) or 0.0),
            put_hedge=float(raw.get('put_hedge', 0.0) or 0.0),
            tech_bet=raw.get('tech_bet', 'status_quo'),
        )
    except Exception:
        return None


def reward_structure(completion: str) -> float:
    """Reward the exact demo contract: closed think block, then JSON, no markdown."""
    text = with_prompt_think(completion).strip()
    score = 0.0

    matches = list(THINK_RE.finditer(text))
    if len(matches) == 1:
        score += 0.10
        think_end = matches[0].end()
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start > think_end and json_end > json_start:
            score += 0.10
            trailing = text[json_end + 1:].strip()
            if not trailing or trailing in {'<|endoftext|>', '</s>', '<|im_end|>'}:
                score += 0.05
    elif '<think>' in text and '</think>' not in text:
        score -= 0.10

    if '```' in text:
        score -= 0.05
    if text.count('{') == 1 and text.count('}') == 1:
        score += 0.05
    return score


def reward_brevity(completion: str) -> float:
    """Favor finished, compact reasoning instead of rambling to max tokens."""
    text = with_prompt_think(completion).strip()
    match = THINK_RE.search(text)
    if not match:
        return -0.10 if len(text) > 1200 else 0.0

    think_words = len(match.group(1).split())
    total_chars = len(text)
    score = 0.0
    if 45 <= think_words <= 180:
        score += 0.12
    elif 25 <= think_words < 45 or 180 < think_words <= 260:
        score += 0.04
    elif think_words > 260:
        score -= 0.12
    else:
        score -= 0.04

    if 350 <= total_chars <= 1200:
        score += 0.08
    elif total_chars > 1600:
        score -= 0.10
    return score


def reward_action_contract(completion: str) -> float:
    """Reward a valid, bounded, non-degenerate PortfolioAction JSON payload."""
    raw = json_payload_from_completion(completion)
    action = parse_action_from_base_completion(completion)
    if raw is None or action is None:
        return -0.30

    score = 0.15
    missing = ACTION_KEYS - set(raw)
    extra = set(raw) - ACTION_KEYS
    if not missing:
        score += 0.10
    else:
        score -= min(0.20, 0.05 * len(missing))
    if extra:
        score -= min(0.09, 0.03 * len(extra))

    weights = raw.get('weights')
    if isinstance(weights, list) and len(weights) == 5:
        try:
            raw_weights = [float(x) for x in weights]
        except (TypeError, ValueError):
            return -0.30
        raw_sum = sum(raw_weights)
        if all(0.0 <= w <= 1.0 for w in raw_weights):
            score += 0.10
        if abs(raw_sum - 1.0) <= 0.03:
            score += 0.10
        elif abs(raw_sum - 1.0) > 0.12:
            score -= 0.10
        if max(raw_weights) <= 0.75 and sum(1 for w in raw_weights if w >= 0.05) >= 2:
            score += 0.05

    for key, lo, hi in (
        ('infra_commit', 0.0, 0.2),
        ('carbon_offset_buy', 0.0, 0.1),
        ('put_hedge', 0.0, 0.05),
    ):
        if key not in raw:
            continue
        try:
            value = float(raw.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            score -= 0.05
            continue
        if lo <= value <= hi:
            score += 0.03
        else:
            score -= 0.08

    if raw.get('tech_bet', 'status_quo') in TECH_BETS:
        score += 0.06
    else:
        score -= 0.10
    return score


def simulate_episode(action, seed: int, phase: int = 1, steps: int = 4, shock_id: str | None = None):
    from portfolio_env import PortfolioEnv
    from portfolio_env.shocks import SHOCKS_BY_ID

    env = PortfolioEnv(phase=phase, seed=seed)
    env.reset(seed=seed)
    if shock_id and getattr(env, '_plan', None) is not None and shock_id in SHOCKS_BY_ID:
        env._plan.shocks_by_quarter[0] = SHOCKS_BY_ID[shock_id]
    for _ in range(steps):
        env.step(action, completion='')
    return env.trajectory


def make_reward_fn(component: str):
    from portfolio_env import r_carbon, r_drawdown, r_format, r_regret, r_sharpe

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        seeds = kwargs.get('seed', [42] * len(completions))
        shock_ids = kwargs.get('shock_id', [None] * len(completions))
        if isinstance(seeds, int):
            seeds = [seeds] * len(completions)
        if isinstance(shock_ids, str) or shock_ids is None:
            shock_ids = [shock_ids] * len(completions)
        scores: list[float] = []
        for raw_completion, seed, shock_id in zip(completions, seeds, shock_ids):
            text = completion_text(raw_completion)
            full_text = with_prompt_think(text)
            if component == 'format':
                scores.append(r_format(full_text))
                continue
            if component == 'structure':
                scores.append(reward_structure(text))
                continue
            if component == 'brevity':
                scores.append(reward_brevity(text))
                continue
            if component == 'action':
                scores.append(reward_action_contract(text))
                continue

            action = parse_action_from_base_completion(text)
            if action is None:
                scores.append(-0.5 if component == 'regret' else 0.0)
                continue

            traj = simulate_episode(action, int(seed), phase=1, steps=4, shock_id=shock_id)
            if component == 'regret':
                scores.append(r_regret(traj))
            elif component == 'sharpe':
                scores.append(r_sharpe(traj))
            elif component == 'drawdown':
                scores.append(r_drawdown(traj))
            elif component == 'carbon':
                scores.append(r_carbon(traj, phase_weight=0.0))
            else:
                scores.append(0.0)
        return scores

    reward_fn.__name__ = f'base_{component}_phase1'
    return reward_fn


def make_prompt_from_news(news: str) -> str:
    from portfolio_env.prompt import build_user_prompt

    return build_user_prompt(news)


def news_from_trace(trace: dict[str, Any]) -> str:
    raw = trace.get('raw')
    if isinstance(raw, dict) and isinstance(raw.get('news'), str):
        return raw['news']

    prompt = str(trace.get('prompt', ''))
    marker = "Today's news:\n"
    if marker in prompt:
        rest = prompt.split(marker, 1)[1]
        return rest.split('\n\nYour <think>', 1)[0].strip()
    return prompt


def sft_completion_tail(completion: str) -> str:
    stripped = completion.lstrip()
    if stripped.startswith(THINK_OPEN):
        return stripped[len(THINK_OPEN):]
    if stripped.startswith('<think>'):
        return stripped[len('<think>'):].lstrip('\n')
    return stripped


def load_sft_dataset(traces_path: Path, tokenizer):
    from datasets import Dataset

    rows = []
    with traces_path.open() as fh:
        for line in fh:
            trace = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': make_prompt_from_news(news_from_trace(trace))}],
                tokenize=False,
                add_generation_prompt=True,
            )
            text = prompt + sft_completion_tail(trace['completion']) + tokenizer.eos_token
            rows.append({'text': text})
    print(f'SFT rows: {len(rows)} from {traces_path}', flush=True)
    return Dataset.from_list(rows)


def build_grpo_dataset(n_prompts: int):
    import numpy as np
    from datasets import Dataset
    from portfolio_env import training_seeds
    from portfolio_env.shocks import shocks_available

    rng = np.random.default_rng(SEED)
    pool = [shock for shock in shocks_available(1) if 'PLACEHOLDER' not in shock.id]
    seeds = training_seeds(rng, n_prompts)
    rows = []
    for seed in seeds:
        shock = pool[int(rng.integers(0, len(pool)))]
        rows.append({
            'prompt': [{'role': 'user', 'content': make_prompt_from_news(shock.news)}],
            'seed': int(seed),
            'shock_id': shock.id,
            'news': shock.news,
        })
    return Dataset.from_list(rows)


def dry_run_template(traces_path: Path, limit: int = 3) -> None:
    """Local, non-GPU check that the custom template ends with `<think>`."""
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        tokenizer = OfflineTemplateTokenizer()
        print('transformers not installed; using offline template renderer.', flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    install_base_chat_template(tokenizer)
    print('Custom template installed.', flush=True)

    with traces_path.open() as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            trace = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': make_prompt_from_news(news_from_trace(trace))}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ok = rendered.endswith(THINK_OPEN)
            print(f'template row {i}: endswith_think={ok} chars={len(rendered)}', flush=True)
            if not ok:
                raise AssertionError('generation template did not end with <think> starter')


def dry_run_rewards(traces_path: Path, limit: int = 3) -> None:
    """Local, non-GPU check for the GRPO reward wrappers."""
    examples: list[tuple[str, str]] = []
    with traces_path.open() as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            trace = json.loads(line)
            examples.append((str(trace.get('id', f'trace_{i}')), trace['completion']))

    examples.extend([
        ('bad_missing_json', '<think>\nThis is unfinished reasoning without an action.'),
        ('bad_overlong', '<think>\n' + ('macro ' * 320) + '</think>\n{"weights":[0.2,0.2,0.2,0.2,0.2]}'),
        ('bad_action', '</think>\n{"weights":[1,1,1], "tech_bet":"moonshot"}'),
    ])

    components = ['format', 'structure', 'brevity', 'action', 'regret']
    reward_fns = {name: make_reward_fn(name) for name in components}
    for label, completion in examples:
        scores = {
            name: reward_fns[name](
                prompts=[[]],
                completions=[completion.removeprefix(THINK_OPEN)],
                seed=[42],
            )[0]
            for name in components
        }
        print(label, json.dumps(scores, sort_keys=True), flush=True)


class OfflineTemplateTokenizer:
    """Small local-only renderer for `--dry-run-template` without transformers."""

    eos_token = '<|endoftext|>'

    def __init__(self) -> None:
        self.chat_template = ''

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        from portfolio_env.prompt import SYSTEM_PROMPT

        if messages and messages[0]['role'] == 'system':
            text = messages[0]['content'] + self.eos_token
            loop_messages = messages[1:]
        else:
            text = SYSTEM_PROMPT + self.eos_token
            loop_messages = messages
        for message in loop_messages:
            if message['role'] == 'user':
                text += message['content']
            elif message['role'] == 'assistant':
                text += message['content'] + self.eos_token
        if add_generation_prompt:
            text += THINK_OPEN
        if tokenize:
            return list(range(len(text.split())))
        return text


def train_and_smoke(token: str, traces_path: Path) -> dict[str, Any]:
    os.environ.setdefault('UNSLOTH_VLLM_STANDBY', '1')

    from unsloth import FastLanguageModel, is_bfloat16_supported
    import numpy as np
    import torch
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    from vllm import SamplingParams

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Loading {MODEL_NAME}', flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=float(os.environ.get('CARBON_ALPHA_GPU_MEMORY_UTILIZATION', '0.9')),
        token=token,
    )
    install_base_chat_template(tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing='unsloth',
        random_state=SEED,
    )
    print(f'VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    sft_dataset = load_sft_dataset(traces_path, tokenizer)
    FastLanguageModel.for_training(model)
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR / 'sft'),
            dataset_text_field='text',
            max_steps=SFT_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            warmup_steps=5,
            logging_steps=5,
            optim='adamw_8bit',
            weight_decay=0.001,
            lr_scheduler_type='linear',
            seed=SEED,
            report_to='none',
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
        ),
    )
    t0 = time.time()
    sft_trainer.train()
    print(f'SFT done in {(time.time() - t0)/60:.1f} min', flush=True)

    sanity = generation_sanity_check(model, tokenizer)
    print('Generation sanity:', json.dumps(sanity, indent=2), flush=True)

    grpo_dataset = build_grpo_dataset(GRPO_PROMPTS)
    token_lengths = [
        len(tokenizer.apply_chat_template(row['prompt'], add_generation_prompt=True, tokenize=True))
        for row in grpo_dataset
    ]
    max_prompt_length = int(np.quantile(token_lengths, 0.9)) + 1
    max_completion_length = min(400, max(64, MAX_SEQ_LEN - max_prompt_length))
    print(
        f'GRPO prompt length p90={max_prompt_length}, max_completion={max_completion_length}',
        flush=True,
    )

    sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=SEED,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    grpo_args = GRPOConfig(
        vllm_sampling_params=sampling_params,
        output_dir=str(OUTPUT_DIR / 'grpo_phase1_smoke'),
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        optim='adamw_8bit',
        logging_steps=1,
        per_device_train_batch_size=GRPO_BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_generations=GRPO_NUM_GENERATIONS,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=GRPO_MAX_STEPS,
        save_steps=GRPO_MAX_STEPS,
        report_to='none',
    )
    reward_fns = [
        make_reward_fn('format'),
        make_reward_fn('structure'),
        make_reward_fn('brevity'),
        make_reward_fn('action'),
        make_reward_fn('regret'),
    ]
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        train_dataset=grpo_dataset,
        args=grpo_args,
    )
    t0 = time.time()
    trainer.train()
    print(f'GRPO smoke done in {(time.time() - t0)/60:.1f} min', flush=True)

    metrics = extract_smoke_metrics(trainer)
    print('Smoke metrics:', json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    gate = smoke_gate_passed(metrics, sanity)
    metrics['smoke_gate_passed'] = gate
    if gate:
        holdout = evaluate_holdout(model, tokenizer)
        metrics['holdout_eval'] = holdout
        mean_regret = holdout.get('mean_regret')
        metrics['beats_v6_sft_mean_regret'] = bool(
            mean_regret is not None and mean_regret > 0.034
        )
        print('Holdout eval:', json.dumps(holdout, indent=2), flush=True)

    final_path = OUTPUT_DIR / 'final_adapter'
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f'Saved adapter to {final_path}', flush=True)
    return {'metrics': metrics, 'artifact_path': str(final_path)}


def generation_sanity_check(model, tokenizer, n: int = 5) -> dict[str, Any]:
    import torch
    from portfolio_env.shocks import shocks_available

    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    shocks = [shock for shock in shocks_available(1) if 'PLACEHOLDER' not in shock.id][:n]
    results = []
    for shock in shocks:
        prompt = [{'role': 'user', 'content': make_prompt_from_news(shock.news)}]
        rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
            )
        completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        action = parse_action_from_base_completion(completion)
        results.append({
            'shock': shock.id,
            'chars': len(completion),
            'tokens': int(out.shape[1] - inputs['input_ids'].shape[1]),
            'valid_action': action is not None,
            'preview': completion[:180],
        })
    valid = sum(1 for row in results if row['valid_action'])
    lengths = [row['tokens'] for row in results]
    return {
        'valid_actions': valid,
        'total': len(results),
        'mean_tokens': sum(lengths) / max(1, len(lengths)),
        'min_tokens': min(lengths) if lengths else 0,
        'max_tokens': max(lengths) if lengths else 0,
        'samples': results,
    }


def evaluate_holdout(model, tokenizer) -> dict[str, Any]:
    import numpy as np
    import torch
    from portfolio_env import holdout_seeds, r_regret
    from portfolio_env.shocks import shocks_available
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    results: dict[int, dict[str, Any]] = {}
    for seed in holdout_seeds():
        rng = np.random.default_rng(seed)
        pool = [shock for shock in shocks_available(3) if 'PLACEHOLDER' not in shock.id]
        shock = pool[int(rng.integers(0, len(pool)))]
        prompt = [{'role': 'user', 'content': make_prompt_from_news(shock.news)}]
        rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
            )
        completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        action = parse_action_from_base_completion(completion)
        if action is None:
            results[int(seed)] = {
                'valid': False,
                'regret': None,
                'shock': shock.id,
                'tokens': int(out.shape[1] - inputs['input_ids'].shape[1]),
            }
            continue
        traj = simulate_episode(action, int(seed), phase=3, steps=12, shock_id=shock.id)
        results[int(seed)] = {
            'valid': True,
            'regret': float(r_regret(traj)),
            'shock': shock.id,
            'final_nav_real': float(traj.nav_real_series[-1]),
            'tokens': int(out.shape[1] - inputs['input_ids'].shape[1]),
        }

    valid_regrets = [row['regret'] for row in results.values() if row['valid']]
    return {
        'valid': len(valid_regrets),
        'total': len(results),
        'mean_regret': float(np.mean(valid_regrets)) if valid_regrets else None,
        'beats_baseline': sum(1 for regret in valid_regrets if regret > 0),
        'v6_sft_mean_regret_bar': 0.034,
        'results': results,
    }


def extract_smoke_metrics(trainer) -> dict[str, Any]:
    rows = getattr(trainer.state, 'log_history', []) or []
    last = rows[-1] if rows else {}
    merged: dict[str, Any] = {'log_rows': len(rows)}
    for row in rows:
        for key, value in row.items():
            merged[key] = value
    merged['last'] = last
    return merged


def metric_float(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is None and isinstance(metrics.get('last'), dict):
            value = metrics['last'].get(name)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(out):
            return out
    return None


def smoke_gate_passed(metrics: dict[str, Any], sanity: dict[str, Any]) -> bool:
    mean_len = metric_float(
        metrics,
        'completions/mean_length',
        'completion_length',
        'completion_length/mean',
    )
    min_len = metric_float(metrics, 'completions/min_length', 'completion_length/min')
    grad = metric_float(metrics, 'grad_norm')
    reward_std = metric_float(
        metrics,
        'rewards/base_regret_phase1/std',
        'rewards/base_action_phase1/std',
        'rewards/base_structure_phase1/std',
        'rewards/base_brevity_phase1/std',
        'rewards/base_format_phase1/std',
        'reward_std',
        'rewards/reward_std',
        'reward/standalone_std',
    )

    if sanity.get('valid_actions', 0) < 3:
        return False
    if mean_len is None or mean_len <= 50:
        return False
    if min_len is None or min_len <= 1:
        return False
    if grad is None or grad <= 0:
        return False
    if reward_std is None or reward_std <= 0:
        return False
    return True


def upload_artifacts(token: str, artifact_path: Path, log_path: Path | None, metrics: dict[str, Any]) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(artifact_path),
        repo_id=MODEL_REPO,
        repo_type='model',
        path_in_repo=RUN_LABEL,
        commit_message=f'{RUN_LABEL}: Qwen3-4B-Base GRPO smoke artifact',
        token=token,
    )
    metrics_path = WORK / 'smoke_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        repo_id=MODEL_REPO,
        repo_type='model',
        path_in_repo=f'{RUN_LABEL}/smoke_metrics.json',
        commit_message=f'{RUN_LABEL}: smoke metrics',
        token=token,
    )
    if log_path and log_path.exists():
        api.upload_file(
            path_or_fileobj=str(log_path),
            repo_id=MODEL_REPO,
            repo_type='model',
            path_in_repo=f'{RUN_LABEL}/training.log',
            commit_message=f'{RUN_LABEL}: training log',
            token=token,
        )
    print(f'Uploaded artifacts to {MODEL_REPO}/{RUN_LABEL}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run-template', action='store_true')
    parser.add_argument('--dry-run-rewards', action='store_true')
    parser.add_argument('--check-hf', action='store_true')
    parser.add_argument('--skip-upload', action='store_true')
    parser.add_argument('--local-code', action='store_true',
                        help='Use current checkout instead of downloading CarbonAlpha-train.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv_for_local()
    token = select_hf_token()

    if args.check_hf:
        check_hf_access(token)

    if args.local_code:
        sys.path.insert(0, str(Path.cwd()))
        traces_path = Path(TRACES)
    else:
        code_dir = download_code_bundle(token)
        traces_path = code_dir / TRACES

    if args.dry_run_template:
        dry_run_template(traces_path)
        return
    if args.dry_run_rewards:
        dry_run_rewards(traces_path)
        return

    check_hf_access(token)
    result = train_and_smoke(token, traces_path)
    artifact = Path(result['artifact_path'])
    metrics = result['metrics']
    if not metrics.get('smoke_gate_passed'):
        print('! Smoke gate failed. Uploading diagnostics only; v6 SFT remains final-safe.', flush=True)
        if not args.skip_upload:
            metrics_path = WORK / 'smoke_metrics_failed.json'
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
            from huggingface_hub import HfApi
            HfApi(token=token).upload_file(
                path_or_fileobj=str(metrics_path),
                repo_id=MODEL_REPO,
                repo_type='model',
                path_in_repo=f'{RUN_LABEL}/smoke_metrics_failed.json',
                commit_message=f'{RUN_LABEL}: failed smoke metrics',
                token=token,
            )
        sys.exit(2)

    if not args.skip_upload:
        upload_artifacts(token, artifact, None, metrics)


if __name__ == '__main__':
    main()
