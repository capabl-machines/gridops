# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "huggingface_hub>=0.34",
#   "openenv-core>=0.2",
#   "fastapi",
#   "pydantic",
#   "uvicorn",
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
"""HF Jobs entrypoint for the Qwen2.5-7B-Instruct SFT rescue run.

This is deliberately SFT-first. Qwen2.5-7B-Instruct is the format-control
pivot: teach the exact `<think>...</think>` + JSON contract on the curriculum
traces, run holdout/demo checks, and only consider GRPO if this beats v6 or
produces clearly better samples.

Recommended launch:
    hf jobs uv run --flavor l40sx1 --secrets HF_API_TOKEN \\
        --env CARBON_ALPHA_TRACES=sft_traces/curriculum_400_e80_m160_h160.jsonl \\
        --env CARBON_ALPHA_RUN_LABEL=sft_qwen25_7b_curriculum_v1 \\
        scripts/hf_sft_qwen25_7b.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


WORK = Path(os.environ.get('CARBON_ALPHA_WORK_DIR', '/tmp/CarbonAlphaQwen25SFT'))
CODE_REPO = os.environ.get('CARBON_ALPHA_CODE_REPO', '77ethers/CarbonAlpha-train')
MODEL_REPO = os.environ.get('CARBON_ALPHA_MODEL_REPO', '77ethers/CarbonAlpha')
MODEL_NAME = os.environ.get('CARBON_ALPHA_BASE_MODEL', 'unsloth/Qwen2.5-7B-Instruct')
TRACES = os.environ.get('CARBON_ALPHA_TRACES', 'sft_traces/curriculum_400_e80_m160_h160.jsonl')
RUN_LABEL = os.environ.get('CARBON_ALPHA_RUN_LABEL', 'sft_qwen25_7b_curriculum_v1')
OUTPUT_DIR = Path(os.environ.get('CARBON_ALPHA_OUTPUT_DIR', str(WORK / 'checkpoints')))

MAX_SEQ_LEN = int(os.environ.get('CARBON_ALPHA_MAX_SEQ_LEN', '4096'))
LORA_RANK = int(os.environ.get('CARBON_ALPHA_LORA_RANK', '16'))
LORA_ALPHA = int(os.environ.get('CARBON_ALPHA_LORA_ALPHA', str(LORA_RANK)))
SFT_STEPS = int(os.environ.get('CARBON_ALPHA_SFT_STEPS', '220'))
BATCH_SIZE = int(os.environ.get('CARBON_ALPHA_BATCH_SIZE', '1'))
GRAD_ACCUM = int(os.environ.get('CARBON_ALPHA_GRAD_ACCUM', '4'))
LR = float(os.environ.get('CARBON_ALPHA_LR', '1e-4'))
SAVE_METHOD = os.environ.get('CARBON_ALPHA_SAVE_METHOD', 'lora')
SEED = int(os.environ.get('CARBON_ALPHA_SEED', '3407'))
V6_MEAN_REGRET = 0.034


def load_dotenv_for_local() -> None:
    env_path = Path.cwd() / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


def select_hf_token() -> str:
    token = os.environ.get('HF_API_TOKEN')
    if not token:
        token = os.environ.get('HF_TOKEN')
        if token:
            print('! HF_API_TOKEN missing; falling back to HF_TOKEN', flush=True)
    if not token:
        raise RuntimeError('HF_API_TOKEN is required for CarbonAlpha private repos')
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


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[0].get('content', '') if completion else ''
    return str(completion)


def parse_action_from_completion(completion: str):
    from portfolio_env import PortfolioAction, parse_json_action

    raw = parse_json_action(completion)
    if raw is None or not isinstance(raw, dict):
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


def simulate_episode(action, seed: int, phase: int = 3, steps: int = 12):
    from portfolio_env import PortfolioEnv

    env = PortfolioEnv(phase=phase, seed=seed)
    env.reset(seed=seed)
    for _ in range(steps):
        env.step(action, completion='')
    return env.trajectory


def make_prompt_from_news(news: str) -> str:
    from portfolio_env.prompt import SYSTEM_PROMPT, build_user_prompt

    return SYSTEM_PROMPT + '\n\n' + build_user_prompt(news)


def load_sft_dataset(traces_path: Path, tokenizer):
    from datasets import Dataset

    rows = []
    by_category: dict[str, int] = {}
    with traces_path.open() as fh:
        for line in fh:
            trace = json.loads(line)
            text = tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': trace['prompt']},
                    {'role': 'assistant', 'content': trace['completion']},
                ],
                tokenize=False,
            )
            rows.append({'text': text})
            category = str(trace.get('seed_category', 'unknown'))
            by_category[category] = by_category.get(category, 0) + 1
    print(f'SFT rows: {len(rows)} from {traces_path}; categories={by_category}', flush=True)
    return Dataset.from_list(rows)


def train_and_eval(token: str, traces_path: Path) -> dict[str, Any]:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    from trl import SFTConfig, SFTTrainer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Loading {MODEL_NAME}', flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=SEED,
    )
    print(f'VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    dataset = load_sft_dataset(traces_path, tokenizer)
    FastLanguageModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR / 'sft'),
            dataset_text_field='text',
            max_seq_length=MAX_SEQ_LEN,
            packing=False,
            max_steps=SFT_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_steps=10,
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
    trainer.train()
    print(f'SFT done in {(time.time() - t0)/60:.1f} min', flush=True)

    sanity = generation_sanity_check(model, tokenizer)
    print('Generation sanity:', json.dumps(sanity, indent=2), flush=True)
    holdout = evaluate_holdout(model, tokenizer)
    print('Holdout eval:', json.dumps(holdout, indent=2), flush=True)

    final_path = OUTPUT_DIR / 'final'
    if SAVE_METHOD == 'merged_16bit':
        model.save_pretrained_merged(str(final_path), tokenizer, save_method='merged_16bit')
    else:
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
    print(f'Saved {SAVE_METHOD} artifact to {final_path}', flush=True)

    metrics = {
        'model_name': MODEL_NAME,
        'run_label': RUN_LABEL,
        'traces': str(traces_path),
        'sft_steps': SFT_STEPS,
        'lora_rank': LORA_RANK,
        'lora_alpha': LORA_ALPHA,
        'save_method': SAVE_METHOD,
        'generation_sanity': sanity,
        'holdout_eval': holdout,
        'beats_v6_sft_mean_regret': (
            holdout.get('mean_regret') is not None and holdout['mean_regret'] > V6_MEAN_REGRET
        ),
        'v6_sft_mean_regret_bar': V6_MEAN_REGRET,
    }
    metrics_path = WORK / 'qwen25_sft_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return {'artifact_path': str(final_path), 'metrics_path': str(metrics_path), 'metrics': metrics}


def generation_sanity_check(model, tokenizer, n: int = 5) -> dict[str, Any]:
    import torch
    from portfolio_env.shocks import shocks_available
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    shocks = shocks_available(1)[:n]
    rows = []
    for shock in shocks:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': make_prompt_from_news(shock.news)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=420,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action = parse_action_from_completion(completion)
        rows.append({
            'shock': shock.id,
            'valid_action': action is not None,
            'tokens': int(out.shape[1] - inputs['input_ids'].shape[1]),
            'chars': len(completion),
            'has_closed_think': '<think>' in completion and '</think>' in completion,
            'preview': completion[:220],
        })
    lengths = [row['tokens'] for row in rows]
    return {
        'valid_actions': sum(1 for row in rows if row['valid_action']),
        'closed_think': sum(1 for row in rows if row['has_closed_think']),
        'total': len(rows),
        'mean_tokens': sum(lengths) / max(1, len(lengths)),
        'min_tokens': min(lengths) if lengths else 0,
        'max_tokens': max(lengths) if lengths else 0,
        'samples': rows,
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
        pool = shocks_available(3)
        shock = pool[int(rng.integers(0, len(pool)))]
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': make_prompt_from_news(shock.news)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=420,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action = parse_action_from_completion(completion)
        if action is None:
            results[int(seed)] = {
                'valid': False,
                'regret': None,
                'shock': shock.id,
                'tokens': int(out.shape[1] - inputs['input_ids'].shape[1]),
                'preview': completion[:220],
            }
            continue
        traj = simulate_episode(action, int(seed), phase=3, steps=12)
        results[int(seed)] = {
            'valid': True,
            'regret': float(r_regret(traj)),
            'shock': shock.id,
            'final_nav_real': float(traj.nav_real_series[-1]),
            'tokens': int(out.shape[1] - inputs['input_ids'].shape[1]),
            'preview': completion[:220],
        }

    valid_regrets = [row['regret'] for row in results.values() if row['valid']]
    return {
        'valid': len(valid_regrets),
        'total': len(results),
        'mean_regret': float(np.mean(valid_regrets)) if valid_regrets else None,
        'beats_baseline': sum(1 for regret in valid_regrets if regret > 0),
        'v6_sft_mean_regret_bar': V6_MEAN_REGRET,
        'results': results,
    }


def upload_artifacts(token: str, artifact_path: Path, metrics_path: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(artifact_path),
        repo_id=MODEL_REPO,
        repo_type='model',
        path_in_repo=RUN_LABEL,
        commit_message=f'{RUN_LABEL}: Qwen2.5-7B-Instruct SFT artifact',
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        repo_id=MODEL_REPO,
        repo_type='model',
        path_in_repo=f'{RUN_LABEL}/metrics.json',
        commit_message=f'{RUN_LABEL}: metrics',
        token=token,
    )
    print(f'Uploaded artifacts to {MODEL_REPO}/{RUN_LABEL}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        return

    if args.local_code:
        sys.path.insert(0, str(Path.cwd()))
        traces_path = Path(TRACES)
    else:
        code_dir = download_code_bundle(token)
        traces_path = code_dir / TRACES

    check_hf_access(token)
    result = train_and_eval(token, traces_path)
    if not args.skip_upload:
        upload_artifacts(token, Path(result['artifact_path']), Path(result['metrics_path']))


if __name__ == '__main__':
    main()
