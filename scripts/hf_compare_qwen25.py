# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "huggingface_hub>=0.34",
#   "openenv-core>=0.2",
#   "fastapi",
#   "pydantic",
#   "uvicorn",
#   "transformers==4.56.2",
#   "unsloth",
#   "torchvision",
#   "bitsandbytes",
#   "xformers",
#   "peft",
#   "accelerate",
#   "numpy",
# ]
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# ///
"""Compare base Qwen2.5-7B-Instruct vs CarbonAlpha trained adapter.

Run on HF Jobs:
    hf jobs uv run --flavor l40sx1 --secrets HF_API_TOKEN scripts/hf_compare_qwen25.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


WORK = Path(os.environ.get('CARBON_ALPHA_WORK_DIR', '/tmp/CarbonAlphaCompare'))
CODE_REPO = os.environ.get('CARBON_ALPHA_CODE_REPO', '77ethers/CarbonAlpha-train')
MODEL_REPO = os.environ.get('CARBON_ALPHA_MODEL_REPO', '77ethers/CarbonAlpha')
BASE_MODEL = os.environ.get('CARBON_ALPHA_BASE_MODEL', 'unsloth/Qwen2.5-7B-Instruct')
ADAPTER_SUBDIR = os.environ.get('CARBON_ALPHA_ADAPTER_SUBDIR', 'sft_qwen25_7b_curriculum400_v1')
MAX_NEW_TOKENS = int(os.environ.get('CARBON_ALPHA_COMPARE_MAX_NEW_TOKENS', '420'))


DEFAULT_NEWS_CASES = [
    {
        'id': 'yen_carry_trade_debacle',
        'news': (
            'The Bank of Japan unexpectedly hikes rates and the yen rallies 9% in two sessions. '
            'Prime brokers report forced deleveraging as global macro funds unwind yen-funded carry trades. '
            'Equity futures fall sharply, credit spreads widen, and sovereign bond futures rally.'
        ),
    },
    {
        'id': 'bitcoin_legalization',
        'news': (
            'The US, EU, Japan, and India finalize laws allowing regulated Bitcoin custody and spot trading '
            'inside major banks and brokerages. No changes are announced to monetary policy, energy policy, '
            'or fiscal spending.'
        ),
    },
    {
        'id': 'political_scandal_noise',
        'news': (
            'A prominent national politician is caught in a personal scandal that dominates cable news. '
            'There are no resignations affecting fiscal policy, no election timeline changes, and no new '
            'trade, energy, or central-bank announcements.'
        ),
    },
    {
        'id': 'bank_credit_stress',
        'news': (
            'Several regional banks and private credit funds halt redemptions after sudden commercial real '
            'estate loan losses. Short-term funding markets tighten, bank equity indices plunge, and Treasury '
            'yields fall on safe-haven demand.'
        ),
    },
    {
        'id': 'energy_supply_shock',
        'news': (
            'A major shipping chokepoint closes after military escalation, removing 3 million barrels per day '
            'of oil supply from global markets. Brent crude jumps 14%, inflation breakevens rise, and central '
            'banks signal policy will stay restrictive.'
        ),
    },
]


def news_cases() -> list[dict[str, str]]:
    raw = os.environ.get('CARBON_ALPHA_COMPARE_CASES_JSON')
    if not raw:
        return DEFAULT_NEWS_CASES
    cases = json.loads(raw)
    if not isinstance(cases, list):
        raise ValueError('CARBON_ALPHA_COMPARE_CASES_JSON must be a JSON list')
    for case in cases:
        if not isinstance(case, dict) or 'id' not in case or 'news' not in case:
            raise ValueError(f'bad compare case: {case}')
    return cases


def token() -> str:
    tok = os.environ.get('HF_API_TOKEN') or os.environ.get('HF_TOKEN')
    if not tok:
        raise RuntimeError('HF_API_TOKEN or HF_TOKEN required')
    os.environ['HF_TOKEN'] = tok
    os.environ['HUGGINGFACE_HUB_TOKEN'] = tok
    return tok


def download_code(tok: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    code_dir = Path(snapshot_download(
        repo_id=CODE_REPO,
        repo_type='dataset',
        token=tok,
        local_dir=str(WORK / 'code'),
    ))
    sys.path.insert(0, str(code_dir))
    return code_dir


def prompt_for_news(news: str) -> str:
    from portfolio_env.prompt import SYSTEM_PROMPT, build_user_prompt

    return SYSTEM_PROMPT + '\n\n' + build_user_prompt(news)


def completion_text(tokenizer, output_ids, input_len: int) -> str:
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def parse_action(completion: str) -> dict[str, Any] | None:
    start = completion.find('{')
    end = completion.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(completion[start:end + 1])
    except Exception:
        return None
    weights = obj.get('weights')
    if not isinstance(weights, list) or len(weights) != 5:
        return None
    return obj


def score_completion(completion: str) -> dict[str, Any]:
    action = parse_action(completion)
    lower = completion.lower()
    return {
        'valid_action': action is not None,
        'closed_think': '<think>' in completion and '</think>' in completion,
        'tokens_approx': len(re.findall(r'\S+', completion)),
        'weights': action.get('weights') if action else None,
        'put_hedge': action.get('put_hedge') if action else None,
        'tech_bet': action.get('tech_bet') if action else None,
        'mentions_not_assuming': 'not assuming' in lower or 'not assume' in lower,
        'unsupported_oil_surge_phrase': any(
            phrase in lower for phrase in ('oil demand surge', 'energy demand surge', 'energy demand surges')
        ),
    }


def generate_all(model, tokenizer, label: str, cases: list[dict[str, str]]) -> dict[str, Any]:
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    results = {}
    for case in cases:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt_for_news(case['news'])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = completion_text(tokenizer, out, inputs['input_ids'].shape[1])
        results[case['id']] = {
            'news': case['news'],
            'completion': completion,
            'score': score_completion(completion),
        }
        print(f'[{label}] {case["id"]}: {json.dumps(results[case["id"]]["score"], sort_keys=True)}', flush=True)
    return results


def load_base(tok: str):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        load_in_4bit=True,
        token=tok,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_trained(tok: str):
    from peft import PeftModel

    model, tokenizer = load_base(tok)
    adapter_dir = Path(snapshot_download(
        repo_id=MODEL_REPO,
        repo_type='model',
        token=tok,
        allow_patterns=[f'{ADAPTER_SUBDIR}/*'],
        local_dir=str(WORK / 'model'),
    )) / ADAPTER_SUBDIR
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    return model, tokenizer


def main() -> None:
    tok = token()
    cases = news_cases()
    download_code(tok)

    print('=== Loading base model ===', flush=True)
    base_model, base_tokenizer = load_base(tok)
    base_results = generate_all(base_model, base_tokenizer, 'base', cases)
    del base_model

    print('=== Loading trained adapter ===', flush=True)
    trained_model, trained_tokenizer = load_trained(tok)
    trained_results = generate_all(trained_model, trained_tokenizer, 'trained', cases)

    report = {
        'base_model': BASE_MODEL,
        'trained_adapter': f'{MODEL_REPO}/{ADAPTER_SUBDIR}',
        'cases': cases,
        'base': base_results,
        'trained': trained_results,
    }
    out_path = WORK / 'qwen25_compare_report.json'
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f'COMPARE_REPORT_JSON={out_path}', flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
