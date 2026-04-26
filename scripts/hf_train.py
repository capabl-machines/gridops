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
"""HF Jobs entrypoint for CarbonAlpha GRPO training.

Pulls the code bundle from HF dataset `77ethers/CarbonAlpha-train`, runs the
canonical Unsloth GRPO recipe (4-bit + vLLM rollout + bf16 training), and
uploads the trained LoRA adapter to HF model repo `77ethers/CarbonAlpha`.

Run via:
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN \\
        scripts/hf_train.py
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi


WORK = Path('/tmp/CarbonAlpha')
WORK.mkdir(parents=True, exist_ok=True)

print('=== Step 1: download code bundle from HF ===', flush=True)
code_dir = snapshot_download(
    repo_id='77ethers/CarbonAlpha-train',
    repo_type='dataset',
    local_dir=str(WORK / 'code'),
)
print(f'  code at: {code_dir}', flush=True)

# Make portfolio_env package importable
sys.path.insert(0, code_dir)
os.chdir(code_dir)

# Configure run from env vars (override-able from `hf jobs uv run --env`)
TRACES = os.environ.get('CARBON_ALPHA_TRACES', 'sft_traces/traces_v2.jsonl')
PHASE = os.environ.get('CARBON_ALPHA_PHASE', 'all')   # 'all' | '1' | '2' | '3' | 'sft-only'
SFT_STEPS = os.environ.get('CARBON_ALPHA_SFT_STEPS', '150')
RUN_LABEL = os.environ.get('CARBON_ALPHA_RUN_LABEL', f'{Path(TRACES).stem}_{PHASE}')

print(f'\n=== Step 2: launch GRPO training (--phase {PHASE}, traces={TRACES}, label={RUN_LABEL}) ===', flush=True)
log_path = WORK / 'train.log'
proc = subprocess.Popen(
    [
        sys.executable, 'notebooks/grpo_training.py',
        '--phase', PHASE,
        '--sft-traces', TRACES,
        '--sft-steps', SFT_STEPS,
    ],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
with log_path.open('w') as fh:
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        fh.write(line)
rc = proc.wait()
print(f'\n  training rc={rc}', flush=True)
if rc != 0:
    sys.exit(rc)

print('\n=== Step 3: upload LoRA adapter to HF ===', flush=True)
checkpoint_dir = Path(code_dir) / '/workspace/checkpoints/final_merged'
# grpo_training.py writes to OUTPUT_DIR=/workspace/checkpoints; on HF Jobs
# that path needs to exist. We'll point at whatever it actually wrote to.
candidates = [
    Path(os.environ.get('CARBON_ALPHA_OUTPUT_DIR', '/workspace/checkpoints')) / 'final_merged',
    Path('/workspace/checkpoints/final_merged'),
    Path(code_dir) / 'checkpoints/final_merged',
    WORK / 'checkpoints/final_merged',
]
adapter_path = next((p for p in candidates if p.exists()), None)
if adapter_path is None:
    print('  ! no adapter dir found, listing /workspace and code_dir:', flush=True)
    for d in ['/workspace', code_dir]:
        for p in Path(d).rglob('adapter_config.json'):
            print(f'    found: {p}', flush=True)
            adapter_path = p.parent
            break
        if adapter_path:
            break

if adapter_path:
    api = HfApi()
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id='77ethers/CarbonAlpha',
        repo_type='model',
        path_in_repo=RUN_LABEL,
        commit_message=f'{RUN_LABEL}: phase={PHASE} traces={TRACES}',
    )
    print(f'  ✓ uploaded {adapter_path} to 77ethers/CarbonAlpha/{RUN_LABEL}', flush=True)
else:
    print('  ✗ no LoRA adapter found to upload', flush=True)
    sys.exit(1)

# Also upload the training log
api = HfApi()
api.upload_file(
    path_or_fileobj=str(log_path),
    path_in_repo=f'{RUN_LABEL}/training.log',
    repo_id='77ethers/CarbonAlpha',
    repo_type='model',
    commit_message=f'{RUN_LABEL}: training log',
)
print('  ✓ uploaded training log', flush=True)
print('\n=== DONE ===', flush=True)
