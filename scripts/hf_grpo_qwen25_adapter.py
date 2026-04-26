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
#   "peft",
#   "bitsandbytes",
#   "datasets",
#   "accelerate",
#   "numpy",
#   "torchvision",
#   "pillow",
#   "matplotlib",
# ]
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# ///
"""HF Jobs GRPO smoke for the trained Qwen2.5-7B SFT adapter.

This is the conservative GRPO rescue path:
- start from the already-good SFT adapter;
- avoid vLLM entirely (`use_vllm=False`) because prior failures were rollout
  / stop-token related;
- run only a tiny smoke by default;
- upload only if the smoke gate passes.

Recommended smoke:
    hf jobs uv run --flavor l40sx1 --secrets HF_API_TOKEN \\
        scripts/hf_grpo_qwen25_adapter.py

Useful overrides:
    --env CARBON_ALPHA_GRPO_STEPS=8
    --env CARBON_ALPHA_GRPO_NUM_GENERATIONS=2
    --env CARBON_ALPHA_GRPO_RUN_LABEL=grpo_qwen25_7b_adapter_smoke_v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


WORK = Path(os.environ.get("CARBON_ALPHA_WORK_DIR", "/tmp/CarbonAlphaQwen25GRPO"))
CODE_REPO = os.environ.get("CARBON_ALPHA_CODE_REPO", "77ethers/CarbonAlpha-train")
MODEL_REPO = os.environ.get("CARBON_ALPHA_MODEL_REPO", "77ethers/CarbonAlpha")
BASE_MODEL = os.environ.get("CARBON_ALPHA_BASE_MODEL", "unsloth/Qwen2.5-7B-Instruct")
SFT_SUBFOLDER = os.environ.get("CARBON_ALPHA_SFT_SUBFOLDER", "sft_qwen25_7b_curriculum400_v1")
RUN_LABEL = os.environ.get("CARBON_ALPHA_GRPO_RUN_LABEL", "grpo_qwen25_7b_adapter_smoke_v1")
OUTPUT_DIR = Path(os.environ.get("CARBON_ALPHA_OUTPUT_DIR", str(WORK / "checkpoints")))

MAX_PROMPT_LENGTH = int(os.environ.get("CARBON_ALPHA_MAX_PROMPT_LENGTH", "1536"))
MAX_COMPLETION_LENGTH = int(os.environ.get("CARBON_ALPHA_MAX_COMPLETION_LENGTH", "420"))
GRPO_STEPS = int(os.environ.get("CARBON_ALPHA_GRPO_STEPS", "8"))
GRPO_PROMPTS = int(os.environ.get("CARBON_ALPHA_GRPO_PROMPTS", "32"))
NUM_GENERATIONS = int(os.environ.get("CARBON_ALPHA_GRPO_NUM_GENERATIONS", "2"))
PER_DEVICE_BATCH = int(os.environ.get("CARBON_ALPHA_GRPO_BATCH", str(NUM_GENERATIONS)))
GRAD_ACCUM = int(os.environ.get("CARBON_ALPHA_GRPO_GRAD_ACCUM", "1"))
LR = float(os.environ.get("CARBON_ALPHA_GRPO_LR", "2e-6"))
SEED = int(os.environ.get("CARBON_ALPHA_SEED", "3407"))

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
TECH_BETS = {"status_quo", "green_leaps", "carbon_priced", "inflationary", "fragmentation"}
ACTION_KEYS = {"weights", "infra_commit", "carbon_offset_buy", "put_hedge", "tech_bet"}
V6_MEAN_REGRET = 0.034


def load_dotenv_for_local() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def select_hf_token() -> str:
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        token = os.environ.get("HF_TOKEN")
        if token:
            print("! HF_API_TOKEN missing; falling back to HF_TOKEN", flush=True)
    if not token:
        raise RuntimeError("HF_API_TOKEN is required for CarbonAlpha private repos")
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


def check_hf_access(token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami(token=token)
    print(f"HF auth OK: {who.get('name')}", flush=True)
    for repo_id, repo_type in ((MODEL_REPO, "model"), (CODE_REPO, "dataset")):
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
        print(
            f"HF access OK: {repo_type}:{repo_id} "
            f"private={getattr(info, 'private', None)} files={len(files)}",
            flush=True,
        )


def download_code_bundle(token: str) -> Path:
    from huggingface_hub import snapshot_download

    WORK.mkdir(parents=True, exist_ok=True)
    code_dir = snapshot_download(
        repo_id=CODE_REPO,
        repo_type="dataset",
        token=token,
        local_dir=str(WORK / "code"),
    )
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)
    print(f"Code bundle: {code_dir}", flush=True)
    return Path(code_dir)


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    return str(completion)


def parse_action(completion: str):
    from portfolio_env import PortfolioAction, parse_json_action

    raw = parse_json_action(completion)
    if raw is None or not isinstance(raw, dict):
        return None
    weights = raw.get("weights")
    if not isinstance(weights, list) or len(weights) != 5:
        return None
    try:
        return PortfolioAction(
            weights=[max(0.0, float(x)) for x in weights],
            infra_commit=float(raw.get("infra_commit", 0.0) or 0.0),
            carbon_offset_buy=float(raw.get("carbon_offset_buy", 0.0) or 0.0),
            put_hedge=float(raw.get("put_hedge", 0.0) or 0.0),
            tech_bet=raw.get("tech_bet", "status_quo"),
        )
    except Exception:
        return None


def reward_format(completions, **kwargs) -> list[float]:
    from portfolio_env import r_format

    return [float(r_format(completion_text(c))) for c in completions]


def reward_action_contract(completions, **kwargs) -> list[float]:
    from portfolio_env import parse_json_action

    scores = []
    for raw_completion in completions:
        text = completion_text(raw_completion)
        raw = parse_json_action(text)
        action = parse_action(text)
        if raw is None or action is None:
            scores.append(-0.40)
            continue

        score = 0.15
        missing = ACTION_KEYS - set(raw)
        extra = set(raw) - ACTION_KEYS
        score += 0.10 if not missing else -min(0.20, 0.05 * len(missing))
        score -= min(0.09, 0.03 * len(extra))

        weights = raw.get("weights")
        if isinstance(weights, list) and len(weights) == 5:
            try:
                raw_weights = [float(x) for x in weights]
            except Exception:
                scores.append(-0.40)
                continue
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
            ("infra_commit", 0.0, 0.2),
            ("carbon_offset_buy", 0.0, 0.1),
            ("put_hedge", 0.0, 0.05),
        ):
            try:
                value = float(raw.get(key, 0.0) or 0.0)
            except Exception:
                score -= 0.05
                continue
            score += 0.03 if lo <= value <= hi else -0.08

        score += 0.06 if raw.get("tech_bet", "status_quo") in TECH_BETS else -0.10
        scores.append(float(score))
    return scores


def reward_reasoning_shape(completions, **kwargs) -> list[float]:
    scores = []
    for raw_completion in completions:
        text = completion_text(raw_completion).strip()
        match = THINK_RE.search(text)
        if not match:
            scores.append(-0.15 if len(text) < 50 else -0.05)
            continue
        words = len(match.group(1).split())
        score = 0.0
        if 45 <= words <= 220:
            score += 0.12
        elif 25 <= words < 45 or 220 < words <= 300:
            score += 0.04
        else:
            score -= 0.08
        if "```" in text:
            score -= 0.05
        scores.append(float(score))
    return scores


def simulate_episode(action, seed: int, shock_id: str | None, phase: int = 1, steps: int = 4):
    from portfolio_env import PortfolioEnv
    from portfolio_env.shocks import SHOCKS_BY_ID

    env = PortfolioEnv(phase=phase, seed=seed)
    env.reset(seed=seed)
    if shock_id and getattr(env, "_plan", None) is not None and shock_id in SHOCKS_BY_ID:
        env._plan.shocks_by_quarter[0] = SHOCKS_BY_ID[shock_id]
    for _ in range(steps):
        env.step(action, completion="")
    return env.trajectory


def reward_regret_phase1(completions, **kwargs) -> list[float]:
    from portfolio_env import r_regret

    seeds = kwargs.get("seed", [42] * len(completions))
    shock_ids = kwargs.get("shock_id", [None] * len(completions))
    if isinstance(seeds, int):
        seeds = [seeds] * len(completions)
    if isinstance(shock_ids, str) or shock_ids is None:
        shock_ids = [shock_ids] * len(completions)

    scores = []
    for raw_completion, seed, shock_id in zip(completions, seeds, shock_ids):
        action = parse_action(completion_text(raw_completion))
        if action is None:
            scores.append(-0.50)
            continue
        traj = simulate_episode(action, int(seed), shock_id, phase=1, steps=4)
        scores.append(float(r_regret(traj)))
    return scores


def reward_carbon_guard(completions, **kwargs) -> list[float]:
    from portfolio_env.constants import CARBON_CAP

    seeds = kwargs.get("seed", [42] * len(completions))
    shock_ids = kwargs.get("shock_id", [None] * len(completions))
    if isinstance(seeds, int):
        seeds = [seeds] * len(completions)
    if isinstance(shock_ids, str) or shock_ids is None:
        shock_ids = [shock_ids] * len(completions)

    scores = []
    for raw_completion, seed, shock_id in zip(completions, seeds, shock_ids):
        action = parse_action(completion_text(raw_completion))
        if action is None:
            scores.append(0.0)
            continue
        traj = simulate_episode(action, int(seed), shock_id, phase=1, steps=4)
        projected = traj.carbon_footprint_accumulated * 3.0
        scores.append(float(-0.10 * max(0.0, projected - CARBON_CAP)))
    return scores


def make_messages(news: str) -> list[dict[str, str]]:
    from portfolio_env.prompt import SYSTEM_PROMPT, build_user_prompt

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(news)},
    ]


def build_grpo_dataset(n_prompts: int):
    import numpy as np
    from datasets import Dataset
    from portfolio_env import training_seeds
    from portfolio_env.shocks import shocks_available

    rng = np.random.default_rng(SEED)
    pool = [shock for shock in shocks_available(1) if "PLACEHOLDER" not in shock.id]
    seeds = training_seeds(rng, n_prompts)
    rows = []
    for seed in seeds:
        shock = pool[int(rng.integers(0, len(pool)))]
        rows.append({
            "prompt": make_messages(shock.news),
            "seed": int(seed),
            "shock_id": shock.id,
            "news": shock.news,
        })
    return Dataset.from_list(rows)


def load_model_and_tokenizer(token: str):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer from {MODEL_REPO}/{SFT_SUBFOLDER}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, subfolder=SFT_SUBFOLDER, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"Loading base model {BASE_MODEL}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        device_map="auto",
        token=token,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    print(f"Loading trainable SFT adapter {MODEL_REPO}/{SFT_SUBFOLDER}", flush=True)
    model = PeftModel.from_pretrained(
        base,
        MODEL_REPO,
        subfolder=SFT_SUBFOLDER,
        token=token,
        is_trainable=True,
    )
    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def patch_trl_preloaded_peft_prepare() -> None:
    """Keep TRL from re-preparing our already-loaded qLoRA adapter.

    TRL 0.22.2 re-runs GRPOConfig.__post_init__ via dataclasses.replace inside
    prepare_peft_model for qLoRA models. Once GRPOConfig has initialized its
    generated batching fields, that second pass can fail validation. We prepare
    the base model for k-bit training ourselves, load the SFT adapter as
    trainable, and then ask GRPOTrainer to use it as-is.
    """

    def _use_preloaded_adapter(model, peft_config, args):
        print("Using preloaded trainable PEFT adapter; skipping TRL prepare_peft_model.", flush=True)
        return model

    import trl.models.utils as trl_model_utils
    import trl.trainer.grpo_trainer as grpo_trainer_mod

    trl_model_utils.prepare_peft_model = _use_preloaded_adapter
    grpo_trainer_mod.prepare_peft_model = _use_preloaded_adapter


def generation_sanity_check(model, tokenizer, n: int = 5) -> dict[str, Any]:
    import torch
    from portfolio_env.shocks import shocks_available

    model.eval()
    rows = []
    for shock in [s for s in shocks_available(1) if "PLACEHOLDER" not in s.id][:n]:
        rendered = tokenizer.apply_chat_template(make_messages(shock.news), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_COMPLETION_LENGTH,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = parse_action(completion)
        rows.append({
            "shock": shock.id,
            "valid_action": action is not None,
            "tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
            "chars": len(completion),
            "has_closed_think": "<think>" in completion and "</think>" in completion,
            "preview": completion[:240],
        })
    model.train()
    lengths = [row["tokens"] for row in rows]
    return {
        "valid_actions": sum(1 for row in rows if row["valid_action"]),
        "closed_think": sum(1 for row in rows if row["has_closed_think"]),
        "total": len(rows),
        "mean_tokens": statistics.mean(lengths) if lengths else 0.0,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "samples": rows,
    }


def evaluate_holdout(model, tokenizer) -> dict[str, Any]:
    import numpy as np
    import torch
    from portfolio_env import holdout_seeds, r_regret
    from portfolio_env.shocks import shocks_available

    model.eval()
    results: dict[int, dict[str, Any]] = {}
    pool = [shock for shock in shocks_available(3) if "PLACEHOLDER" not in shock.id]
    for seed in holdout_seeds():
        rng = np.random.default_rng(seed)
        shock = pool[int(rng.integers(0, len(pool)))]
        rendered = tokenizer.apply_chat_template(make_messages(shock.news), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_COMPLETION_LENGTH,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = parse_action(completion)
        if action is None:
            results[int(seed)] = {
                "valid": False,
                "regret": None,
                "shock": shock.id,
                "tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
                "preview": completion[:240],
            }
            continue
        traj = simulate_episode(action, int(seed), shock.id, phase=3, steps=12)
        results[int(seed)] = {
            "valid": True,
            "regret": float(r_regret(traj)),
            "shock": shock.id,
            "tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
            "final_nav_real": float(traj.nav_real_series[-1]),
            "preview": completion[:240],
        }
    model.train()
    valid_regrets = [row["regret"] for row in results.values() if row["valid"]]
    return {
        "valid": len(valid_regrets),
        "total": len(results),
        "mean_regret": float(sum(valid_regrets) / len(valid_regrets)) if valid_regrets else None,
        "beats_baseline": sum(1 for regret in valid_regrets if regret > 0),
        "v6_sft_mean_regret_bar": V6_MEAN_REGRET,
        "results": results,
    }


def smoke_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    sanity = metrics.get("post_grpo_sanity", {})
    holdout = metrics.get("post_grpo_holdout", {})
    if sanity.get("mean_tokens", 0) <= 50:
        reasons.append("mean completion length <= 50")
    if sanity.get("min_tokens", 0) <= 1 and sanity.get("max_tokens", 0) <= 1:
        reasons.append("completion length stuck at 1")
    if sanity.get("valid_actions", 0) < 3:
        reasons.append("fewer than 3/5 sanity completions parsed")
    if holdout.get("valid", 0) < 3:
        reasons.append("fewer than 3/5 holdout completions parsed")
    return not reasons, reasons


def train_and_eval(token: str) -> dict[str, Any]:
    import torch
    from trl import GRPOConfig, GRPOTrainer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(token)

    pre_sanity = generation_sanity_check(model, tokenizer)
    print("Pre-GRPO sanity:", json.dumps(pre_sanity, indent=2), flush=True)

    dataset = build_grpo_dataset(GRPO_PROMPTS)
    reward_funcs = [
        reward_format,
        reward_action_contract,
        reward_reasoning_shape,
        reward_regret_phase1,
        reward_carbon_guard,
    ]

    args = GRPOConfig(
        output_dir=str(OUTPUT_DIR / "grpo"),
        max_steps=GRPO_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.9,
        top_p=0.95,
        top_k=None,
        min_p=0.05,
        generation_kwargs={"min_new_tokens": 32},
        learning_rate=LR,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        optim="paged_adamw_8bit",
        weight_decay=0.001,
        logging_steps=1,
        save_steps=GRPO_STEPS,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False,
        use_vllm=False,
        beta=0.02,
        loss_type="dapo",
        mask_truncated_completions=True,
        reward_weights=[0.7, 1.0, 0.3, 1.2, 0.4],
        seed=SEED,
        remove_unused_columns=False,
    )

    patch_trl_preloaded_peft_prepare()
    trainer = GRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
    )
    t0 = time.time()
    trainer.train()
    print(f"GRPO smoke done in {(time.time() - t0) / 60:.1f} min", flush=True)

    post_sanity = generation_sanity_check(model, tokenizer)
    print("Post-GRPO sanity:", json.dumps(post_sanity, indent=2), flush=True)
    holdout = evaluate_holdout(model, tokenizer)
    print("Post-GRPO holdout:", json.dumps(holdout, indent=2), flush=True)

    final_path = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    metrics = {
        "base_model": BASE_MODEL,
        "sft_subfolder": SFT_SUBFOLDER,
        "run_label": RUN_LABEL,
        "grpo_steps": GRPO_STEPS,
        "num_generations": NUM_GENERATIONS,
        "per_device_batch": PER_DEVICE_BATCH,
        "lr": LR,
        "pre_grpo_sanity": pre_sanity,
        "post_grpo_sanity": post_sanity,
        "post_grpo_holdout": holdout,
    }
    passed, reasons = smoke_gate(metrics)
    metrics["smoke_gate_passed"] = passed
    metrics["smoke_gate_reasons"] = reasons
    metrics_path = WORK / "qwen25_grpo_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print("Smoke gate:", json.dumps({"passed": passed, "reasons": reasons}, indent=2), flush=True)
    return {"artifact_path": str(final_path), "metrics_path": str(metrics_path), "metrics": metrics}


def upload_artifacts(token: str, artifact_path: Path, metrics_path: Path, passed: bool) -> None:
    from huggingface_hub import HfApi

    if not passed and os.environ.get("CARBON_ALPHA_UPLOAD_FAILED_GRPO", "0") != "1":
        print("Smoke gate failed; not uploading adapter. Metrics will still upload.", flush=True)
        HfApi(token=token).upload_file(
            path_or_fileobj=str(metrics_path),
            repo_id=MODEL_REPO,
            repo_type="model",
            path_in_repo=f"{RUN_LABEL}/metrics.json",
            commit_message=f"{RUN_LABEL}: failed smoke metrics",
            token=token,
        )
        return

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(artifact_path),
        repo_id=MODEL_REPO,
        repo_type="model",
        path_in_repo=RUN_LABEL,
        commit_message=f"{RUN_LABEL}: Qwen2.5 SFT adapter GRPO smoke",
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        repo_id=MODEL_REPO,
        repo_type="model",
        path_in_repo=f"{RUN_LABEL}/metrics.json",
        commit_message=f"{RUN_LABEL}: metrics",
        token=token,
    )
    print(f"Uploaded artifacts to {MODEL_REPO}/{RUN_LABEL}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-hf", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--local-code", action="store_true")
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
    else:
        download_code_bundle(token)

    check_hf_access(token)
    result = train_and_eval(token)
    if not args.skip_upload:
        upload_artifacts(
            token,
            Path(result["artifact_path"]),
            Path(result["metrics_path"]),
            bool(result["metrics"].get("smoke_gate_passed")),
        )


if __name__ == "__main__":
    main()
