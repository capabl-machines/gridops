# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.56.2",
#   "trl>=0.22.2",
#   "peft>=0.17.1",
#   "datasets>=4.0",
#   "accelerate>=1.0",
#   "bitsandbytes",
#   "huggingface_hub>=0.34,<2.0",
# ]
# ///
"""HF Jobs/Colab DPO script for GridOps strategy-json model."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi, upload_folder
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer

try:
    from trl import DPOConfig
except ImportError:  # pragma: no cover - older TRL compatibility.
    DPOConfig = None


MODEL_REPO = os.environ.get("GRIDOPS_MODEL_REPO", "77ethers/gridops-models")
BASE_MODEL = os.environ.get("GRIDOPS_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
INIT_ADAPTER = os.environ.get("GRIDOPS_INIT_ADAPTER", "77ethers/gridops-models/sft_qwen25_15b_gridops_strategy_v7")
PAIR_PATH = os.environ.get("GRIDOPS_DPO_PAIR_PATH", "sft_traces/gridops_strategy_dpo_pairs_v1.jsonl")
RUN_LABEL = os.environ.get("GRIDOPS_RUN_LABEL", "dpo_qwen25_15b_gridops_strategy_v72")
MAX_STEPS = int(os.environ.get("GRIDOPS_DPO_STEPS", "80"))
HF_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
PER_DEVICE_BATCH_SIZE = int(os.environ.get("GRIDOPS_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRIDOPS_GRAD_ACCUM", "8"))
MAX_LENGTH = int(os.environ.get("GRIDOPS_MAX_LENGTH", "1024"))
MAX_PROMPT_LENGTH = int(os.environ.get("GRIDOPS_MAX_PROMPT_LENGTH", "768"))
LEARNING_RATE = float(os.environ.get("GRIDOPS_LEARNING_RATE", "5e-6"))
BETA = float(os.environ.get("GRIDOPS_DPO_BETA", "0.1"))
UPLOAD_TO_HF = os.environ.get("GRIDOPS_UPLOAD", "1").lower() not in {"0", "false", "no"}
GRADIENT_CHECKPOINTING = os.environ.get("GRIDOPS_GRADIENT_CHECKPOINTING", "1").lower() not in {"0", "false", "no"}


def model_path_kwargs(path: str) -> tuple[str, dict[str, str]]:
    if Path(path).exists():
        return path, {}
    parts = path.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2]), {"subfolder": "/".join(parts[2:])}
    return path, {}


def load_rows(path: str, tokenizer) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        rows.append(
            {
                "prompt": prompt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "id": row.get("id", ""),
            }
        )
    return rows


def build_dpo_config(out_dir: Path):
    train_args = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        "logging_steps": 10,
        "save_steps": max(40, MAX_STEPS),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": [],
        "remove_unused_columns": False,
    }
    if DPOConfig is None:
        from transformers import TrainingArguments

        return TrainingArguments(**train_args)

    config_params = inspect.signature(DPOConfig).parameters
    if "beta" in config_params:
        train_args["beta"] = BETA
    if "max_length" in config_params:
        train_args["max_length"] = MAX_LENGTH
    if "max_prompt_length" in config_params:
        train_args["max_prompt_length"] = MAX_PROMPT_LENGTH
    return DPOConfig(**train_args)


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Set HF_API_TOKEN or HF_TOKEN")
    if not INIT_ADAPTER:
        raise RuntimeError("Set GRIDOPS_INIT_ADAPTER to a strategy SFT adapter")

    adapter_id, adapter_kwargs = model_path_kwargs(INIT_ADAPTER)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=HF_TOKEN, **adapter_kwargs)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(load_rows(PAIR_PATH, tokenizer))
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant,
        device_map="auto",
        token=HF_TOKEN,
    )
    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model = PeftModel.from_pretrained(model, adapter_id, token=HF_TOKEN, is_trainable=True, **adapter_kwargs)

    out_dir = Path("outputs") / RUN_LABEL
    args = build_dpo_config(out_dir)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
    }
    trainer_params = inspect.signature(DPOTrainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = {
        "base_model": BASE_MODEL,
        "init_adapter": INIT_ADAPTER,
        "pair_path": PAIR_PATH,
        "run_label": RUN_LABEL,
        "dpo_steps": MAX_STEPS,
        "dataset_rows": len(dataset),
        "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
    }
    (out_dir / "gridops_dpo_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    upload_target = ""
    if UPLOAD_TO_HF:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
        upload_folder(
            folder_path=str(out_dir),
            repo_id=MODEL_REPO,
            repo_type="model",
            path_in_repo=RUN_LABEL,
            token=HF_TOKEN,
            commit_message=f"Upload GridOps strategy DPO adapter {RUN_LABEL}",
        )
        upload_target = f"{MODEL_REPO}/{RUN_LABEL}"
    print(json.dumps({"uploaded_to": upload_target, **metrics}, indent=2))


if __name__ == "__main__":
    main()
