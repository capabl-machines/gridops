# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.45",
#   "trl>=0.12",
#   "peft>=0.13",
#   "datasets>=2.20",
#   "accelerate>=1.0",
#   "bitsandbytes",
#   "huggingface_hub>=0.34,<1.0",
# ]
# ///
"""HF Jobs/Colab SFT script for GridOps JSON-action model.

Writes a new adapter subfolder and never overwrites existing artifacts.
"""

from __future__ import annotations

import json
import os
import inspect
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi, upload_folder
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:  # TRL < 0.8 compatibility.
    SFTConfig = None


MODEL_REPO = os.environ.get("GRIDOPS_MODEL_REPO", "77ethers/gridops-models")
BASE_MODEL = os.environ.get("GRIDOPS_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TRACE_PATH = os.environ.get("GRIDOPS_TRACE_PATH", "sft_traces/gridops_curriculum_1200.jsonl")
RUN_LABEL = os.environ.get("GRIDOPS_RUN_LABEL", "sft_qwen25_3b_gridops_curriculum1200_v1")
MAX_STEPS = int(os.environ.get("GRIDOPS_SFT_STEPS", "300"))
HF_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
PER_DEVICE_BATCH_SIZE = int(os.environ.get("GRIDOPS_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRIDOPS_GRAD_ACCUM", "8"))
MAX_LENGTH = int(os.environ.get("GRIDOPS_MAX_LENGTH", "1536"))
LORA_R = int(os.environ.get("GRIDOPS_LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("GRIDOPS_LORA_ALPHA", str(LORA_R * 2)))
LORA_DROPOUT = float(os.environ.get("GRIDOPS_LORA_DROPOUT", "0.05"))
LEARNING_RATE = float(os.environ.get("GRIDOPS_LEARNING_RATE", "2e-4"))
UPLOAD_TO_HF = os.environ.get("GRIDOPS_UPLOAD", "1").lower() not in {"0", "false", "no"}
GRADIENT_CHECKPOINTING = os.environ.get("GRIDOPS_GRADIENT_CHECKPOINTING", "1").lower() not in {"0", "false", "no"}


def load_rows(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = list(row["messages"]) + [{"role": "assistant", "content": row["completion"]}]
        rows.append({"messages": messages})
    return rows


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Set HF_API_TOKEN or HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

    dataset = Dataset.from_list(load_rows(TRACE_PATH)).map(render)

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

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    out_dir = Path("outputs") / RUN_LABEL
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        "logging_steps": 10,
        "save_steps": max(50, MAX_STEPS),
        "bf16": use_bf16,
        "fp16": torch.cuda.is_available() and not use_bf16,
        "report_to": [],
        "remove_unused_columns": False,
    }

    if SFTConfig is not None:
        config_params = inspect.signature(SFTConfig).parameters
        if "dataset_text_field" in config_params:
            train_args["dataset_text_field"] = "text"
        if "max_length" in config_params:
            train_args["max_length"] = MAX_LENGTH
        elif "max_seq_length" in config_params:
            train_args["max_seq_length"] = MAX_LENGTH
        if "packing" in config_params:
            train_args["packing"] = False
        args = SFTConfig(**train_args)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=peft_config,
            args=args,
        )
    else:
        args = TrainingArguments(**train_args)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=MAX_LENGTH,
            peft_config=peft_config,
            args=args,
        )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = {
        "base_model": BASE_MODEL,
        "trace_path": TRACE_PATH,
        "run_label": RUN_LABEL,
        "sft_steps": MAX_STEPS,
        "dataset_rows": len(dataset),
        "max_length": MAX_LENGTH,
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
    }
    (out_dir / "gridops_sft_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

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
            commit_message=f"Upload GridOps SFT adapter {RUN_LABEL}",
        )
        upload_target = f"{MODEL_REPO}/{RUN_LABEL}"
    print(json.dumps({"uploaded_to": upload_target, **metrics}, indent=2))


if __name__ == "__main__":
    main()
