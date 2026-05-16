#!/usr/bin/env python3
"""Publish the Capabl Machines GridOps HF release.

This script expects a Hugging Face token that can write to the
`capabl-machines` organization. The current 77ethers-only token can read the
source artifacts but cannot create repos under the org.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD = REPO_ROOT / "hf_release" / "capabl_machines" / "model_README.md"
SPACE_README = REPO_ROOT / "hf_release" / "capabl_machines" / "space_README.md"
SPACE_DOCKERFILE = REPO_ROOT / "hf_release" / "capabl_machines" / "space_Dockerfile"


def load_token() -> str:
    for key in ("HF_API_TOKEN", "HF_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"HF_API_TOKEN", "HF_TOKEN"}:
                return value.strip().strip("\"'")
    raise RuntimeError("Set HF_API_TOKEN with write access to capabl-machines.")


def copytree(src: Path, dst: Path, ignore: list[str] | None = None) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*(ignore or [])))


def stage_model(source_repo: str, source_subfolder: str, target: Path, token: str) -> None:
    patterns = [
        f"{source_subfolder}/adapter_config.json",
        f"{source_subfolder}/adapter_model.safetensors",
        f"{source_subfolder}/chat_template.jinja",
        f"{source_subfolder}/gridops_dpo_metrics.json",
        f"{source_subfolder}/tokenizer.json",
        f"{source_subfolder}/tokenizer_config.json",
        f"{source_subfolder}/training_args.bin",
        f"{source_subfolder}/evals/**",
    ]
    downloaded = Path(
        snapshot_download(
            repo_id=source_repo,
            repo_type="model",
            allow_patterns=patterns,
            token=token,
        )
    )
    target.mkdir(parents=True, exist_ok=True)
    for item in (downloaded / source_subfolder).iterdir():
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)
    shutil.copy2(MODEL_CARD, target / "README.md")


def stage_space(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SPACE_README, target / "README.md")
    shutil.copy2(SPACE_DOCKERFILE, target / "Dockerfile")
    for file_name in ["pyproject.toml", "openenv.yaml", "inference.py"]:
        shutil.copy2(REPO_ROOT / file_name, target / file_name)
    copytree(REPO_ROOT / "gridops", target / "gridops", ignore=["__pycache__", "*.pyc"])
    copytree(REPO_ROOT / "server", target / "server", ignore=["__pycache__", "*.pyc"])
    copytree(REPO_ROOT / "assets", target / "assets", ignore=["*.psd", "*.ai"])
    eval_target = target / "evals"
    eval_target.mkdir(exist_ok=True)
    for file_name in [
        "gridops_v7_strategy_controller_holdout_7001_7003.json",
        "gridops_v7_strategy_controller_extended_7201_7210.json",
        "gridops_v7_optimizer_holdout_7001_7003.json",
    ]:
        source = REPO_ROOT / "evals" / file_name
        if source.exists():
            shutil.copy2(source, eval_target / file_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", default="capabl-machines")
    parser.add_argument("--model-repo-name", default="gridops-strategy-selector-v7")
    parser.add_argument("--space-repo-name", default="gridops-demo")
    parser.add_argument("--source-repo", default="77ethers/gridops-models")
    parser.add_argument("--source-subfolder", default="dpo_qwen25_15b_gridops_strategy_v73_crisis")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = load_token()
    api = HfApi(token=token)
    model_repo = f"{args.org}/{args.model_repo_name}"
    space_repo = f"{args.org}/{args.space_repo_name}"

    with tempfile.TemporaryDirectory(prefix="gridops-capabl-release-") as tmp:
        tmp_path = Path(tmp)
        model_dir = tmp_path / "model"
        space_dir = tmp_path / "space"
        stage_model(args.source_repo, args.source_subfolder, model_dir, token)
        stage_space(space_dir)

        print(f"Prepared model release: {model_dir}")
        print(f"Prepared space release: {space_dir}")
        if args.dry_run:
            print(f"Dry run only. Would publish model={model_repo}, space={space_repo}")
            return

        api.create_repo(repo_id=model_repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=model_repo,
            repo_type="model",
            folder_path=str(model_dir),
            commit_message="Release GridOps Strategy Selector v7",
        )

        api.create_repo(
            repo_id=space_repo,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=space_repo,
            repo_type="space",
            folder_path=str(space_dir),
            commit_message="Release GridOps demo Space",
        )

    print(f"Model: https://huggingface.co/{model_repo}")
    print(f"Space: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
