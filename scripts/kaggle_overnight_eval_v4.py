"""Run GridOps v4 evals overnight on Kaggle and upload evidence immediately.

This wrapper is intentionally defensive:

- every eval writes to /kaggle/working/gridops/evals;
- stdout/stderr is mirrored to a log file;
- a compact digest of invalid examples and valid samples is written;
- artifacts are uploaded to the model repo after each eval finishes.

Use it after cloning the repo in Kaggle and setting a Secret named HF_API_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "77ethers/gridops-models"
DEFAULT_RUN_LABEL = "sft_qwen25_3b_gridops_kimi_reason_action_v4"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = f"{DEFAULT_MODEL_REPO}/{DEFAULT_RUN_LABEL}"


def ensure_hf_token() -> str:
    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_API_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        return token

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret("HF_API_TOKEN")
    except Exception as exc:  # pragma: no cover - only runs on Kaggle.
        raise SystemExit("Set a Kaggle Secret named HF_API_TOKEN before running this script.") from exc

    if not token:
        raise SystemExit("Kaggle Secret HF_API_TOKEN is empty.")
    os.environ["HF_API_TOKEN"] = token
    os.environ["HF_TOKEN"] = token
    return token


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, command)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def collect_invalids(report: dict[str, Any]) -> list[dict[str, Any]]:
    invalids: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        for item in row.get("invalid_examples", []):
            invalids.append(
                {
                    "task_id": row.get("task_id"),
                    "seed": row.get("seed"),
                    "score": row.get("score"),
                    **item,
                }
            )
    return invalids


def collect_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        for item in row.get("samples", []):
            samples.append(
                {
                    "task_id": row.get("task_id"),
                    "seed": row.get("seed"),
                    "score": row.get("score"),
                    **item,
                }
            )
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_markdown_summary(path: Path, run_name: str, report: dict[str, Any], invalid_count: int) -> None:
    by_task = report.get("by_task", {})
    lines = [
        f"# {run_name}",
        "",
        f"- model: `{report.get('name', '')}`",
        f"- average_score: `{report.get('average_score')}`",
        f"- valid_action_rate: `{report.get('valid_action_rate')}`",
        f"- invalid_examples_saved: `{invalid_count}`",
        "",
        "| Task | Score | Valid action rate | Blackout kWh | Diesel kWh | Cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task_id, row in by_task.items():
        lines.append(
            "| {task} | {score} | {valid} | {blackout} | {diesel} | {cost} |".format(
                task=task_id,
                score=row.get("score"),
                valid=row.get("valid_action_rate"),
                blackout=row.get("blackout_kwh"),
                diesel=row.get("diesel_kwh"),
                cost=row.get("cost"),
            )
        )
    lines.extend(
        [
            "",
            "## Next-step interpretation",
            "",
            "- If valid action rate is below `0.99`, inspect `invalid_examples.jsonl`.",
            "- If task 3 diesel is near zero, the model has fallen back into the v3 shortcut.",
            "- If task 3 score is above `0.60` with nonzero diesel, v4 is a meaningful policy improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def upload_file(api: Any, repo_id: str, local_path: Path, remote_path: str) -> None:
    print(f"\nUploading {local_path} -> {repo_id}/{remote_path}", flush=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="model",
    )


def upload_artifacts(api: Any, repo_id: str, run_label: str, run_name: str, paths: list[Path]) -> None:
    for path in paths:
        upload_file(api, repo_id, path, f"{run_label}/evals/{run_name}/{path.name}")


def write_manifest(path: Path, args: argparse.Namespace) -> None:
    nvidia = subprocess.run(
        ["nvidia-smi"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "model_repo": args.model_repo,
        "run_label": args.run_label,
        "nvidia_smi": nvidia,
    }
    write_json(path, payload)


def run_eval(
    args: argparse.Namespace,
    run_name: str,
    seeds: str,
    tasks: str,
    max_new_tokens: int,
    horizon: int,
    no_4bit: bool,
) -> list[Path]:
    output = args.output_dir / f"{run_name}.json"
    log_path = args.output_dir / f"{run_name}.log"
    invalid_path = args.output_dir / f"{run_name}_invalid_examples.jsonl"
    sample_path = args.output_dir / f"{run_name}_valid_samples.jsonl"
    markdown_path = args.output_dir / f"{run_name}_summary.md"

    if output.exists() and not args.force:
        print(f"\nSkipping {run_name}; {output} already exists. Use --force to rerun.", flush=True)
    else:
        command = [
            sys.executable,
            "scripts/evaluate_gridops_adapter.py",
            "--base-model",
            args.base_model,
            "--adapter-path",
            args.adapter_path,
            "--prompt-mode",
            "reason_action",
            "--max-new-tokens",
            str(max_new_tokens),
            "--seeds",
            seeds,
            "--tasks",
            tasks,
            "--sample-limit",
            str(args.sample_limit),
            "--horizon",
            str(horizon),
            "--output",
            str(output),
        ]
        if no_4bit:
            command.append("--no-4bit")
        started = time.time()
        run_command(command, log_path)
        print(f"\n{run_name} finished in {(time.time() - started) / 60:.1f} minutes.", flush=True)

    report = read_json(output)
    invalids = collect_invalids(report)
    samples = collect_samples(report)
    write_jsonl(invalid_path, invalids)
    write_jsonl(sample_path, samples)
    write_markdown_summary(markdown_path, run_name, report, len(invalids))
    return [output, log_path, invalid_path, sample_path, markdown_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.environ.get("GRIDOPS_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter-path", default=os.environ.get("GRIDOPS_ADAPTER_PATH", DEFAULT_ADAPTER))
    parser.add_argument("--model-repo", default=os.environ.get("GRIDOPS_MODEL_REPO", DEFAULT_MODEL_REPO))
    parser.add_argument("--run-label", default=os.environ.get("GRIDOPS_RUN_LABEL", DEFAULT_RUN_LABEL))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/kaggle_overnight_v4"))
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-long-decode", action="store_true")
    parser.add_argument("--run-long-decode", action="store_true")
    args = parser.parse_args()

    ensure_hf_token()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_API_TOKEN"])
    manifest = args.output_dir / "run_manifest.json"
    write_manifest(manifest, args)
    upload_artifacts(api, args.model_repo, args.run_label, "manifest", [manifest])

    eval_plan = [
        {
            "run_name": "smoke_7001_tokens220",
            "seeds": "7001",
            "tasks": "task_1_normal,task_2_heatwave,task_3_crisis",
            "max_new_tokens": 220,
            "horizon": 72,
        }
    ]
    if not args.skip_full:
        eval_plan.append(
            {
                "run_name": "holdout_7001_7003_tokens220",
                "seeds": "7001,7002,7003",
                "tasks": "task_1_normal,task_2_heatwave,task_3_crisis",
                "max_new_tokens": 220,
                "horizon": 72,
            }
        )
    if args.run_long_decode and not args.skip_long_decode:
        eval_plan.append(
            {
                "run_name": "holdout_7001_7003_tokens320",
                "seeds": "7001,7002,7003",
                "tasks": "task_1_normal,task_2_heatwave,task_3_crisis",
                "max_new_tokens": 320,
                "horizon": 72,
            }
        )

    all_paths: list[Path] = []
    failures: list[dict[str, str]] = []
    for item in eval_plan:
        try:
            paths = run_eval(args, no_4bit=args.no_4bit, **item)
            upload_artifacts(api, args.model_repo, args.run_label, item["run_name"], paths)
            all_paths.extend(paths)
        except Exception as exc:
            failure_path = args.output_dir / f"{item['run_name']}_failure.json"
            failure = {"run_name": item["run_name"], "error": repr(exc)}
            failures.append(failure)
            write_json(failure_path, failure)
            upload_artifacts(api, args.model_repo, args.run_label, item["run_name"], [failure_path])
            print(f"\nFAILED {item['run_name']}: {exc!r}", flush=True)

    index = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded_to": f"https://huggingface.co/{args.model_repo}/tree/main/{args.run_label}/evals",
        "artifact_count": len(all_paths),
        "failures": failures,
        "eval_runs": [item["run_name"] for item in eval_plan],
    }
    index_path = args.output_dir / "overnight_eval_index.json"
    write_json(index_path, index)
    upload_artifacts(api, args.model_repo, args.run_label, "index", [index_path])
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
