"""Create GridOps evaluation plots from holdout JSON reports."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


TASKS = ["task_1_normal", "task_2_heatwave", "task_3_crisis"]
TASK_LABELS = ["Normal", "Heatwave", "Crisis"]
POLICY_LABELS = {
    "do_nothing": "Do-nothing",
    "sft": "SFT v1",
    "oracle": "Oracle",
}


def task_rows(report: dict, task_id: str) -> list[dict]:
    return [row for row in report["rows"] if row["task_id"] == task_id]


def mean_metric(report: dict, task_id: str, key: str) -> float:
    rows = task_rows(report, task_id)
    if key == "score":
        values = [row["score"] for row in rows]
    else:
        values = [(row["grade"] or {}).get(key, 0.0) for row in rows]
    return sum(values) / max(len(values), 1)


def bar_plot(data: dict[str, list[float]], title: str, ylabel: str, output: Path) -> None:
    x = range(len(TASKS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    offsets = [-width, 0, width]
    colors = ["#8b8f98", "#6d5dfc", "#19a974"]
    for idx, (name, values) in enumerate(data.items()):
        ax.bar([i + offsets[idx] for i in x], values, width=width, label=POLICY_LABELS[name], color=colors[idx])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x), TASK_LABELS)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_training_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        clean = ansi_re.sub("", line).strip()
        if not clean.startswith("{") or "loss" not in clean:
            continue
        try:
            parsed = ast.literal_eval(clean)
        except (SyntaxError, ValueError):
            continue
        if "loss" in parsed and "epoch" in parsed:
            rows.append(parsed)
    return rows


def line_plot(metrics: list[dict], output: Path) -> None:
    if not metrics:
        return
    x = list(range(10, 10 * len(metrics) + 1, 10))
    loss = [float(row["loss"]) for row in metrics]
    acc = [float(row.get("mean_token_accuracy", 0.0)) for row in metrics]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, loss, color="#6d5dfc", marker="o", label="loss")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("SFT loss", color="#6d5dfc")
    ax1.tick_params(axis="y", labelcolor="#6d5dfc")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, acc, color="#19a974", marker="s", label="token accuracy")
    ax2.set_ylabel("Mean token accuracy", color="#19a974")
    ax2.tick_params(axis="y", labelcolor="#19a974")
    ax1.set_title("GridOps SFT Training Curve")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="evals/gridops_sft_mixed1418_v1_holdout_7001_7003.json")
    parser.add_argument("--do-nothing", default="evals/gridops_do_nothing_holdout_7001_7003.json")
    parser.add_argument("--oracle", default="evals/gridops_oracle_holdout_7001_7003.json")
    parser.add_argument("--training-log", default="training_logs/outputs_sft_gridops_mixed1418_v1.log")
    parser.add_argument("--output-dir", default="evals/plots")
    args = parser.parse_args()

    reports = {
        "do_nothing": json.loads(Path(args.do_nothing).read_text()),
        "sft": json.loads(Path(args.sft).read_text()),
        "oracle": json.loads(Path(args.oracle).read_text()),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bar_plot(
        {name: [mean_metric(report, task, "score") for task in TASKS] for name, report in reports.items()},
        "GridOps Holdout Score by Task",
        "Score",
        output_dir / "gridops_holdout_scores.png",
    )
    bar_plot(
        {name: [mean_metric(report, task, "battery_throughput_kwh") for task in TASKS] for name, report in reports.items()},
        "Battery Throughput Shows Real Dispatch",
        "Battery throughput (kWh)",
        output_dir / "gridops_battery_throughput.png",
    )
    bar_plot(
        {name: [mean_metric(report, task, "total_blackout_kwh") for task in TASKS] for name, report in reports.items()},
        "Blackout Reduction vs Do-nothing",
        "Blackout energy (kWh)",
        output_dir / "gridops_blackout_kwh.png",
    )
    training_metrics = parse_training_metrics(Path(args.training_log))
    line_plot(training_metrics, output_dir / "gridops_sft_training_curve.png")
    if training_metrics:
        (output_dir / "gridops_sft_training_metrics.json").write_text(json.dumps(training_metrics, indent=2) + "\n")

    summary = {
        name: {
            "average_score": report["average_score"],
            "valid_action_rate": report["valid_action_rate"],
            "by_task": {
                task: {
                    "score": round(mean_metric(report, task, "score"), 4),
                    "battery_throughput_kwh": round(mean_metric(report, task, "battery_throughput_kwh"), 2),
                    "blackout_kwh": round(mean_metric(report, task, "total_blackout_kwh"), 2),
                    "diesel_kwh": round(mean_metric(report, task, "total_diesel_kwh"), 2),
                    "cost": round(mean_metric(report, task, "actual_cost"), 2),
                }
                for task in TASKS
            },
        }
        for name, report in reports.items()
    }
    if training_metrics:
        summary["training"] = {
            "logged_points": len(training_metrics),
            "first_loss": float(training_metrics[0]["loss"]),
            "final_loss": float(training_metrics[-1]["loss"]),
            "final_mean_token_accuracy": float(training_metrics[-1].get("mean_token_accuracy", 0.0)),
        }
    (output_dir / "gridops_holdout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
