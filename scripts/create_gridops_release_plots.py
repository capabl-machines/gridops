#!/usr/bin/env python3
"""Create static release plots for the GridOps Hugging Face model card."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf_release" / "capabl_machines" / "model_assets"

SYSTEMS = {
    "v5.1 direct\nmodel": {
        "average": 0.7354,
        "task_1_normal": 0.7896,
        "task_2_heatwave": 0.7681,
        "task_3_crisis": 0.6484,
        "lp_capture": None,
    },
    "v7 deterministic\ncontroller": {
        "average": 0.7907,
        "task_1_normal": 0.7995,
        "task_2_heatwave": 0.8224,
        "task_3_crisis": 0.7503,
        "lp_capture": 0.9604,
    },
    "Untuned 1.5B\n+ harness": {
        "average": 0.7911,
        "task_1_normal": 0.7993,
        "task_2_heatwave": 0.8223,
        "task_3_crisis": 0.7517,
        "lp_capture": 0.9609,
    },
    "v7.1 SFT\nselector": {
        "average": 0.7880,
        "task_1_normal": 0.7994,
        "task_2_heatwave": 0.8224,
        "task_3_crisis": 0.7421,
        "lp_capture": 0.9571,
    },
    "v7.3 DPO\nselector": {
        "average": 0.7888,
        "task_1_normal": 0.7993,
        "task_2_heatwave": 0.8223,
        "task_3_crisis": 0.7449,
        "lp_capture": 0.9581,
    },
    "Full LP\nceiling": {
        "average": 0.8233,
        "task_1_normal": 0.8372,
        "task_2_heatwave": 0.8416,
        "task_3_crisis": 0.7912,
        "lp_capture": 1.0,
    },
}

FOOTPRINT = {
    "v7 deterministic\ncontroller": {
        "blackout_kwh": 338.7,
        "diesel_kwh": 760.2,
        "cost_rs": 217_690.0,
    },
    "Untuned 1.5B\n+ harness": {
        "blackout_kwh": 356.85,
        "diesel_kwh": 757.0,
        "cost_rs": 216_568.16,
    },
    "v7.3 DPO\nselector": {
        "blackout_kwh": 404.68,
        "diesel_kwh": 760.2,
        "cost_rs": 222_097.46,
    },
    "Full LP\nceiling": {
        "blackout_kwh": 62.02,
        "diesel_kwh": 792.0,
        "cost_rs": 183_761.90,
    },
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfbf8",
            "axes.facecolor": "#fbfbf8",
            "axes.edgecolor": "#d7d7d0",
            "axes.labelcolor": "#242424",
            "xtick.color": "#242424",
            "ytick.color": "#242424",
            "text.color": "#242424",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "savefig.facecolor": "#fbfbf8",
            "savefig.bbox": "tight",
        }
    )


def annotate_bars(ax, bars, fmt="{:.3f}", offset=0.006) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_task_scores() -> None:
    labels = list(SYSTEMS)
    tasks = [
        ("task_1_normal", "Task 1 normal"),
        ("task_2_heatwave", "Task 2 heatwave"),
        ("task_3_crisis", "Task 3 crisis"),
    ]
    x = np.arange(len(labels))
    width = 0.24
    colors = ["#2f8f83", "#e8a33a", "#d95f59"]

    fig, ax = plt.subplots(figsize=(14.2, 6.2))
    for index, (key, label) in enumerate(tasks):
        values = [SYSTEMS[name][key] for name in labels]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=colors[index])
        annotate_bars(ax, bars, offset=0.004)

    ax.set_ylim(0.60, 0.87)
    ax.set_ylabel("Environment score")
    ax.set_title("GridOps Holdout Scores by Task")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.text(
        0.01,
        0.01,
        "Holdout seeds: 7001, 7002, 7003. Higher is better.",
        fontsize=9,
        color="#555",
    )
    fig.savefig(OUT / "gridops_v7_task_scores.png", dpi=180)
    plt.close(fig)


def plot_lp_capture() -> None:
    labels = [label for label, row in SYSTEMS.items() if row["lp_capture"] is not None]
    values = [SYSTEMS[label]["lp_capture"] * 100 for label in labels]
    colors = ["#2f8f83", "#d6a13b", "#61a6d8", "#8d73d8", "#242424"]
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    bars = ax.bar(labels, values, color=colors)
    annotate_bars(ax, bars, fmt="{:.2f}%", offset=0.5)
    ax.set_ylim(90, 102)
    ax.set_ylabel("LP ceiling capture")
    ax.set_title("How Close The Release Gets To The LP Ceiling")
    ax.grid(axis="y", alpha=0.22)
    fig.text(
        0.01,
        0.01,
        "The strategy harness is the main unlock: even an untuned 1.5B model reaches 96.09% LP capture with strict strategy JSON.",
        fontsize=9,
        color="#555",
    )
    fig.savefig(OUT / "gridops_v7_lp_capture.png", dpi=180)
    plt.close(fig)


def plot_operational_footprint() -> None:
    labels = list(FOOTPRINT)
    metrics = [
        ("blackout_kwh", "Blackout kWh", "#d95f59", 1.0),
        ("diesel_kwh", "Diesel kWh", "#68615b", 1.0),
        ("cost_rs", "Cost, Rs lakh", "#2f8f83", 100_000.0),
    ]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    for index, (key, name, color, scale) in enumerate(metrics):
        values = [FOOTPRINT[label][key] / scale for label in labels]
        bars = ax.bar(x + (index - 1) * width, values, width, label=name, color=color)
        annotate_bars(ax, bars, fmt="{:.1f}", offset=max(values) * 0.015)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Operational Footprint on Crisis Holdout")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.text(
        0.01,
        0.01,
        "Task 3 crisis averages over seeds 7001-7003. Cost is shown in Rs lakh for readability.",
        fontsize=9,
        color="#555",
    )
    fig.savefig(OUT / "gridops_v7_crisis_footprint.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    plot_task_scores()
    plot_lp_capture()
    plot_operational_footprint()
    print(f"wrote plots to {OUT}")


if __name__ == "__main__":
    main()
