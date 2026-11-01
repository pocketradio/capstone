"""Comparison plots generated from persisted experiment metrics."""

from __future__ import annotations

from pathlib import Path


def plot_cost_comparison(metrics_csv: str | Path, output_path: str | Path) -> None:
    """Plot closed-loop/oracle costs and threshold baselines by model."""
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(metrics_csv)
    pivot = frame.pivot_table(index="model", columns="experiment", values="total_cost", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_ylabel("Total cost (points; lower is better)")
    ax.set_xlabel("Fraud-score model")
    ax.set_title("Decision cost under shared feedback simulation")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(target, dpi=160)
    plt.close(ax.figure)


def plot_detection_comparison(metrics_csv: str | Path, output_path: str | Path) -> None:
    """Plot ranking and operating metrics for each score provider."""
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(metrics_csv).drop_duplicates(subset=["model"])
    metrics = [c for c in ("pr_auc", "roc_auc", "precision_at_0_5", "recall_at_0_5") if c in frame]
    ax = frame.set_index("model")[metrics].plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_ylim(0, 1)
    ax.set_title("Detection diagnostics (accuracy is not primary)")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(target, dpi=160)
    plt.close(ax.figure)

