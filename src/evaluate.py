"""src/evaluate.py
Reads result.json files produced by all experimental variations, aggregates them
into a Pandas DataFrame, prints comparison metrics to stdout (JSON-formatted),
and generates publication-quality PDF figures for accuracy and training loss
trajectories.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Matplotlib global style suitable for papers
plt.style.use("seaborn-v0_8-paper")
sns.set_context("paper", font_scale=1.4)

###########################################################################
# --------------------------- Helper functions -------------------------- #
###########################################################################

def _annotate_bars(ax):
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_height()
        ax.text(_x, _y, f"{_y:.3f}", ha="center", va="bottom", fontsize=9)


def _annotate_lines(ax, x_vals, y_vals):
    ax.text(x_vals[-1], y_vals[-1], f"{y_vals[-1]:.3f}", va="center", ha="left", fontsize=8)

###########################################################################
# --------------------------- Core Evaluation --------------------------- #
###########################################################################

def load_results(results_dir: Path) -> List[Dict]:
    runs = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        res_file = run_dir / "results.json"
        if not res_file.exists():
            continue
        with open(res_file, "r", encoding="utf-8") as fh:
            runs.append(json.load(fh))
    if not runs:
        raise RuntimeError(f"No results.json files found under {results_dir}")
    return runs


def generate_accuracy_figure(df: pd.DataFrame, out_path: Path):
    ax = sns.barplot(x="run_id", y="best_val_acc", data=df)
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0.0, 1.0)
    _annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def generate_loss_curves(df_epoch: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for run_id, sub in df_epoch.groupby("run_id"):
        ax.plot(sub["epoch"], sub["val_loss"], label=run_id)
        _annotate_lines(ax, sub["epoch"].values, sub["val_loss"].values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

###########################################################################
# --------------------------- Main Interface --------------------------- #
###########################################################################

def evaluate(results_dir: Path) -> None:
    runs = load_results(results_dir)

    # Flatten summary metrics into a DataFrame
    summary_rows = []
    epoch_rows = []
    for r in runs:
        metrics = r["metrics"]
        summary_rows.append({
            "run_id": r["run_id"],
            **metrics,
        })
        for em in r["epoch_metrics"]:
            epoch_rows.append(em)

    df = pd.DataFrame(summary_rows)
    df_epoch = pd.DataFrame(epoch_rows)

    # -------------------------------------------------------------- #
    # Print human- & machine-readable summary to stdout
    # -------------------------------------------------------------- #
    comparison_json: Dict[str, Dict] = {
        row["run_id"]: {
            "best_val_acc": row["best_val_acc"],
            "total_time_sec": row["total_time_sec"],
        }
        for _, row in df.iterrows()
    }
    print(json.dumps({"comparison": comparison_json}, indent=2))

    # -------------------------------------------------------------- #
    # Generate Figures
    # -------------------------------------------------------------- #
    generate_accuracy_figure(df, results_dir / "accuracy.pdf")
    generate_loss_curves(df_epoch, results_dir / "training_loss.pdf")

    # Inform the caller which figures have been produced
    figure_names = ["accuracy.pdf", "training_loss.pdf"]
    print(json.dumps({"figures": figure_names}))

###########################################################################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate <results_dir>")
        sys.exit(1)
    evaluate(Path(sys.argv[1]))
