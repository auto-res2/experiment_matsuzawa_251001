"""src/evaluate.py
Aggregates the results of multiple experiment variations and produces
comparison figures that are *identical* for all future specialised runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import pandas as pd


PRIMARY_METRIC_KEY = "best_primary_metric"  # training runs
HPO_FINAL_KEY = "final_best_score"          # hpo runs


def _collect_results(results_dir: Path) -> List[Dict]:
    runs = []
    for sub in results_dir.iterdir():
        res_file = sub / "results.json"
        if res_file.exists():
            with open(res_file, "r") as f:
                runs.append(json.load(f))
    return runs


def _produce_bar(runs: List[Dict], results_dir: Path, key: str, ylabel: str, fig_topic: str):
    names = [r["run_id"] for r in runs]
    values = [r[key] for r in runs]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.3f}",
                 ha='center', va='bottom')
    plt.tight_layout()
    fig_path = results_dir / f"{fig_topic}.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return str(fig_path)


def main():
    parser = argparse.ArgumentParser(description="Aggregate & compare experiment results.")
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = _collect_results(results_dir)

    if not runs:
        print("No results found to evaluate.")
        return

    # -------------------- experiment description --------------------
    print("==== Evaluation Description ====")
    print(f"Found {len(runs)} runs in {results_dir}")

    # Separate training vs HPO runs for metric-specific evaluation.
    training_runs = [r for r in runs if r["mode"] == "training"]
    hpo_runs = [r for r in runs if r["mode"] == "hpo"]

    figures = []

    if training_runs:
        figures.append(_produce_bar(training_runs, results_dir,
                                    key=PRIMARY_METRIC_KEY,
                                    ylabel="Best Validation Metric",
                                    fig_topic="primary_metric_comparison"))

    if hpo_runs:
        figures.append(_produce_bar(hpo_runs, results_dir,
                                    key=HPO_FINAL_KEY,
                                    ylabel="Best Score Reached",
                                    fig_topic="hpo_score_comparison"))

    # -------- Summary JSON (printed) --------
    summary = {
        "num_runs": len(runs),
        "figures": figures,
        "runs": runs  # raw details for downstream parsing
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
