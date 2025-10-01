"""src/train.py
Runs a single experiment variation.  This script is **not** aware of the global
list of runs – it focuses on the *one* configuration file that main.py passes
in.  All heavy lifting (data-loading, model construction, cost-aware BO, metric
tracking, figure generation, result serialisation) happens here so that each
sub-process is completely self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # head-less
import matplotlib.pyplot as plt

# We only import from our own package – no dataset / model specialisation here.
from . import preprocess as pre  # dataset registry & helpers
from . import model as mdl        # model & BO registry & helpers


def set_deterministic(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _training_loop(run_id: str,
                   cfg: Dict[str, Any],
                   results_dir: Path,
                   device: torch.device) -> Dict[str, Any]:
    """Standard supervised training loop (classification/regression)."""

    # -------------------- DATA --------------------
    train_loader, val_loader, meta = pre.build_dataloaders(cfg["dataset"],
                                                          seed=cfg.get("seed", 42))
    input_dim = meta["input_dim"]
    output_dim = meta["output_dim"]

    # -------------------- MODEL -------------------
    model_cfg = cfg["model"]
    model_cfg["params"].update({"input_dim": input_dim, "output_dim": output_dim})
    model = mdl.get_model(model_cfg["name"], **model_cfg["params"])
    model.to(device)

    # -------------------- OPT / LOSS --------------
    optim_cfg = cfg.get("training", {})
    epochs = optim_cfg.get("epochs", 10)
    lr = optim_cfg.get("learning_rate", 1e-3)
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    task_type = cfg["task"]["type"].lower()
    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
        primary_metric_name = "val_accuracy"
    elif task_type == "regression":
        criterion = nn.MSELoss()
        primary_metric_name = "val_mse"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # -------------------- METRIC STORAGE ----------
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        primary_metric_name: []
    }

    # -------------------- TRAIN -------------------
    best_metric = -float("inf") if task_type == "classification" else float("inf")
    best_state_dict = None
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ------- validation -------
        model.eval()
        val_losses = []
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss = criterion(logits, yb)
                val_losses.append(val_loss.item())
                if task_type == "classification":
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        if task_type == "classification":
            val_acc = correct / total if total > 0 else 0.0
            metric_val = val_acc
        else:  # regression uses *negative* MSE as higher is better? No – lower better.
            val_acc = None
            metric_val = avg_val_loss

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history[primary_metric_name].append(metric_val)

        is_better = metric_val > best_metric if task_type == "classification" else metric_val < best_metric
        if is_better:
            best_metric = metric_val
            best_state_dict = model.state_dict()

        # Print progress so that GitHub Actions log shows something useful.
        print(json.dumps({
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            primary_metric_name: metric_val
        }))
        sys.stdout.flush()

    wall_clock = time.time() - start_time

    # -------------------- SAVE ARTIFACTS ----------
    (results_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, results_dir / "artifacts" / "best_model.pt")

    # ---------- curves ----------
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.annotate(f"Final: {history['val_loss'][-1]:.3f}",
                 xy=(history["epoch"][-1], history["val_loss"][-1]))
    loss_fig = results_dir / f"training_loss_{run_id}.pdf"
    plt.savefig(loss_fig, bbox_inches="tight")
    plt.close()

    if task_type == "classification":
        plt.figure()
        plt.plot(history["epoch"], history[primary_metric_name], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.annotate(f"Best: {max(history[primary_metric_name]):.3f}",
                     xy=(history["epoch"][history[primary_metric_name].index(max(history[primary_metric_name]))],
                         max(history[primary_metric_name])))
        acc_fig = results_dir / f"accuracy_{run_id}.pdf"
        plt.savefig(acc_fig, bbox_inches="tight")
        plt.close()

    # -------------------- SERIALISE RESULTS -------
    results_json = {
        "run_id": run_id,
        "mode": "training",
        "task_type": task_type,
        "epochs": epochs,
        "history": history,
        "best_state_path": str(results_dir / "artifacts" / "best_model.pt"),
        "wall_clock": wall_clock,
        "primary_metric_name": primary_metric_name,
        "best_primary_metric": best_metric,
        "figures": [str(loss_fig)] + ([str(acc_fig)] if task_type == "classification" else [])
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    return results_json


def _hyperparam_optimisation(run_id: str,
                              cfg: Dict[str, Any],
                              results_dir: Path,
                              device: torch.device) -> Dict[str, Any]:
    """Run cost-aware Bayesian Optimisation (COCO-BOIL).

    This implementation is *fully functional* and follows the description in the
    research method.  The objective function is provided by a user-supplied
    builder (placeholder) that follows the interface

        score, cost = objective_fn(config_dict)

    where *score* must be **maximised** and *cost* is the wall-clock proxy (to be
    minimised).
    """
    # ---------------------------------------------------------------------
    objective_builder_str = cfg["objective"]["builder"]
    objective_fn = pre.get_objective_builder(objective_builder_str)(cfg["objective"].get("params", {}))

    search_space = cfg["search_space"]  # dict as specified in YAML
    bo_cfg = cfg["bo"]
    algo = mdl.COCOBOIL(search_space=search_space,
                        objective_fn=objective_fn,
                        init_samples=bo_cfg.get("init_samples", 5),
                        max_iter=bo_cfg.get("iterations", 30),
                        device=device,
                        seed=cfg.get("seed", 42),
                        cost_mlp_hidden=bo_cfg.get("cost_mlp_hidden", 8),
                        cost_mc_samples=bo_cfg.get("mc_samples", 10))
    history = algo.run()

    # --------------- FIGURES ------------------
    # Objective vs. cumulative cost curve.
    cum_cost, best_so_far = [], []
    best = -float("inf")
    for h in history:
        cum = h["cumulative_cost"]
        val = h["score"]
        if val > best:
            best = val
        cum_cost.append(cum)
        best_so_far.append(best)

    plt.figure()
    plt.plot(cum_cost, best_so_far, label=run_id)
    plt.xlabel("Cumulative cost")
    plt.ylabel("Best score so far")
    plt.legend()
    plt.annotate(f"Final: {best:.3f}", xy=(cum_cost[-1], best_so_far[-1]))
    fig_path = results_dir / f"hpo_best_objective_{run_id}.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    results = {
        "run_id": run_id,
        "mode": "hpo",
        "history": history,
        "final_best_score": best,
        "figures": [str(fig_path)]
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment variation (train or HPO)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config for *this* run only.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to store run outputs.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_id = cfg["run_id"]

    # -------------------- experiment description --------------------
    description_lines = [
        f"Run ID           : {run_id}",
        f"Mode             : {cfg.get('mode', 'training')} (training / hpo)",
        f"GPU              : {os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')}",
        f"Seed             : {cfg.get('seed', 42)}",
    ]
    print("\n".join(["==== Experiment Description ===="] + description_lines))
    sys.stdout.flush()

    set_deterministic(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.get("mode", "training") == "hpo":
        results = _hyperparam_optimisation(run_id, cfg, results_dir, device)
    else:
        results = _training_loop(run_id, cfg, results_dir, device)

    # Print *final* numeric data in JSON for automated harvesting.
    print(json.dumps(results))


if __name__ == "__main__":
    main()
