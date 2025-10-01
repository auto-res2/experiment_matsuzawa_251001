"""src/train.py
This script runs a SINGLE experimental variation ("run").  It must be executed by
src/main.py and therefore **never** allocates a GPU by itself – the orchestrator
already guarantees that only one visible GPU is available inside the process.
The script is *dataset / model agnostic*; all experiment–specific information is
read from the YAML config file created by main.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .preprocess import get_dataloaders
from .model import get_model

###########################################################################
# ------------------------------ Utilities ------------------------------ #
###########################################################################

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures determinism where possible (may decrease speed slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Returns *scalar* accuracy value on CPU."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def epoch_step(model: nn.Module,
               loader: DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer | None,
               device: torch.device) -> Tuple[float, float]:
    """Runs exactly one epoch over `loader`.

    Returns
    -------
    loss_mean : float  – average loss of the epoch
    acc_mean  : float  – average accuracy of the epoch
    """
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    running_acc = 0.0
    batches = 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(xb)
            loss = criterion(outputs, yb)
            acc = accuracy_from_logits(outputs.detach(), yb.detach())

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.detach().item()
        running_acc += acc
        batches += 1

    return running_loss / batches, running_acc / batches

###########################################################################
# ------------------------------ Main logic ----------------------------- #
###########################################################################

def run_experiment(cfg: Dict[str, Any], results_dir: Path, smoke_test: bool) -> None:
    """Core routine executed for a single experimental variation."""
    run_id: str = cfg["id"]

    # ------------------------------------------------------------------ #
    # House-keeping: create directories & deterministic seeds
    # ------------------------------------------------------------------ #
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(cfg.get("seed", 42))

    # ------------------------------------------------------------------ #
    # Data pipeline (placeholder-aware but *functional* for smoke test)
    # ------------------------------------------------------------------ #
    train_loader, val_loader, num_classes, input_shape = get_dataloaders(cfg["dataset"], smoke_test=smoke_test)

    # ------------------------------------------------------------------ #
    # Model, criterion, optimiser
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg["model"], input_shape=input_shape, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimiser_cfg = cfg.get("optimizer", {"name": "SGD", "lr": 0.01, "momentum": 0.9})
    if optimiser_cfg["name"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=optimiser_cfg.get("lr", 1e-3))
    else:
        optimizer = optim.SGD(model.parameters(), lr=optimiser_cfg.get("lr", 0.01),
                              momentum=optimiser_cfg.get("momentum", 0.9))

    # Scheduler (optional placeholder)
    scheduler = None
    if "scheduler" in cfg:
        sched_cfg = cfg["scheduler"]
        if sched_cfg["name"].lower() == "steplr":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg.get("step_size", 30),
                                                  gamma=sched_cfg.get("gamma", 0.1))

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    epochs: int = cfg.get("training", {}).get("epochs", 1 if smoke_test else 100)
    best_val_acc = 0.0
    epoch_metrics = []
    wall_clock_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = epoch_step(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = epoch_step(model, val_loader, criterion, optimizer=None, device=device)

        if scheduler is not None:
            scheduler.step()

        best_val_acc = max(best_val_acc, val_acc)

        epoch_log = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        }
        epoch_metrics.append(epoch_log)
        # Log epoch metrics as JSON line (for real-time streaming in main)
        print(json.dumps(epoch_log), flush=True)

    wall_clock_total = time.perf_counter() - wall_clock_start

    # ------------------------------------------------------------------ #
    # Serialize results
    # ------------------------------------------------------------------ #
    results = {
        "run_id": run_id,
        "config": cfg,
        "metrics": {
            "best_val_acc": best_val_acc,
            "final_val_acc": val_acc,
            "final_val_loss": val_loss,
            "total_time_sec": wall_clock_total,
        },
        "epoch_metrics": epoch_metrics,
    }

    with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Also save model state_dict for potential later analysis
    torch.save(model.state_dict(), run_dir / "model.pth")


###########################################################################
# ------------------------------- CLI ----------------------------------- #
###########################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single experimental variation (train phase)")
    p.add_argument("--config-path", type=str, required=True, help="Path to YAML containing ALL experiments")
    p.add_argument("--run-id", type=str, required=True, help="Unique identifier of the variation to run (matches YAML id)")
    p.add_argument("--results-dir", type=str, required=True, help="Root directory where outputs will be written")
    p.add_argument("--smoke-test", action="store_true", help="Activate ultra-light experiment settings for CI smoke test")
    return p.parse_args()


def main() -> None:
    import yaml  # local import to keep global namespace clean

    args = parse_args()

    with open(args.config_path, "r", encoding="utf-8") as fh:
        full_cfg = yaml.safe_load(fh)

    # Find the requested experiment
    experiments = {exp["id"]: exp for exp in full_cfg["experiments"]}
    if args.run_id not in experiments:
        raise ValueError(f"Run-id '{args.run_id}' not found in {args.config_path}")
    exp_cfg = experiments[args.run_id]

    # Run experiment
    run_experiment(exp_cfg, Path(args.results_dir), smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
