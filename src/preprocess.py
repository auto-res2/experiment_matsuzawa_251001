"""src/preprocess.py
Dataset preprocessing & DataLoader creation **common** to all experiments.
Only the *dataset/model specific* parts are placeholders that will be filled in
future steps.
"""
from __future__ import annotations

from typing import Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

###########################################################################
# ----------------------- Generic / Placeholder Datasets --------------- #
###########################################################################

class RandomClassificationDataset(Dataset):
    """A *generic* synthetic classification dataset used exclusively for smoke-tests.
    It guarantees that the base infrastructure can execute end-to-end without any
    real dataset present.
    """
    def __init__(self, num_samples: int, input_shape: Tuple[int, ...], num_classes: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.data = torch.from_numpy(rng.randn(num_samples, *input_shape)).float()
        self.targets = torch.from_numpy(rng.randint(0, num_classes, size=(num_samples,))).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

###########################################################################
# -------------------------- DataLoader Factory ------------------------ #
###########################################################################

def _build_random_loaders(ds_cfg: Dict[str, Any], smoke_test: bool):
    """Internal helper to create loaders for the synthetic dataset."""
    # When smoke_test == True we keep the dataset *tiny* regardless of cfg.
    num_samples = 256 if smoke_test else ds_cfg.get("num_samples", 10_000)
    input_shape = tuple(ds_cfg.get("input_shape", (3, 32, 32)))
    num_classes = ds_cfg.get("num_classes", 10)
    train_frac = ds_cfg.get("train_fraction", 0.8)
    batch_size = ds_cfg.get("batch_size", 32)

    dataset = RandomClassificationDataset(num_samples=num_samples,
                                          input_shape=input_shape,
                                          num_classes=num_classes,
                                          seed=ds_cfg.get("seed", 0))

    n_train = int(train_frac * num_samples)
    n_val = num_samples - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, num_classes, input_shape


def get_dataloaders(dataset_cfg: Dict[str, Any], smoke_test: bool):
    """Public entry-point used by src.train.

    Parameters
    ----------
    dataset_cfg : Dict[str, Any]
        Section of the YAML configuration dedicated to the dataset.
    smoke_test : bool
        Whether we are running the CI smoke-test (forces tiny dataset + 1 epoch).

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
    val_loader   : torch.utils.data.DataLoader
    num_classes  : int
    input_shape  : Tuple[int, ...]
    """
    name = dataset_cfg["name"].upper()

    # ------------------------------------------------------------------ #
    # 1) Synthetic dataset for smoke test / default quick sanity checks
    # ------------------------------------------------------------------ #
    if name == "RANDOM_CLASSIFICATION":
        return _build_random_loaders(dataset_cfg, smoke_test)

    # ------------------------------------------------------------------ #
    # 2) Placeholder for *real* datasets – must be filled in derived code
    # ------------------------------------------------------------------ #
    if name == "DATASET_PLACEHOLDER":
        raise NotImplementedError(
            "DATASET_PLACEHOLDER must be replaced with actual dataset loading logic "
            "in the derived, dataset-specific experiment code.")

    # ------------------------------------------------------------------ #
    # 3) Fallback – allow future extensions via registry mechanism
    # ------------------------------------------------------------------ #
    raise ValueError(f"Unknown dataset name '{name}'. Please register it in preprocess.py")
