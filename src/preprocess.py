"""src/preprocess.py
Common pre-processing & data-loading *framework*.

The key idea is a registry-based design:
    1. BaseDatasetBuilder defines the interface.
    2. `@register_dataset(name)` registers concrete implementations.

`build_dataloaders` is the single public entry-point used by train.py.

We also expose a similar mechanism for *objective builders* used by the HPO
algorithm so that expensive training/train-then-eval pipelines can be plugged in
without modifying the core framework.
"""
from __future__ import annotations

import abc
from typing import Tuple, Dict, Any, Callable, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split

_DATASET_REGISTRY: Dict[str, "BaseDatasetBuilder"] = {}
_OBJECTIVE_REGISTRY: Dict[str, Callable] = {}


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------
class BaseDatasetBuilder(abc.ABC):
    """Abstract base class – every dataset must implement this interface."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abc.abstractmethod
    def get_dataset(self) -> Tuple[Dataset, Dict[str, Any]]:
        """Return full *torch.utils.data.Dataset* **and** meta-information.

        Meta MUST contain at least:
            * input_dim  – total flattened feature dimension expected by models
            * output_dim – number of classes (classification) or output size
        """

    def train_val_split(self, ds: Dataset, val_fraction: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset]:
        g = torch.Generator().manual_seed(seed)
        val_len = int(len(ds) * val_fraction)
        train_len = len(ds) - val_len
        return random_split(ds, [train_len, val_len], generator=g)


@dataclass
class DummyClassificationDataset(Dataset):
    num_samples: int
    input_dim: int
    num_classes: int

    def __post_init__(self):
        self.x = torch.randn(self.num_samples, self.input_dim)
        self.y = torch.randint(0, self.num_classes, (self.num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DummyDatasetBuilder(BaseDatasetBuilder):
    """Random toy data – *fully functional* yet data-agnostic."""

    def get_dataset(self):
        n = self.params.get("num_samples", 1000)
        d = self.params.get("input_dim", 20)
        k = self.params.get("num_classes", 3)
        ds = DummyClassificationDataset(num_samples=n, input_dim=d, num_classes=k)
        meta = {"input_dim": d, "output_dim": k}
        return ds, meta


def register_dataset(name: str):
    def decorator(cls):
        if name in _DATASET_REGISTRY:
            raise ValueError(f"Dataset name '{name}' already registered")
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


# Register the dummy dataset so smoke-tests run out-of-the-box.
_DATASET_REGISTRY["DUMMY_CLASSIFICATION"] = DummyDatasetBuilder


# ---------------------------------------------------------------------------
# Objective (black-box) builders for BO experiments
# ---------------------------------------------------------------------------
class DummyObjective:
    """Maps a config dict to (score, cost).  Completely synthetic but reproducible."""
    def __init__(self, params: Dict[str, Any]):
        self.dim = params.get("dim", 5)

    def __call__(self, cfg: Dict[str, float]):
        import math, random
        x = torch.tensor([cfg[f"x{i}"] for i in range(self.dim)])
        # Simple non-convex function – negated rastrigin (max @ 0)
        score = -(10 * self.dim + torch.sum(x ** 2 - 10 * torch.cos(2 * math.pi * x))).item()
        # Cost is quadratic in the L2-norm (simulates longer training for large hyper-params)
        cost = (x.norm().item() ** 2) / self.dim + random.random() * 0.1  # tiny noise
        return score, cost


_OBJECTIVE_REGISTRY["DUMMY_OBJECTIVE"] = DummyObjective


# Public helpers ----------------------------------------------------------------

def get_dataset_builder(name: str):
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset builder '{name}'.  Available: {list(_DATASET_REGISTRY)}")
    return _DATASET_REGISTRY[name]


def build_dataloaders(dataset_cfg: Dict[str, Any], seed: int = 42):
    builder_cls = get_dataset_builder(dataset_cfg["name"])
    builder = builder_cls(dataset_cfg.get("params", {}))
    full_ds, meta = builder.get_dataset()
    train_ds, val_ds = builder.train_val_split(full_ds, val_fraction=dataset_cfg.get("val_fraction", 0.2), seed=seed)

    batch_size = dataset_cfg.get("params", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, meta


def register_objective(name: str):
    def decorator(cls):
        if name in _OBJECTIVE_REGISTRY:
            raise ValueError(f"Objective name '{name}' already registered")
        _OBJECTIVE_REGISTRY[name] = cls
        return cls
    return decorator


def get_objective_builder(name: str):
    if name not in _OBJECTIVE_REGISTRY:
        raise KeyError(f"Unknown objective builder '{name}'.  Available: {list(_OBJECTIVE_REGISTRY)}")
    return _OBJECTIVE_REGISTRY[name]
