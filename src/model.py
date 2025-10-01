"""src/model.py
Model registry, *plus* the core COCO-BOIL implementation.

We deliberately keep dataset-specific architectures out – they will be plugged
in later through the `@register_model` mechanism.  For now we include a *dummy*
MLP so that the framework is runnable end-to-end.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Any, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


# ---------------------------------------------------------------------------
# Generic helper – model registry
# ---------------------------------------------------------------------------

def register_model(name: str):
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'.  Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Simple dummy model so that smoke tests run without any user plug-ins.
# ---------------------------------------------------------------------------
@register_model("DUMMY_MLP")
class DummyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Cost surrogate models for BOIL / COCO-BOIL
# ---------------------------------------------------------------------------
class LinearCostModel(nn.Module):
    """y = w^T x + b"""
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (x * self.w).sum(dim=-1, keepdim=True) + self.b


class TinyCostNet(nn.Module):
    """One-hidden-layer MLP with 8 ReLU units – as per research method."""
    def __init__(self, dim: int, hidden: int = 8, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training: bool = False):
        h = F.relu(self.fc1(x))
        if training:
            h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Tiny Gaussian Process implementation using closed-form equations.
# ---------------------------------------------------------------------------
class SimpleGP:
    def __init__(self, noise: float = 1e-6, lengthscale: float = 1.0, signal: float = 1.0):
        self.noise = noise
        self.ls = lengthscale
        self.sig = signal
        self.X, self.y = None, None
        self.K_inv = None

    def _kernel(self, Xa: torch.Tensor, Xb: torch.Tensor):
        # RBF kernel
        sqdist = ((Xa[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
        return self.sig ** 2 * torch.exp(-0.5 / self.ls ** 2 * sqdist)

    def update(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.clone().detach()
        self.y = y.clone().detach()
        K = self._kernel(self.X, self.X) + self.noise * torch.eye(len(self.X))
        self.K_inv = torch.linalg.inv(K)

    def predict(self, Xtest: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Return mean, std."""
        K_s = self._kernel(Xtest, self.X)
        K_ss = self._kernel(Xtest, Xtest)
        mu = K_s @ self.K_inv @ self.y
        var = K_ss - K_s @ self.K_inv @ K_s.T
        std = var.diag().clamp(min=1e-9).sqrt()
        return mu.squeeze(-1), std


# ---------------------------------------------------------------------------
# COCO-BOIL implementation
# ---------------------------------------------------------------------------
class COCOBOIL:
    """Cost-Conditioned BOIL (COCO-BOIL).

    This implementation follows the *two-line* modification described in the
    research method yet is *fully functional* and generic.
    """
    def __init__(self,
                 search_space: Dict[str, Dict[str, Any]],
                 objective_fn: Callable[[Dict[str, float]], tuple],
                 init_samples: int = 5,
                 max_iter: int = 50,
                 device: torch.device | str = "cpu",
                 seed: int = 42,
                 cost_mlp_hidden: int = 8,
                 cost_mc_samples: int = 10):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.init_samples = init_samples
        self.max_iter = max_iter
        self.device = torch.device(device)
        self.rng = torch.Generator().manual_seed(seed)
        self.dim = len(search_space)
        self.cost_mc_samples = cost_mc_samples

        # Surrogate models
        self.gp = SimpleGP(noise=1e-6)
        self.cost_net = TinyCostNet(self.dim, hidden=cost_mlp_hidden).to(self.device)
        self.cost_opt = torch.optim.Adam(self.cost_net.parameters(), lr=1e-3)

        # Storage
        self.X = []   # hyper-parameters tried
        self.y = []   # objective scores
        self.c = []   # observed costs

    # ---------------- Helper functions ------------------
    def _sample_random(self) -> Dict[str, float]:
        cfg = {}
        for i, (name, info) in enumerate(self.search_space.items()):
            if info["type"] == "continuous":
                low, high = info["bounds"]
                cfg[name] = torch.empty(1).uniform_(low, high, generator=self.rng).item()
            elif info["type"] == "discrete":
                choices = info["values"]
                idx = torch.randint(0, len(choices), (1,), generator=self.rng).item()
                cfg[name] = choices[idx]
            else:
                raise ValueError("Unsupported param type")
        return cfg

    def _cfg_to_tensor(self, cfg: Dict[str, float]) -> torch.Tensor:
        return torch.tensor([cfg[name] for name in self.search_space], dtype=torch.float32)

    # ---------------- Main public method --------------
    def run(self) -> List[Dict[str, Any]]:
        history = []
        cumulative_cost = 0.0

        # ---------- initial random design ----------
        for _ in range(self.init_samples):
            cfg = self._sample_random()
            score, cost = self.objective_fn(cfg)
            cumulative_cost += cost
            self._append_observation(cfg, score, cost)
            history.append({"cfg": cfg, "score": score, "cost": cost, "cumulative_cost": cumulative_cost})

        # ---------- main BO loop ----------
        for it in range(self.max_iter - self.init_samples):
            # Fit surrogate models --------------------------------------------------
            X_tensor = torch.stack([self._cfg_to_tensor(c) for c in self.X])
            y_tensor = torch.tensor(self.y).unsqueeze(-1)
            c_tensor = torch.tensor(self.c).unsqueeze(-1)

            # GP for objective
            self.gp.update(X_tensor, y_tensor)

            # Cost net (MLP) – tiny, so train from scratch each iteration
            for _ in range(20):
                pred = self.cost_net(X_tensor, training=True)
                loss = F.mse_loss(pred, c_tensor)
                self.cost_opt.zero_grad()
                loss.backward()
                self.cost_opt.step()

            # Suggest next ---------------------------------------------------------
            next_cfg = self._suggest()

            # Evaluate objective
            score, cost = self.objective_fn(next_cfg)
            cumulative_cost += cost
            self._append_observation(next_cfg, score, cost)
            history.append({"cfg": next_cfg, "score": score, "cost": cost, "cumulative_cost": cumulative_cost})

        return history

    # ----------------------------------------------------------------------
    def _append_observation(self, cfg: Dict[str, float], score: float, cost: float):
        self.X.append(cfg)
        self.y.append(score)
        self.c.append(cost)

    # ----------------------------------------------------------------------
    def _suggest(self) -> Dict[str, float]:
        """Optimise acquisition on a *random* candidate set – simple yet effective."""
        candidate_cfgs = [self._sample_random() for _ in range(2048)]
        X_candidates = torch.stack([self._cfg_to_tensor(c) for c in candidate_cfgs])

        with torch.no_grad():
            mu, sigma = self.gp.predict(X_candidates)
            best_so_far = max(self.y)
            # Expected Improvement (EI)
            z = (mu - best_so_far) / (sigma + 1e-9)
            from torch.distributions import Normal
            normal = Normal(0, 1)
            ei = (mu - best_so_far) * normal.cdf(z) + sigma * normal.log_prob(z).exp()

            # ---- COCO modification (cost-aware) ----
            cost_samples = []
            for _ in range(self.cost_mc_samples):
                cost_pred = self.cost_net(X_candidates, training=True)  # dropout active
                cost_samples.append(cost_pred.squeeze(-1))
            cost_samples = torch.stack(cost_samples)  # [K, N]
            expected_log_cost = cost_samples.mean(0).log()
            acquisition = torch.log(ei.clamp(min=1e-12)) - expected_log_cost

            best_idx = torch.argmax(acquisition)
            return candidate_cfgs[int(best_idx)]
