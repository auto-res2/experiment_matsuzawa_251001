"""src/model.py
Model registry with concrete architectures required by DyGraM-HPO experiments.
Currently implemented models
• RESNET18 / RESNET50 (vision) – torchvision
• BILSTM            (text)    – simple 2-layer BiLSTM classifier
• LINEAR_REGRESSION (tabular) – single linear layer for YearPrediction
A DummyNet is kept for smoke tests.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

###########################################################################
# ------------------------------ Registry ------------------------------ #
###########################################################################

_MODEL_REGISTRY: Dict[str, nn.Module] = {}


def register_model(name: str):
    def _wrap(cls):
        _MODEL_REGISTRY[name.upper()] = cls
        return cls
    return _wrap

###########################################################################
# ------------------------- Vision backbones --------------------------- #
###########################################################################

@register_model("RESNET18")
class ResNet18Classifier(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, **kwargs):
        super().__init__()
        self.model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # Replace FC to match num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.model(x)


@register_model("RESNET50")
class ResNet50Classifier(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, **kwargs):
        super().__init__()
        self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.model(x)

###########################################################################
# ------------------------------ Bi-LSTM ------------------------------- #
###########################################################################

@register_model("BILSTM")
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, embed_dim: int = 300,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3, **kwargs):
        super().__init__()
        seq_len = input_shape[0]
        vocab_size = kwargs.get("vocab_size", 50_000)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        x = self.embed(x)
        _, (hn, _) = self.lstm(x)
        # Concatenate last layer’s forward & backward hidden state
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        hn = self.dropout(hn)
        return self.fc(hn)

###########################################################################
# --------------------------- Tabular model ---------------------------- #
###########################################################################

@register_model("LINEAR_REGRESSION")
class LinearRegression(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, **kwargs):
        super().__init__()
        in_features = input_shape[0]
        self.lin = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.lin(x)

###########################################################################
# ---------------------------- Dummy Net ------------------------------- #
###########################################################################

@register_model("DUMMYNET")
class DummyNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, **kwargs):
        super().__init__()
        in_features = 1
        for d in input_shape:
            in_features *= d
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.net(x)

###########################################################################
# -------------------------- Construction API -------------------------- #
###########################################################################

def get_model(model_cfg: Dict[str, Any], *, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
    name = model_cfg["name"].upper()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](input_shape=input_shape, num_classes=num_classes, **model_cfg.get("params", {}))
