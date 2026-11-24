"""Focal loss and the class-weighted variant from the novelty experiment."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class WeightedFocalLoss(nn.Module):
    def __init__(self, weights, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return (self.weights[labels] * ((1 - pt) ** self.gamma) * ce).mean()


def class_weights_from_labels(labels, num_classes: int, power: float = 0.25) -> np.ndarray:
    """weight_c ∝ count_c^-power, normalised to mean 1 (exactly the notebook's formula).
    Computed from TRAINING labels only - never from test labels."""
    counts = np.bincount(np.asarray(labels).reshape(-1), minlength=num_classes)
    weights = 1.0 / np.power(np.maximum(counts, 1), power)
    return weights / weights.mean()
