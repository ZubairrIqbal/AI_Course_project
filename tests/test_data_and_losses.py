import numpy as np
import torch

from mednet.data import stratified_indices
from mednet.losses import FocalLoss, WeightedFocalLoss, class_weights_from_labels


def test_stratified_indices_size_coverage_and_determinism():
    labels = np.array([0] * 900 + [1] * 90 + [2] * 10)
    idx = stratified_indices(labels, 100)
    assert len(idx) == 100 and set(labels[idx]) == {0, 1, 2}
    assert np.array_equal(idx, stratified_indices(labels, 100))


def test_stratified_none_returns_everything():
    assert len(stratified_indices(np.arange(10) % 2, None)) == 10


def test_class_weights_normalised_and_favour_rare_classes():
    w = class_weights_from_labels([0] * 100 + [1] * 10, 2)
    assert abs(w.mean() - 1) < 1e-9 and w[1] > w[0]


def test_weighted_loss_with_equal_weights_equals_focal():
    logits, y = torch.randn(16, 5), torch.randint(0, 5, (16,))
    assert torch.allclose(FocalLoss()(logits, y), WeightedFocalLoss(np.ones(5))(logits, y))
