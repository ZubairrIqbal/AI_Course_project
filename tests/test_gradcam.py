import numpy as np
import torch

from mednet.gradcam import gradcam_for_stage, overlay
from mednet.model import MedNet


def test_gradcam_all_stages():
    model = MedNet(7).eval()
    x = torch.randn(3, 28, 28)
    for stage in range(1, 6):
        heat, cls = gradcam_for_stage(model, x, stage)
        assert heat.shape == (28, 28) and 0.0 <= heat.min() and heat.max() <= 1.0 and 0 <= cls < 7


def test_gradcam_leaves_no_hooks_behind():
    model = MedNet(4).eval()
    gradcam_for_stage(model, torch.randn(3, 28, 28), 3)
    assert len(model.stage3._forward_hooks) == 0


def test_overlay_is_uint8_rgb():
    out = overlay(np.zeros((28, 28, 3), np.uint8), np.random.rand(28, 28))
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
