"""Grad-CAM for any MedNet stage (same method as the notebook, packaged for reuse)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import MedNet


def gradcam_for_stage(model: MedNet, image_tensor: torch.Tensor, stage_index: int,
                      class_id: int | None = None) -> tuple[np.ndarray, int]:
    """Return (heatmap in [0,1] at input resolution, class explained). stage_index is 1..5."""
    stage = model.stages[stage_index - 1]
    saved: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        saved["activation"] = output
        output.register_hook(lambda grad: saved.update({"gradient": grad}))

    handle = stage.register_forward_hook(forward_hook)
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            logits = model(image_tensor.unsqueeze(0))
            if class_id is None:
                class_id = int(logits.argmax(dim=1).item())
            logits[0, class_id].backward()
        activation = saved["activation"].detach()
        gradient = saved["gradient"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_id
    finally:
        handle.remove()


def overlay(display_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, size: int = 224) -> np.ndarray:
    """Blend a jet heatmap on the (nearest-neighbour upscaled) input. Returns uint8 RGB."""
    from matplotlib import colormaps

    base = Image.fromarray(display_rgb).resize((size, size), Image.Resampling.NEAREST)
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    colour = (colormaps["jet"](np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * colour.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)
