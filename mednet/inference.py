"""Model loading, image preprocessing and prediction for the web app."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps

from .config import DATASETS, IMAGE_SIZE, MODELS, checkpoint_path
from .model import MedNet

torch.set_num_threads(1)  # keeps memory/CPU predictable on small free hosts


@dataclass
class Preprocessed:
    tensor: torch.Tensor      # (3, 28, 28) float, normalised to [-1, 1]
    display: np.ndarray       # (28, 28, 3) uint8, what the model actually sees


def load_model(model_key: str) -> MedNet:
    spec = MODELS[model_key]
    path = checkpoint_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = MedNet(DATASETS[spec.dataset].n_classes, spec.use_cbam)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def preprocess(image: Image.Image, dataset_key: str, center_crop: bool = True) -> Preprocessed:
    """Replicates the notebook pipeline: 28x28, grey->3ch, scale to [0,1], (x-0.5)/0.5."""
    spec = DATASETS[dataset_key]
    img = ImageOps.exif_transpose(image)
    img = img.convert("L") if spec.mode == "gray" else img.convert("RGB")
    if center_crop:
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float() / 255.0
    return Preprocessed(tensor=(x - 0.5) / 0.5, display=arr)


@torch.no_grad()
def predict(model: MedNet, tensor: torch.Tensor) -> np.ndarray:
    """Softmax probabilities for one image, shape (n_classes,)."""
    model.eval()
    logits = model(tensor.unsqueeze(0))
    return torch.softmax(logits, dim=1)[0].numpy()
