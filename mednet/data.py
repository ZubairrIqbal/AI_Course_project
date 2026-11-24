"""Dataset helpers used by the training / evaluation / sample-export scripts.

`medmnist` is imported lazily so the *web app* never needs it (smaller deploy).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import IMAGE_SIZE

SEED = 42
TRAIN_LIMITS = {"bloodmnist": 4000, "octmnist": 8000, "dermamnist": 4000}
VAL_LIMITS = {"bloodmnist": 1000, "octmnist": 2000, "dermamnist": 1000}


def load_split(dataset: str, split: str, root: str | None = None):
    """Return (images, labels) from MedMNIST (downloads on first use)."""
    import medmnist
    from medmnist import INFO

    cls = getattr(medmnist, INFO[dataset]["python_class"])
    ds = cls(split=split, download=True, size=IMAGE_SIZE, root=root)
    return ds.imgs, ds.labels.reshape(-1)


class SimpleMedDataset(Dataset):
    def __init__(self, images, labels, indices=None):
        if indices is not None:
            images, labels = images[indices], labels[indices]
        self.images, self.labels = images, np.asarray(labels).reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        return (x - 0.5) / 0.5, int(self.labels[idx])


def stratified_indices(labels, limit, seed: int = SEED) -> np.ndarray:
    """Class-proportional subset (>=1 per class), identical to the notebook."""
    labels = np.asarray(labels).reshape(-1)
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    if limit < len(classes):
        raise ValueError("Subset must contain at least one example per class")
    quotas = counts * limit / counts.sum()
    allocation = np.minimum(counts, np.maximum(1, np.floor(quotas).astype(int)))
    while allocation.sum() < limit:
        choices = np.flatnonzero(allocation < counts)
        allocation[choices[np.argmax(quotas[choices] - allocation[choices])]] += 1
    while allocation.sum() > limit:
        choices = np.flatnonzero(allocation > 1)
        allocation[choices[np.argmax(allocation[choices] - quotas[choices])]] -= 1
    chosen = []
    for cls, number in zip(classes, allocation):
        pool = np.flatnonzero(labels == cls)
        chosen.extend(rng.choice(pool, int(number), replace=False).tolist())
    return np.sort(np.array(chosen, dtype=int))


def make_loader(images, labels, indices, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SimpleMedDataset(images, labels, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)
