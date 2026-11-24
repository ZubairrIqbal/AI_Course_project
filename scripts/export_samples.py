#!/usr/bin/env python
"""Save a few real test images per class so visitors can try the app without uploading.

python scripts/export_samples.py --per-class 2
Files: assets/samples/<dataset>/c<class_id>_<k>.png  (28x28, tiny - safe to commit)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mednet.config import DATASETS, samples_dir  # noqa: E402
from mednet.data import load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for name, spec in DATASETS.items():
        images, labels = load_split(name, "test", args.data_dir)
        out = samples_dir() / name
        out.mkdir(parents=True, exist_ok=True)
        for cls in range(spec.n_classes):
            pool = np.flatnonzero(labels == cls)
            for k, idx in enumerate(rng.choice(pool, min(args.per_class, len(pool)), replace=False)):
                Image.fromarray(images[idx]).save(out / f"c{cls}_{k}.png")
        print(f"{name}: wrote samples to {out}")


if __name__ == "__main__":
    main()
