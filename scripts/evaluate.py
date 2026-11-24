#!/usr/bin/env python
"""Evaluate checkpoints on the FULL official test split and write assets/metrics/<model>.json.

python scripts/evaluate.py --all                 # every checkpoint found in models/
python scripts/evaluate.py --model dermamnist_cbam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mednet.config import MODELS, checkpoint_path  # noqa: E402
from mednet.data import load_split, make_loader  # noqa: E402
from mednet.inference import load_model  # noqa: E402
from mednet.metrics import compute_metrics, save_metrics  # noqa: E402


@torch.no_grad()
def run_test(model, loader):
    probs, labels = [], []
    for images, y in loader:
        probs.append(torch.softmax(model(images), dim=1).numpy())
        labels.extend(y.numpy())
    return np.asarray(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    keys = [k for k, s in MODELS.items() if checkpoint_path(s).is_file()] if args.all else [args.model]
    if not keys:
        sys.exit("No checkpoints found in models/ - copy your .pt files there first.")
    for key in keys:
        spec = MODELS[key]
        images, labels = load_split(spec.dataset, "test", args.data_dir)
        loader = make_loader(images, labels, None, 128, shuffle=False)
        y_true, y_prob = run_test(load_model(key), loader)
        result = compute_metrics(key, y_true, y_prob)
        save_metrics(result)
        o = result["overall"]
        print(f"{key:22s} N={result['test_n']} acc={o['accuracy']:.4f} macroF1={o['macro_f1']:.4f} auc={o['auc']:.4f}")


if __name__ == "__main__":
    main()
