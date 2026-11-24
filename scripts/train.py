#!/usr/bin/env python
"""Train MedNet on a MedMNIST dataset (same recipe as the reproduction notebook).

Examples
--------
python scripts/train.py --dataset dermamnist                       # CBAM + focal loss
python scripts/train.py --dataset dermamnist --no-cbam             # ablation
python scripts/train.py --dataset dermamnist --loss weighted_focal # class-balanced experiment
python scripts/train.py --dataset bloodmnist --train-limit 0 --epochs 30   # 0 = full training set
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mednet.config import DATASETS, checkpoint_filename  # noqa: E402
from mednet.data import SEED, TRAIN_LIMITS, VAL_LIMITS, load_split, make_loader, stratified_indices  # noqa: E402
from mednet.losses import FocalLoss, WeightedFocalLoss, class_weights_from_labels  # noqa: E402
from mednet.model import MedNet  # noqa: E402


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--no-cbam", action="store_true", help="disable CBAM (ablation)")
    ap.add_argument("--loss", default="focal", choices=["focal", "weighted_focal"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-limit", type=int, default=None, help="0 = full split (default: notebook value)")
    ap.add_argument("--val-limit", type=int, default=None, help="0 = full split (default: notebook value)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--data-dir", default=None, help="where MedMNIST .npz files are stored")
    args = ap.parse_args()

    use_cbam = not args.no_cbam
    spec = DATASETS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_limit = TRAIN_LIMITS[args.dataset] if args.train_limit is None else (args.train_limit or None)
    val_limit = VAL_LIMITS[args.dataset] if args.val_limit is None else (args.val_limit or None)

    print(f"Loading {args.dataset} ...")
    x_tr, y_tr = load_split(args.dataset, "train", args.data_dir)
    x_va, y_va = load_split(args.dataset, "val", args.data_dir)
    tr_idx = stratified_indices(y_tr, train_limit, args.seed)
    va_idx = stratified_indices(y_va, val_limit, args.seed)
    train_loader = make_loader(x_tr, y_tr, tr_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(x_va, y_va, va_idx, args.batch_size, shuffle=False)
    print(f"train={len(tr_idx)} val={len(va_idx)} device={device} cbam={use_cbam} loss={args.loss}")

    model = MedNet(spec.n_classes, use_cbam).to(device)
    if args.loss == "weighted_focal":
        weights = class_weights_from_labels(y_tr[tr_idx], spec.n_classes)
        print("class weights:", np.round(weights, 3))
        criterion = WeightedFocalLoss(weights).to(device)
    else:
        criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / checkpoint_filename(args.dataset, use_cbam, args.loss)
    best, waiting, history, start = -1.0, 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, train_acc=tr_acc, val_acc=va_acc))
        print(f"epoch {epoch:02d}/{args.epochs} | train acc {tr_acc:.3f} | val acc {va_acc:.3f} | {time.time()-start:.0f}s")
        if va_acc > best:
            best, waiting = va_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            waiting += 1
            if waiting >= args.patience:
                print("early stopping")
                break
    Path("runs").mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(Path("runs") / (ckpt.stem + "_history.csv"), index=False)
    print(f"best val acc {best:.4f} -> saved {ckpt}")


if __name__ == "__main__":
    main()
