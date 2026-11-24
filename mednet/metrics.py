"""Compute / load the metrics that the app shows on the 'Model metrics' tab."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from .config import DATASETS, MODELS, metrics_dir


def compute_metrics(model_key: str, y_true, y_prob) -> dict:
    """Full test-set metrics from true labels and class probabilities."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                                 precision_score, recall_score, roc_auc_score)

    spec = MODELS[model_key]
    classes = DATASETS[spec.dataset].classes
    labels = np.arange(len(classes))
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = []
    for i, name in enumerate(classes):
        try:
            auc_i = float(roc_auc_score((y_true == i).astype(int), y_prob[:, i]))
        except ValueError:
            auc_i = None
        per_class.append({"class": name, "precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i]), "auc": auc_i})
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels))
    except ValueError:
        auc = None
    return {
        "model_key": model_key, "dataset": spec.dataset, "variant": spec.label,
        "test_n": int(len(y_true)),
        "overall": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc": auc,
        },
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "source": "scripts/evaluate.py (full official test split)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_metrics(result: dict) -> None:
    metrics_dir().mkdir(parents=True, exist_ok=True)
    (metrics_dir() / f"{result['model_key']}.json").write_text(json.dumps(result, indent=2))


def load_metrics(model_key: str) -> dict | None:
    path = metrics_dir() / f"{model_key}.json"
    return json.loads(path.read_text()) if path.is_file() else None
