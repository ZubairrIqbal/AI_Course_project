import json

import numpy as np
import pytest

from mednet.config import DATASETS, MODELS, metrics_dir
from mednet.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    pytest.importorskip("sklearn")
    n = DATASETS["dermamnist"].n_classes
    y = np.repeat(np.arange(n), 5)
    result = compute_metrics("dermamnist_cbam", y, np.eye(n)[y])
    assert result["overall"]["accuracy"] == 1.0 and result["overall"]["macro_f1"] == 1.0
    assert np.trace(np.array(result["confusion_matrix"])) == len(y)


def test_shipped_metrics_files_are_consistent_with_config():
    for key, spec in MODELS.items():
        data = json.loads((metrics_dir() / f"{key}.json").read_text())
        assert data["dataset"] == spec.dataset
        assert {"accuracy", "macro_f1", "auc"} <= data["overall"].keys()
        if data["per_class"]:
            assert [r["class"] for r in data["per_class"]] == list(DATASETS[spec.dataset].classes)
