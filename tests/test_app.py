"""Smoke-test the real Streamlit app with random-weight checkpoints (no training needed)."""
import numpy as np
import pytest
import torch
from PIL import Image

st_testing = pytest.importorskip("streamlit.testing.v1")

from mednet.config import DATASETS, MODELS  # noqa: E402
from mednet.model import MedNet  # noqa: E402

APP = "app.py"


def test_app_without_checkpoints_shows_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDNET_MODEL_DIR", str(tmp_path))
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("No trained weights" in e.value for e in at.error)


def test_app_end_to_end_with_sample_image(tmp_path, monkeypatch):
    models, samples = tmp_path / "models", tmp_path / "samples" / "dermamnist"
    models.mkdir(); samples.mkdir(parents=True)
    spec = MODELS["dermamnist_cbam"]
    torch.save(MedNet(DATASETS["dermamnist"].n_classes, True).state_dict(), models / spec.checkpoint)
    Image.fromarray(np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)).save(samples / "c4_0.png")
    monkeypatch.setenv("MEDNET_MODEL_DIR", str(models))
    monkeypatch.setenv("MEDNET_SAMPLES_DIR", str(tmp_path / "samples"))

    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.radio[0].set_value("Use a sample test image").run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Predicted class" in labels and "Confidence" in labels
    assert any("Ground truth" in (s.value) for s in list(at.success) + list(at.error))
