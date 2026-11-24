"""Central registry: datasets, class names, model checkpoints, file locations.

Everything that other modules need to know about *which* model belongs to *which*
dataset lives here, so the app, training script and evaluation script never
disagree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = 28


def model_dir() -> Path:
    """Folder holding .pt checkpoints (override with MEDNET_MODEL_DIR)."""
    return Path(os.environ.get("MEDNET_MODEL_DIR", ROOT / "models"))


def samples_dir() -> Path:
    """Folder with example test images (override with MEDNET_SAMPLES_DIR)."""
    return Path(os.environ.get("MEDNET_SAMPLES_DIR", ROOT / "assets" / "samples"))


def metrics_dir() -> Path:
    """Folder holding per-model metrics JSON (override with MEDNET_METRICS_DIR)."""
    return Path(os.environ.get("MEDNET_METRICS_DIR", ROOT / "assets" / "metrics"))


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    modality: str
    mode: str                 # "rgb" or "gray" (gray is repeated to 3 channels)
    classes: tuple[str, ...]  # index == label id in MedMNIST
    paper_acc_28: float       # published 28x28 reference accuracy (from your notebook)
    note: str

    @property
    def n_classes(self) -> int:
        return len(self.classes)


DATASETS: dict[str, DatasetSpec] = {
    "bloodmnist": DatasetSpec(
        key="bloodmnist",
        title="Blood cells (BloodMNIST)",
        modality="Microscopy of peripheral blood cells",
        mode="rgb",
        classes=(
            "basophil", "eosinophil", "erythroblast",
            "immature granulocytes",  # official: immature granulocytes(myelocytes, metamyelocytes and promyelocytes)
            "lymphocyte", "monocyte", "neutrophil", "platelet",
        ),
        paper_acc_28=0.961,
        note="Upload a single, centred blood-cell microscopy image.",
    ),
    "octmnist": DatasetSpec(
        key="octmnist",
        title="Retinal OCT (OCTMNIST)",
        modality="Optical coherence tomography of the retina (greyscale)",
        mode="gray",
        classes=("choroidal neovascularization", "diabetic macular edema", "drusen", "normal"),
        paper_acc_28=0.927,
        note="Upload a retinal OCT B-scan. Colour images are converted to greyscale.",
    ),
    "dermamnist": DatasetSpec(
        key="dermamnist",
        title="Skin lesions (DermaMNIST)",
        modality="Dermatoscopic images of pigmented skin lesions",
        mode="rgb",
        classes=(
            "actinic keratoses & intraepithelial carcinoma",
            "basal cell carcinoma",
            "benign keratosis-like lesions",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular lesions",
        ),
        paper_acc_28=0.777,
        note="Upload a dermatoscopic lesion image (not an ordinary phone photo).",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    dataset: str
    checkpoint: str
    use_cbam: bool
    loss: str    # "focal" or "weighted_focal"
    label: str


def checkpoint_filename(dataset: str, use_cbam: bool, loss: str = "focal") -> str:
    """File names match the ones written by your Kaggle notebook."""
    if loss == "weighted_focal":
        if not use_cbam:
            raise ValueError("The weighted-focal experiment only exists for CBAM")
        return f"{dataset}_cbam_weighted_focal_best.pt"
    return f"{dataset}_{'CBAM' if use_cbam else 'No_CBAM'}_best.pt"


def _spec(dataset: str, use_cbam: bool, loss: str, label: str) -> ModelSpec:
    suffix = {"focal": "cbam" if use_cbam else "no_cbam", "weighted_focal": "weighted"}[loss]
    return ModelSpec(f"{dataset}_{suffix}", dataset,
                     checkpoint_filename(dataset, use_cbam, loss), use_cbam, loss, label)


MODELS: dict[str, ModelSpec] = {}
for _ds in DATASETS:
    for _spec_obj in (
        _spec(_ds, True, "focal", "MedNet + CBAM (focal loss)"),
        _spec(_ds, False, "focal", "MedNet without CBAM (ablation)"),
    ):
        MODELS[_spec_obj.key] = _spec_obj
_w = _spec("dermamnist", True, "weighted_focal", "MedNet + CBAM (class-weighted focal loss)")
MODELS[_w.key] = _w


def models_for(dataset: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.dataset == dataset]


def checkpoint_path(spec: ModelSpec) -> Path:
    return model_dir() / spec.checkpoint


def available_models_for(dataset: str) -> list[ModelSpec]:
    return [m for m in models_for(dataset) if checkpoint_path(m).is_file()]
