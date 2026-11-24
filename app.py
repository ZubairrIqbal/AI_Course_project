"""MedNet Medical Image Classifier - Streamlit app.

Upload a medical image (or pick a sample), choose the modality, and get:
class probabilities, a Grad-CAM explanation, and the model's full test-set metrics.
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from mednet import __version__  # noqa: E402
from mednet.config import (DATASETS, MODELS, available_models_for, checkpoint_path,  # noqa: E402
                           model_dir, samples_dir)
from mednet.gradcam import gradcam_for_stage, overlay  # noqa: E402
from mednet.inference import load_model, predict, preprocess  # noqa: E402
from mednet.metrics import load_metrics  # noqa: E402
from mednet.model import MedNet, count_parameters  # noqa: E402

st.set_page_config(page_title="MedNet Medical Image Classifier", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading model…", max_entries=2)
def get_model(model_key: str, model_folder: str):
    # model_folder is part of the cache key so changing MEDNET_MODEL_DIR never serves a stale model
    return load_model(model_key)


def pct(value, digits=1):
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def sample_files(dataset_key: str) -> list:
    folder = samples_dir() / dataset_key
    return sorted(folder.glob("c*_*.png")) if folder.is_dir() else []


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 MedNet")
st.sidebar.caption(f"Residual depthwise-separable CNN + CBAM · v{__version__}")

dataset_key = st.sidebar.selectbox("1. Image type", list(DATASETS), format_func=lambda k: DATASETS[k].title)
dataset = DATASETS[dataset_key]
st.sidebar.caption(dataset.modality)

available = available_models_for(dataset_key)
model_spec = None
if available:
    model_spec = st.sidebar.selectbox("2. Model", available, format_func=lambda m: m.label)
else:
    st.sidebar.warning("No checkpoint found for this image type.")

stage = st.sidebar.selectbox("3. Grad-CAM layer", [1, 2, 3, 4, 5], index=2,
                             format_func=lambda s: f"Stage {s}" + (" (default)" if s == 3 else ""),
                             help="Early stages = fine detail, late stages = coarse, more semantic regions.")
center_crop = st.sidebar.checkbox("Centre-crop to square", value=True,
                                  help="Avoids stretching non-square images when shrinking to 28×28.")
st.sidebar.divider()
st.sidebar.error("**Research demo only.** Not a medical device. Never use for diagnosis or treatment decisions.")

st.title("MedNet Medical Image Classifier")
st.caption("Portfolio project · PyTorch · Streamlit · deployed on Render · trained on 28×28 MedMNIST images")

tab_classify, tab_metrics, tab_arch, tab_about = st.tabs(
    ["🔍 Classify", "📊 Model metrics", "🧠 Architecture", "ℹ️ About & limits"])

# ------------------------------------------------------------------ classify
with tab_classify:
    if model_spec is None:
        st.error(
            f"No trained weights found for **{dataset.title}** in `{model_dir()}`.\n\n"
            "Copy your `.pt` files from the Kaggle notebook output into the `models/` folder "
            "(see README step 1), or train with `python scripts/train.py`."
        )
    else:
        st.info(dataset.note)
        sources = ["Upload an image"] + (["Use a sample test image"] if sample_files(dataset_key) else [])
        source = st.radio("Image source", sources, horizontal=True, label_visibility="collapsed")

        image, true_label, name = None, None, None
        if source == "Upload an image":
            up = st.file_uploader("Image file", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
            if up is not None:
                try:
                    image, name = Image.open(up), up.name
                    image.load()
                except Exception:
                    st.error("That file could not be read as an image.")
                    image = None
        else:
            files = sample_files(dataset_key)
            choice = st.selectbox(
                "Sample", files,
                format_func=lambda p: f"{dataset.classes[int(p.stem.split('_')[0][1:])]} (sample {int(p.stem.split('_')[1]) + 1})")
            image, name = Image.open(choice), choice.name
            true_label = int(choice.stem.split("_")[0][1:])

        if image is None:
            st.write("⬆️ Provide an image to see the prediction.")
        else:
            model = get_model(model_spec.key, str(model_dir()))
            pre = preprocess(image, dataset_key, center_crop)
            t0 = time.perf_counter()
            probs = predict(model, pre.tensor)
            latency_ms = (time.perf_counter() - t0) * 1000
            top = int(np.argmax(probs))
            heat, _ = gradcam_for_stage(model, pre.tensor, stage, class_id=top)

            c1, c2, c3 = st.columns(3)
            c1.markdown("**Your image**")
            c1.image(image.convert("RGB"), width=224)
            c2.markdown("**Model input (28×28)**")
            c2.image(Image.fromarray(pre.display).resize((224, 224), Image.Resampling.NEAREST), width=224)
            c3.markdown(f"**Grad-CAM · stage {stage}**")
            c3.image(overlay(pre.display, heat), width=224)

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted class", dataset.classes[top])
            m2.metric("Confidence", pct(float(probs[top])))
            m3.metric("Inference time", f"{latency_ms:.0f} ms")

            if true_label is not None:
                ok = true_label == top
                (st.success if ok else st.error)(
                    f"Ground truth: **{dataset.classes[true_label]}** → "
                    f"{'correct ✅' if ok else 'incorrect ❌ (the model is imperfect – see metrics tab)'}")
            if probs[top] < 0.5:
                st.warning("Low confidence: the model is unsure. This often happens for images unlike the "
                           "training data (e.g. a phone photo instead of a dermatoscope image).")
            st.caption("The model always outputs *some* class, even for images it was never trained on. "
                       "Confidence is not the same as correctness.")

            table = pd.DataFrame({"Class": dataset.classes, "Probability": probs * 100}).sort_values(
                "Probability", ascending=False).reset_index(drop=True)
            st.dataframe(table, hide_index=True, column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=100.0, format="%.1f%%")})

            with st.expander("Grad-CAM at all five stages"):
                cols = st.columns(5)
                for s, col in zip(range(1, 6), cols):
                    h, _ = gradcam_for_stage(model, pre.tensor, s, class_id=top)
                    col.image(overlay(pre.display, h, size=160), caption=f"Stage {s}", width=160)
                st.caption("Grad-CAM shows where the network looked; it does not prove clinical reasoning.")

            result = {"file": name, "dataset": dataset_key, "model": model_spec.key,
                      "prediction": dataset.classes[top], "confidence": float(probs[top]),
                      "probabilities": {c: float(p) for c, p in zip(dataset.classes, probs)}}
            st.download_button("Download result (JSON)", json.dumps(result, indent=2),
                               file_name="prediction.json", mime="application/json")

# ------------------------------------------------------------------ metrics
with tab_metrics:
    all_for_ds = [m for m in MODELS.values() if m.dataset == dataset_key]
    spec_m = st.selectbox("Model", all_for_ds, format_func=lambda m: m.label,
                          index=all_for_ds.index(model_spec) if model_spec in all_for_ds else 0, key="metrics_model")
    data = load_metrics(spec_m.key)
    if data is None:
        st.info("No metrics file for this model yet. Run `python scripts/evaluate.py --all`.")
    else:
        o = data["overall"]
        st.caption(f"Evaluated on **{data['test_n']:,}** official test images. Source: {data['source']}")
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"),
                                            ("Macro precision", "macro_precision"),
                                            ("Macro recall", "macro_recall"), ("AUC (OvR)", "auc")]):
            col.metric(label, "–" if o.get(key) is None else f"{o[key]:.3f}")

        st.markdown("##### Per-class results")
        pc = pd.DataFrame(data["per_class"]).rename(columns={
            "class": "Class", "precision": "Precision", "recall": "Recall (sensitivity)",
            "f1": "F1", "support": "Test images", "auc": "AUC"})
        pc = pc.dropna(axis=1, how="all")
        st.dataframe(pc, hide_index=True)
        if "Recall (sensitivity)" in pc and pc["Recall (sensitivity)"].min() < 0.3:
            weak = pc.loc[pc["Recall (sensitivity)"] < 0.3, "Class"].tolist()
            st.warning("Weak classes (recall < 30%): " + ", ".join(weak) +
                       ". Overall accuracy hides this – look at per-class recall and macro F1.")

        cm = data.get("confusion_matrix")
        if cm:
            st.markdown("##### Confusion matrix (row-normalised)")
            cm = np.array(cm, dtype=float)
            cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(dataset.classes)), dataset.classes, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(dataset.classes)), dataset.classes, fontsize=8)
            for i in range(cmn.shape[0]):
                for j in range(cmn.shape[1]):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if cmn[i, j] > 0.5 else "black")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("Confusion matrix appears after running `scripts/evaluate.py`.")

    st.markdown("##### Variant comparison on this dataset")
    rows = []
    for m in all_for_ds:
        d = load_metrics(m.key)
        if d:
            rows.append({"Model": m.label, "Accuracy": d["overall"]["accuracy"],
                         "Macro F1": d["overall"]["macro_f1"], "AUC": d["overall"]["auc"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Published 28×28 reference accuracy for {dataset.title}: **{dataset.paper_acc_28:.1%}**. "
               "This project used a reduced CPU training budget (subset of the training data, 15 epochs, single seed), "
               "so it reproduces the *architecture*, not the paper's full result.")

# ------------------------------------------------------------------ architecture
with tab_arch:
    st.markdown(
        "**MedNet** stacks five residual blocks. Each block = two *depthwise-separable convolutions* "
        "(cheap spatial + channel mixing) → **CBAM** attention (channel then spatial) → residual shortcut. "
        "Global average pooling feeds a 256-unit dropout layer and the classifier.")
    rows = [("Input", "3 × 28 × 28", "–"), ("Stage 1", "64 × 28 × 28", "stride 1"), ("Stage 2", "128 × 14 × 14", "stride 2"),
            ("Stage 3", "256 × 7 × 7", "stride 2"), ("Stage 4", "512 × 4 × 4", "stride 2"),
            ("Stage 5", "1024 × 2 × 2", "stride 2"), ("Global avg pool → FC 256 → Dropout 0.4 → FC", f"{dataset.n_classes} classes", "–")]
    st.dataframe(pd.DataFrame(rows, columns=["Layer", "Output shape", "Note"]), hide_index=True)
    c1, c2 = st.columns(2)
    c1.metric("Parameters (with CBAM)", f"{count_parameters(MedNet(dataset.n_classes, True)):,}")
    c2.metric("Parameters (no CBAM)", f"{count_parameters(MedNet(dataset.n_classes, False)):,}")
    st.markdown("**Training recipe:** focal loss (γ=2) · AdamW (lr 3e-4, wd 1e-4) · cosine schedule · "
                "batch 16 · ≤15 epochs · early stopping (patience 5) · seed 42. "
                "The DermaMNIST *class-weighted* variant scales the focal loss by inverse-frequency^0.25 class weights.")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        """
**What this is.** A portfolio project that turns a research-paper reproduction (MedNet) into a deployable web app.

**Honest limitations**
- Trained and tested on **28×28 thumbnails** from MedMNIST; real clinical images are far higher resolution.
- Trained with a reduced budget on a CPU. OCT accuracy (~61%) and DermaMNIST minority-class recall (e.g. melanoma ~23%) are **well below clinical usefulness**.
- One model per image type: choose the correct type in the sidebar. Out-of-distribution images still get a prediction.
- Single training seed; CBAM vs no-CBAM differences are small and not statistically conclusive.

**Data.** MedMNIST v2 (Yang et al., *Scientific Data*, 2023). BloodMNIST and OCTMNIST are CC BY 4.0;
DermaMNIST derives from HAM10000 and is **CC BY-NC 4.0** (non-commercial). Check licences before any other use.

**Not medical advice.** Do not upload patient-identifiable images to a public demo.
        """)
    st.caption("Checkpoints present: " + (", ".join(m.key for m in MODELS.values() if checkpoint_path(m).is_file()) or "none"))
