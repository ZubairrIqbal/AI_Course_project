# 🩺 MedNet Medical Image Classifier

A deployable web app built around **MedNet** – a five-stage residual depthwise-separable CNN with **CBAM** attention – reproduced from a research paper and trained on three [MedMNIST](https://medmnist.com/) datasets.

Upload an image → get the predicted class, all class probabilities, a **Grad-CAM** explanation, and the model's **full test-set metrics** (accuracy, macro F1, precision, recall, AUC, per-class table, confusion matrix, CBAM vs no-CBAM ablation).

| Image type | Dataset | Classes | Test accuracy (this repo) | Test macro F1 |
|---|---|---|---|---|
| Blood cells | BloodMNIST | 8 | 88.1 % | 0.866 |
| Retinal OCT | OCTMNIST | 4 | 60.9 % | 0.545 |
| Skin lesions | DermaMNIST | 7 | 72.1 % (class-weighted variant: 71.4 %, macro F1 0.431) | 0.411 |

> ⚠️ **Research / portfolio demo – not a medical device.** Models were trained on 28×28 thumbnails with a reduced CPU budget. See [Limitations](#limitations).

**Stack:** PyTorch · Streamlit · Matplotlib · pytest · GitHub Actions · Render (free tier)

---

## Project structure

```
mednet-classifier/
├── app.py                    # Streamlit UI (4 tabs: Classify, Metrics, Architecture, About)
├── mednet/                   # importable package (no UI code)
│   ├── config.py             #   dataset/class/checkpoint registry
│   ├── model.py              #   MedNet + CBAM (identical to the notebook)
│   ├── inference.py          #   load_model / preprocess / predict
│   ├── gradcam.py            #   Grad-CAM for any of the 5 stages
│   ├── losses.py             #   FocalLoss, WeightedFocalLoss
│   ├── data.py               #   MedMNIST loading, stratified subsets
│   └── metrics.py            #   compute/save/load metrics JSON
├── scripts/
│   ├── train.py              # retrain any model from the command line
│   ├── evaluate.py           # full test-set metrics -> assets/metrics/*.json
│   └── export_samples.py     # save example test images for the demo
├── models/                   # ← put your .pt checkpoints here
├── assets/metrics/           # metrics JSON shown in the app
├── assets/samples/           # example images (created by export_samples.py)
├── tests/                    # pytest suite (architecture, preprocessing, Grad-CAM, app smoke test)
├── .github/workflows/ci.yml  # tests on every push
├── render.yaml               # Render Blueprint (one-click deploy config)
└── requirements.txt          # slim runtime deps (CPU PyTorch)
```

---

## Step-by-step guide

### Step 0 – Prerequisites
- [Git](https://git-scm.com/), Python **3.11** (3.10–3.12 also work), a [GitHub](https://github.com) account, a [Render](https://render.com) account (sign up with GitHub).

### Step 1 – Get your trained weights from Kaggle
The notebook saved checkpoints in `/kaggle/working/mednet_results/`. In your Kaggle notebook open the **Output** tab, download these files and copy them into this repo's `models/` folder:

| Required (3) | Optional |
|---|---|
| `bloodmnist_CBAM_best.pt` | `bloodmnist_No_CBAM_best.pt` |
| `octmnist_CBAM_best.pt` | `octmnist_No_CBAM_best.pt` |
| `dermamnist_CBAM_best.pt` | `dermamnist_No_CBAM_best.pt` |
| `dermamnist_cbam_weighted_focal_best.pt` | |

Each file is ≈ 13 MB, so plain Git works (no Git LFS needed). The app only lists models whose file exists.

> No Kaggle files? Train locally instead (Step 9) – the app is identical either way.

### Step 2 – Local setup
```bash
git clone <your-repo-url> mednet-classifier && cd mednet-classifier   # or just cd into the folder
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # runtime + medmnist + scikit-learn + pytest
```

### Step 3 – Generate real metrics and sample images (recommended)
```bash
python scripts/evaluate.py --all          # full test split -> assets/metrics/*.json (adds confusion matrices, per-class AUC)
python scripts/export_samples.py          # 2 real test images per class -> assets/samples/
```
This downloads the MedMNIST `.npz` files once (~ 200 MB total, stored in `~/.medmnist`). The repo already ships metrics seeded from your notebook output, so the app works even if you skip this step – but running it replaces them with exact, full-precision numbers.

### Step 4 – Run the tests
```bash
pytest -q
```
Includes a check that parameter counts equal the notebook's (3,269,838 / 3,268,810 / 3,269,581) – proof the architecture was ported faithfully.

### Step 5 – Run the app locally
```bash
streamlit run app.py
```
Open http://localhost:8501. Try a sample image, then upload your own.

### Step 6 – Push to GitHub
```bash
git init
git add .
git commit -m "MedNet classifier: Streamlit app, tests, Render config"
git branch -M main
git remote add origin https://github.com/<you>/mednet-classifier.git
git push -u origin main
```
Check the **Actions** tab – the CI badge should turn green.

### Step 7 – Deploy on Render (free)
**Option A – Blueprint (easiest):** Render dashboard → **New +** → **Blueprint** → connect the repo → Render reads `render.yaml` → **Apply**.

**Option B – manual:** **New +** → **Web Service** → connect the repo and set:

| Field | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Instance type | **Free** |
| Environment variable | `PYTHON_VERSION` = `3.11.9` |
| Health check path | `/_stcore/health` |

The first build takes ~5–8 minutes (it downloads CPU-only PyTorch). When it says **Live**, open the `*.onrender.com` URL.

### Step 8 – Know the free-tier behaviour
- The service **sleeps after ~15 min idle**; the next visit takes ~30–60 s to wake. Open it once before showing it to anyone.
- **512 MB RAM** limit. The app is built for it (CPU torch, 1 thread, at most 2 models cached). If you ever see out-of-memory restarts, deploy only the 3 required checkpoints.
- Every `git push` to `main` auto-redeploys (`autoDeploy: true`).

### Step 9 – (Optional) retrain from scratch
```bash
python scripts/train.py --dataset dermamnist                         # CBAM + focal loss (notebook recipe)
python scripts/train.py --dataset dermamnist --no-cbam               # ablation
python scripts/train.py --dataset dermamnist --loss weighted_focal   # class-balanced experiment
python scripts/train.py --dataset bloodmnist --train-limit 0 --epochs 30   # full training set (GPU recommended)
python scripts/evaluate.py --all                                     # refresh metrics
```
Checkpoints are written to `models/` using the same filenames the app expects.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| App says "No trained weights found" | Put the `.pt` files in `models/` (exact filenames in Step 1) and `git add models/*.pt`. |
| `Missing key(s) in state_dict` | The checkpoint belongs to a different dataset/variant than its filename claims (class count or CBAM mismatch). |
| Render build fails on torch | Make sure `PYTHON_VERSION` is set to `3.11.9` and `requirements.txt` still starts with the `--extra-index-url …/cpu` line. |
| Render service restarts / OOM | Remove optional `No_CBAM` checkpoints from the deploy; keep the 3–4 required ones. |
| Predictions look random for my photo | Expected: models were trained on 28×28 research datasets. Use the sample images, or dermatoscopic/OCT/blood-smear images. |
| `git push` rejects a file > 100 MB | Not applicable here (≈ 13 MB each); if you add larger files use Git LFS. |

---

## Limitations
- **28×28 inputs only.** Uploaded images are centre-cropped and shrunk, discarding almost all clinical detail.
- **Reduced training budget** (subset of training data, ≤ 15 epochs, CPU, one seed) – results are *below* the published numbers (e.g. OCT 60.9 % vs 92.7 % reported). This is an architecture reproduction, not a full experimental reproduction.
- **Class imbalance:** DermaMNIST accuracy (72 %) is driven by the majority class (melanocytic nevi); melanoma recall is only ≈ 23 %. The class-weighted focal loss improved macro F1 by +0.019 and some minority classes but hurt others (e.g. basal cell carcinoma recall 69 % → 48 %).
- CBAM vs no-CBAM differences are small and based on a single seed – not statistically conclusive.
- The model always returns a class, even for out-of-distribution images; softmax confidence ≠ correctness.
- Never upload patient-identifiable data to a public demo.

## How to present this project
- *"Ported a research architecture (MedNet: depthwise-separable residual blocks + CBAM) to PyTorch, verified against the original with parameter-count tests, and shipped it as a Streamlit app deployed on Render with CI via GitHub Actions."*
- *"Evaluated on full official test sets with macro-F1, per-class recall, AUC and Grad-CAM; ran a controlled CBAM ablation and a class-weighted focal-loss experiment, and reported negative results honestly."*
- Talking points: why accuracy misleads under class imbalance; why Grad-CAM ≠ clinical reasoning; free-tier constraints (CPU torch wheel, 512 MB, cold starts); separation of UI (`app.py`) from logic (`mednet/`) for testability.

## Credits & licences
- Code: MIT (see `LICENSE`).
- Data: MedMNIST v2 – Yang et al., *Scientific Data* 2023. BloodMNIST & OCTMNIST: CC BY 4.0. **DermaMNIST (HAM10000): CC BY-NC 4.0 – non-commercial use only.**
- MedNet architecture: from the paper you reproduced – add its full citation here.
