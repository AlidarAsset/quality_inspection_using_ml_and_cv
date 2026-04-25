# PaDiM Add-on + Interactive Demo

This pack adds **PaDiM** (third anomaly detection model) and an **interactive web demo**
to your existing project. Nothing here overwrites your trained PatchCore or EfficientAD
checkpoints.

## What's inside

| File | Purpose |
|------|---------|
| `padim.py` | PaDiM model class (same interface as PatchCore/EfficientAD) |
| `train_padim.py` | Train PaDiM only (~5–10 min on RTX 3050) |
| `evaluate_padim.py` | Evaluate PaDiM, **adds row to `metrics.csv`** without overwriting |
| `visualize_padim.py` | Generate PaDiM heatmaps + combined ROC / score plots |
| `demo_app.py` | Interactive Streamlit web app with random image picker |

## Setup

Drop all 5 files into your existing project folder (next to `dataset.py`,
`patchcore.py`, etc.). No changes to existing files needed.

```bash
pip install streamlit  # for the demo app (other deps already installed)
```

## 1️⃣ Train PaDiM (5–10 min)

```bash
python train_padim.py --data_path "C:/Users/alidar/Desktop/aitu/data/vial"
```

This creates `checkpoints/padim.pkl`. Your existing `patchcore.pkl` and
`efficientad.pt` are untouched.

**Tip:** for higher accuracy use `--backbone wide_resnet50` (slower, more VRAM).
Default `resnet18` is faster and usually accurate enough for vials.

## 2️⃣ Evaluate PaDiM

```bash
python evaluate_padim.py --data_path "C:/Users/alidar/Desktop/aitu/data/vial"
```

This **adds** a PaDiM row to `results/metrics.csv` — your PatchCore and
EfficientAD rows stay there. Output looks like:

```
            model  image_auroc  pixel_auroc  f1_score  threshold  pro_score
        PatchCore       0.8414       0.9412    0.9211     0.4890     0.8135
      EfficientAD       0.5339       0.5382    0.8607  81107.531     0.1726
            PaDiM       0.XXXX       0.XXXX    0.XXXX     0.XXXX     0.XXXX
```

## 3️⃣ Generate visualisations

```bash
python visualize_padim.py --data_path "C:/Users/alidar/Desktop/aitu/data/vial"
```

Creates:
- `results/predictions/PaDiM_predictions.png` — 6 sample heatmaps for PaDiM
- `results/roc_curves_all.png` — ROC for **all 3 models** on one plot
- `results/score_distribution_all.png` — score histograms for all 3

## 4️⃣ Run the interactive demo

```bash
streamlit run demo_app.py
```

Opens at `http://localhost:8501`. Features:

- **Pick a model** from a dropdown (PatchCore / EfficientAD / PaDiM)
- **🎲 Random image button** — click and get a random test image analysed instantly
- **Pick by index** — dial in any specific test sample
- **Upload your own image** — drag-and-drop any PNG/JPG to test
- **PASS/FAIL verdict** with the anomaly score and the chosen threshold
- **Side-by-side comparison** across all 3 models on the same image
- **Adjustable threshold slider** — see how the contour changes

### Cross-industry use

The demo works out of the box for **any MVTec-style dataset**. Just change the
"Dataset path" field in the sidebar to point at:

- PCB defect data (electronics)
- Fabric defect images (textiles)
- Food packaging (beverages)
- Metal castings (automotive)

The same trained models won't work directly (they're trained on vials), but you
can retrain in 5–10 min by pointing `train_padim.py` at the new data folder.

## For your diploma — what to show on demo day

1. Open `demo_app.py` in a browser on your laptop.
2. Click **🎲 Random image** a few times — show that the system correctly flags
   anomalies and passes good vials.
3. Switch the model dropdown to compare PatchCore vs PaDiM vs EfficientAD on
   the same image — visual proof that PatchCore wins.
4. Upload a custom image (e.g., a vial photo from your phone) to demonstrate
   the system handles real-world inputs.
5. Change the dataset path to a different MVTec category to show
   cross-industry capability.

This covers the **"Demonstration of Practical Results" — 10 points** part of
the rubric.
