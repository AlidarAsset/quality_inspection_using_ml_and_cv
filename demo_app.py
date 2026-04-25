"""
demo_app.py - Interactive Anomaly Detection Demo (Streamlit)

Features:
  - Pick any trained model: PatchCore / EfficientAD / PaDiM
  - Random test image button - one click to analyse a random sample
  - Pick by index or upload your own image
  - Side-by-side visualisation: original | heatmap | anomaly contour
  - Image-level PASS/FAIL verdict with adjustable threshold
  - Model comparison panel when multiple checkpoints are available
  - Cross-industry: point the dataset path at any MVTec-style folder

Usage:
  pip install streamlit
  streamlit run demo_app.py
"""

import os
import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.cm as cm
import cv2

from dataset import VialTestDataset, get_transforms
from patchcore import PatchCore
from efficientad import EfficientAD

try:
    from padim import PaDiM
    HAS_PADIM = True
except ImportError:
    HAS_PADIM = False



# Page configuration


st.set_page_config(
    page_title="Anomaly Detection Demo",
    page_icon="microscope",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .verdict-pass {
        background: #e8f5e9;
        border: 1.5px solid #43a047;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .verdict-fail {
        background: #ffebee;
        border: 1.5px solid #e53935;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
    }
    div[data-testid="stImage"] img {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)



# Utility functions


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    return np.clip(img * std + mean, 0, 1)


def score_to_heatmap(score_map):
    norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    return cm.jet(norm)[:, :, :3]


def draw_contours(img_np, score_map, threshold_percentile=93,
                  contour_color=(0.96, 0.26, 0.21), thickness=2):
    thresh = np.percentile(score_map, threshold_percentile)
    binary = (score_map >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = (img_np * 255).astype(np.uint8).copy()
    color_u8 = tuple(int(c * 255) for c in contour_color)
    cv2.drawContours(out, contours, -1, color_u8, thickness)
    overlay = out.copy()
    cv2.drawContours(overlay, contours, -1, color_u8, cv2.FILLED)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    return out.astype(np.float32) / 255.0



# Model and dataset loading (cached)


@st.cache_resource
def load_models(checkpoint_dir: str, img_size: int = 256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}

    pc_path = os.path.join(checkpoint_dir, "patchcore.pkl")
    if os.path.exists(pc_path):
        m = PatchCore(device=device, img_size=img_size)
        m.load(pc_path)
        models["PatchCore"] = m

    ead_path = os.path.join(checkpoint_dir, "efficientad.pt")
    if os.path.exists(ead_path):
        m = EfficientAD(device=device, img_size=img_size)
        m.load(ead_path)
        models["EfficientAD"] = m

    if HAS_PADIM:
        padim_path = os.path.join(checkpoint_dir, "padim.pkl")
        if os.path.exists(padim_path):
            m = PaDiM(device=device, img_size=img_size)
            m.load(padim_path)
            models["PaDiM"] = m

    return models, device


@st.cache_resource
def load_dataset(data_path: str, img_size: int = 256):
    return VialTestDataset(data_path, img_size=img_size)


def analyse_image(model, img_tensor, device):
    """Run inference on a single image tensor [3, H, W]."""
    batch = img_tensor.unsqueeze(0).to(device)
    image_scores, anomaly_maps = model.predict(batch)
    return float(image_scores[0]), anomaly_maps[0, 0]


def transform_pil(pil_img, img_size=256):
    """Apply the same preprocessing as VialTestDataset."""
    transform = get_transforms(img_size, split="test")
    return transform(pil_img.convert("RGB"))



# Sidebar


with st.sidebar:
    st.markdown("### Settings")

    data_path = st.text_input(
        "Dataset path",
        value=r"C:\Users\alidar\Desktop\aitu\data\vial",
        help="Path to any MVTec-style dataset. Change for cross-industry use.",
    )
    checkpoint_dir = st.text_input("Checkpoints folder", value="checkpoints")
    img_size = st.selectbox("Image size", [256, 224, 320], index=0)

    st.markdown("---")
    st.markdown(
        "**Cross-industry use:** point the dataset path at any folder "
        "following the MVTec layout (`train/good/`, `test_public/good/`, "
        "`test_public/bad/`). The same interface works for PCBs, fabric, "
        "food packaging, castings, and more."
    )
    st.markdown("---")
    st.caption(
        "PatchCore: Roth et al., 2022  \n"
        "EfficientAD: Batzner et al., 2024  \n"
        "PaDiM: Defard et al., 2021  \n"
        "Dataset: MVTec AD2, Bauer et al., 2024"
    )



# Header


st.title("Quality Inspection — Anomaly Detection Demo")
st.caption(
    "Unsupervised deep learning for surface defect detection. "
    "Trained only on normal vials; no defect labels required."
)


# Load models and data


try:
    models, device = load_models(checkpoint_dir, img_size)
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

if not models:
    st.error(
        f"No model checkpoints found in `{checkpoint_dir}/`. "
        "Run `train.py` and/or `train_padim.py` first."
    )
    st.stop()

device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
st.info(
    f"Loaded {len(models)} model(s): **{', '.join(models.keys())}** "
    f"— running on **{device_label}**"
)

try:
    test_ds = load_dataset(data_path, img_size)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()


# Controls


col_model, col_source, col_thresh = st.columns([2, 2, 2])

with col_model:
    selected_model = st.selectbox("Model", list(models.keys()), index=0)

with col_source:
    image_source = st.radio(
        "Image source",
        ["Random from test set", "Pick by index", "Upload your own"],
    )

with col_thresh:
    threshold_pct = st.slider(
        "Contour threshold (percentile)",
        min_value=80, max_value=99, value=93, step=1,
        help="Higher = only the most extreme anomalies are outlined",
    )

st.markdown("---")


# Image selection


img_tensor  = None
true_label  = None
img_path_str = None

if image_source == "Random from test set":
    if "rand_idx" not in st.session_state:
        st.session_state.rand_idx = random.randint(0, len(test_ds) - 1)
    if st.button("Pick a random image", use_container_width=False):
        st.session_state.rand_idx = random.randint(0, len(test_ds) - 1)

    idx = st.session_state.rand_idx
    img_tensor, true_label, mask, defect_type, img_path_str = test_ds[idx]
    st.caption(
        f"Sample #{idx}  |  defect type: `{defect_type}`  |  "
        f"ground truth: **{'ANOMALY' if true_label == 1 else 'NORMAL'}**"
    )

elif image_source == "Pick by index":
    idx = st.number_input(
        "Test sample index",
        min_value=0, max_value=len(test_ds) - 1, value=0, step=1,
    )
    img_tensor, true_label, mask, defect_type, img_path_str = test_ds[idx]
    st.caption(
        f"Sample #{idx}  |  defect type: `{defect_type}`  |  "
        f"ground truth: **{'ANOMALY' if true_label == 1 else 'NORMAL'}**"
    )

else:
    uploaded = st.file_uploader("Upload an image (PNG / JPG)", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        pil = Image.open(uploaded)
        img_tensor = transform_pil(pil, img_size)
        true_label = None
        st.caption(f"Uploaded: `{uploaded.name}` — no ground truth available")


# Inference and results


if img_tensor is not None:
    model = models[selected_model]

    with st.spinner(f"Running {selected_model}..."):
        score, score_map = analyse_image(model, img_tensor, device)

    img_np    = denormalize(img_tensor)
    heatmap   = score_to_heatmap(score_map)
    contoured = draw_contours(img_np, score_map, threshold_percentile=threshold_pct)

    # Per-model thresholds derived from evaluation metrics
    verdict_thresholds = {
        "PatchCore":   0.489,
        "EfficientAD": 81107.5,
        "PaDiM":       10.547,
    }
    thresh = verdict_thresholds.get(selected_model, score * 0.8)
    is_anomaly = score > thresh

    # Result metrics
    st.markdown("#### Result")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Anomaly score", f"{score:.4f}")
    with m2:
        st.metric("Decision threshold", f"{thresh:.4f}")
    with m3:
        if is_anomaly:
            st.markdown(
                "<div class='verdict-fail'>"
                "<div style='font-size:1.3rem;font-weight:700;color:#c62828;'>FAIL</div>"
                "<div style='font-size:0.85rem;color:#555;'>Anomaly detected</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='verdict-pass'>"
                "<div style='font-size:1.3rem;font-weight:700;color:#2e7d32;'>PASS</div>"
                "<div style='font-size:0.85rem;color:#555;'>Within normal range</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    # Comparison with ground truth
    if true_label is not None:
        truth_str = "ANOMALY" if true_label == 1 else "NORMAL"
        pred_str  = "ANOMALY" if is_anomaly else "NORMAL"
        correct   = (true_label == 1) == is_anomaly
        if correct:
            st.success(f"Correct prediction: **{pred_str}** | Ground truth: **{truth_str}**")
        else:
            st.warning(f"Mismatch: predicted **{pred_str}** | Ground truth: **{truth_str}**")

    st.markdown("---")

    # Visualisation panels
    st.markdown("#### Visualisation")
    pan_a, pan_b, pan_c = st.columns(3)

    with pan_a:
        st.markdown("<div class='section-header'>Original image</div>", unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)

    with pan_b:
        st.markdown("<div class='section-header'>Anomaly heatmap</div>", unsafe_allow_html=True)
        st.image(heatmap, use_container_width=True)

    with pan_c:
        st.markdown("<div class='section-header'>Detected region</div>", unsafe_allow_html=True)
        st.image(contoured, use_container_width=True)

    # Cross-model comparison
    if len(models) > 1:
        st.markdown("---")
        st.markdown("#### Compare all models on this image")
        cmp_cols = st.columns(len(models))
        for col, (name, m) in zip(cmp_cols, models.items()):
            with col:
                cmp_score, cmp_map = analyse_image(m, img_tensor, device)
                cmp_heat  = score_to_heatmap(cmp_map)
                cmp_thresh = verdict_thresholds.get(name, cmp_score * 0.8)
                cmp_anom  = cmp_score > cmp_thresh

                st.markdown(f"**{name}**")
                st.image(cmp_heat, use_container_width=True)
                st.caption(f"Score: {cmp_score:.4f}")
                if cmp_anom:
                    st.markdown(
                        "<span style='color:#c62828;font-weight:600;'>FAIL</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:#2e7d32;font-weight:600;'>PASS</span>",
                        unsafe_allow_html=True,
                    )
