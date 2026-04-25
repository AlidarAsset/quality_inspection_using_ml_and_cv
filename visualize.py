import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from dataset import VialTestDataset
from patchcore import PatchCore
from efficientad import EfficientAD


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   type=str,
                        default=r"C:\Users\alidar\Desktop\aitu\data\vial")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--n_samples",   type=int, default=6)
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    return np.clip(img * std + mean, 0, 1)


def score_to_heatmap(score_map):
    norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    return cm.jet(norm)[:, :, :3]


def draw_contours(img_np, score_map, threshold_percentile=93,
                  contour_color=(1.0, 0.1, 0.1), thickness=2):
    thresh = np.percentile(score_map, threshold_percentile)
    binary = (score_map >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = (img_np * 255).astype(np.uint8).copy()
    color_uint8 = tuple(int(c * 255) for c in contour_color)
    cv2.drawContours(out, contours, -1, color_uint8, thickness)
    overlay = out.copy()
    cv2.drawContours(overlay, contours, -1, color_uint8, cv2.FILLED)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    return out.astype(np.float32) / 255.0


def collect_predictions(model, test_loader, device):
    records = []
    for imgs, labels, masks, defect_types, paths in tqdm(test_loader, desc="  Inference"):
        imgs = imgs.to(device)
        img_scores, anomaly_maps = model.predict(imgs)
        for i in range(len(imgs)):
            records.append({
                "img":       imgs[i].cpu(),
                "label":     int(labels[i]),
                "mask":      masks[i].squeeze(0).numpy(),
                "score_map": anomaly_maps[i, 0],
                "img_score": float(img_scores[i]),
                "defect":    defect_types[i],
                "path":      paths[i],
            })
    return records


def plot_sample_predictions(records, model_name, save_dir, n_samples=6, seed=42):
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)

    anomalous = [r for r in records if r["label"] == 1]
    if not anomalous:
        print(f"[Viz] No anomalous samples for {model_name}")
        return

    samples = random.sample(anomalous, min(n_samples, len(anomalous)))
    n = len(samples)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Original Image", "Anomaly Heatmap", "Detected Anomaly Region"]
    for col_idx, title in enumerate(col_titles):
        axes[0][col_idx].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for row, r in enumerate(samples):
        img_np    = denormalize(r["img"])
        score_map = r["score_map"]
        heatmap   = score_to_heatmap(score_map)
        contoured = draw_contours(img_np, score_map, threshold_percentile=93)

        axes[row][0].imshow(img_np)
        axes[row][0].axis("off")
        axes[row][0].set_ylabel(f"Sample {row+1}", fontsize=10, rotation=90, labelpad=5)

        axes[row][1].imshow(heatmap)
        axes[row][1].axis("off")
        axes[row][1].set_xlabel(f"Score: {r['img_score']:.4f}", fontsize=9)

        axes[row][2].imshow(contoured)
        axes[row][2].axis("off")

    # Colorbar
    cbar_ax = fig.add_axes([0.37, 0.02, 0.27, 0.015])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                 label="Normalised Anomaly Score")

    red_patch = mpatches.Patch(color=(1, 0.1, 0.1), alpha=0.7,
                               label="Detected anomaly region")
    fig.legend(handles=[red_patch], loc="lower right", fontsize=10,
               bbox_to_anchor=(0.99, 0.02))

    fig.suptitle(f"{model_name} — Vial Anomaly Detection Results",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    fname = os.path.join(save_dir, f"{model_name}_predictions.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {fname}")


def plot_roc_curves(all_records, save_path):
    plt.figure(figsize=(7, 6))
    colors = ["steelblue", "tomato"]

    for (model_name, records), color in zip(all_records.items(), colors):
        labels = np.array([r["label"] for r in records])
        scores = np.array([r["img_score"] for r in records])
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.5,
                 label=f"{model_name}  (AUROC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — Vial Anomaly Detection", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Viz] ROC curves → {save_path}")


def plot_score_distribution(all_records, save_path):
    n = len(all_records)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (model_name, records) in zip(axes, all_records.items()):
        normal  = [r["img_score"] for r in records if r["label"] == 0]
        anomaly = [r["img_score"] for r in records if r["label"] == 1]
        ax.hist(normal,  bins=30, alpha=0.6, color="steelblue", label="Normal",  density=True)
        ax.hist(anomaly, bins=30, alpha=0.6, color="tomato",    label="Anomaly", density=True)
        ax.set_title(f"{model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Anomaly Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Anomaly Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Viz] Score distribution → {save_path}")


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Viz] Device: {device}")

    test_ds = VialTestDataset(args.data_path, img_size=args.img_size)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    all_records = {}

    pc_ckpt = os.path.join(args.checkpoints, "patchcore.pkl")
    if os.path.exists(pc_ckpt):
        print("\n[PatchCore] Loading...")
        pc = PatchCore(device=str(device), img_size=args.img_size)
        pc.load(pc_ckpt)
        all_records["PatchCore"] = collect_predictions(pc, test_loader, device)

    ead_ckpt = os.path.join(args.checkpoints, "efficientad.pt")
    if os.path.exists(ead_ckpt):
        print("\n[EfficientAD] Loading...")
        ead = EfficientAD(device=str(device), img_size=args.img_size)
        ead.load(ead_ckpt)
        all_records["EfficientAD"] = collect_predictions(ead, test_loader, device)

    if not all_records:
        print("[!] No checkpoints found. Run train.py first.")
        return

    for model_name, records in all_records.items():
        plot_sample_predictions(records, model_name,
                                save_dir=os.path.join(args.results_dir, "predictions"),
                                n_samples=args.n_samples, seed=args.seed)

    plot_roc_curves(all_records,
                    save_path=os.path.join(args.results_dir, "roc_curves.png"))
    plot_score_distribution(all_records,
                             save_path=os.path.join(args.results_dir, "score_distribution.png"))

    print(f"\n All visualizations saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
