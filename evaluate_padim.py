"""
evaluate_padim.py — Evaluate ONLY PaDiM on the MVTec AD2 vial test set.
Adds PaDiM row to the existing metrics.csv (does not overwrite other models).

Usage:
  python evaluate_padim.py --data_path "C:/Users/alidar/Desktop/aitu/data/vial"
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VialTestDataset
from padim import PaDiM
from metrics import evaluate_model, print_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   type=str,
                        default=r"C:\Users\alidar\Desktop\aitu\data\vial") #You can change this to your dataset path
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--no_pro",      action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def run_inference(model, test_loader, device):
    all_labels, all_image_scores = [], []
    all_masks,  all_score_maps   = [], []
    for imgs, labels, masks, defect_types, paths in tqdm(test_loader,
                                                          desc="[PaDiM] Evaluating"):
        imgs = imgs.to(device)
        image_scores, anomaly_maps = model.predict(imgs)
        all_labels.append(labels.numpy())
        all_image_scores.append(image_scores)
        all_masks.append(masks.squeeze(1).numpy())
        all_score_maps.append(anomaly_maps.squeeze(1))
    return (
        np.concatenate(all_labels),
        np.concatenate(all_image_scores),
        np.concatenate(all_masks),
        np.concatenate(all_score_maps),
    )


def merge_into_metrics_csv(new_result: dict, csv_path: str):
    """
    Append PaDiM row into existing metrics.csv if present, else create it.
    Removes any old PaDiM row first to avoid duplicates.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["model"] != "PaDiM"]
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[PaDiM] Updated {csv_path}")
    return df


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)

    print(f"\n[Evaluate Config — PaDiM]")
    print(f"  Data path : {args.data_path}")
    print(f"  Device    : {args.device}")

    test_ds = VialTestDataset(args.data_path, img_size=args.img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.num_workers > 0),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    padim_ckpt = os.path.join(args.checkpoints, "padim.pkl")
    if not os.path.exists(padim_ckpt):
        print(f"[!] PaDiM checkpoint not found: {padim_ckpt}")
        print(f"    Run: python train_padim.py first.")
        return

    print(f"\n[PaDiM] Loading checkpoint...")
    model = PaDiM(device=args.device, img_size=args.img_size)
    model.load(padim_ckpt)

    labels, img_scores, masks, score_maps = run_inference(model, test_loader, device)
    res = evaluate_model(
        "PaDiM", labels, img_scores, masks, score_maps,
        compute_pro=not args.no_pro,
    )

    print_results([res])

    # Save / merge into metrics.csv
    csv_path = os.path.join(args.results_dir, "metrics.csv")
    df = merge_into_metrics_csv(res, csv_path)

    
    print("  ALL MODELS — UPDATED METRICS")
    print(df.to_string(index=False))
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
