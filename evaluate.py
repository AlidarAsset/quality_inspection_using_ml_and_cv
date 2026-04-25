import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VialTestDataset
from patchcore import PatchCore
from efficientad import EfficientAD
from metrics import evaluate_model, print_results, save_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   type=str,
                        default=r"C:\Users\alidar\Desktop\aitu\data\vial")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--no_pro",      action="store_true",
                        help="Skip PRO score (slow for large test sets)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (use 0 if errors on Windows)")
    return parser.parse_args()


def print_gpu_info(device: torch.device):
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        total_mb = props.total_memory / 1024 ** 2
        print(f"  GPU : {props.name}  ({total_mb:.0f} MB VRAM)")
    else:
        print("  GPU : NOT available — running on CPU")


def run_inference(model, test_loader, device):
    """Run model over all test batches, return scores, maps, labels, masks."""
    all_labels, all_image_scores = [], []
    all_masks,  all_score_maps   = [], []

    for imgs, labels, masks, defect_types, paths in tqdm(test_loader, desc="Evaluating"):
        imgs = imgs.to(device)

        image_scores, anomaly_maps = model.predict(imgs)
        # model.predict() already returns numpy arrays on CPU

        all_labels.append(labels.numpy())
        all_image_scores.append(image_scores)
        all_masks.append(masks.squeeze(1).numpy())        # [B, H, W]
        all_score_maps.append(anomaly_maps.squeeze(1))   # [B, H, W]

    return (
        np.concatenate(all_labels),
        np.concatenate(all_image_scores),
        np.concatenate(all_masks),
        np.concatenate(all_score_maps),
    )


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)

    print(f"\n[Evaluate Config]")
    print(f"  Data path : {args.data_path}")
    print(f"  Device    : {args.device}")
    print_gpu_info(device)

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

    results = []

    
    #  PatchCore
    
    patchcore_ckpt = os.path.join(args.checkpoints, "patchcore.pkl")
    if os.path.exists(patchcore_ckpt):
        print("\n[PatchCore] Loading checkpoint...")
        pc = PatchCore(device=args.device, img_size=args.img_size)
        pc.load(patchcore_ckpt)   # load() now auto-caches memory bank on GPU

        labels, img_scores, masks, score_maps = run_inference(pc, test_loader, device)
        res = evaluate_model(
            "PatchCore", labels, img_scores, masks, score_maps,
            compute_pro=not args.no_pro
        )
        results.append(res)
    else:
        print(f"[!] PatchCore checkpoint not found: {patchcore_ckpt}")

    
    #  EfficientAD
    
    ead_ckpt = os.path.join(args.checkpoints, "efficientad.pt")
    if os.path.exists(ead_ckpt):
        print("\n[EfficientAD] Loading checkpoint...")
        ead = EfficientAD(device=args.device, img_size=args.img_size)
        ead.load(ead_ckpt)        # load() now restores stats to GPU

        labels, img_scores, masks, score_maps = run_inference(ead, test_loader, device)
        res = evaluate_model(
            "EfficientAD", labels, img_scores, masks, score_maps,
            compute_pro=not args.no_pro
        )
        results.append(res)
    else:
        print(f"[!] EfficientAD checkpoint not found: {ead_ckpt}")

    if results:
        df = print_results(results)
        save_results(results, os.path.join(args.results_dir, "metrics.csv"))
        print("Evaluation complete!")
    else:
        print("[!] No checkpoints found. Run train.py first.")


if __name__ == "__main__":
    main()
