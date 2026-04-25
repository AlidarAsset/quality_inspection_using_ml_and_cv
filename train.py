import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import VialTrainDataset
from patchcore import PatchCore
from efficientad import EfficientAD


def get_args():
    parser = argparse.ArgumentParser(description="Vial Anomaly Detection — Train")
    parser.add_argument("--model",     type=str, default="both",
                        choices=["patchcore", "efficientad", "both"],
                        help="Which model to train")
    parser.add_argument("--data_path", type=str,
                        default=r"C:\Users\alidar\Desktop\aitu\data\vial",
                        help="Path to vial dataset root")
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--epochs",    type=int, default=50,
                        help="Epochs for EfficientAD (PatchCore doesn't need epochs)")
    parser.add_argument("--coreset_ratio", type=float, default=0.1,
                        help="PatchCore coreset ratio (0.1 = 10% of patches)")
    parser.add_argument("--save_dir",  type=str, default="checkpoints")
    parser.add_argument("--device",    type=str, default="cuda")
    # num_workers=0 is safest on Windows; increase on Linux
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (use 0 if errors on Windows)")
    return parser.parse_args()


def print_gpu_info(device: torch.device):
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        total_mb = props.total_memory / 1024 ** 2
        print(f"  GPU       : {props.name}")
        print(f"  VRAM      : {total_mb:.0f} MB")
        print(f"  CUDA      : {torch.version.cuda}")
    else:
        print("  GPU       : NOT available — running on CPU")


def make_train_loader(args) -> DataLoader:
    train_ds = VialTrainDataset(args.data_path, img_size=args.img_size)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.num_workers > 0),   # pin_memory needs workers > 0
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return loader


def train_patchcore(args, train_loader): 
    print("  Training PatchCore")

    model = PatchCore(
        backbone="wide_resnet50",
        coreset_ratio=args.coreset_ratio,
        device=args.device,
        img_size=args.img_size,
    )
    model.fit(train_loader)

    save_path = os.path.join(args.save_dir, "patchcore.pkl")
    model.save(save_path)
    print(f"[PatchCore] Done! Saved to {save_path}")
    return model


def train_efficientad(args, train_loader):
    print("  Training EfficientAD")

    model = EfficientAD(
        device=args.device,
        img_size=args.img_size,
        use_autoencoder=True,
    )
    model.fit(train_loader, epochs=args.epochs)

    save_path = os.path.join(args.save_dir, "efficientad.pt")
    model.save(save_path)
    print(f"[EfficientAD] Done! Saved to {save_path}")
    return model


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)   # normalise so models receive "cuda" or "cpu"

    print(f"\n[Config]")
    print(f"  Data path : {args.data_path}")
    print(f"  Image size: {args.img_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device    : {args.device}")
    print(f"  Model(s)  : {args.model}")
    print_gpu_info(device)

    train_loader = make_train_loader(args)

    if args.model in ("patchcore", "both"):
        train_patchcore(args, train_loader)

    if args.model in ("efficientad", "both"):
        train_efficientad(args, train_loader)

    print("\nTraining complete! Run evaluate.py next.")


if __name__ == "__main__":
    main()
