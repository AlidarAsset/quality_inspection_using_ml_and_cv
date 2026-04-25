import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms



#  Default transforms


def get_transforms(img_size: int = 256, split: str = "train"):
    """Return transforms for train (augmented) or test (clean)."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  # test / val
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_mask_transform(img_size: int = 256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])



#  Dataset classes


class VialTrainDataset(Dataset):
    """Only good/normal images for unsupervised training."""

    def __init__(self, data_root: str, img_size: int = 256):
        self.root = Path(data_root)
        self.img_size = img_size
        self.transform = get_transforms(img_size, split="train")

        good_dir = self.root / "train" / "good"
        assert good_dir.exists(), f"Directory not found: {good_dir}"

        self.image_paths = sorted(list(good_dir.glob("*.png")) +
                                  list(good_dir.glob("*.jpg")) +
                                  list(good_dir.glob("*.bmp")))
        print(f"[Train] Found {len(self.image_paths)} normal images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class VialTestDataset(Dataset):
    """
    Test set for MVTec AD2 vial structure:
      test_public/good/        → normal images
      test_public/bad/         → anomaly images
      test_public/ground_truth/bad/ → masks

    Returns (image_tensor, label, mask_tensor, defect_type, image_path)
    label: 0 = normal, 1 = anomaly
    """

    def __init__(self, data_root: str, img_size: int = 256,
                 split: str = "test_public"):
        self.root = Path(data_root)
        self.img_size = img_size
        self.split = split
        self.transform = get_transforms(img_size, split="test")
        self.mask_transform = get_mask_transform(img_size)

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        test_dir = self.root / self.split
        gt_dir   = test_dir / "ground_truth" / "bad"

        # Normal images
        good_dir = test_dir / "good"
        if good_dir.exists():
            for ext in ("*.png", "*.jpg", "*.bmp"):
                for p in sorted(good_dir.glob(ext)):
                    self.samples.append((p, 0, None, "good"))

        # Anomaly images
        bad_dir = test_dir / "bad"
        if bad_dir.exists():
            for ext in ("*.png", "*.jpg", "*.bmp"):
                for p in sorted(bad_dir.glob(ext)):
                    # Try to find matching mask (same stem or stem + _mask)
                    mask_path = gt_dir / (p.stem + "_mask.png")
                    if not mask_path.exists():
                        mask_path = gt_dir / p.name
                    if not mask_path.exists():
                        mask_path = None
                    self.samples.append((p, 1, mask_path, "bad"))

        print(f"[Test]  Found {len(self.samples)} images  "
              f"({sum(1 for s in self.samples if s[1]==0)} normal, "
              f"{sum(1 for s in self.samples if s[1]==1)} anomaly)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path, defect_type = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros(1, self.img_size, self.img_size)

        return img_tensor, label, mask_tensor, defect_type, str(img_path)



#  Quick test

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\alidar\Desktop\aitu\data\vial"

    train_ds = VialTrainDataset(DATA_PATH)
    test_ds  = VialTestDataset(DATA_PATH)

    img = train_ds[0]
    print(f"Train sample shape: {img.shape}")

    if len(test_ds) > 0:
        img, label, mask, dtype, path = test_ds[0]
        print(f"Test  sample shape: {img.shape}, label={label}, defect={dtype}")
    else:
        print("[!] Test dataset is empty — check folder structure")
