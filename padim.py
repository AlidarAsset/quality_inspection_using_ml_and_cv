"""
padim.py — PaDiM (Patch Distribution Modeling) Anomaly Detection
Paper: "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection
        and Localization" (Defard et al., 2021)

Key idea:
  1. Extract patch features from a pretrained CNN (ResNet18 / WideResNet50)
  2. For each spatial position (i, j), model the distribution of normal patch
     features as a multivariate Gaussian: N(mu_ij, Sigma_ij)
  3. At test time: anomaly score = Mahalanobis distance to that position's Gaussian
  4. Optionally: random feature dimension reduction for memory efficiency
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18, ResNet18_Weights,
    wide_resnet50_2, Wide_ResNet50_2_Weights,
)
from tqdm import tqdm
import pickle
from pathlib import Path


class PaDiM:
    """
    PaDiM with optional random feature dimension reduction.
    Backbone choice: 'resnet18' (faster, smaller) or 'wide_resnet50' (more accurate).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list = ["layer1", "layer2", "layer3"],
        d_reduced: int = 100,           # reduce feature dim with random selection
        device: str = "cuda",
        img_size: int = 256,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.d_reduced = d_reduced
        self.img_size = img_size
        self.backbone_name = backbone

        # Statistics
        self.mean = None         # [C_red, h*w] on GPU after fit
        self.cov_inv = None      # [h*w, C_red, C_red] on GPU after fit
        self.feature_size = None # (h, w)
        self.idx_select = None   # Random feature indices for dim reduction

        self._build_backbone(backbone)
        print(f"[PaDiM] Running on {self.device} | backbone: {backbone}")

    # ------------------------------------------------------------------ #
    #  Backbone
    # ------------------------------------------------------------------ #

    def _build_backbone(self, backbone: str):
        if backbone == "resnet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            channels_per_layer = {"layer1": 64,  "layer2": 128, "layer3": 256}
        elif backbone == "wide_resnet50":
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            channels_per_layer = {"layer1": 256, "layer2": 512, "layer3": 1024}
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Total channels = sum of selected layers' channels
        self.total_channels = sum(channels_per_layer[l] for l in self.layers)

        # Pick random feature indices for dimension reduction
        rng = np.random.default_rng(seed=42)
        self.idx_select = rng.choice(
            self.total_channels, self.d_reduced, replace=False
        )
        self.idx_select = torch.tensor(self.idx_select, dtype=torch.long).to(self.device)

        # Hooks
        self.features = {}
        self.hooks = []

        def make_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for name, module in model.named_modules():
            if name in self.layers:
                self.hooks.append(module.register_forward_hook(make_hook(name)))

        self.model = model.to(self.device)
        self.model.eval()

    
    #  Feature extraction
    

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        Returns: [B, d_reduced, h, w] — concatenated multi-layer features
        """
        with torch.no_grad():
            self.features = {}
            _ = self.model(x)

        # Collect feature maps, upsample all to layer1 resolution
        target_size = self.features[self.layers[0]].shape[-2:]

        maps = []
        for layer_name in self.layers:
            feat = self.features[layer_name]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode="bilinear", align_corners=False
                )
            maps.append(feat)

        # Concatenate channel-wise
        combined = torch.cat(maps, dim=1)  # [B, total_channels, h, w]

        # Random feature dimension reduction
        combined = combined[:, self.idx_select, :, :]  # [B, d_reduced, h, w]

        self.feature_size = combined.shape[-2:]
        return combined

    
    #  Training: estimate Gaussian parameters per spatial position
    

    def fit(self, train_loader: DataLoader):
        """Estimate mean and inverse covariance for each spatial position."""
        all_feats = []

        for batch in tqdm(train_loader, desc="[PaDiM] Extracting features"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            feats = self._extract_features(x)  # [B, C_red, h, w]
            all_feats.append(feats.cpu())

        # Stack: [N, C_red, h, w]
        all_feats = torch.cat(all_feats, dim=0)
        N, C, h, w = all_feats.shape
        print(f"[PaDiM] Total samples: {N}, feature dim: {C}, spatial: {h}x{w}")

        # Reshape to [C, N, h*w] for per-position statistics
        feats_flat = all_feats.permute(1, 0, 2, 3).reshape(C, N, h * w)

        # Compute mean per position: [C, h*w]
        mean = feats_flat.mean(dim=1)  # [C, h*w]

        # Compute covariance per position: [h*w, C, C]
        # cov_ij = (1/(N-1)) * sum_n (x_n - mu)(x_n - mu)^T
        feats_centered = feats_flat - mean.unsqueeze(1)  # [C, N, h*w]

        # Use einsum for batched covariance
        # For each position p: cov[p] = (1/(N-1)) * X[:, :, p] @ X[:, :, p].T
        cov = torch.zeros(h * w, C, C)
        I = torch.eye(C) * 0.01  # regularisation for stability

        print("[PaDiM] Computing covariance matrices...")
        for p in tqdm(range(h * w), desc="  Per-position covariance"):
            x_p = feats_flat[:, :, p]  # [C, N]
            x_p_centered = x_p - x_p.mean(dim=1, keepdim=True)
            cov[p] = (x_p_centered @ x_p_centered.T) / (N - 1) + I

        # Invert covariances (needed for Mahalanobis at test time)
        print("[PaDiM] Inverting covariance matrices...")
        cov_inv = torch.linalg.inv(cov)  # [h*w, C, C]

        # Move to GPU for fast inference
        self.mean = mean.to(self.device)            # [C, h*w]
        self.cov_inv = cov_inv.to(self.device)      # [h*w, C, C]

        print(f"[PaDiM] Fit complete. mean: {tuple(self.mean.shape)}, "
              f"cov_inv: {tuple(self.cov_inv.shape)}")

    
    #  Inference: Mahalanobis distance per position
    

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Returns:
            image_scores: np.ndarray [B]
            anomaly_maps: np.ndarray [B, 1, H, W]

        Mahalanobis distance: d(x, p) = sqrt((x-mu_p)^T * Sigma_p^-1 * (x-mu_p))
        """
        feats = self._extract_features(x)  # [B, C, h, w]
        B, C, h, w = feats.shape
        feats_flat = feats.reshape(B, C, h * w)  # [B, C, h*w]

        # Center: [B, C, h*w]
        delta = feats_flat - self.mean.unsqueeze(0)

        # Reshape for batched matmul: [h*w, B, C]
        delta_perm = delta.permute(2, 0, 1)  # [h*w, B, C]

        # cov_inv: [h*w, C, C]
        # We want, for each position p, batch b:
        #   d^2 = delta[b] @ cov_inv[p] @ delta[b]
        # Compute: [h*w, B, C] @ [h*w, C, C] = [h*w, B, C]
        intermediate = torch.bmm(delta_perm, self.cov_inv)
        # then sum over channels with delta itself
        sq_dist = (intermediate * delta_perm).sum(dim=2)  # [h*w, B]
        sq_dist = sq_dist.T  # [B, h*w]

        # Take sqrt for actual Mahalanobis distance
        dist = torch.sqrt(torch.clamp(sq_dist, min=0))  # [B, h*w]

        # Reshape to spatial grid
        score_map = dist.reshape(B, 1, h, w)

        # Upsample to original image size
        anomaly_maps = F.interpolate(
            score_map, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )

        # Smooth with Gaussian-like averaging (commonly done in PaDiM)
        # Approximated by another bilinear upsample-down cycle
        kernel_size = 5
        padding = kernel_size // 2
        anomaly_maps = F.avg_pool2d(
            anomaly_maps, kernel_size=kernel_size, stride=1, padding=padding
        )

        image_scores = anomaly_maps.reshape(B, -1).max(dim=1).values

        return image_scores.cpu().numpy(), anomaly_maps.cpu().numpy()

    
    #  Save / Load
   

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "backbone":     self.backbone_name,
            "layers":       self.layers,
            "d_reduced":    self.d_reduced,
            "img_size":     self.img_size,
            "mean":         self.mean.cpu(),
            "cov_inv":      self.cov_inv.cpu(),
            "idx_select":   self.idx_select.cpu(),
            "feature_size": self.feature_size,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[PaDiM] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.backbone_name = state["backbone"]
        self.layers        = state["layers"]
        self.d_reduced     = state["d_reduced"]
        self.img_size      = state["img_size"]
        self.feature_size  = state["feature_size"]
        self.mean          = state["mean"].to(self.device)
        self.cov_inv       = state["cov_inv"].to(self.device)
        self.idx_select    = state["idx_select"].to(self.device)
        print(f"[PaDiM] Loaded from {path} "
              f"(backbone: {self.backbone_name}, d_reduced: {self.d_reduced})")
