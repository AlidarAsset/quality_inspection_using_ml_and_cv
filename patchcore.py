import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from tqdm import tqdm
import pickle
from pathlib import Path


class PatchCore:
    """
    Simplified PatchCore using WideResNet50 features from layers layer2 & layer3.
    Uses greedy coreset subsampling to keep memory manageable.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        layers: list = ["layer2", "layer3"],
        patch_size: int = 3,          # neighbourhood aggregation kernel
        coreset_ratio: float = 0.1,   # keep 10% of patches in memory
        device: str = "cuda",
        img_size: int = 256,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.patch_size = patch_size
        self.coreset_ratio = coreset_ratio
        self.img_size = img_size

        self.memory_bank = None          # numpy array [N, C] — for save/load
        self.memory_bank_gpu = None      # GPU tensor [N, C] — for fast inference
        self.feature_map_size = None     # (H, W) of the feature map

        self._build_backbone(backbone)
        print(f"[PatchCore] Running on {self.device}")

    
    #  Backbone
    

    def _build_backbone(self, backbone: str):
        if backbone == "wide_resnet50":
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Register forward hooks on selected layers
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
        Returns: [B, C_agg, h, w]  — neighbourhood-aggregated patch features (on GPU)
        """
        with torch.no_grad():
            self.features = {}
            _ = self.model(x)

        # Grab feature maps from each layer, upsample to common size
        maps = []
        target_size = None

        for layer_name in self.layers:
            feat = self.features[layer_name]  # [B, C, h, w]
            if target_size is None:
                target_size = feat.shape[-2:]
            else:
                feat = F.interpolate(feat, size=target_size, mode="bilinear",
                                     align_corners=False)
            maps.append(feat)

        # Concatenate along channel dim → [B, C_total, h, w]
        combined = torch.cat(maps, dim=1)

        # Neighbourhood aggregation via avg pooling
        if self.patch_size > 1:
            combined = F.avg_pool2d(
                combined,
                kernel_size=self.patch_size,
                stride=1,
                padding=self.patch_size // 2,
            )

        # L2 normalise
        combined = F.normalize(combined, p=2, dim=1)

        self.feature_map_size = combined.shape[-2:]
        return combined   # [B, C, h, w]  — stays on GPU

    
    #  Coreset subsampling (greedy)
   

    @staticmethod
    def _greedy_coreset(features: np.ndarray, ratio: float) -> np.ndarray:
        """
        Greedy k-center coreset selection.
        features: [N, C]  (float32)
        Returns subset of shape [k, C]
        """
        n = len(features)
        k = max(1, int(n * ratio))

        # Start with a random point
        rng = np.random.default_rng(42)
        selected = [rng.integers(n)]
        min_dists = np.full(n, np.inf)

        for _ in tqdm(range(k - 1), desc="Coreset subsampling", leave=False):
            last = features[selected[-1]]
            dists = np.linalg.norm(features - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))

        return features[selected]

    
    #  Training (memory bank construction)
    

    def fit(self, train_loader: DataLoader):
        all_patches = []

        for batch in tqdm(train_loader, desc="Extracting train features"):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)
            feats = self._extract_features(x)          # [B, C, h, w] on GPU
            B, C, h, w = feats.shape
            # Reshape to [B*h*w, C] — move to CPU only for coreset (numpy)
            patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
            all_patches.append(patches.cpu().numpy())

        all_patches = np.concatenate(all_patches, axis=0).astype(np.float32)
        print(f"[PatchCore] Total patches before coreset: {len(all_patches):,}")

        # Coreset subsampling
        self.memory_bank = self._greedy_coreset(all_patches, self.coreset_ratio)
        print(f"[PatchCore] Memory bank size after coreset: {len(self.memory_bank):,}")

        # Cache memory bank on GPU for fast inference
        self._cache_memory_bank_on_gpu()

    def _cache_memory_bank_on_gpu(self):
        """Upload memory bank to GPU tensor for fast NN search."""
        self.memory_bank_gpu = torch.tensor(
            self.memory_bank, dtype=torch.float32
        ).to(self.device)
        print(f"[PatchCore] Memory bank cached on {self.device} "
              f"({self.memory_bank_gpu.shape[0]:,} × {self.memory_bank_gpu.shape[1]} floats, "
              f"{self.memory_bank_gpu.numel() * 4 / 1024**2:.1f} MB)")

    
    #  Inference
    

    def predict(self, x: torch.Tensor):
        """
        x: [B, 3, H, W]
        Returns:
            image_scores: np.ndarray [B]  — image-level anomaly score (max patch dist)
            anomaly_maps: np.ndarray [B, 1, H, W]  — pixel-level heatmap (upsampled)

        GPU optimization: patches and memory bank both on GPU — no CPU round-trip.
        """
        feats = self._extract_features(x)          # [B, C, h, w] on GPU
        B, C, h, w = feats.shape

        # Reshape to [B*h*w, C] — stays on GPU
        patches = feats.permute(0, 2, 3, 1).reshape(-1, C)  # [B*h*w, C]

        # Use cached GPU memory bank
        memory = self.memory_bank_gpu  # [M, C] on GPU

        # Batch NN search fully on GPU to avoid OOM
        chunk = 2048  # increased from 1024 since we're on GPU
        nn_dists = []
        for i in range(0, len(patches), chunk):
            q = patches[i:i + chunk]                           # [c, C]
            dists = torch.cdist(q.unsqueeze(0), memory.unsqueeze(0))[0]  # [c, M]
            nn_dists.append(dists.min(dim=1).values)

        nn_dists = torch.cat(nn_dists)                        # [B*h*w] on GPU
        score_map = nn_dists.reshape(B, 1, h, w)              # [B, 1, h, w]

        # Upsample to original image size (stays on GPU)
        anomaly_maps = F.interpolate(
            score_map, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        )                                                      # [B, 1, H, W]

        image_scores = anomaly_maps.reshape(B, -1).max(dim=1).values  # [B]

        # Move to CPU/numpy only at the very end
        return image_scores.cpu().numpy(), anomaly_maps.cpu().numpy()

    
    #  Save / Load
    

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "memory_bank":      self.memory_bank,        # numpy — compact for disk
            "feature_map_size": self.feature_map_size,
            "coreset_ratio":    self.coreset_ratio,
            "img_size":         self.img_size,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[PatchCore] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.memory_bank      = state["memory_bank"]
        self.feature_map_size = state["feature_map_size"]
        self.coreset_ratio    = state["coreset_ratio"]
        self.img_size         = state["img_size"]
        print(f"[PatchCore] Loaded from {path}  "
              f"(memory bank: {len(self.memory_bank):,} patches)")

        # Re-cache on GPU after loading
        self._cache_memory_bank_on_gpu()
