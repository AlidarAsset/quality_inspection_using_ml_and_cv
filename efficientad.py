import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path



#  Patch Description Network (PDN) — lightweight teacher/student


class PatchDescriptionNetwork(nn.Module):
    """
    Small CNN used as both teacher (pretrained) and student (trained from scratch).
    Produces dense feature maps for anomaly scoring.
    """

    def __init__(self, out_channels: int = 384):
        super().__init__()
        self.conv1  = nn.Conv2d(3,   128, kernel_size=4, stride=1, padding=3)
        self.conv2  = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3  = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4  = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.bn     = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn(x)
        return x



#  Autoencoder (global anomaly branch)


class AnomalyAutoencoder(nn.Module):

    def __init__(self, in_channels: int = 384):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64,  3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64,  32,  3, padding=1), nn.ReLU(),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 64,  kernel_size=2, stride=2), nn.ReLU(),
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, in_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))



#  EfficientAD


class EfficientAD:

    def __init__(
        self,
        out_channels: int = 384,
        device: str = "cuda",
        img_size: int = 256,
        lr: float = 1e-4,
        use_autoencoder: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.use_autoencoder = use_autoencoder

        # Teacher (frozen after pretrain-style init)
        self.teacher = PatchDescriptionNetwork(out_channels).to(self.device)
        self._init_teacher_weights()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Student
        self.student = PatchDescriptionNetwork(out_channels).to(self.device)

        # Autoencoder
        if use_autoencoder:
            self.autoencoder = AnomalyAutoencoder(out_channels).to(self.device)

        # Teacher channel statistics — kept on GPU for zero-copy inference
        self.teacher_mean: torch.Tensor | None = None
        self.teacher_std:  torch.Tensor | None = None

        print(f"[EfficientAD] Running on {self.device}")

    

    def _init_teacher_weights(self):
        """Initialise teacher with small random weights (no ImageNet here)."""
        for m in self.teacher.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    #  Training
    

    def fit(self, train_loader: DataLoader, epochs: int = 50):
        params = list(self.student.parameters())
        if self.use_autoencoder:
            params += list(self.autoencoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)

        # Step 1: Compute teacher channel statistics on training set
        print("[EfficientAD] Computing teacher statistics...")
        self._compute_teacher_stats(train_loader)

        # Step 2: Train student (+ autoencoder)
        print("[EfficientAD] Training student network...")
        self.student.train()
        if self.use_autoencoder:
            self.autoencoder.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                with torch.no_grad():
                    t_out = self.teacher(x)
                    # Normalise teacher output — teacher_mean/std already on GPU
                    t_out = (t_out - self.teacher_mean) / (self.teacher_std + 1e-8)

                s_out = self.student(x)

                # Student loss: match teacher on normal images
                loss = F.mse_loss(s_out, t_out)

                if self.use_autoencoder:
                    ae_out = self.autoencoder(t_out.detach())
                    loss  += F.mse_loss(ae_out, t_out.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(train_loader)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1:3d}/{epochs}]  Loss: {avg:.6f}")

        self.student.eval()
        if self.use_autoencoder:
            self.autoencoder.eval()
        print("[EfficientAD] Training complete.")

    def _compute_teacher_stats(self, loader: DataLoader):
        """
        Compute per-channel mean & std of teacher features over the training set.
        Results are kept on GPU as tensors for zero-copy use in predict().
        """
        all_feats = []
        self.teacher.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                feats = self.teacher(x)       # [B, C, h, w] on GPU
                all_feats.append(feats)

        all_feats = torch.cat(all_feats, dim=0)   # [N, C, h, w] — stays on GPU
        # keepdim so broadcasting works in predict()
        self.teacher_mean = all_feats.mean(dim=(0, 2, 3), keepdim=True)  # on GPU
        self.teacher_std  = all_feats.std( dim=(0, 2, 3), keepdim=True)  # on GPU
        print(f"[EfficientAD] Teacher stats computed on {self.device}  "
              f"(mean shape: {list(self.teacher_mean.shape)})")

   
    #  Inference
    

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Returns:
            image_scores: np.ndarray [B]
            anomaly_maps: np.ndarray [B, 1, H, W]

        GPU optimization: entire pipeline runs on GPU — CPU only for final output.
        """
        x = x.to(self.device)

        t_out  = self.teacher(x)
        # teacher_mean / teacher_std are GPU tensors — no transfer needed
        t_norm = (t_out - self.teacher_mean) / (self.teacher_std + 1e-8)
        s_out  = self.student(x)

        # Student-teacher anomaly map
        st_map = (s_out - t_norm).pow(2).mean(dim=1, keepdim=True)  # [B,1,h,w]

        if self.use_autoencoder:
            ae_out    = self.autoencoder(t_norm)
            ae_map    = (ae_out - t_norm).pow(2).mean(dim=1, keepdim=True)
            score_map = (st_map + ae_map) / 2.0
        else:
            score_map = st_map

        # Upsample to original resolution (still on GPU)
        anomaly_maps = F.interpolate(
            score_map, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        )  # [B, 1, H, W]

        image_scores = anomaly_maps.reshape(len(x), -1).max(dim=1).values  # [B]

        # Move to CPU/numpy only at the very end
        return image_scores.cpu().numpy(), anomaly_maps.cpu().numpy()

    # 
    #  Save / Load
    

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "student":      self.student.state_dict(),
            # Move stats to CPU before serialising — avoids CUDA device mismatch on load
            "teacher_mean": self.teacher_mean.cpu() if self.teacher_mean is not None else None,
            "teacher_std":  self.teacher_std.cpu()  if self.teacher_std  is not None else None,
            "img_size":     self.img_size,
        }
        if self.use_autoencoder:
            state["autoencoder"] = self.autoencoder.state_dict()
        torch.save(state, path)
        print(f"[EfficientAD] Saved to {path}")

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")   # load to CPU first
        self.student.load_state_dict(state["student"])
        self.img_size = state["img_size"]

        # Restore stats and move to GPU immediately
        self.teacher_mean = state["teacher_mean"].to(self.device) \
            if state["teacher_mean"] is not None else None
        self.teacher_std  = state["teacher_std"].to(self.device) \
            if state["teacher_std"]  is not None else None

        if self.use_autoencoder and "autoencoder" in state:
            self.autoencoder.load_state_dict(state["autoencoder"])

        self.student.eval()
        if self.use_autoencoder:
            self.autoencoder.eval()
        print(f"[EfficientAD] Loaded from {path}  "
              f"(stats on {self.device})")
