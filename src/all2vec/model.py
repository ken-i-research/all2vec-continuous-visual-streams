"""
ALL2Vec: Core Architecture

This module implements the continuous predictive state space model for real-time visual stream processing.

Reference:
Ken I. (2025). ALL2Vec: Continuous Predictive Representations 
for Dynamic Visual Streams. Preprint.
https://doi.org/10.5281/zenodo.17513405
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


@dataclass
class ModelConfig:
    """Configuration container for ALL2Vec live demo."""

    DU: int = 256
    B: int = 49
    patch_size: int = 7
    lr: float = 1e-4
    eta_s: float = 0.08
    u_inj: float = 0.05
    noise_std: float = 0.0
    frame_skip: int = 2
    clamp_val: float = 5.0
    eps: float = 1e-6
    stability_lambda: float = 0.0
    smooth_lambda: float = 0.0
    recon_lambda: float = 1.0
    dyn_lambda: float = 1.0
    pca_ref: int = 2000
    pca_seed: int = 42
    save_interval: int = 100
    log_dir: str = "all2vec_logs_7x7"
    preferred_cameras: Tuple[int, ...] = (1, 0)
    fallback_cameras: Tuple[int, ...] = tuple(range(2, 6))
    projection_seed: int = 42


class FeatureExtractor:
    """Wraps MobileNetV3 and exposes B projected patches per frame."""

    def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype

        with torch.no_grad():
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            ).to(device).eval()

        self.preproc = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.selected_layer_idx: int | None = None
        self.n_patches: int | None = None
        self.channels: int | None = None
        self.height: int | None = None
        self.width: int | None = None
        self.proj_W: torch.Tensor | None = None

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.preproc(rgb).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        if self.selected_layer_idx is None:
            self._select_layer(x)

        fmap = self._forward_to_selected_layer(x)

        C = self.channels  # type: ignore[arg-type]
        H = self.height    # type: ignore[arg-type]
        W = self.width     # type: ignore[arg-type]
        n_patches = self.n_patches  # type: ignore[arg-type]

        patches = fmap.view(C, H * W).T  # [n_patches, C]

        if self.config.B <= n_patches:
            indices = torch.linspace(0, n_patches - 1, self.config.B, device=self.device).long()
            particle_feats = patches[indices]
        else:  # 再利用（通常は発生しないはず）
            indices = (
                torch.linspace(0, n_patches - 1, self.config.B, device=self.device)
                .long()
                % n_patches
            )
            particle_feats = patches[indices]

        if self.proj_W is None or self.proj_W.size(1) != C:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.config.projection_seed)
            self.proj_W = torch.randn(
                self.config.DU, C, generator=gen, device=self.device, dtype=self.dtype
            ) / np.sqrt(C)

        U = F.linear(particle_feats, self.proj_W)
        U = U / (U.norm(dim=-1, keepdim=True) + self.config.eps)

        return U

    def _select_layer(self, x: torch.Tensor) -> None:
        layer_candidates = []
        temp_x = x.clone()
        with torch.no_grad():
            for i, layer in enumerate(self.backbone.features):
                temp_x = layer(temp_x)
                _, C, H, W = temp_x.shape
                layer_candidates.append(
                    {
                        "idx": i,
                        "n_patches": H * W,
                        "channels": C,
                        "height": H,
                        "width": W,
                    }
                )

        valid_layers = [l for l in layer_candidates if l["n_patches"] >= self.config.B]
        if valid_layers:
            selected = valid_layers[-1]
        else:  # fall back to the deepest layer
            selected = layer_candidates[-1]

        self.selected_layer_idx = selected["idx"]
        self.n_patches = selected["n_patches"]
        self.channels = selected["channels"]
        self.height = selected["height"]
        self.width = selected["width"]

        print(
            f"[INFO] Selected layer {self.selected_layer_idx}: "
            f"{self.channels}×{self.height}×{self.width}"
            f" ({self.n_patches} patches, required {self.config.B})"
        )

    def _forward_to_selected_layer(self, x: torch.Tensor) -> torch.Tensor:
        assert self.selected_layer_idx is not None
        with torch.no_grad():
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                if i == self.selected_layer_idx:
                    break
        return x.squeeze(0)


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Predictor(nn.Module):
    def __init__(self, du: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(du * 2, du),
            nn.GELU(),
            nn.Linear(du, du),
        )

    def forward(self, S: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([S, U], dim=-1))


class Reconstructor(nn.Module):
    def __init__(self, du: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(du, du),
            nn.GELU(),
            nn.Linear(du, du),
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.net(S)


class ALL2VecModel(nn.Module):
    def __init__(self, du: int):
        super().__init__()
        self.projector = Projector()
        self.pred = Predictor(du)
        self.recon = Reconstructor(du)

    def forward_step(self, S: torch.Tensor, U_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U_proj = self.projector(U_batch)
        S_hat = self.pred(S, U_proj)
        recon = self.recon(S_hat)
        U_mean = U_proj.mean(dim=0, keepdim=True)
        return S_hat, recon, U_mean

    def predict(self, S: torch.Tensor, U_batch: torch.Tensor) -> torch.Tensor:
        return self.pred(S, self.projector(U_batch))


def init_state(config: ModelConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    S = torch.randn(config.B, config.DU, device=device, dtype=dtype)
    return S / (S.norm(dim=-1, keepdim=True) + config.eps)


__all__ = [
    "ModelConfig",
    "FeatureExtractor",
    "ALL2VecModel",
    "init_state",
]

