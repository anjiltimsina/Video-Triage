"""Fusion model = video backbone + pose MLP + classification head.   (Contract: CONTRACTS.md)

    model(video, pose)        -> dict(logits (B,3), pose_logits (B,3))
    model.predict(video, pose)-> dict(probs, calibrated_probs)        both (B,3), order RED/YELLOW/GREEN
    model.explain(video, pose)-> heatmaps (B, T, H, W) in [0, 1]      (stub until Aayush P's GradCAM lands)

OWNER: Anjil.
"""
from __future__ import annotations

import torch
from torch import nn

from ..constants import NUM_CLASSES, POSE_DIM
from .backbones import build_backbone


class PoseEncoder(nn.Module):
    """(B, T, 17, 3) -> per-frame MLP (51 -> 64) -> mean over time -> (B, 64)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(POSE_DIM, 128), nn.ReLU(inplace=True),
                                 nn.Linear(128, hidden), nn.ReLU(inplace=True))
        self.out_dim = hidden

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        b, t = pose.shape[:2]
        return self.mlp(pose.reshape(b, t, POSE_DIM)).mean(dim=1)


class FusionTriageModel(nn.Module):
    def __init__(self, backbone: str = "tiny", pretrained: str | None = None,
                 pose_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        self.pose_encoder = PoseEncoder(pose_hidden)
        self.pose_head = nn.Linear(pose_hidden, NUM_CLASSES)           # auxiliary pose-only head
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(self.backbone.out_dim + pose_hidden, NUM_CLASSES))
        self.register_buffer("temperature", torch.ones(1))             # set by calibration.fit_temperature

    def set_train_mode(self, mode: str) -> None:
        self.backbone.set_train_mode(mode)

    def forward(self, video: torch.Tensor, pose: torch.Tensor) -> dict:
        v = self.backbone(video)
        p = self.pose_encoder(pose)
        return {"logits": self.head(torch.cat([v, p], dim=1)), "pose_logits": self.pose_head(p)}

    @torch.no_grad()
    def predict(self, video: torch.Tensor, pose: torch.Tensor) -> dict:
        self.eval()
        logits = self.forward(video, pose)["logits"]
        return {
            "probs": torch.softmax(logits, dim=1),
            "calibrated_probs": torch.softmax(logits / self.temperature.clamp_min(1e-3), dim=1),
        }

    def explain(self, video: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """STUB - replaced by GradCAM (Aayush P, Phase 3). Returns uniform heatmaps."""
        b, t, _, h, w = video.shape
        return torch.full((b, t, h, w), 0.5)
