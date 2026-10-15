"""Video backbones. Every backbone maps  video (B, T, 3, H, W) in [0, 1]  ->  features (B, D)
and exposes `.out_dim`, so the rest of the model does not care which one is plugged in.

  * TinyVideoBackbone  - small 3D CNN, trains on CPU in seconds. For tests / pipeline checks.
  * TimeSformerBackbone - the real model from the proposal (HuggingFace). Needs a GPU for training.

OWNER: Anjil.
"""
from __future__ import annotations

import torch
from torch import nn


class TinyVideoBackbone(nn.Module):
    out_dim = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.net(video.permute(0, 2, 1, 3, 4))      # Conv3d wants (B, C, T, H, W)

    def set_train_mode(self, mode: str) -> None:
        trainable = mode != "frozen"
        for p in self.parameters():
            p.requires_grad = trainable


class TimeSformerBackbone(nn.Module):
    """HuggingFace TimeSformer (divided space-time attention). Output = CLS token (768-dim).

    Needs `pip install transformers`, 224x224 frames and 8 frames (for the k400 checkpoint).
    """
    out_dim = 768
    _MEAN, _STD = 0.45, 0.225      # normalisation used by facebook/timesformer-base-finetuned-k400

    def __init__(self, pretrained: str = "facebook/timesformer-base-finetuned-k400"):
        super().__init__()
        from transformers import TimesformerModel   # lazy import: not needed for the tiny backbone
        self.model = TimesformerModel.from_pretrained(pretrained)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pixel_values = (video - self._MEAN) / self._STD     # (B, T, 3, H, W)
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0]

    def set_train_mode(self, mode: str) -> None:
        """frozen -> nothing trains | last4 -> last 4 transformer blocks | full -> everything."""
        for p in self.model.parameters():
            p.requires_grad = mode == "full"
        if mode == "last4":
            for block in self.model.encoder.layer[-4:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.model.layernorm.parameters():
                p.requires_grad = True


def build_backbone(name: str, pretrained: str | None = None) -> nn.Module:
    if name == "tiny":
        return TinyVideoBackbone()
    if name == "timesformer":
        return TimeSformerBackbone(pretrained or "facebook/timesformer-base-finetuned-k400")
    raise ValueError(f"unknown backbone {name!r} (use 'tiny' or 'timesformer')")
