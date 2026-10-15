"""Stub model with the SAME interface as FusionTriageModel but random output.

Used on Day 1 so the API + dashboard can be built before any real model exists, and as the
fallback in serving when no checkpoint is configured.

OWNER: Anjil.
"""
from __future__ import annotations

import torch

from ..constants import NUM_CLASSES


class StubTriageModel:
    def __init__(self, seed: int = 0):
        self.gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def predict(self, video: torch.Tensor, pose: torch.Tensor) -> dict:
        probs = torch.softmax(torch.randn(video.shape[0], NUM_CLASSES, generator=self.gen) * 2, dim=1)
        return {"probs": probs, "calibrated_probs": probs}

    def explain(self, video: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = video.shape
        return torch.rand(b, t, h, w, generator=self.gen)
