"""Loss functions.   L_total = alpha*L_focal + beta*L_pose + gamma*L_consistency   (proposal eq. 3)

Notes / decisions the proposal left open (write them in your report):
  * The proposal says "BCE" for the pose stream; with 3 mutually exclusive classes the correct
    multi-class form is cross-entropy, so L_pose = CE(pose_logits, y).
  * The proposal does not define L_consistency. We define it as the mean squared difference
    between the fused-head and pose-head probability vectors (encourages the two streams to agree).

OWNER: Anjil.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weights (handles RED being ~5% of patients)."""

    def __init__(self, gamma: float = 2.0, class_weights: list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", w)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, target[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.class_weights is not None:
            loss = loss * self.class_weights[target]
        return loss.mean()


class TriageLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.2, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.focal = FocalLoss(focal_gamma, class_weights)

    def forward(self, out: dict, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l_focal = self.focal(out["logits"], target)
        l_pose = F.cross_entropy(out["pose_logits"], target)
        l_cons = F.mse_loss(torch.softmax(out["logits"], 1), torch.softmax(out["pose_logits"], 1))
        total = self.alpha * l_focal + self.beta * l_pose + self.gamma * l_cons
        return total, {"focal": l_focal.item(), "pose": l_pose.item(), "consistency": l_cons.item()}
