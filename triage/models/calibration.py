"""Temperature scaling:  p_cal = softmax(z / T), with one scalar T fitted on the VALIDATION set.
Target from the proposal: Expected Calibration Error (ECE) < 0.05  (ECE itself lives in eval/metrics.py).

OWNER: Anjil.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Return the T > 0 that minimises validation NLL. logits (N,3), labels (N,)."""
    log_t = torch.zeros(1, requires_grad=True)             # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())
