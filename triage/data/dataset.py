"""PyTorch Dataset driven entirely by the manifest.

Returns, per clip (see CONTRACTS.md):
    video : float32 (T, 3, H, W) in [0, 1]
    pose  : float32 (T, 17, 3)   (x, y, confidence)
    label : int64   0=RED 1=YELLOW 2=GREEN

OWNER: Aayush B.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import LABEL2ID
from .manifest import load_manifest


def _sample_indices(n_available: int, n_wanted: int) -> np.ndarray:
    """Uniformly sample `n_wanted` frame indices out of `n_available` (e.g. 8 out of ~150)."""
    if n_available <= 0:
        raise ValueError("clip has no frames")
    return np.linspace(0, n_available - 1, n_wanted).round().astype(int)


class TriageDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, num_frames: int = 8,
                 image_size: int | None = None, sources: list[str] | None = None):
        self.root = Path(manifest_path).parent
        df = load_manifest(manifest_path, sources)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].map(LABEL2ID).to_numpy()

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        video = np.load(self.root / row["video_path"])          # (T0, H, W, 3) uint8
        pose = np.load(self.root / row["pose_path"])            # (T0', 17, 3) float32

        video = video[_sample_indices(len(video), self.num_frames)]
        pose = pose[_sample_indices(len(pose), self.num_frames)]

        if self.image_size and video.shape[1] != self.image_size:
            video = np.stack([cv2.resize(f, (self.image_size, self.image_size)) for f in video])

        video_t = torch.from_numpy(video).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
        return {
            "video": video_t,
            "pose": torch.from_numpy(pose).float(),
            "label": torch.tensor(LABEL2ID[row["label"]], dtype=torch.long),
        }
