"""Synthetic data generator so the whole pipeline runs BEFORE any real data exists.

The clips are fake but class-dependent, so a model can actually learn something and we can
check that training / evaluation / calibration behave sensibly:

  * RED    -> slumped posture, big sway/jitter, slow movement
  * YELLOW -> in between
  * GREEN  -> upright posture, small sway, normal movement

Per-clip severity is noisy (random jitter) so the task is not trivially separable.

Output on disk (per clip):
    <root>/clips/<clip_id>_video.npy   uint8   (T, H, W, 3)
    <root>/clips/<clip_id>_pose.npy    float32 (T, 17, 3)   (x, y, confidence) in [0, 1]
and one manifest CSV at <root>/manifest.csv   (schema in CONTRACTS.md).

OWNER: Aayush B. When real data arrives, this file is NOT used - the real-data path writes
the same manifest format from CVAT annotations instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import CLASSES, MANIFEST_COLUMNS, NUM_KEYPOINTS

# Standing skeleton template in normalised image coordinates (COCO-17 order).
_TEMPLATE = np.array([
    [0.50, 0.15], [0.48, 0.13], [0.52, 0.13], [0.46, 0.14], [0.54, 0.14],   # head
    [0.42, 0.28], [0.58, 0.28], [0.38, 0.42], [0.62, 0.42], [0.36, 0.55], [0.64, 0.55],  # arms
    [0.45, 0.55], [0.55, 0.55], [0.45, 0.75], [0.55, 0.75], [0.45, 0.95], [0.55, 0.95],  # legs
], dtype=np.float32)
_UPPER_BODY = list(range(0, 11))
_SEVERITY = {"RED": 0.9, "YELLOW": 0.5, "GREEN": 0.1}


def make_pose(severity: float, T: int, rng: np.random.Generator) -> np.ndarray:
    pose = np.zeros((T, NUM_KEYPOINTS, 3), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    sway = (0.01 + 0.04 * severity) * np.sin(2 * np.pi * t / T * 1.5 + phase)
    for k in range(NUM_KEYPOINTS):
        xy = _TEMPLATE[k].copy()
        pose[:, k, 0] = xy[0] + sway
        pose[:, k, 1] = xy[1]
    # slump: upper body drops and leans as severity rises
    pose[:, _UPPER_BODY, 1] += 0.15 * severity
    pose[:, _UPPER_BODY, 0] += 0.08 * severity
    # jitter grows with severity (gait instability)
    pose[:, :, :2] += rng.normal(0, 0.004 + 0.02 * severity, size=(T, NUM_KEYPOINTS, 2))
    pose[:, :, 2] = np.clip(rng.normal(0.9 - 0.2 * severity, 0.08, size=(T, NUM_KEYPOINTS)), 0.05, 1.0)
    pose[:, :, :2] = np.clip(pose[:, :, :2], 0.0, 1.0)
    return pose


def make_video(severity: float, T: int, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    video = rng.integers(0, 40, size=(T, H, W, 3), dtype=np.uint8)
    size = max(4, H // 5)
    speed = (1.0 - 0.8 * severity) * (W - size) / max(T - 1, 1)   # severe -> slow
    y0 = int((0.2 + 0.4 * severity) * (H - size))                  # severe -> lower in frame
    x0 = int(rng.integers(0, max(1, (W - size) // 4)))
    for i in range(T):
        x = int(min(W - size, x0 + speed * i))
        video[i, y0:y0 + size, x:x + size, :] = 220
    return video


def generate_synthetic_dataset(
    root: str | Path,
    n_clips: int = 300,
    num_frames: int = 8,
    image_size: int = 64,
    class_probs: list[float] | None = None,
    seed: int = 42,
) -> Path:
    """Create clips + manifest.csv under `root`. Returns the manifest path."""
    root = Path(root)
    (root / "clips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    class_probs = class_probs or [0.15, 0.30, 0.55]

    labels = rng.choice(len(CLASSES), size=n_clips, p=class_probs)
    splits = rng.choice(["train", "val", "test"], size=n_clips, p=[0.70, 0.15, 0.15])

    rows = []
    for i, lab in enumerate(labels):
        name = CLASSES[lab]
        severity = float(np.clip(_SEVERITY[name] + rng.normal(0, 0.2), 0.0, 1.0))
        clip_id = f"syn_{i:05d}"
        np.save(root / "clips" / f"{clip_id}_video.npy",
                make_video(severity, num_frames, image_size, image_size, rng))
        np.save(root / "clips" / f"{clip_id}_pose.npy", make_pose(severity, num_frames, rng))
        rows.append({
            "clip_id": clip_id,
            "video_path": f"clips/{clip_id}_video.npy",
            "pose_path": f"clips/{clip_id}_pose.npy",
            "label": name,
            "split": splits[i],
            "age": int(np.clip(rng.normal(35 + 30 * severity, 15), 18, 95)),
            "sex": rng.choice(["M", "F"]),
            "arrival_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 90))),
            "source": "synthetic",
            "annotator": "synthetic",
        })

    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    return manifest_path
