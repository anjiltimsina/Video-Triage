"""Multi-person tracking: video -> one Track per person, each with per-frame box + 17 keypoints.

Why this exists: real ED footage has many people in view at once (patient, family, nurses,
doctors, other patients). The single-person extractor in extract.py follows only one guess.
Here every person gets a persistent ID (ByteTrack via ultralytics), so later steps can
  * cut one person-centred clip PER TRACK,
  * score how trustworthy each track is (occlusion, small, low confidence),
  * decide who is the patient (role filter - next step) instead of guessing.

The tracker is a small interface (`reset()`, `track_frame(frame) -> list[TrackedDet]`), so
tests use a scripted fake and real runs use YoloTracker.

Known limitation: ByteTrack can swap IDs when people cross closely or are fully hidden for a
while; the quality score's `continuity` and `occluded_frac` are there to flag those tracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TrackedDet:
    track_id: int
    keypoints: np.ndarray   # (17, 3) pixels: x, y, confidence
    box: np.ndarray         # (4,) x1, y1, x2, y2 in pixels
    score: float


@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)            # global frame indices where seen
    keypoints: list[np.ndarray] = field(default_factory=list)  # pixel (17, 3) per seen frame
    boxes: list[np.ndarray] = field(default_factory=list)      # pixel (4,) per seen frame
    scores: list[float] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.frames[0]

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    def add(self, frame_idx: int, det: TrackedDet) -> None:
        self.frames.append(frame_idx)
        self.keypoints.append(det.keypoints)
        self.boxes.append(det.box)
        self.scores.append(det.score)


@dataclass
class TrackingResult:
    tracks: dict[int, Track]
    fps: float
    frame_size: tuple[int, int]     # (width, height)
    n_frames: int

    def __post_init__(self):
        # frame index -> {track_id: box}; used to find who overlaps whom
        self._index: dict[int, dict[int, np.ndarray]] = {}
        for tid, tr in self.tracks.items():
            for f, b in zip(tr.frames, tr.boxes):
                self._index.setdefault(f, {})[tid] = b

    def boxes_at(self, frame_idx: int) -> dict[int, np.ndarray]:
        return self._index.get(frame_idx, {})


class YoloTracker:
    """YOLOv8-pose + ByteTrack. Weights (about 6 MB) download on first use."""

    def __init__(self, weights: str = "yolov8n-pose.pt", device: str | None = None,
                 conf: float = 0.25, imgsz: int = 640, tracker: str = "bytetrack.yaml"):
        self.weights, self.device, self.conf, self.imgsz, self.tracker = weights, device, conf, imgsz, tracker
        self.reset()

    def reset(self) -> None:
        from ultralytics import YOLO          # lazy import
        self.model = YOLO(self.weights)       # fresh model = fresh tracker state

    def track_frame(self, frame_bgr: np.ndarray) -> list[TrackedDet]:
        r = self.model.track(frame_bgr, persist=True, tracker=self.tracker, conf=self.conf,
                             imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            return []
        ids = r.boxes.id.int().cpu().tolist()
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy()
        return [TrackedDet(i, k, b, float(s)) for i, k, b, s in zip(ids, kps, boxes, scores)]


def track_video(video_path: str | Path, tracker, max_seconds: float | None = None,
                progress_every: int = 0) -> TrackingResult:
    """Run the tracker over a whole video and collect one Track per person ID."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    limit = int(max_seconds * fps) if max_seconds else None

    tracker.reset()
    tracks: dict[int, Track] = {}
    idx = 0
    while limit is None or idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        for det in tracker.track_frame(frame):
            tracks.setdefault(det.track_id, Track(det.track_id)).add(idx, det)
        idx += 1
        if progress_every and idx % progress_every == 0:
            print(f"  ... {idx} frames, {len(tracks)} track(s) so far")
    cap.release()
    return TrackingResult(tracks, float(fps), size, idx)