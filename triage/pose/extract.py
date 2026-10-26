"""Pose extraction: video clip  ->  (T, 17, 3) array of (x, y, confidence), x/y normalised to [0, 1].

How it works (one pass over the video, in small batches so memory stays low):
  1. YOLOv8-pose finds every person in each frame and gives 17 COCO keypoints per person.
  2. We pick ONE person to follow: the biggest one in the first frame, then in each later frame
     the detection whose box centre is closest to where the person was last seen.
  3. Pixel coordinates are divided by the frame width/height, so the pose does not depend on resolution.
  4. Frames where nobody was detected are filled from the nearest detected frame, with confidence 0
     so the model (and you) can tell the difference between "seen" and "guessed".

Known limitation: in a crowded frame the tracker can jump to a neighbour. Step 2 of the project
(MOG2 clip segmentation) cuts one clip per entering patient, which keeps scenes simple.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..constants import NUM_KEYPOINTS

# Skeleton edges between COCO keypoint indices, used only for drawing the overlay.
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class Detection:
    keypoints: np.ndarray   # (17, 3) pixels: x, y, confidence
    box: np.ndarray         # (4,)  x1, y1, x2, y2 in pixels
    score: float            # person-detection confidence

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))


@dataclass
class PoseResult:
    pose: np.ndarray        # (T, 17, 3) float32, normalised
    detected: np.ndarray    # (T,) bool - True where a person was really found in that frame
    fps: float
    frame_size: tuple[int, int]   # (width, height) of the source video

    @property
    def coverage(self) -> float:
        return float(self.detected.mean()) if len(self.detected) else 0.0

    @property
    def mean_confidence(self) -> float:
        return float(self.pose[self.detected][..., 2].mean()) if self.detected.any() else 0.0


class YoloPoseDetector:
    """Thin wrapper around ultralytics YOLOv8-pose. Downloads weights on first use (about 6 MB)."""

    def __init__(self, weights: str = "yolov8n-pose.pt", device: str | None = None,
                 conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO          # lazy import: only needed when really detecting
        self.model = YOLO(weights)
        self.device, self.conf, self.imgsz = device, conf, imgsz

    def detect(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        results = self.model.predict(frames_bgr, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)
        out = []
        for r in results:
            dets: list[Detection] = []
            if r.keypoints is not None and r.boxes is not None and len(r.boxes):
                kps = r.keypoints.data.cpu().numpy()          # (n, 17, 3)
                boxes = r.boxes.xyxy.cpu().numpy()            # (n, 4)
                scores = r.boxes.conf.cpu().numpy()           # (n,)
                dets = [Detection(k, b, float(s)) for k, b, s in zip(kps, boxes, scores)]
            out.append(dets)
        return out


def select_person(dets: list[Detection], prev_center: np.ndarray | None) -> Detection | None:
    """Pick the person to follow. First frame: biggest (area x score). After that: nearest to last position."""
    if not dets:
        return None
    if prev_center is None:
        return max(dets, key=lambda d: d.area * d.score)
    return min(dets, key=lambda d: float(np.linalg.norm(d.center - prev_center)))


def normalise_keypoints(kp_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixels -> [0, 1] for x and y; confidence is left unchanged."""
    out = kp_px.astype(np.float32).copy()
    out[:, 0] = np.clip(out[:, 0] / width, 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1] / height, 0.0, 1.0)
    out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
    return out


def fill_missing(poses: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a list with None gaps into a dense (T, 17, 3) array + a 'really detected' mask.

    A gap takes the nearest earlier detected frame (or the nearest later one if the clip starts
    with a gap). Filled frames get confidence 0. If nobody was ever detected, everything is zeros.
    """
    T = len(poses)
    detected = np.array([p is not None for p in poses], dtype=bool)
    dense = np.zeros((T, NUM_KEYPOINTS, 3), dtype=np.float32)
    if not detected.any():
        return dense, detected
    valid = np.flatnonzero(detected)
    for t in range(T):
        if detected[t]:
            dense[t] = poses[t]
        else:
            before = valid[valid < t]
            src = before[-1] if len(before) else valid[0]
            dense[t] = poses[src]
            dense[t, :, 2] = 0.0
    return dense, detected


def draw_pose(frame_bgr: np.ndarray, pose_norm: np.ndarray, seen: bool = True,
              kp_conf_min: float = 0.3) -> np.ndarray:
    """Draw one skeleton on a copy of the frame. Green = detected, orange = filled-in guess."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    color = (0, 220, 0) if seen else (0, 140, 255)
    pts = [(int(x * w), int(y * h), c) for x, y, c in pose_norm]
    for a, b in COCO_EDGES:
        if pts[a][2] >= kp_conf_min and pts[b][2] >= kp_conf_min:
            cv2.line(img, pts[a][:2], pts[b][:2], color, 2, cv2.LINE_AA)
    for x, y, c in pts:
        if c >= kp_conf_min:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
    if not seen:
        cv2.putText(img, "no person detected", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    return img


def process_video(video_path: str | Path, detector, overlay_path: str | Path | None = None,
                  batch_size: int = 16, stride: int = 1) -> PoseResult:
    """Run pose extraction over a whole video file.

    `detector` is anything with .detect(list_of_bgr_frames) -> list[list[Detection]]
    (YoloPoseDetector in real use, a fake one in tests). `stride=2` keeps every 2nd frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if overlay_path:
        Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (width, height))

    poses: list[np.ndarray | None] = []
    prev_center: np.ndarray | None = None
    frame_idx, batch = 0, []

    def flush():
        nonlocal prev_center
        if not batch:
            return
        for frame, dets in zip(batch, detector.detect(batch)):
            chosen = select_person(dets, prev_center)
            if chosen is None:
                poses.append(None)
            else:
                prev_center = chosen.center
                poses.append(normalise_keypoints(chosen.keypoints, width, height))
            if writer is not None:
                shown = poses[-1] if poses[-1] is not None else np.zeros((NUM_KEYPOINTS, 3), np.float32)
                writer.write(draw_pose(frame, shown, seen=poses[-1] is not None))
        batch.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            batch.append(frame)
            if len(batch) == batch_size:
                flush()
        frame_idx += 1
    flush()
    cap.release()
    if writer is not None:
        writer.release()

    pose, detected = fill_missing(poses)
    return PoseResult(pose=pose, detected=detected, fps=fps / stride, frame_size=(width, height))