"""De-identification: blur every tracked person's head in every frame that gets STORED.

WHY KEYPOINTS AND NOT A FACE DETECTOR: in an ED, faces are covered in blood, masks, oxygen
tubes, bandages, turned away or lying down - exactly the cases where face detectors fail and
pose estimators (which see a whole body) still work. So the head location comes from the body
keypoints, and a face detector is used only as an independent QA check afterwards.

FAIL-CLOSED RULES (a missed blur is a privacy breach; an extra blur costs nothing):
  1. head box from nose/eyes/ears, sized by the LARGEST of several estimates, padded x1.6;
  2. UNIONED with a generous region at the top of the person's body box (or both ends if the
     person is lying down), so a wrong keypoint still lands inside a blurred area;
  3. if the head keypoints are not visible at all, only the generous region is used;
  4. frames where the tracker lost the person are blurred at BOTH neighbouring positions, and
     the blur is held for `hold_frames` before the first / after the last detection;
  5. pixelation to ~6x6 cells + smoothing (cannot be sharpened back), not a light Gaussian blur.

KNOWN LIMITS (write them in the report; the QA step measures them):
  * a person the detector NEVER found has no blur. `verify_no_faces` is the safety net: run it
    on every output clip and treat any hit as a failed clip.
  * tiny/far people are blurred less reliably; faces in photos, screens or mirrors are not touched.
  * the original raw video still contains faces: keep it on the secured hospital storage only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pose.tracking import TrackingResult

HEAD_KPS = (0, 1, 2, 3, 4)          # nose, left eye, right eye, left ear, right ear


def _top_region_boxes(box: np.ndarray) -> list[np.ndarray]:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    # Upright reading: a head is about 1/3 of the body-box WIDTH, but only ~1/7 of its HEIGHT when the
    # whole body is visible. Someone close to the camera is cut off at the frame edge (short, wide box),
    # so the height rule alone under-covers. Use whichever is larger, capped so we never blur >60%.
    hh = min(0.6 * h, max(0.30 * h, 0.8 * w))
    boxes = [np.array([x1 + 0.05 * w, y1, x2 - 0.05 * w, y1 + hh])]
    # A WIDE box is ambiguous: lying down (head at either END) or arms stretched out (head in the MIDDLE).
    # Fail closed: cover both readings.
    if w > 1.3 * h:
        boxes += [np.array([x1, y1, x1 + 0.3 * w, y2]), np.array([x2 - 0.3 * w, y1, x2, y2])]
    return boxes


def head_boxes_for_person(kp: np.ndarray, box: np.ndarray, conf_min: float = 0.3, pad: float = 1.6,
                          conservative: bool = True) -> tuple[list[np.ndarray], str]:
    """Boxes (x1, y1, x2, y2 in pixels) to blur for ONE person in ONE frame, and where they came from."""
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    big = max(w, h, 1.0)
    vis = [i for i in HEAD_KPS if kp[i, 2] >= conf_min]
    boxes: list[np.ndarray] = []
    if len(vis) >= 2:
        centre = kp[vis, :2].mean(axis=0)
        sizes = [0.13 * big]                                     # a head is ~1/7.5 of body length: never smaller
        if kp[3, 2] >= conf_min and kp[4, 2] >= conf_min:        # ears: ear-to-ear ~ head width
            sizes.append(1.5 * float(np.linalg.norm(kp[3, :2] - kp[4, :2])))
        if kp[1, 2] >= conf_min and kp[2, 2] >= conf_min:        # eyes: eye distance ~ 0.4 head width
            sizes.append(3.5 * float(np.linalg.norm(kp[1, :2] - kp[2, :2])))
        if kp[5, 2] >= conf_min and kp[6, 2] >= conf_min:        # shoulders: head ~ 0.55 shoulder width
            sizes.append(0.55 * float(np.linalg.norm(kp[5, :2] - kp[6, :2])))
        side = max(sizes) * pad
        cx, cy = float(centre[0]), float(centre[1]) - 0.1 * side  # hair/forehead sit above the face keypoints
        boxes.append(np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]))
        source = "keypoints"
    else:
        source = "fallback"
    if source == "fallback" or conservative:
        boxes += _top_region_boxes(box)
    return boxes, source


def build_head_timeline(result: TrackingResult, hold_frames: int = 15, conf_min: float = 0.3,
                        pad: float = 1.6, conservative: bool = True
                        ) -> tuple[dict[int, list[np.ndarray]], dict[tuple[int, int], str]]:
    """frame index -> boxes to blur (ALL tracked people), and (track_id, frame) -> where the box came from."""
    per_frame: dict[int, list[np.ndarray]] = {}
    sources: dict[tuple[int, int], str] = {}

    def add(f: int, boxes: list[np.ndarray]) -> None:
        per_frame.setdefault(f, []).extend(boxes)

    for tid, tr in result.tracks.items():
        det = {f: head_boxes_for_person(kp, b, conf_min, pad, conservative)
               for f, kp, b in zip(tr.frames, tr.keypoints, tr.boxes)}
        frames = sorted(det)
        for f in frames:
            add(f, det[f][0])
            sources[(tid, f)] = det[f][1]
        for a, b in zip(frames[:-1], frames[1:]):                 # tracker lost the person: blur where they were AND went
            for g in range(a + 1, b):
                add(g, det[a][0] + det[b][0])
                sources[(tid, g)] = "gap"
        for k in range(1, hold_frames + 1):                       # blur slightly before / after the track exists
            if frames[0] - k >= 0:
                add(frames[0] - k, det[frames[0]][0])
                sources[(tid, frames[0] - k)] = "hold"
            if frames[-1] + k < result.n_frames:
                add(frames[-1] + k, det[frames[-1]][0])
                sources[(tid, frames[-1] + k)] = "hold"
    return per_frame, sources


def pixelate_region(frame: np.ndarray, box, blocks: int = 6) -> None:
    """In place: replace the box by a `blocks` x `blocks` mosaic, then smooth the mosaic edges."""
    H, W = frame.shape[:2]
    x1, y1 = int(max(0, np.floor(box[0]))), int(max(0, np.floor(box[1])))
    x2, y2 = int(min(W, np.ceil(box[2]))), int(min(H, np.ceil(box[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (min(blocks, w), min(blocks, h)), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(3, (min(w, h) // 4) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(mosaic, (k, k), 0)


def deidentify_frame(frame: np.ndarray, boxes: list[np.ndarray], blocks: int = 6) -> np.ndarray:
    """Return a de-identified COPY of the frame (the original is untouched). No boxes -> same frame object."""
    if not boxes:
        return frame
    out = frame.copy()
    for b in boxes:
        pixelate_region(out, b, blocks)
    return out


def clip_deid_stats(track_id: int, start: int, n: int, sources: dict[tuple[int, int], str]) -> dict:
    """How the head boxes of one clip were obtained: share from keypoints vs. weaker fallbacks."""
    s = [sources.get((track_id, f), "none") for f in range(start, start + n)]
    n = max(1, len(s))
    kp = sum(x == "keypoints" for x in s) / n
    return {"deid_kp_frac": round(kp, 3), "deid_weak_frac": round(1.0 - kp, 3)}


# ----------------------------------------------------------------------------- independent QA

class MtcnnFaceChecker:
    """Independent face detector used ONLY to check the output (not to place the blur).

    Optional install (it pins an old torch, so skip its dependency check):
        pip install --no-deps facenet-pytorch
        pip install torchvision pillow
    The MTCNN weights ship inside that package (no download).
    """

    def __init__(self, min_prob: float = 0.9, device: str = "cpu"):
        from facenet_pytorch import MTCNN
        self.det = MTCNN(keep_all=True, device=device, post_process=False)
        self.min_prob = min_prob

    def count_faces(self, frame_bgr: np.ndarray) -> int:
        from PIL import Image
        boxes, probs = self.det.detect(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        if boxes is None:
            return 0
        return int(sum(float(p) >= self.min_prob for p in np.asarray(probs, dtype=float)))


def verify_no_faces(video_path: str | Path, checker, every: int = 5, max_frames: int = 60) -> dict:
    """Run a face checker on sampled frames of an OUTPUT clip. frames_with_face > 0 = FAILED de-identification."""
    cap = cv2.VideoCapture(str(video_path))
    checked = with_face = faces = idx = 0
    while checked < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            n = checker.count_faces(frame)
            checked += 1
            with_face += n > 0
            faces += n
        idx += 1
    cap.release()
    return {"frames_checked": checked, "frames_with_face": with_face, "faces_total": faces}


def qa_rows(rows: list[dict], out_dir: str | Path, checker, every: int = 5) -> list[dict]:
    """Add residual_face_frames / deid_ok to every written clip's row (deid_ok=False -> do not use the clip)."""
    for r in rows:
        if not r.get("written"):
            continue
        s = verify_no_faces(Path(out_dir) / r["clip_file"], checker, every)
        r["residual_face_frames"] = s["frames_with_face"]
        r["deid_ok"] = s["frames_with_face"] == 0
    return rows