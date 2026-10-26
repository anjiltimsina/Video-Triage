"""Per-track quality scoring and person-centred clip writing.

For every tracked person we:
  1. choose a window  = the first `clip_seconds` after they appear (the pre-triage window);
  2. score the track's quality and attach FLAGS (we flag, we never silently drop);
  3. cut a smoothed crop of that person's padded box onto a `size` x `size` canvas with black bars
     (so the video model sees ONE person, not the whole crowd), and grey-out other people's boxes;
  4. save the pose RE-NORMALISED to the crop: (T, 17, 3) with x, y in [0, 1] relative to the crop.

Quality flags (all thresholds in QualityConfig, tune on real footage):
  short      - person visible for less than min_seconds
  small      - person is a tiny part of the frame (far from camera)
  low_conf   - the pose model is unsure about the keypoints (blur, blood, darkness, odd posture)
  few_kp     - too few keypoints are visible (body cut off or hidden)
  occluded   - the person is often overlapped by someone else
  gappy      - tracker lost the person for many frames (ID may have switched)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..constants import NUM_KEYPOINTS
from .extract import fill_missing
from .tracking import Track, TrackingResult


@dataclass
class QualityConfig:
    min_seconds: float = 2.0
    min_box_height_frac: float = 0.15
    min_kp_conf: float = 0.40
    kp_visible_thresh: float = 0.30
    min_visible_kp_frac: float = 0.50
    occlusion_ioa: float = 0.30          # box overlap (over OWN area) that counts as occluded
    max_occluded_frac: float = 0.40
    min_continuity: float = 0.70


@dataclass
class ClipPlan:
    track_id: int
    start_frame: int
    n_frames: int
    crop: np.ndarray        # (n, 5): centre x, centre y, canvas side, visible width, visible height (source pixels)
    pose: np.ndarray        # (n, 17, 3) normalised to the crop, float32
    detected: np.ndarray    # (n,) bool - person really seen in that frame


def box_ioa(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area divided by the area of `a` (how much of A is covered by B)."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1])))


def select_window(track: Track, fps: float, clip_seconds: float) -> tuple[int, int]:
    """(start_frame, n_frames): first `clip_seconds` of the track's life."""
    start = track.first_frame
    end = min(track.last_frame + 1, start + int(round(clip_seconds * fps)))
    return start, end - start


def track_quality(track: Track, result: TrackingResult, start: int, n: int,
                  cfg: QualityConfig = QualityConfig()) -> dict:
    """Quality metrics + flags for the frames [start, start+n) of one track."""
    fps = result.fps
    W, H = result.frame_size
    idx = [i for i, f in enumerate(track.frames) if start <= f < start + n]
    if not idx:
        return {"n_seen": 0, "flags": ["short"]}

    kps = np.stack([track.keypoints[i] for i in idx])                  # (k, 17, 3)
    boxes = np.stack([track.boxes[i] for i in idx])                    # (k, 4)
    seen_frames = [track.frames[i] for i in idx]

    occluded = 0
    for f, box in zip(seen_frames, boxes):
        if any(box_ioa(box, other) > cfg.occlusion_ioa
               for tid, other in result.boxes_at(f).items() if tid != track.track_id):
            occluded += 1

    q = {
        "n_seen": len(idx),
        "duration_s": round(n / fps, 2),
        "continuity": round(len(idx) / max(1, n), 3),
        "mean_kp_conf": round(float(kps[..., 2].mean()), 3),
        "visible_kp_frac": round(float((kps[..., 2] >= cfg.kp_visible_thresh).mean()), 3),
        "box_height_frac": round(float(np.median(boxes[:, 3] - boxes[:, 1]) / H), 3),
        "occluded_frac": round(occluded / len(idx), 3),
        "first_center_x": round(float((boxes[0, 0] + boxes[0, 2]) / 2 / W), 3),
        "first_center_y": round(float((boxes[0, 1] + boxes[0, 3]) / 2 / H), 3),
    }
    flags = []
    if n / fps < cfg.min_seconds:
        flags.append("short")
    if q["box_height_frac"] < cfg.min_box_height_frac:
        flags.append("small")
    if q["mean_kp_conf"] < cfg.min_kp_conf:
        flags.append("low_conf")
    if q["visible_kp_frac"] < cfg.min_visible_kp_frac:
        flags.append("few_kp")
    if q["occluded_frac"] > cfg.max_occluded_frac:
        flags.append("occluded")
    if q["continuity"] < cfg.min_continuity:
        flags.append("gappy")
    q["flags"] = flags
    return q


def _smooth(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Zero-lag exponential smoothing (forward + backward average) so the crop does not jitter."""
    fwd = x.copy()
    for i in range(1, len(x)):
        fwd[i] = alpha * x[i] + (1 - alpha) * fwd[i - 1]
    bwd = x.copy()
    for i in range(len(x) - 2, -1, -1):
        bwd[i] = alpha * x[i] + (1 - alpha) * bwd[i + 1]
    return (fwd + bwd) / 2


def plan_clip(track: Track, start: int, n: int, pad: float = 0.3, min_side: float = 32.0) -> ClipPlan:
    """Crop box per frame (interpolated over gaps, then smoothed) + pose re-normalised to that crop."""
    seen = {f: i for i, f in enumerate(track.frames)}
    frames = np.arange(start, start + n)
    have = np.array([f in seen for f in frames])
    have_idx = np.flatnonzero(have)

    bx = [track.boxes[seen[f]] for f in frames[have]]
    cx = np.array([(b[0] + b[2]) / 2 for b in bx])
    cy = np.array([(b[1] + b[3]) / 2 for b in bx])
    wp = np.array([(b[2] - b[0]) * (1 + pad) for b in bx])        # padded box width
    hp = np.array([(b[3] - b[1]) * (1 + pad) for b in bx])        # padded box height
    raw = np.stack([np.interp(np.arange(n), have_idx, v) for v in (cx, cy, np.maximum(wp, hp), wp, hp)], axis=1)
    sm = np.stack([_smooth(raw[:, k]) for k in range(5)], axis=1)
    # crop columns: centre x, centre y, square canvas side (= max(w, h)), visible width, visible height
    crop = np.stack([sm[:, 0], sm[:, 1], np.maximum(sm[:, 2], min_side),
                     np.maximum(sm[:, 3], 8.0), np.maximum(sm[:, 4], 8.0)], axis=1)

    px_pose, detected = fill_missing([track.keypoints[seen[f]] if f in seen else None for f in frames])
    x0, y0, s = crop[:, 0] - crop[:, 2] / 2, crop[:, 1] - crop[:, 2] / 2, crop[:, 2]
    pose = px_pose.copy()
    pose[..., 0] = np.clip((px_pose[..., 0] - x0[:, None]) / s[:, None], 0.0, 1.0)
    pose[..., 1] = np.clip((px_pose[..., 1] - y0[:, None]) / s[:, None], 0.0, 1.0)
    return ClipPlan(track.track_id, start, n, crop.astype(np.float32), pose.astype(np.float32), detected)


def _isolate_target(frame: np.ndarray, target_box: np.ndarray | None, others: list[np.ndarray],
                    fill: int = 114) -> np.ndarray:
    """Grey-out every OTHER person's box, then put the target's own pixels back (so a neighbour
    who overlaps the target never hides part of the target)."""
    H, W = frame.shape[:2]
    out = frame.copy()
    for b in others:
        x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])), int(min(W, b[2])), int(min(H, b[3])))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = fill
    if target_box is not None:
        x1, y1, x2, y2 = (int(max(0, target_box[0])), int(max(0, target_box[1])),
                          int(min(W, target_box[2])), int(min(H, target_box[3])))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out


def render_clips(video_path: str | Path, plans: list[ClipPlan], out_paths: list[Path],
                 fps: float, size: int = 224, result: TrackingResult | None = None,
                 isolate: bool = True, head_boxes: dict | None = None) -> None:
    """Write all planned clips in ONE sequential pass over the source video.

    Each clip is the person's padded box, scaled to fit a size x size canvas, with BLACK BARS
    outside the box (so a tall standing person does not drag half the room along). With
    `isolate=True` (needs `result`) other people's boxes are greyed out inside the clip.
    `head_boxes` (from deid.build_head_timeline) are pixelated in the FULL frame BEFORE cropping,
    so every tracked person's head is blurred, not only the clip's own person.
    """
    from ..deid import deidentify_frame
    if not plans:
        return
    cap = cv2.VideoCapture(str(video_path))
    writers: dict[int, cv2.VideoWriter] = {}
    last = max(p.start_frame + p.n_frames for p in plans)
    idx = 0
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        if head_boxes is not None and any(p.start_frame <= idx < p.start_frame + p.n_frames for p in plans):
            frame = deidentify_frame(frame, head_boxes.get(idx, []))
        for k, p in enumerate(plans):
            if p.start_frame <= idx < p.start_frame + p.n_frames:
                cx, cy, side, vw, vh = p.crop[idx - p.start_frame]
                src = frame
                if isolate and result is not None:
                    at = result.boxes_at(idx)
                    src = _isolate_target(frame, at.get(p.track_id),
                                          [b for tid, b in at.items() if tid != p.track_id])
                sc = size / side
                M = np.array([[sc, 0, -(cx - side / 2) * sc], [0, sc, -(cy - side / 2) * sc]], dtype=np.float32)
                crop = cv2.warpAffine(src, M, (size, size), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                # black bars: keep only the padded person box (centred in the canvas)
                hw, hh = int(round(vw * sc / 2)), int(round(vh * sc / 2))
                x0, x1, y0, y1 = size // 2 - hw, size // 2 + hw, size // 2 - hh, size // 2 + hh
                crop[:, :max(0, x0)] = 0
                crop[:, min(size, x1):] = 0
                crop[:max(0, y0)] = 0
                crop[min(size, y1):] = 0
                if k not in writers:
                    writers[k] = cv2.VideoWriter(str(out_paths[k]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
                writers[k].write(crop)
        idx += 1
    cap.release()
    for w in writers.values():
        w.release()


def build_clips(result: TrackingResult, video_path: str | Path, out_dir: str | Path,
                clip_seconds: float = 5.0, size: int = 224, pad: float = 0.3,
                cfg: QualityConfig = QualityConfig(), isolate: bool = True,
                roles: dict | None = None, deid: bool = True, head_hold_frames: int = 15) -> list[dict]:
    """For every track: quality row; for tracks that last >= min_seconds also a clip + pose file.

    Flagged tracks are still written (with their flags in the row): filter later, never lose data.
    `roles` (from roles.assign_roles) adds role columns; excluded tracks are STILL written, the
    `score_this_track` column is what downstream scoring must respect.
    `deid=True` (the DEFAULT, keep it on for anything that comes from a hospital camera) pixelates
    every tracked person's head in the stored clips and adds deid_* columns to each row.
    """
    from ..deid import build_head_timeline, clip_deid_stats
    if roles:
        from .roles import roles_to_columns          # optional module: only needed when roles are passed
    else:
        roles_to_columns = lambda info: {}           # noqa: E731
    head_boxes, head_sources = build_head_timeline(result, head_hold_frames) if deid else (None, {})
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    rows, plans, vpaths = [], [], []

    for tid, tr in sorted(result.tracks.items()):
        start, n = select_window(tr, result.fps, clip_seconds)
        q = track_quality(tr, result, start, n, cfg)
        written = n / result.fps >= cfg.min_seconds and q["n_seen"] > 0
        clip_id = f"{stem}_t{tid:03d}"
        row = {"clip_id": clip_id, "source_video": video_path.name, "track_id": tid,
               "start_s": round(start / result.fps, 2), "end_s": round((start + n) / result.fps, 2),
               **{k: v for k, v in q.items() if k != "flags"},
               "flags": ";".join(q["flags"]), "written": written,
               "clip_file": f"{clip_id}.mp4" if written else "", "pose_file": f"{clip_id}_pose.npy" if written else "",
               **roles_to_columns((roles or {}).get(tid)),
               "deid_applied": bool(deid and written),
               **(clip_deid_stats(tid, start, n, head_sources) if deid and written else {})}
        rows.append(row)
        if written:
            plan = plan_clip(tr, start, n, pad)
            np.save(out_dir / f"{clip_id}_pose.npy", plan.pose)
            plans.append(plan)
            vpaths.append(out_dir / f"{clip_id}.mp4")

    render_clips(video_path, plans, vpaths, result.fps, size, result, isolate, head_boxes)
    return rows