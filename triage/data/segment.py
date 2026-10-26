"""Clip segmentation: long CCTV video  ->  one short clip (default 5 s) per person who enters.

How it works:
  1. MOG2 background subtraction learns what the empty scene looks like. Anything that moves
     in front of it becomes "foreground".
  2. Per frame we measure ACTIVITY = the area of the biggest foreground blob, as a fraction of
     the frame. A person walking in makes this jump up; an empty scene keeps it near 0.
  3. Frames where activity stays above a threshold for a minimum time form an EVENT (= someone
     entered). Events separated by a short gap are merged into one.
  4. For each event we cut a clip starting a little BEFORE the person appears (pre-roll), so the
     first steps are included. Pre-triage window = the first seconds after entry.

Known limitations (write these in your report):
  * Two people entering together/closely become ONE event, so one clip.
  * A person who stands still for a long time is slowly absorbed into the background (MOG2 history).
  * Sudden lighting changes (door opens, lights flicker) can look like motion: tune --min-area.
  * Thresholds are camera-specific: tune them on real PAHS footage with the activity plot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Segment:
    start_frame: int
    end_frame: int            # exclusive
    peak_activity: float      # biggest blob fraction seen inside the event (how clear the detection was)

    def seconds(self, fps: float) -> tuple[float, float]:
        return self.start_frame / fps, self.end_frame / fps


def compute_activity(video_path: str | Path, proc_width: int = 320, history: int = 500,
                     var_threshold: float = 25.0, warmup_s: float = 2.0) -> tuple[np.ndarray, float, int]:
    """Return (activity per frame, fps, n_frames). activity[i] = largest foreground blob / frame area."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    activity: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (proc_width, max(1, int(h * proc_width / w))))
        mask = bg.apply(small)
        mask = np.where(mask == 255, 255, 0).astype(np.uint8)          # 127 = shadow -> ignore
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)           # remove speckle noise
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max((cv2.contourArea(c) for c in contours), default=0.0)
        activity.append(biggest / mask.size)
    cap.release()

    act = np.array(activity, dtype=np.float32)
    act[: int(warmup_s * fps)] = 0.0       # MOG2 is still learning the background: ignore
    return act, float(fps), len(act)


def find_events(activity: np.ndarray, fps: float, min_area: float = 0.01,
                min_duration_s: float = 0.5, merge_gap_s: float = 2.0) -> list[tuple[int, int, float]]:
    """Return [(start_frame, end_frame_exclusive, peak_activity)] for each detected entry."""
    active = activity >= min_area
    runs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(active)])

    merged: list[list[int]] = []
    gap = int(merge_gap_s * fps)
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    min_len = int(min_duration_s * fps)
    return [(s, e, float(activity[s:e].max())) for s, e in merged if e - s >= min_len]


def events_to_segments(events: list[tuple[int, int, float]], fps: float, n_frames: int,
                       clip_seconds: float = 5.0, pre_roll_s: float = 0.5,
                       min_clip_fraction: float = 0.6) -> list[Segment]:
    """One fixed-length clip per event. Clips that would be cut short by the end of the video
    (shorter than min_clip_fraction of the wanted length) are dropped."""
    clip_len = int(round(clip_seconds * fps))
    segments = []
    for start, _end, peak in events:
        s = max(0, start - int(round(pre_roll_s * fps)))
        e = min(n_frames, s + clip_len)
        if e - s >= min_clip_fraction * clip_len:
            segments.append(Segment(s, e, peak))
    return segments


def cut_clips(video_path: str | Path, segments: list[Segment], out_dir: str | Path,
              prefix: str) -> list[Path]:
    """Write each segment as <out_dir>/<prefix>_<k>.mp4 using one sequential pass over the video
    (more reliable than seeking). Overlapping segments are fine."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    paths = [out_dir / f"{prefix}_{k:03d}.mp4" for k in range(len(segments))]
    writers: dict[int, cv2.VideoWriter] = {}

    idx = 0
    last_end = max((s.end_frame for s in segments), default=0)
    while idx < last_end:
        ok, frame = cap.read()
        if not ok:
            break
        for k, seg in enumerate(segments):
            if seg.start_frame <= idx < seg.end_frame:
                if k not in writers:
                    writers[k] = cv2.VideoWriter(str(paths[k]), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                writers[k].write(frame)
        idx += 1
    cap.release()
    for w in writers.values():
        w.release()
    return [p for k, p in enumerate(paths) if k in writers]


def save_activity_plot(activity: np.ndarray, fps: float, events: list[tuple[int, int, float]],
                       min_area: float, out_path: str | Path, title: str = "") -> None:
    """Plot activity over time with the detected events shaded. Use it to tune --min-area."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(activity)) / fps
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, activity, lw=1)
    ax.axhline(min_area, color="red", ls="--", lw=1, label=f"min_area = {min_area}")
    for s, e, _ in events:
        ax.axvspan(s / fps, e / fps, color="green", alpha=0.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("biggest blob / frame")
    ax.set_title(title or f"{len(events)} event(s) detected (green)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def segment_video(video_path: str | Path, out_dir: str | Path, clip_seconds: float = 5.0,
                  pre_roll_s: float = 0.5, min_area: float = 0.01, min_duration_s: float = 0.5,
                  merge_gap_s: float = 2.0, warmup_s: float = 2.0, var_threshold: float = 25.0,
                  plot_path: str | Path | None = None) -> list[dict]:
    """Whole pipeline for one video. Returns one dict per clip (ready to write to segments.csv)."""
    video_path = Path(video_path)
    activity, fps, n_frames = compute_activity(video_path, var_threshold=var_threshold, warmup_s=warmup_s)
    events = find_events(activity, fps, min_area, min_duration_s, merge_gap_s)
    segments = events_to_segments(events, fps, n_frames, clip_seconds, pre_roll_s)
    if plot_path:
        save_activity_plot(activity, fps, events, min_area, plot_path, title=video_path.name)
    paths = cut_clips(video_path, segments, out_dir, video_path.stem)
    rows = []
    for seg, p in zip(segments, paths):
        s, e = seg.seconds(fps)
        rows.append({"clip_file": p.name, "source_video": video_path.name, "start_s": round(s, 2),
                     "end_s": round(e, 2), "start_frame": seg.start_frame, "end_frame": seg.end_frame,
                     "fps": round(fps, 2), "peak_activity": round(seg.peak_activity, 4)})
    return rows