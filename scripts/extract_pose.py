"""Extract pose from one video or a whole folder of clips.

  One video, with an overlay so you can SEE if the skeleton is right:
    python scripts/extract_pose.py --video my_walk.mp4 --out data/pose_test/my_walk_pose.npy --overlay data/pose_test/my_walk_overlay.mp4

  A folder of clips (writes <name>_pose.npy for each clip into --out):
    python scripts/extract_pose.py --video-dir data/mock/clips --out data/mock/poses

Output per clip: float32 (T, 17, 3) = (x, y, confidence), x/y in [0, 1]  (see CONTRACTS.md).
ALWAYS watch at least a few overlay videos before trusting the numbers.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from triage.pose.extract import YoloPoseDetector, process_video

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def run_one(video: Path, out_npy: Path, overlay: Path | None, detector, stride: int) -> None:
    res = process_video(video, detector, overlay, stride=stride)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, res.pose)
    flag = "" if res.coverage >= 0.8 else "   <-- LOW COVERAGE, check this clip"
    print(f"{video.name}: frames={len(res.pose)} person_found={res.coverage:.0%} "
          f"mean_kp_conf={res.mean_confidence:.2f} -> {out_npy}{flag}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=Path)
    g.add_argument("--video-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="file (single video) or folder (--video-dir)")
    ap.add_argument("--overlay", type=Path, default=None, help="overlay mp4 (single video) or folder (--video-dir)")
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0 (first GPU). Default: auto")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1, help="use every N-th frame")
    a = ap.parse_args()

    detector = YoloPoseDetector(a.weights, a.device, a.conf)
    if a.video:
        run_one(a.video, a.out, a.overlay, detector, a.stride)
    else:
        videos = sorted(p for p in a.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXT)
        if not videos:
            sys.exit(f"no videos found in {a.video_dir}")
        for v in videos:
            ov = (a.overlay / f"{v.stem}_overlay.mp4") if a.overlay else None
            run_one(v, a.out / f"{v.stem}_pose.npy", ov, detector, a.stride)


if __name__ == "__main__":
    main()