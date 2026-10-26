"""Cut long videos into one short clip per person who enters (MOG2 background subtraction).

  One video, with an activity plot to help you tune the threshold:
    python scripts/segment_clips.py --video data/raw/cam1.mp4 --out data/mock/clips --plot data/mock/plots

  A whole folder of long videos:
    python scripts/segment_clips.py --video-dir data/raw --out data/mock/clips --plot data/mock/plots

Writes  <out>/<video>_000.mp4, _001.mp4 ...  and  <out>/segments.csv  (what was cut, from where).
Next step:  python scripts/extract_pose.py --video-dir <out> --out data/mock/poses

TUNING: open the plot. The red dashed line is --min-area. A person entering should push the blue
line clearly above it; an empty scene should stay clearly below. Adjust --min-area until the green
(detected) regions match the real entries. ALWAYS check a few clips by eye.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from triage.data.segment import segment_video

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=Path)
    g.add_argument("--video-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="folder for the clips")
    ap.add_argument("--plot", type=Path, default=None, help="folder for activity plots (recommended)")
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--pre-roll", type=float, default=0.5, help="seconds kept BEFORE the person appears")
    ap.add_argument("--min-area", type=float, default=0.01, help="blob size (fraction of frame) that counts as a person")
    ap.add_argument("--min-duration", type=float, default=0.5, help="seconds a blob must persist")
    ap.add_argument("--merge-gap", type=float, default=2.0, help="gaps shorter than this merge into one event")
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds at the start ignored while MOG2 learns the background")
    ap.add_argument("--var-threshold", type=float, default=25.0, help="MOG2 sensitivity (lower = more sensitive)")
    a = ap.parse_args()

    videos = [a.video] if a.video else sorted(p for p in a.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        sys.exit(f"no videos found in {a.video_dir}")

    all_rows = []
    for v in videos:
        plot = (a.plot / f"{v.stem}_activity.png") if a.plot else None
        rows = segment_video(v, a.out, a.clip_seconds, a.pre_roll, a.min_area, a.min_duration,
                             a.merge_gap, a.warmup, a.var_threshold, plot)
        print(f"{v.name}: {len(rows)} clip(s)" + ("" if rows else "   <-- nothing detected, lower --min-area or check the plot"))
        all_rows += rows

    if all_rows:
        a.out.mkdir(parents=True, exist_ok=True)
        csv = a.out / "segments.csv"
        new = pd.DataFrame(all_rows)
        if csv.exists():                       # keep earlier videos' rows, replace re-run ones
            old = pd.read_csv(csv)
            new = pd.concat([old[~old["source_video"].isin(new["source_video"])], new], ignore_index=True)
        new.to_csv(csv, index=False)
        print(f"wrote {len(all_rows)} clip(s) to {a.out}  (index: {csv})")


if __name__ == "__main__":
    main()