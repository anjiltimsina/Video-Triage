"""Multi-person clips: track EVERY person in a video and cut one person-centred clip per track.

  python scripts/track_clips.py --video data/raw/crowd.mp4 --out data/tracks/clips

For each person (track) it writes, into --out:
  <video>_t003.mp4        224x224 clip centred on that person (first 5 s after they appear)
  <video>_t003_pose.npy   (T, 17, 3)  x, y in [0,1] RELATIVE TO THE CROP, confidence
and one row per track in tracks.csv with quality numbers and FLAGS
(short, small, low_conf, few_kp, occluded, gappy). Flagged tracks are still written - decide
later (in the manifest builder) what to keep. Watch some clips before trusting any threshold.

With --roles configs/roles.yaml it also adds role columns (role_hint, staff_prob, score_this_track,
needs_review, group_id, reasons). Only people CONFIDENTLY identified as staff (2+ independent cues)
get score_this_track=False; everybody else is scored. Nurse corrections: --overrides overrides.csv
with columns clip_id,role (patient | staff | companion | visitor).

PRIVACY: heads of every tracked person are pixelated in the stored clips BY DEFAULT (triage/deid.py).
--unsafe-no-deid turns that off: only for debugging with footage of people who consented, never for
hospital video. --qa-faces runs an independent face detector over every output clip and marks
clips where a face is still found (deid_ok=False in tracks.csv; do NOT use those clips).
The raw input video still contains faces: keep it on secured hospital storage.

Needs:  pip install ultralytics lapx
Optional for --qa-faces:  pip install --no-deps facenet-pytorch   and   pip install torchvision pillow
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from triage.pose.track_clips import QualityConfig, build_clips
from triage.pose.tracking import YoloTracker, track_video

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=Path)
    g.add_argument("--video-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=224, help="clip side in pixels (224 for TimeSformer)")
    ap.add_argument("--pad", type=float, default=0.3, help="extra margin around the person box")
    ap.add_argument("--no-isolate", action="store_true", help="do NOT grey-out other people inside each clip")
    ap.add_argument("--unsafe-no-deid", action="store_true", help="DISABLE head blurring (never use on hospital video)")
    ap.add_argument("--qa-faces", action="store_true", help="check output clips with an independent face detector (needs facenet-pytorch)")
    ap.add_argument("--head-hold", type=int, default=15, help="frames a head stays blurred before/after the tracker sees the person")
    ap.add_argument("--roles", type=Path, default=None, help="role-filter config (configs/roles.yaml): adds role columns")
    ap.add_argument("--overrides", type=Path, default=None, help="CSV clip_id,role  (patient|staff|companion|visitor) from a nurse")
    ap.add_argument("--role-weights", type=Path, default=None, help="learned weights JSON from fit_staff_model (optional)")
    ap.add_argument("--min-seconds", type=float, default=2.0, help="tracks shorter than this are not written")
    ap.add_argument("--max-seconds", type=float, default=None, help="only process the first N seconds (quick tests)")
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    ap.add_argument("--device", default=None, help="cpu, or 0 for the first GPU")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    videos = [a.video] if a.video else sorted(p for p in a.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        sys.exit(f"no videos found in {a.video_dir}")

    checker = None
    if a.qa_faces:
        from triage.deid import MtcnnFaceChecker
        checker = MtcnnFaceChecker(device="cpu")
    if a.unsafe_no_deid:
        print("WARNING: head blurring is DISABLED. Do not store or share these clips if they show real patients.")

    tracker = YoloTracker(a.weights, a.device, a.conf)
    cfg = QualityConfig(min_seconds=a.min_seconds)
    all_rows = []
    for v in videos:
        print(f"{v.name}: tracking ...")
        result = track_video(v, tracker, a.max_seconds, progress_every=300)
        roles = None
        if a.roles:
            from triage.pose.roles import RoleConfig, assign_roles, load_overrides, load_weights   # optional module
            roles = assign_roles(result, v, RoleConfig.from_yaml(a.roles),
                                 load_weights(a.role_weights) if a.role_weights else None,
                                 load_overrides(a.overrides))
        rows = build_clips(result, v, a.out, a.clip_seconds, a.size, a.pad, cfg, isolate=not a.no_isolate, roles=roles,
                           deid=not a.unsafe_no_deid, head_hold_frames=a.head_hold)
        if checker is not None:
            from triage.deid import qa_rows
            qa_rows(rows, a.out, checker)
        n_w = sum(r["written"] for r in rows)
        print(f"{v.name}: {len(rows)} track(s), {n_w} clip(s) written")
        for r in rows:
            note = f"flags={r['flags']}" if r["flags"] else "ok"
            role = ""
            if roles:
                role = f"  role={r['role_hint']} score={'YES' if r['score_this_track'] else 'NO'}" + \
                       ("  NEEDS-REVIEW" if r["needs_review"] else "")
            deid = ""
            if r.get("deid_applied"):
                deid = f"  head-blur: {r['deid_kp_frac']:.0%} from keypoints"
            if "deid_ok" in r:
                deid += "  QA: faces-found=" + str(r["residual_face_frames"]) + ("  OK" if r["deid_ok"] else "  *** DE-ID FAILED: DO NOT USE THIS CLIP ***")
            print(f"   track {r['track_id']:>3}: {r['start_s']}-{r['end_s']}s  kp_conf={r.get('mean_kp_conf')}  "
                  f"occluded={r.get('occluded_frac')}  {note}{role}{deid}{'' if r['written'] else '  (not written: too short)'}")
        all_rows += rows

    if all_rows:
        csv = a.out / "tracks.csv"
        new = pd.DataFrame(all_rows)
        if csv.exists():
            old = pd.read_csv(csv)
            new = pd.concat([old[~old["source_video"].isin(new["source_video"])], new], ignore_index=True)
        new.to_csv(csv, index=False)
        print(f"index written to {csv}")


if __name__ == "__main__":
    main()