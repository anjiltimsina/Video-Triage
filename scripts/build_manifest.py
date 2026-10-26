"""Turn tracked clips + human labels into the training manifest.

  python scripts/build_manifest.py --tracks data/tracks/walk_clips --labels data/labels.csv --out data/hospital/manifest.csv

--tracks may be given several times (one folder per batch of videos).
labels.csv:  clip_id,label[,patient_id,age,sex,arrival_time,annotator]   (label = RED|YELLOW|GREEN)
Writes manifest.csv and rejected.csv (every clip left out + the reason) in the --out folder.
Then:  python scripts/train.py --set data.manifest=data/hospital/manifest.csv   (see README)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage.data.build_manifest import DEFAULT_DROP_FLAGS, build_manifest
from triage.data.manifest import validate_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, action="append", required=True, help="folder with tracks.csv (repeatable)")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="hospital", help="hospital | mock | ntu | ucf | synthetic")
    ap.add_argument("--drop-flags", default=",".join(DEFAULT_DROP_FLAGS), help="quality flags that exclude a clip ('' = keep all)")
    ap.add_argument("--allow-no-deid", action="store_true", help="accept clips made with --unsafe-no-deid (NEVER for hospital data)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rep = build_manifest(a.tracks, a.labels, a.out, a.source, tuple(f for f in a.drop_flags.split(",") if f),
                         a.allow_no_deid, seed=a.seed)
    print(f"manifest: {rep['manifest']}   clips={rep['n_clips']}   patients={rep['n_patients']}")
    print(rep["counts"].to_string())
    print("left out:", rep["rejected"] or "nothing", "(details in rejected.csv)")
    for w in rep["warnings"]:
        print("WARNING:", w)
    problems = validate_manifest(a.out)
    print("validation:", problems or "OK")


if __name__ == "__main__":
    main()