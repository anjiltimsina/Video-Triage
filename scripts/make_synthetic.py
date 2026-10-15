"""Generate the synthetic dataset + manifest.   python scripts/make_synthetic.py [--set key=value ...]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage.config import load_config
from triage.data.manifest import summarize_manifest, validate_manifest
from triage.data.synthetic import generate_synthetic_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. data.synthetic_n_clips=500")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    d = cfg["data"]
    path = generate_synthetic_dataset(d["synthetic_root"], d["synthetic_n_clips"], d["num_frames"],
                                      d["image_size"], d["synthetic_class_probs"], cfg["seed"])
    problems = validate_manifest(path)
    print(f"manifest: {path}\nproblems: {problems or 'none'}\n{summarize_manifest(path)}")


if __name__ == "__main__":
    main()
