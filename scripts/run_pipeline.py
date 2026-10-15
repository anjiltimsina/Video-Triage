"""One command that exercises the whole pipeline:  data -> train -> calibrate -> evaluate.

    python scripts/run_pipeline.py --source synthetic
    python scripts/run_pipeline.py --source synthetic --set train.epochs=2     (quick check)

Friday integration run: if this fails, fixing it is Monday's first task.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage.config import load_config
from triage.data.manifest import summarize_manifest, validate_manifest
from triage.data.synthetic import generate_synthetic_dataset
from triage.training import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", help="synthetic | or any manifest name you set in the config")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    d = cfg["data"]

    if a.source == "synthetic" and not Path(d["manifest"]).exists():
        print("[pipeline] generating synthetic data ...")
        generate_synthetic_dataset(d["synthetic_root"], d["synthetic_n_clips"], d["num_frames"],
                                   d["image_size"], d["synthetic_class_probs"], cfg["seed"])

    problems = validate_manifest(d["manifest"])
    if problems:
        sys.exit(f"[pipeline] manifest problems: {problems}")
    print(f"[pipeline] manifest OK\n{summarize_manifest(d['manifest'])}")

    results = train(cfg)
    t = results.get("test")
    if t:
        print(f"\n[pipeline] TEST  macro_auc={t['macro_auc']:.3f}  RED_sens={t['red_sensitivity']:.3f}  "
              f"RED_spec={t['red_specificity']:.3f}  ECE={t['ece']:.3f}  (n={t['n']}, RED={t['n_red']})")
    print(f"[pipeline] outputs in {cfg['train']['out_dir']}/  (best.pt, metrics.json, history.csv)")


if __name__ == "__main__":
    main()
