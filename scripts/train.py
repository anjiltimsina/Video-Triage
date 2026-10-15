"""Train.   python scripts/train.py --config configs/default.yaml --set train.epochs=10"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage.config import load_config
from triage.training import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=2")
    a = ap.parse_args()
    results = train(load_config(a.config, a.set))
    t = results.get("test")
    if t:
        print(f"TEST  macro_auc={t['macro_auc']:.3f}  RED_sens={t['red_sensitivity']:.3f}  "
              f"ECE={t['ece']:.3f}  (n={t['n']}, RED={t['n_red']})")


if __name__ == "__main__":
    main()
