"""Evaluate a checkpoint on a split and write a report (metrics, reliability table, subgroups).

    python scripts/evaluate.py --checkpoint runs/synthetic/best.pt --split test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from triage.eval.metrics import evaluate_all, reliability_table, subgroup_report
from triage.training import collect_logits, get_device, load_checkpoint, make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--manifest", default=None, help="override the manifest (use this for the data-swap dry run)")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = get_device("auto")
    model, ckpt = load_checkpoint(a.checkpoint, device)
    cfg = ckpt["config"]
    if a.manifest:
        cfg["data"]["manifest"] = a.manifest
    tau = a.tau if a.tau is not None else cfg["serving"]["tau"]

    ds = make_dataset(cfg, a.split)
    logits, labels = collect_logits(model, DataLoader(ds, batch_size=16), device)
    probs = torch.softmax(logits / model.temperature.item(), 1).numpy()
    y = labels.numpy()

    out = Path(a.out or Path(a.checkpoint).parent / f"eval_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    report = evaluate_all(y, probs, tau)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    reliability_table(y, probs).to_csv(out / "reliability.csv", index=False)
    subgroup_report(ds.df, y, probs, tau).to_csv(out / "subgroups.csv", index=False)
    print(json.dumps({k: report[k] for k in ("n", "n_red", "macro_auc", "red_sensitivity", "ece")}, indent=2))
    print(f"report written to {out}/")


if __name__ == "__main__":
    main()
