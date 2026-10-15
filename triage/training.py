"""Config-driven training / evaluation core. scripts/train.py and scripts/evaluate.py call this.

What `train(cfg)` does, in order:
  1. build train/val/test datasets from the manifest in cfg["data"]["manifest"]
  2. build the model (optionally continue from cfg["train"]["init_checkpoint"], i.e. the previous phase)
  3. set which layers train (frozen | last4 | full) and train with focal loss (+ RED oversampling)
  4. keep the checkpoint with the best validation macro AUC
  5. fit the temperature (calibration) on the validation set
  6. evaluate on val + test, write metrics.json and history.csv to out_dir

OWNER: Anjil (Aayush P owns the metrics it calls).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .constants import CLASSES, RED_ID
from .data.dataset import TriageDataset
from .eval.metrics import choose_tau_for_sensitivity, evaluate_all, macro_auc
from .models.calibration import fit_temperature
from .models.fusion import FusionTriageModel
from .models.losses import TriageLoss


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg: dict) -> FusionTriageModel:
    m = cfg["model"]
    return FusionTriageModel(m["backbone"], m.get("pretrained"), m["pose_hidden"], m["dropout"])


def make_dataset(cfg: dict, split: str) -> TriageDataset:
    d = cfg["data"]
    return TriageDataset(d["manifest"], split, d["num_frames"], d["image_size"], d.get("sources"))


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        out = model(batch["video"].to(device), batch["pose"].to(device))
        logits.append(out["logits"].float().cpu())
        labels.append(batch["label"])
    return torch.cat(logits), torch.cat(labels)


def save_checkpoint(path: Path, model, cfg: dict, extra: dict | None = None) -> None:
    torch.save({"model_state": model.state_dict(), "config": cfg, "classes": CLASSES, **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FusionTriageModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def train(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    t, device = cfg["train"], get_device(cfg["train"]["device"])
    out_dir = Path(t["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = (make_dataset(cfg, s) for s in ("train", "val", "test"))
    if len(train_ds) == 0:
        raise RuntimeError("No training clips found - check data.manifest / data.sources in the config.")

    sample_w = np.where(train_ds.labels == RED_ID, t["oversample_red"], 1.0)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
    mk = lambda ds, **kw: DataLoader(ds, batch_size=t["batch_size"], num_workers=t["num_workers"], **kw)
    train_loader = mk(train_ds, sampler=sampler)
    val_loader, test_loader = mk(val_ds), mk(test_ds)

    model = build_model(cfg).to(device)
    if t.get("init_checkpoint"):
        state = torch.load(t["init_checkpoint"], map_location=device, weights_only=False)["model_state"]
        model.load_state_dict(state)
        print(f"[train] continuing from {t['init_checkpoint']}")
    model.set_train_mode(cfg["model"]["train_mode"])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    criterion = TriageLoss(**cfg["loss"]).to(device)
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history, best_auc, best_path = [], -1.0, out_dir / "best.pt"
    print(f"[train] device={device} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"trainable_params={sum(p.numel() for p in params):,}")

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in train_loader:
            video, pose, y = batch["video"].to(device), batch["pose"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, _ = criterion(model(video, pose), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if len(val_ds):
            logits, labels = collect_logits(model, val_loader, device)
            row["val_macro_auc"] = macro_auc(labels.numpy(), torch.softmax(logits, 1).numpy())
        history.append(row)
        print("[train]", row)

        score = row.get("val_macro_auc", -row["train_loss"])
        if np.isnan(score):
            score = -row["train_loss"]
        if score > best_auc:
            best_auc = score
            save_checkpoint(best_path, model, cfg)

    # ---- reload best weights, then calibrate on validation, evaluate on val + test ----
    model, _ = load_checkpoint(best_path, device)
    results: dict = {"history": history}
    tau = cfg["serving"]["tau"]
    if len(val_ds):
        v_logits, v_labels = collect_logits(model, val_loader, device)
        temp = fit_temperature(v_logits, v_labels)
        model.temperature.fill_(temp)
        v_probs = torch.softmax(v_logits / temp, 1).numpy()
        tau_val = choose_tau_for_sensitivity(v_labels.numpy(), v_probs, 0.85)
        results["temperature"], results["tau_from_val_for_sens_0.85"] = temp, tau_val
        results["val"] = evaluate_all(v_labels.numpy(), v_probs, tau)
    save_checkpoint(best_path, model, cfg, {"tau_val": results.get("tau_from_val_for_sens_0.85")})

    if len(test_ds):
        te_logits, te_labels = collect_logits(model, test_loader, device)
        te_probs = torch.softmax(te_logits / model.temperature.item(), 1).numpy()
        results["test"] = evaluate_all(te_labels.numpy(), te_probs, tau)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[train] done. best checkpoint: {best_path}")
    return results
