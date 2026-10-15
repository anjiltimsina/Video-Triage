"""End-to-end smoke test + the 'data swap' guarantee: a second manifest (a fake 'hospital' folder)
must work with the same code, changing only the config."""
import json
import shutil

import pandas as pd

from triage.config import load_config
from triage.data.synthetic import generate_synthetic_dataset
from triage.training import load_checkpoint, train


def _cfg(manifest, out_dir):
    return load_config("configs/default.yaml", [
        f"data.manifest={manifest}", "data.image_size=32", "train.epochs=2",
        f"train.out_dir={out_dir}", "train.batch_size=8"])


def test_train_evaluate_and_reload(synthetic_manifest, tmp_path):
    results = train(_cfg(synthetic_manifest, tmp_path / "run"))
    assert {"history", "temperature", "val", "test"} <= set(results)
    assert 0.0 <= results["test"]["macro_auc"] <= 1.0
    assert (tmp_path / "run" / "best.pt").exists()
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert metrics["test"]["n"] == results["test"]["n"]
    model, ckpt = load_checkpoint(tmp_path / "run" / "best.pt")
    assert model.temperature.item() > 0 and ckpt["classes"] == ["RED", "YELLOW", "GREEN"]


def test_data_swap_to_other_source(tmp_path):
    """Pretend the PAHS data arrived: new folder, new manifest with source=hospital. Nothing else changes."""
    root = tmp_path / "hospital"
    path = generate_synthetic_dataset(root, n_clips=60, num_frames=8, image_size=32, seed=7)
    df = pd.read_csv(path)
    df["source"] = "hospital"
    df.to_csv(path, index=False)
    cfg = _cfg(path, tmp_path / "run_h")
    cfg["data"]["sources"] = ["hospital"]
    assert train(cfg)["test"]["n"] > 0


def test_continue_from_previous_phase(synthetic_manifest, tmp_path):
    train(_cfg(synthetic_manifest, tmp_path / "p1"))
    cfg = _cfg(synthetic_manifest, tmp_path / "p2")
    cfg["train"]["init_checkpoint"] = str(tmp_path / "p1" / "best.pt")
    cfg["model"]["train_mode"] = "frozen"
    assert "test" in train(cfg)
