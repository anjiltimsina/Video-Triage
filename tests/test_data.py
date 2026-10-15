import pandas as pd
import torch

from triage.constants import MANIFEST_COLUMNS
from triage.data.dataset import TriageDataset
from triage.data.manifest import validate_manifest


def test_manifest_is_valid(synthetic_manifest):
    assert validate_manifest(synthetic_manifest) == []
    df = pd.read_csv(synthetic_manifest)
    assert list(df.columns) == MANIFEST_COLUMNS


def test_validate_catches_bad_label(synthetic_manifest, tmp_path):
    df = pd.read_csv(synthetic_manifest)
    df.loc[0, "label"] = "PURPLE"
    bad = synthetic_manifest.parent / "bad.csv"
    df.to_csv(bad, index=False)
    assert any("unknown labels" in p for p in validate_manifest(bad))


def test_dataset_item_matches_contract(synthetic_manifest):
    ds = TriageDataset(synthetic_manifest, "train", num_frames=8, image_size=32)
    item = ds[0]
    assert item["video"].shape == (8, 3, 32, 32) and item["video"].dtype == torch.float32
    assert 0.0 <= item["video"].min() and item["video"].max() <= 1.0
    assert item["pose"].shape == (8, 17, 3)
    assert item["label"].item() in (0, 1, 2)


def test_dataset_resamples_frames_and_size(synthetic_manifest):
    ds = TriageDataset(synthetic_manifest, "train", num_frames=4, image_size=16)
    item = ds[0]
    assert item["video"].shape == (4, 3, 16, 16) and item["pose"].shape == (4, 17, 3)
