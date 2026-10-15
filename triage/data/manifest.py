"""Manifest loading + validation. The manifest CSV is the ONLY thing that tells the training
code which data to use, so swapping synthetic -> mock -> NTU -> hospital data is a config change.

OWNER: Aayush B.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import CLASSES, MANIFEST_COLUMNS, SOURCES, SPLITS


def load_manifest(path: str | Path, sources: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sources:
        df = df[df["source"].isin(sources)].reset_index(drop=True)
    return df


def validate_manifest(path: str | Path, check_files: bool = True) -> list[str]:
    """Return a list of human-readable problems (empty list == manifest is valid)."""
    path = Path(path)
    problems: list[str] = []
    df = pd.read_csv(path)

    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    if df["clip_id"].duplicated().any():
        problems.append("duplicate clip_id values")
    bad = sorted(set(df["label"]) - set(CLASSES))
    if bad:
        problems.append(f"unknown labels: {bad}")
    bad = sorted(set(df["split"]) - set(SPLITS))
    if bad:
        problems.append(f"unknown splits: {bad}")
    bad = sorted(set(df["source"]) - set(SOURCES))
    if bad:
        problems.append(f"unknown sources: {bad}")

    if check_files:
        root = path.parent
        for col in ("video_path", "pose_path"):
            absent = [p for p in df[col] if not (root / p).exists()]
            if absent:
                problems.append(f"{len(absent)} files missing for {col}, e.g. {absent[0]}")
    return problems


def summarize_manifest(path: str | Path) -> pd.DataFrame:
    """Counts per split x label - run this whenever new data arrives."""
    df = pd.read_csv(path)
    return pd.crosstab(df["split"], df["label"]).reindex(columns=CLASSES, fill_value=0)
