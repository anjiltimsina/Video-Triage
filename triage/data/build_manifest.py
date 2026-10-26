"""Manifest builder: tracks.csv (from scripts/track_clips.py) + a labels file -> the manifest CSV used for training.

WHAT IT DECIDES (all of it is reported, nothing is silently dropped):
  * a clip is only eligible if it was written, was de-identified (deid_applied) and did not fail the
    face QA (deid_ok != False)  -> a clip that may show a face NEVER reaches training;
  * clips whose pose quality flags are in `drop_flags` (default low_conf, few_kp, small) are left out,
    but occluded / gappy clips are KEPT: that is what a real ED looks like and the model must see it;
  * only clips a human labelled get in; the rest are listed as "unlabelled";
  * every rejected clip goes to rejected.csv with a reason, so nothing is lost and everything can be revisited;
  * the split is by PATIENT (patient_id, or the source video if absent), never by clip: two clips of the
    same person in train and test would leak and inflate every metric. RED patients are spread over all
    splits first (stratified), because the RED sensitivity target is the number the project is judged on.

labels.csv columns:  clip_id,label[,patient_id,age,sex,arrival_time,annotator]
  clip_id = the id printed by track_clips.py (e.g. cam1_t003); label = RED | YELLOW | GREEN.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import CLASSES, MANIFEST_COLUMNS, NUM_KEYPOINTS, SOURCES

DEFAULT_DROP_FLAGS = ("low_conf", "few_kp", "small")
MIN_CLIPS_FOR_METRIC = 10          # fewer RED clips than this in val/test -> sensitivity is meaningless


def _rel(target: Path, root: Path) -> str:
    return Path(os.path.relpath(target, root)).as_posix()


def load_tracks(tracks_dirs: list[str | Path]) -> pd.DataFrame:
    """Read tracks.csv of every folder; remember where each clip's files live."""
    parts = []
    for d in tracks_dirs:
        d = Path(d)
        df = pd.read_csv(d / "tracks.csv")
        df["_dir"] = str(d)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    if out["clip_id"].duplicated().any():
        dup = out.loc[out["clip_id"].duplicated(), "clip_id"].iloc[0]
        raise ValueError(f"clip_id '{dup}' appears in more than one tracks.csv - rename a source video")
    return out


def load_labels(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("clip_id", "label"):
        if c not in df.columns:
            raise ValueError(f"labels file needs a '{c}' column")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    bad = sorted(set(df["label"]) - set(CLASSES))
    if bad:
        raise ValueError(f"unknown labels {bad}; allowed: {CLASSES}")
    if df["clip_id"].duplicated().any():
        raise ValueError("a clip_id is labelled twice in the labels file: " + str(df.loc[df["clip_id"].duplicated(), "clip_id"].tolist()))
    return df


def _flag_set(s) -> set[str]:
    return {f for f in str(s).split(";") if f and f != "nan"}


def eligibility(row: pd.Series, drop_flags, allow_no_deid: bool) -> str:
    """'' if the clip may be used, otherwise the reason it may not."""
    if not bool(row.get("written", False)):
        return "not_written"
    if not allow_no_deid and not bool(row.get("deid_applied", False)):
        return "no_deid"
    if "deid_ok" in row and pd.notna(row["deid_ok"]) and not bool(row["deid_ok"]):
        return "deid_qa_failed"
    hit = sorted(_flag_set(row.get("flags", "")) & set(drop_flags))
    if hit:
        return "quality:" + ",".join(hit)
    return ""


def pose_problem(path: Path) -> str:
    try:
        p = np.load(path)
    except Exception as e:                                         # missing / corrupt
        return f"pose_unreadable:{type(e).__name__}"
    if p.ndim != 3 or p.shape[1] != NUM_KEYPOINTS or p.shape[2] != 3 or not np.isfinite(p).all():
        return f"pose_bad_shape:{p.shape}"
    return ""


def split_by_group(groups: pd.DataFrame, ratios=(0.7, 0.15, 0.15), seed: int = 0) -> dict[str, str]:
    """groups: columns group, severity (RED/YELLOW/GREEN = the worst label of that patient), n_clips.
    Returns group -> split. Stratified by severity so each split gets RED patients where possible."""
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for sev in CLASSES:                                            # RED first: scarcest class gets spread first
        g = groups[groups["severity"] == sev]["group"].tolist()
        rng.shuffle(g)
        n = len(g)
        n_val, n_test = int(round(n * ratios[1])), int(round(n * ratios[2]))
        if n >= 3:                                                 # at least one patient in val and test
            n_val, n_test = max(1, n_val), max(1, n_test)
        elif n == 2:
            n_val, n_test = 1, 0
        else:
            n_val = n_test = 0
        for i, name in enumerate(g):
            out[name] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return out


def build_manifest(tracks_dirs: list[str | Path], labels_path: str | Path, out_path: str | Path,
                   source: str = "hospital", drop_flags=DEFAULT_DROP_FLAGS, allow_no_deid: bool = False,
                   ratios=(0.7, 0.15, 0.15), seed: int = 0) -> dict:
    """Write manifest.csv + rejected.csv next to out_path and return a report dict."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    out_path = Path(out_path)
    root = out_path.parent
    root.mkdir(parents=True, exist_ok=True)
    tracks, labels = load_tracks(tracks_dirs), load_labels(labels_path)

    known = set(tracks["clip_id"])
    unknown_labels = sorted(set(labels["clip_id"]) - known)        # a typo here would silently lose a labelled clip
    df = tracks.merge(labels, on="clip_id", how="left", suffixes=("", "_lab"))

    keep, rejected = [], []
    for _, r in df.iterrows():
        if pd.isna(r["label"]):
            rejected.append((r["clip_id"], "unlabelled"))
            continue
        why = eligibility(r, drop_flags, allow_no_deid)
        if not why:
            why = pose_problem(Path(r["_dir"]) / r["pose_file"])
        if not why and not (Path(r["_dir"]) / r["clip_file"]).exists():
            why = "video_missing"
        (rejected.append((r["clip_id"], why)) if why else keep.append(r))

    kept = pd.DataFrame(keep)
    if kept.empty:
        pd.DataFrame(rejected, columns=["clip_id", "reason"]).to_csv(root / "rejected.csv", index=False)
        raise ValueError("no usable labelled clips - see rejected.csv")

    def col(name):                                                 # optional label columns
        return kept[name] if name in kept.columns else pd.Series([np.nan] * len(kept), index=kept.index)

    kept["group"] = col("patient_id").astype("object").where(col("patient_id").notna(), kept["source_video"]).astype(str)
    rank = {c: i for i, c in enumerate(CLASSES)}                   # RED = most severe = lowest index
    sev = kept.groupby("group")["label"].agg(lambda s: min(s, key=rank.get)).rename("severity").reset_index()
    sev["n_clips"] = sev["group"].map(kept["group"].value_counts())
    split_of = split_by_group(sev, ratios, seed)

    m = pd.DataFrame({
        "clip_id": kept["clip_id"].values,
        "video_path": [_rel(Path(d) / f, root) for d, f in zip(kept["_dir"], kept["clip_file"])],
        "pose_path": [_rel(Path(d) / f, root) for d, f in zip(kept["_dir"], kept["pose_file"])],
        "label": kept["label"].values,
        "split": kept["group"].map(split_of).values,
        "age": col("age").values, "sex": col("sex").values, "arrival_time": col("arrival_time").values,
        "source": source, "annotator": col("annotator").values,
        "patient_id": kept["group"].values,
    })[MANIFEST_COLUMNS + ["patient_id"]]
    m.to_csv(out_path, index=False)
    rej = pd.DataFrame(rejected, columns=["clip_id", "reason"])
    rej.to_csv(root / "rejected.csv", index=False)

    counts = pd.crosstab(m["split"], m["label"]).reindex(index=["train", "val", "test"], columns=CLASSES, fill_value=0)
    warnings = []
    if unknown_labels:
        warnings.append(f"{len(unknown_labels)} labelled clip_id(s) not found in tracks.csv (typo?): {unknown_labels[:5]}")
    for sp in ("val", "test"):
        if counts.loc[sp, "RED"] < MIN_CLIPS_FOR_METRIC:
            warnings.append(f"only {counts.loc[sp, 'RED']} RED clips in {sp}: RED sensitivity there is not statistically meaningful")
    if (counts.loc["train"] == 0).any():
        warnings.append("train has a class with zero clips")
    if kept["group"].nunique() < 6:
        warnings.append(f"only {kept['group'].nunique()} patients: splits are not trustworthy yet")
    if "patient_id" not in labels.columns:
        warnings.append("no patient_id in labels: split uses the source video as the patient (safe, but coarse)")
    reasons = rej["reason"].str.split(":").str[0].value_counts().to_dict() if len(rej) else {}
    return {"manifest": str(out_path), "n_clips": len(m), "n_patients": int(kept["group"].nunique()),
            "counts": counts, "rejected": reasons, "warnings": warnings}