"""Evaluation metrics from the proposal: macro AUC, RED sensitivity/specificity, bootstrapped
95% CIs, Expected Calibration Error (ECE), subgroup analysis, and tau selection.

All functions take numpy arrays:  y_true (N,) int class ids,  probs (N,3) probabilities.

OWNER: Aayush P.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..constants import CLASSES, NUM_CLASSES, RED_ID


def macro_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """One-vs-rest AUC averaged over the classes that have both positives and negatives."""
    aucs = []
    for c in range(NUM_CLASSES):
        pos = (y_true == c)
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def red_sensitivity_specificity(y_true: np.ndarray, probs: np.ndarray, tau: float = 0.5) -> tuple[float, float]:
    """Alert rule used by the system: RED alert when p_RED > tau."""
    is_red = y_true == RED_ID
    alert = probs[:, RED_ID] > tau
    sens = float(alert[is_red].mean()) if is_red.any() else float("nan")
    spec = float((~alert[~is_red]).mean()) if (~is_red).any() else float("nan")
    return sens, spec


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_table(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Data for a reliability diagram (plot confidence vs accuracy per bin)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                     "confidence": float(conf[m].mean()) if m.any() else np.nan,
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def bootstrap_ci(metric_fn, y_true: np.ndarray, probs: np.ndarray, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of metric_fn(y_true, probs)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric_fn(y_true[idx], probs[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def choose_tau_for_sensitivity(y_true: np.ndarray, probs: np.ndarray, target: float = 0.85) -> float:
    """Highest tau whose RED sensitivity is still >= target (fewest false alarms for the target).
    Fit this on the VALIDATION set, never on the test set."""
    for tau in np.linspace(0.99, 0.01, 99):
        sens, _ = red_sensitivity_specificity(y_true, probs, tau)
        if not np.isnan(sens) and sens >= target:
            return float(tau)
    return 0.01


def evaluate_all(y_true: np.ndarray, probs: np.ndarray, tau: float = 0.5, n_boot: int = 500) -> dict:
    sens, spec = red_sensitivity_specificity(y_true, probs, tau)
    return {
        "n": int(len(y_true)),
        "n_red": int((y_true == RED_ID).sum()),
        "tau": tau,
        "macro_auc": macro_auc(y_true, probs),
        "macro_auc_ci95": bootstrap_ci(macro_auc, y_true, probs, n_boot),
        "red_sensitivity": sens,
        "red_sensitivity_ci95": bootstrap_ci(lambda y, p: red_sensitivity_specificity(y, p, tau)[0], y_true, probs, n_boot),
        "red_specificity": spec,
        "ece": expected_calibration_error(y_true, probs),
        "accuracy": float((probs.argmax(1) == y_true).mean()),
        "confusion_matrix": pd.crosstab(
            pd.Categorical(y_true, categories=range(NUM_CLASSES)),
            pd.Categorical(probs.argmax(1), categories=range(NUM_CLASSES)), dropna=False,
        ).to_numpy().tolist(),
        "class_order": CLASSES,
    }


def subgroup_report(df: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, tau: float = 0.5) -> pd.DataFrame:
    """Subgroup analysis by age band, sex and arrival hour (df = the split's manifest rows, same order)."""
    d = df.reset_index(drop=True).copy()
    d["age_band"] = pd.cut(d["age"], [17, 40, 65, 120], labels=["18-40", "41-65", "66+"])
    d["arrival_shift"] = pd.cut(pd.to_datetime(d["arrival_time"]).dt.hour, [-1, 7, 15, 23],
                                labels=["night(0-7)", "day(8-15)", "evening(16-23)"])
    rows = []
    for col in ("age_band", "sex", "arrival_shift"):
        for val, g in d.groupby(col, observed=True):
            idx = g.index.to_numpy()
            sens, spec = red_sensitivity_specificity(y_true[idx], probs[idx], tau)
            rows.append({"group": col, "value": str(val), "n": len(idx),
                         "macro_auc": macro_auc(y_true[idx], probs[idx]),
                         "red_sensitivity": sens, "red_specificity": spec})
    return pd.DataFrame(rows)
