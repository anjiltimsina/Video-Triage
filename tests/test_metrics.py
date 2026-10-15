import numpy as np

from triage.eval.metrics import (bootstrap_ci, choose_tau_for_sensitivity, evaluate_all,
                                 expected_calibration_error, macro_auc, red_sensitivity_specificity)


def _perfect(n=60):
    y = np.arange(n) % 3
    probs = np.eye(3)[y] * 0.9 + 0.05
    return y, probs


def test_perfect_predictions():
    y, p = _perfect()
    assert macro_auc(y, p) == 1.0
    assert red_sensitivity_specificity(y, p, 0.5) == (1.0, 1.0)


def test_random_predictions_auc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    p = rng.dirichlet(np.ones(3), 3000)
    assert abs(macro_auc(y, p) - 0.5) < 0.05


def test_ece_perfectly_calibrated_is_small_and_overconfident_is_large():
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 3, n)
    overconf = np.full((n, 3), 0.01)
    overconf[np.arange(n), rng.integers(0, 3, n)] = 0.98     # always 98% sure, right ~1/3 of the time
    assert expected_calibration_error(y, overconf) > 0.5
    y2, p2 = _perfect()
    assert expected_calibration_error(y2, p2) < 0.1


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    p = rng.dirichlet(np.ones(3), 400) + np.eye(3)[y] * 0.4
    p = p / p.sum(1, keepdims=True)
    lo, hi = bootstrap_ci(macro_auc, y, p, n_boot=200)
    assert lo <= macro_auc(y, p) <= hi


def test_choose_tau_meets_target_sensitivity():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 600)
    p = rng.dirichlet(np.ones(3), 600) + np.eye(3)[y] * 0.3
    p = p / p.sum(1, keepdims=True)
    tau = choose_tau_for_sensitivity(y, p, target=0.85)
    assert red_sensitivity_specificity(y, p, tau)[0] >= 0.85


def test_evaluate_all_keys():
    y, p = _perfect()
    r = evaluate_all(y, p, tau=0.5, n_boot=50)
    assert {"macro_auc", "macro_auc_ci95", "red_sensitivity", "ece", "confusion_matrix"} <= set(r)
