"""Tests for statistical metrics, calibration ECE, bootstrap CIs, and McNemar test."""

import numpy as np

from piwpct.metrics import (
    compute_ece,
    compute_working_point_metrics,
    mcnemar_test,
    paired_bootstrap_comparison,
)


def test_working_point_metrics():
    # Synthetic logits with perfect separation for b-jets
    logits = np.array([
        [-5.0, -5.0, 5.0],  # true b
        [-5.0, -5.0, 5.0],  # true b
        [5.0, -5.0, -5.0],  # true light
        [5.0, -5.0, -5.0],  # true light
        [-5.0, 5.0, -5.0],  # true charm
    ])
    labels = np.array([2, 2, 0, 0, 1])

    m = compute_working_point_metrics(logits, labels, target_wp=0.70)
    assert m["eps_b"] == 1.0
    assert m["r_light"] == 0.0
    assert m["r_charm"] == 0.0
    assert m["rej_light"] >= 1e5


def test_ece_computation():
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    labels = np.array([0, 0, 1, 2])
    ece = compute_ece(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_paired_bootstrap_and_mcnemar():
    rng = np.random.RandomState(42)
    labels = rng.randint(0, 3, size=200)

    # Regime A: moderate noise
    logits_a = rng.normal(0, 1, size=(200, 3))
    # Regime B: better performance
    logits_b = logits_a.copy()
    for i in range(200):
        logits_b[i, labels[i]] += 1.5

    def _metric_fn(logits, labels):
        return compute_working_point_metrics(logits, labels, target_wp=0.70)

    comp = paired_bootstrap_comparison(logits_a, logits_b, labels, _metric_fn, n_bootstraps=100, seed=42)
    assert "rej_light" in comp
    assert "mean_diff" in comp["rej_light"]

    preds_a = np.argmax(logits_a, axis=-1)
    preds_b = np.argmax(logits_b, axis=-1)
    mcn = mcnemar_test(preds_a, preds_b, labels)
    assert "mcnemar_statistic" in mcn
    assert "p_value" in mcn
