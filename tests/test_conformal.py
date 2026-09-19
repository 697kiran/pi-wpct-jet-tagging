"""Tests for Mondrian (class-conditional) conformal calibration."""

import numpy as np
import pytest

from piwpct.conformal import MondrianConformalCalibrator


def test_mondrian_conformal_finite_sample_coverage():
    """Verify that Mondrian calibrator achieves >= 1 - alpha coverage on held-out test data."""
    rng = np.random.RandomState(42)
    n_cal = 1500
    n_test = 2000
    alpha = 0.10  # 90% target coverage

    # Synthetic calibrated logits: class with highest logit is correlated with true label
    def _make_data(n):
        labels = rng.randint(0, 3, size=n)
        logits = rng.normal(0, 1, size=(n, 3))
        # Add signal boost to true class
        for i in range(n):
            logits[i, labels[i]] += 2.0
        return logits, labels

    cal_logits, cal_labels = _make_data(n_cal)
    test_logits, test_labels = _make_data(n_test)

    calibrator = MondrianConformalCalibrator(alpha=alpha, score_type="residual")
    calibrator.calibrate(cal_logits, cal_labels)

    eval_out = calibrator.evaluate(test_logits, test_labels)

    assert eval_out.marginal_coverage >= 0.88  # Within finite-sample confidence band of 0.90
    for k in range(3):
        assert eval_out.per_class_coverage[k] >= 0.87
    assert eval_out.mean_set_size >= 1.0


def test_raps_conformal_scoring():
    """Verify RAPS scoring functionality and prediction set construction."""
    rng = np.random.RandomState(101)
    cal_logits = rng.normal(0, 1, size=(500, 3))
    cal_labels = rng.randint(0, 3, size=500)
    test_logits = rng.normal(0, 1, size=(200, 3))
    test_labels = rng.randint(0, 3, size=200)

    calibrator = MondrianConformalCalibrator(alpha=0.15, score_type="raps", raps_kreg=1, raps_lam=0.05)
    calibrator.calibrate(cal_logits, cal_labels)

    eval_out = calibrator.evaluate(test_logits, test_labels)
    assert len(eval_out.prediction_sets) == 200
    assert eval_out.marginal_coverage > 0.0


def test_distribution_shift_bound():
    """Verify Barber-Candès-Ramdas-Tibshirani theoretical shift bound."""
    calibrator = MondrianConformalCalibrator(alpha=0.10)
    tv_delta = 0.05
    bound = calibrator.compute_distribution_shift_bound(tv_delta)
    assert np.isclose(bound, 0.85)
