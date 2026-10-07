"""Regression coverage for temperature-scaled conformal scores on both splits."""

import numpy as np
import pytest

from piwpct.conformal import MondrianConformalCalibrator
from piwpct.train import apply_temperature, scale_calibration_and_test_logits


def test_calibration_and_test_logits_receive_identical_temperature_transform():
    calibration_logits = np.array(
        [[4.0, 0.0, -1.0], [0.0, 3.0, -1.0], [-1.0, 0.0, 2.0]], dtype=np.float32
    )
    calibration_labels = np.array([0, 1, 2])
    test_logits = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    test_labels = np.array([0, 1])
    temperature = 2.5

    scaled_calibration, scaled_test = scale_calibration_and_test_logits(
        calibration_logits, test_logits, temperature
    )
    np.testing.assert_allclose(scaled_calibration, calibration_logits / temperature)
    np.testing.assert_allclose(scaled_test, test_logits / temperature)
    np.testing.assert_allclose(
        apply_temperature(np.concatenate([calibration_logits, test_logits]), temperature),
        np.concatenate([scaled_calibration, scaled_test]),
    )

    calibrator = MondrianConformalCalibrator(alpha=0.2)
    calibrator.calibrate(scaled_calibration, calibration_labels)
    actual = calibrator.evaluate_batches([(scaled_test, test_labels)])

    reference = MondrianConformalCalibrator(alpha=0.2)
    reference.calibrate(calibration_logits / temperature, calibration_labels)
    expected = reference.evaluate(test_logits / temperature, test_labels)
    assert actual.class_quantiles == expected.class_quantiles
    assert actual.marginal_coverage == expected.marginal_coverage
    assert actual.mean_set_size == expected.mean_set_size

    unscaled_calibrator = MondrianConformalCalibrator(alpha=0.2)
    unscaled_calibrator.calibrate(calibration_logits, calibration_labels)
    assert unscaled_calibrator.class_quantiles != calibrator.class_quantiles


def test_temperature_must_be_positive_and_finite():
    with pytest.raises(ValueError, match="positive and finite"):
        apply_temperature(np.zeros((2, 3)), 0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        apply_temperature(np.zeros((2, 3)), np.nan)


def test_streaming_conformal_quantiles_and_metrics_match_array_api():
    rng = np.random.RandomState(8)
    cal_logits = rng.normal(size=(100, 3)).astype(np.float32)
    cal_labels = rng.randint(0, 3, size=100)
    test_logits = rng.normal(size=(80, 3)).astype(np.float32)
    test_labels = rng.randint(0, 3, size=80)
    direct = MondrianConformalCalibrator(alpha=0.15)
    direct.calibrate(cal_logits, cal_labels)
    streamed = MondrianConformalCalibrator(alpha=0.15)
    streamed.calibrate_batches([
        (cal_logits[:25], cal_labels[:25]),
        (cal_logits[25:60], cal_labels[25:60]),
        (cal_logits[60:], cal_labels[60:]),
    ])
    assert streamed.class_quantiles == direct.class_quantiles
    expected = direct.evaluate(test_logits, test_labels)
    actual = streamed.evaluate_batches([
        (test_logits[:19], test_labels[:19]),
        (test_logits[19:51], test_labels[19:51]),
        (test_logits[51:], test_labels[51:]),
    ])
    assert actual.marginal_coverage == expected.marginal_coverage
    assert actual.per_class_coverage == expected.per_class_coverage
    assert actual.mean_set_size == expected.mean_set_size
