"""Named JetClass feature contract and physically consistent perturbations."""

import numpy as np
import pytest

from piwpct.evaluation.robustness import apply_jes_shift, apply_ip_smearing, apply_jer_smearing
from piwpct.schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RAW_PARTICLE_BRANCHES,
    RAW_PARTICLE_INDEX,
    build_features,
    build_four_vectors,
    build_points,
    feature_index,
    feature_schema_document,
)


def _raw_particle(deta=0.1, dphi=-0.2, pt=10.0, energy=15.0):
    raw = np.zeros((1, len(RAW_PARTICLE_BRANCHES)), dtype=np.float32)
    raw[0, RAW_PARTICLE_INDEX["part_deta"]] = deta
    raw[0, RAW_PARTICLE_INDEX["part_dphi"]] = dphi
    raw[0, RAW_PARTICLE_INDEX["part_px"]] = pt * np.cos(0.3)
    raw[0, RAW_PARTICLE_INDEX["part_py"]] = pt * np.sin(0.3)
    raw[0, RAW_PARTICLE_INDEX["part_pz"]] = pt * np.sinh(deta)
    raw[0, RAW_PARTICLE_INDEX["part_energy"]] = energy
    raw[0, RAW_PARTICLE_INDEX["part_d0val"]] = 0.25
    raw[0, RAW_PARTICLE_INDEX["part_d0err"]] = 1.5
    raw[0, RAW_PARTICLE_INDEX["part_dzval"]] = -0.5
    raw[0, RAW_PARTICLE_INDEX["part_dzerr"]] = -0.2
    raw[0, RAW_PARTICLE_INDEX["part_charge"]] = -1
    raw[0, RAW_PARTICLE_INDEX["part_isChargedHadron"]] = 1
    return raw


def test_authoritative_schema_order_transform_dtype_and_four_vectors():
    raw = _raw_particle()
    features = build_features(raw, np.array([100.0]), np.array([200.0]))
    schema = feature_schema_document()

    assert len(FEATURE_NAMES) == 17
    assert schema["schema_version"] == FEATURE_SCHEMA_VERSION
    assert schema["ordering"] == list(FEATURE_NAMES)
    assert schema["features"][0]["index"] == 0
    assert schema["features"][2]["transformation"] == "(log(hypot(px, py)) - 1.7) * 0.7"
    assert features.dtype == np.float32
    assert features.shape == (1, 17)
    assert features[0, feature_index("part_deta")] == pytest.approx(0.1)
    assert features[0, feature_index("part_dphi")] == pytest.approx(-0.2)
    assert features[0, feature_index("part_pt_log")] == pytest.approx((np.log(10.0) - 1.7) * 0.7)
    assert features[0, feature_index("part_e_log")] == pytest.approx((np.log(15.0) - 2.0) * 0.7)
    assert features[0, feature_index("part_d0")] == pytest.approx(np.tanh(0.25))
    assert features[0, feature_index("part_dz")] == pytest.approx(np.tanh(-0.5))
    assert features[0, feature_index("part_logptrel")] == pytest.approx((np.log(0.1) + 4.7) * 0.7)
    assert features[0, feature_index("part_logerel")] == pytest.approx((np.log(15.0 / 200.0) + 4.7) * 0.7)
    assert features[0, feature_index("part_deltaR")] == pytest.approx((np.hypot(0.1, -0.2) - 0.2) * 4.0)
    assert features[0, feature_index("part_d0err")] == 1.0
    assert features[0, feature_index("part_dzerr")] == 0.0
    np.testing.assert_allclose(build_points(raw), [[0.1, -0.2]])
    vectors = build_four_vectors(raw)
    assert vectors.shape == (1, 4)
    assert vectors[0, 0] == pytest.approx(10.0 * np.cos(0.3))
    assert vectors[0, 3] == pytest.approx(15.0)


def test_feature_builder_rejects_invalid_positive_kinematics():
    raw = _raw_particle(pt=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        build_features(raw, np.array([100.0]), np.array([200.0]))


def test_jes_and_jer_update_dependent_named_features_and_preserve_padding():
    features = np.zeros((1, 2, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, 0, feature_index("part_pt_log")] = 1.0
    features[0, 0, feature_index("part_e_log")] = 2.0
    features[0, 0, feature_index("part_logptrel")] = -1.0
    features[0, 0, feature_index("part_logerel")] = -2.0
    vectors = np.ones((1, 2, 4), dtype=np.float32)
    mask = np.array([[1.0, 0.0]], dtype=np.float32)

    jes, jes_vectors = apply_jes_shift(features, vectors, 1.1, mask)
    expected_delta = 0.7 * np.log(1.1)
    assert jes[0, 0, feature_index("part_pt_log")] == pytest.approx(1.0 + expected_delta)
    assert jes[0, 0, feature_index("part_e_log")] == pytest.approx(2.0 + expected_delta)
    assert jes[0, 0, feature_index("part_logptrel")] == pytest.approx(-1.0)
    assert np.all(jes[0, 1] == 0.0)
    assert jes_vectors[0, 0, 0] == pytest.approx(1.1)

    jer, jer_vectors = apply_jer_smearing(features, vectors, 0.1, seed=17, mask=mask)
    noise = np.random.RandomState(17).normal(1.0, 0.1, size=(1, 2))
    noise = np.clip(noise, 0.5, 2.0)
    delta = 0.7 * np.log(noise[0, 0])
    assert jer[0, 0, feature_index("part_pt_log")] == pytest.approx(1.0 + delta)
    assert jer[0, 0, feature_index("part_logptrel")] == pytest.approx(-1.0 + delta)
    assert np.all(jer[0, 1] == 0.0)
    assert jer_vectors[0, 0, 0] == pytest.approx(noise[0, 0])


def test_ip_smearing_uses_named_fields_and_keeps_padding_zero():
    features = np.zeros((1, 2, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, 0, feature_index("part_d0")] = np.tanh(0.5)
    features[0, 0, feature_index("part_dz")] = np.tanh(-0.5)
    mask = np.array([[1.0, 0.0]], dtype=np.float32)
    result = apply_ip_smearing(features, 0.1, seed=3, mask=mask)
    assert result.shape == features.shape
    assert result[0, 1, feature_index("part_d0")] == 0.0
    assert result[0, 1, feature_index("part_dz")] == 0.0
