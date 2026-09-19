"""Tests for data generation, impact parameter transforms, and canonical splits."""

import json
import os
import numpy as np
import pytest

from piwpct.data import create_canonical_splits, generate_pilot_dataset, transform_impact_parameters
from piwpct.utils.split_integrity import SplitAuditor, check_splits_disjoint


def test_transform_impact_parameters():
    """Verify tanh(d0) and tanh(dz) compression and bounded properties."""
    d0 = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    dz = np.array([-50.0, -0.5, 0.0, 0.5, 50.0])
    t_d0, t_dz = transform_impact_parameters(d0, dz, clip_val=10.0)

    assert np.all(t_d0 >= -1.0) and np.all(t_d0 <= 1.0)
    assert np.all(t_dz >= -1.0) and np.all(t_dz <= 1.0)
    assert np.isclose(t_d0[2], 0.0)
    assert np.isclose(t_dz[2], 0.0)


def test_generate_pilot_dataset():
    """Verify pilot dataset shapes, classes, and provisional composition."""
    data = generate_pilot_dataset(n_samples=600, num_particles=16, feature_dim=16, seed=42)

    assert data["features"].shape == (600, 16, 16)
    assert data["points"].shape == (600, 16, 2)
    assert data["mask"].shape == (600, 16)
    assert data["vectors"].shape == (600, 16, 4)
    assert data["high_level"].shape == (600, 12)
    assert data["labels"].shape == (600,)

    classes = np.unique(data["labels"])
    assert set(classes) == {0, 1, 2}

    # Pilot class ratio is approximately 4:1:1
    n_light = np.sum(data["labels"] == 0)
    n_charm = np.sum(data["labels"] == 1)
    n_b = np.sum(data["labels"] == 2)
    assert n_light > n_charm and n_light > n_b


def test_canonical_splits_creation_and_disjointness(tmp_path):
    """Verify that canonical 4-way split is strictly disjoint and content-hashed."""
    data = generate_pilot_dataset(n_samples=1000, num_particles=16, feature_dim=16, seed=42)
    out_dir = str(tmp_path / "splits")

    manifest = create_canonical_splits(
        data=data,
        output_dir=out_dir,
        fractions=(0.60, 0.15, 0.10, 0.15),
        seed=42,
        is_pilot=True,
    )

    assert manifest["is_pilot"] is True
    assert "PILOT SUBSET" in manifest["pilot_composition_warning"]
    assert os.path.exists(os.path.join(out_dir, "canonical_split_manifest.json"))

    # Verify that files exist and hashes are valid SHA-256 strings
    for split_name in ["train", "validation", "calibration", "test"]:
        fpath = manifest["files"][split_name]
        assert os.path.exists(fpath)
        sha = manifest["sha256_checksums"][split_name]
        assert len(sha) == 64
        # Verify hash matches on-disk file
        assert SplitAuditor.compute_sha256(fpath) == sha

    # Test Auditor disjointness verification
    auditor = SplitAuditor(os.path.join(out_dir, "canonical_split_manifest.json"))
    auditor.verify_disjointness()

    # Verify that total count matches sum of splits
    total_samples = sum(manifest["counts"].values())
    assert total_samples == 1000
