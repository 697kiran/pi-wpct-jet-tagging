"""Automated split leakage prevention tests.

Enforces:
1. Calibration split is NEVER accessed in training or risk allocation.
2. Test split is NEVER accessed in training or risk allocation.
3. Splits are strictly pairwise disjoint.
4. Any violation raises SplitIntegrityViolation and fails immediately.
"""

import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from piwpct.backbones.particlenet_lite import ParticleNetLite
from piwpct.losses import BaselineCELoss
from piwpct.risk_allocator import RiskAllocator
from piwpct.train import train_epoch
from piwpct.utils.split_integrity import SplitAuditor, SplitIntegrityViolation


class MockSplitDataset(Dataset):
    """Dataset with explicit split identity for leakage auditing."""
    def __init__(self, split_name: str, size: int = 20):
        self.split_name = split_name
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "features": torch.randn(16, 16),
            "points": torch.randn(16, 2),
            "mask": torch.ones(16),
            "vectors": torch.zeros(16, 4),
            "labels": torch.tensor(idx % 3, dtype=torch.long),
        }


def test_risk_allocator_rejects_calibration_split():
    """Verify that RiskAllocator raises SplitIntegrityViolation if calibration split is passed."""
    allocator = RiskAllocator()
    model = ParticleNetLite(input_dim=16, num_classes=3)
    cal_loader = DataLoader(MockSplitDataset(split_name="calibration"), batch_size=10)

    with pytest.raises(SplitIntegrityViolation, match="RiskAllocator was passed the calibration split"):
        allocator.evaluate_validation_logits(model, cal_loader, device=torch.device("cpu"))


def test_risk_allocator_rejects_test_split():
    """Verify that RiskAllocator raises SplitIntegrityViolation if test split is passed."""
    allocator = RiskAllocator()
    model = ParticleNetLite(input_dim=16, num_classes=3)
    test_loader = DataLoader(MockSplitDataset(split_name="test"), batch_size=10)

    with pytest.raises(SplitIntegrityViolation, match="RiskAllocator was passed the test split"):
        allocator.evaluate_validation_logits(model, test_loader, device=torch.device("cpu"))


def test_train_epoch_rejects_calibration_split():
    """Verify that train_epoch raises SplitIntegrityViolation if calibration split is passed."""
    model = ParticleNetLite(input_dim=16, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = BaselineCELoss()
    cal_loader = DataLoader(MockSplitDataset(split_name="calibration"), batch_size=10)

    with pytest.raises(SplitIntegrityViolation, match="Calibration split used in train_epoch"):
        train_epoch(model, cal_loader, optimizer, loss_fn, device=torch.device("cpu"), target_wp=0.70)


def test_train_epoch_rejects_test_split():
    """Verify that train_epoch raises SplitIntegrityViolation if test split is passed."""
    model = ParticleNetLite(input_dim=16, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = BaselineCELoss()
    test_loader = DataLoader(MockSplitDataset(split_name="test"), batch_size=10)

    with pytest.raises(SplitIntegrityViolation, match="Test split used in train_epoch"):
        train_epoch(model, test_loader, optimizer, loss_fn, device=torch.device("cpu"), target_wp=0.70)


def test_split_auditor_detects_index_overlap(tmp_path):
    """Verify that SplitAuditor detects overlapping indices across splits."""
    manifest_data = {
        "splits": {
            "indices": {
                "train": [0, 1, 2, 3],
                "validation": [4, 5, 6],
                "calibration": [7, 8],
                "test": [3, 9, 10],  # 3 leaks from train to test!
            }
        }
    }
    manifest_file = tmp_path / "leaky_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)

    auditor = SplitAuditor()
    with pytest.raises(SplitIntegrityViolation, match="FATAL SPLIT LEAKAGE"):
        auditor.load_manifest(str(manifest_file))


def test_split_auditor_asserts_indices_belong_strictly_to_split(tmp_path):
    """Verify membership assertions for individual batches."""
    manifest_data = {
        "splits": {
            "indices": {
                "train": [0, 1, 2],
                "validation": [3, 4],
                "calibration": [5],
                "test": [6, 7],
            }
        }
    }
    manifest_file = tmp_path / "clean_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)

    auditor = SplitAuditor(str(manifest_file))
    # Clean indices
    auditor.assert_indices_belong_strictly_to_split([0, 1], expected_split="train")

    # Contaminated indices: index 4 belongs to validation, not train
    with pytest.raises(SplitIntegrityViolation, match="SPLIT LEAKAGE"):
        auditor.assert_indices_belong_strictly_to_split([0, 4], expected_split="train")
