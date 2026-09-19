"""Unit tests for the non-differentiable RiskAllocator math and edge cases."""

import numpy as np
import pytest
import torch

from piwpct.backbones.particlenet_lite import ParticleNetLite
from piwpct.risk_allocator import RiskAllocator


def test_closed_form_sensitivities_against_numerical_derivative():
    """Verify that analytic dP/dr_i matches finite-difference numerical derivative."""
    s_yield = 20000.0
    y_l = 100000.0
    y_c = 25000.0
    r_l = 0.05
    r_c = 0.15

    def purity_fn(rl, rc):
        b_l = y_l * rl
        b_c = y_c * rc
        return s_yield / (s_yield + b_l + b_c)

    # Numerical derivatives
    delta = 1e-6
    num_dp_drl = (purity_fn(r_l + delta, r_c) - purity_fn(r_l - delta, r_c)) / (2 * delta)
    num_dp_drc = (purity_fn(r_l, r_c + delta) - purity_fn(r_l, r_c - delta)) / (2 * delta)

    # Closed-form analytic sensitivities
    denom_sq = (s_yield + y_l * r_l + y_c * r_c) ** 2
    analytic_dp_drl = - (s_yield * y_l) / denom_sq
    analytic_dp_drc = - (s_yield * y_c) / denom_sq

    assert np.isclose(num_dp_drl, analytic_dp_drl, rtol=1e-5)
    assert np.isclose(num_dp_drc, analytic_dp_drc, rtol=1e-5)


def test_variance_stabilization_floor():
    """Verify minimum variance floor is strictly enforced when counts are 0 or extreme."""
    allocator = RiskAllocator(min_variance_floor=1e-7, variance_stabilization="min_floor")
    var_zero, applied = allocator.compute_variance(k=0, n=100)
    assert var_zero >= 1e-7
    assert applied is True

    var_normal, applied_normal = allocator.compute_variance(k=50, n=100)
    expected = (0.5 * 0.5) / 100
    assert np.isclose(var_normal, expected)
    assert applied_normal is False


def test_wilson_interval_variance():
    """Verify Wilson interval regularization produces positive non-zero variance."""
    allocator = RiskAllocator(variance_stabilization="wilson_floor", min_variance_floor=1e-7)
    var_val, applied = allocator.compute_variance(k=0, n=50)
    assert var_val > 0.0
    assert var_val >= 1e-7


def test_zero_sum_guard_edge_case():
    """Verify division-by-zero guard when risks sum to zero."""
    allocator = RiskAllocator(lambda_bg=1.0, zero_sum_guard_epsilon=1e-12)

    # Mock logits where everything is identical
    logits = np.zeros((30, 3))
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

    # Force zero risk by mocking yields to 0
    res = allocator.compute_risk_allocation(logits, labels, target_wp=0.70, yields={"light": 0.0, "charm": 0.0, "b": 100.0})

    assert not np.isnan(res.lambda_light)
    assert not np.isnan(res.lambda_charm)
    assert np.isclose(res.lambda_light, 0.5)
    assert np.isclose(res.lambda_charm, 0.5)


def test_risk_allocator_is_strictly_non_differentiable():
    """Verify that RiskAllocator forward pass does not record PyTorch autograd graph."""
    model = ParticleNetLite(input_dim=16, num_classes=3)
    # Ensure model parameters have requires_grad=True
    for p in model.parameters():
        assert p.requires_grad is True

    allocator = RiskAllocator()

    # Create dummy DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    dummy_x = torch.randn(20, 16, 16)
    dummy_y = torch.tensor([0] * 7 + [1] * 7 + [2] * 6, dtype=torch.long)
    dummy_ds = TensorDataset(dummy_x, dummy_y)

    class DummyBatchDataset:
        def __init__(self, ds):
            self.ds = ds
            self.split_name = "validation"
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            return {
                "features": x,
                "labels": y,
                "mask": torch.ones(16),
                "points": x[:, :2],
                "vectors": torch.zeros(16, 4),
            }

    val_loader = DataLoader(DummyBatchDataset(dummy_ds), batch_size=10)

    # Run refresh
    device = torch.device("cpu")
    result = allocator.refresh(model, val_loader, target_wp=0.70, device=device)

    assert isinstance(result.lambda_light, float)
    assert isinstance(result.lambda_charm, float)
    # Model parameters should have zero gradients
    for p in model.parameters():
        assert p.grad is None
