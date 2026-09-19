"""Tests for all loss functions (Baseline CE, WPCT, PI-WPCT, ConfTr)."""

import torch
import pytest

from piwpct.losses import BaselineCELoss, ConfTrLoss, PIWPCTLoss, WPCTLoss


def test_baseline_ce_loss():
    loss_fn = BaselineCELoss()
    logits = torch.randn(10, 3, requires_grad=True)
    labels = torch.randint(0, 3, (10,))

    loss, metrics = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert "ce_loss" in metrics
    assert float(loss.item()) > 0


def test_wpct_loss():
    loss_fn = WPCTLoss(target_wp=0.70, lambda_eff=2.0, lambda_light=0.7, lambda_charm=0.3)
    logits = torch.randn(30, 3, requires_grad=True)
    # Ensure all 3 classes are present
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10, dtype=torch.long)

    loss, metrics = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert "eff_penalty" in metrics
    assert "soft_eps_b" in metrics
    assert "soft_r_light" in metrics
    assert "soft_r_charm" in metrics
    assert metrics["lambda_light"] == 0.7
    assert metrics["lambda_charm"] == 0.3


def test_pi_wpct_loss_weight_injection():
    loss_fn = PIWPCTLoss(target_wp=0.70, lambda_eff=2.0, initial_lambda_light=0.5, initial_lambda_charm=0.5)
    logits = torch.randn(30, 3, requires_grad=True)
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10, dtype=torch.long)

    # Initial forward
    _, m1 = loss_fn(logits, labels)
    assert m1["lambda_light"] == 0.5
    assert m1["lambda_charm"] == 0.5

    # Update weights via external non-differentiable scalar injection
    loss_fn.update_weights(0.85, 0.15)
    loss2, m2 = loss_fn(logits, labels)
    assert m2["lambda_light"] == 0.85
    assert m2["lambda_charm"] == 0.15

    loss2.backward()
    assert logits.grad is not None


def test_conftr_loss():
    loss_fn = ConfTrLoss(alpha=0.10)
    logits = torch.randn(20, 3, requires_grad=True)
    labels = torch.randint(0, 3, (20,))

    loss, metrics = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert "conftr_cov_loss" in metrics
    assert "soft_coverage" in metrics
    assert "soft_set_size" in metrics
