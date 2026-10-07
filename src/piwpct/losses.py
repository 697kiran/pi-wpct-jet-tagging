"""Loss functions for jet flavour tagging regimes.

Includes:
- Baseline Cross-Entropy Loss
- Working-Point Conformal Training (WPCT) Loss with static background mistag penalties
- Physics-Impact Weighted WPCT (PI-WPCT) Loss with dynamically injected weights
- Differentiable Conformal Training (ConfTr) Loss (Stutz et al., ICLR 2022)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCELoss(nn.Module):
    """Standard multi-class Cross-Entropy Loss."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = self.criterion(logits, labels)
        return loss, {"ce_loss": float(loss.item())}


class WPCTLoss(nn.Module):
    """Working-Point Conformal Training (WPCT) loss with fixed/manually-tuned weights.

    L(w) = L_CE + lambda_eff * |epsilon_b - w| + lambda_light(w) * L_light + lambda_charm(w) * L_charm
    """

    def __init__(
        self,
        target_wp: float = 0.70,
        lambda_eff: float = 2.0,
        lambda_bg: float = 1.0,
        lambda_light: float = 0.7,
        lambda_charm: float = 0.3,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.target_wp = target_wp
        self.lambda_eff = lambda_eff
        self.lambda_bg = lambda_bg
        self.lambda_light = lambda_light
        self.lambda_charm = lambda_charm
        self.temperature = temperature
        self.ce_criterion = nn.CrossEntropyLoss()

    def compute_soft_wp_penalties(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_wp: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute differentiable soft b-efficiency and light/charm mistag rates.

        Classes: 0: light, 1: charm, 2: b
        """
        probs = F.softmax(logits, dim=-1)
        b_scores = probs[:, 2]  # [B]

        is_b = (labels == 2)
        is_light = (labels == 0)
        is_charm = (labels == 1)

        b_count = is_b.sum()
        light_count = is_light.sum()
        charm_count = is_charm.sum()

        if b_count > 0:
            b_scores_true = b_scores[is_b]
            # Quantile (1 - target_wp) gives threshold where fraction target_wp passes
            q_val = float(np_quantile_safe(b_scores_true.detach().cpu().numpy(), 1.0 - target_wp))
            tau = torch.tensor(q_val, device=logits.device, dtype=logits.dtype)
        else:
            tau = torch.tensor(0.5, device=logits.device, dtype=logits.dtype)

        # Sigmoidal smooth step function: sigma((s - tau) / T)
        soft_pass = torch.sigmoid((b_scores - tau) / self.temperature)

        # Soft b-efficiency: mean soft_pass for true b-jets
        if b_count > 0:
            soft_eps_b = soft_pass[is_b].mean()
        else:
            soft_eps_b = torch.tensor(target_wp, device=logits.device)

        # Soft mistag rates
        soft_r_light = soft_pass[is_light].mean() if light_count > 0 else torch.tensor(0.0, device=logits.device)
        soft_r_charm = soft_pass[is_charm].mean() if charm_count > 0 else torch.tensor(0.0, device=logits.device)

        return soft_eps_b, soft_r_light, soft_r_charm

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_wp: Optional[float] = None,
        lambda_light: Optional[float] = None,
        lambda_charm: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        wp = target_wp if target_wp is not None else self.target_wp
        l_l = lambda_light if lambda_light is not None else self.lambda_light
        l_c = lambda_charm if lambda_charm is not None else self.lambda_charm

        ce_loss = self.ce_criterion(logits, labels)

        soft_eps_b, soft_r_l, soft_r_c = self.compute_soft_wp_penalties(logits, labels, wp)

        eff_penalty = torch.abs(soft_eps_b - wp)
        loss = ce_loss + self.lambda_eff * eff_penalty + l_l * soft_r_l + l_c * soft_r_c

        metrics = {
            "loss": float(loss.item()),
            "ce_loss": float(ce_loss.item()),
            "eff_penalty": float(eff_penalty.item()),
            "soft_eps_b": float(soft_eps_b.item()),
            "soft_r_light": float(soft_r_l.item()),
            "soft_r_charm": float(soft_r_c.item()),
            "lambda_light": float(l_l),
            "lambda_charm": float(l_c),
        }
        return loss, metrics


class PIWPCTLoss(WPCTLoss):
    """Physics-Impact Weighted WPCT Loss.

    Identical loss skeleton to WPCTLoss, but lambda_light(w) and lambda_charm(w)
    are injected as read-only scalars from the current state maintained by RiskAllocator.
    The loss module NEVER computes or differentiates through the risk allocation.
    """

    def __init__(
        self,
        target_wp: float = 0.70,
        lambda_eff: float = 2.0,
        lambda_bg: float = 1.0,
        initial_lambda_light: float = 0.5,
        initial_lambda_charm: float = 0.5,
        temperature: float = 0.05,
    ):
        super().__init__(
            target_wp=target_wp,
            lambda_eff=lambda_eff,
            lambda_bg=lambda_bg,
            lambda_light=initial_lambda_light,
            lambda_charm=initial_lambda_charm,
            temperature=temperature,
        )
        self.current_lambda_light = initial_lambda_light
        self.current_lambda_charm = initial_lambda_charm

    def update_weights(self, lambda_light: float, lambda_charm: float) -> None:
        """Inject new scalar weights from validation RiskAllocator.

        Non-differentiable: these are purely scalar constants.
        """
        self.current_lambda_light = float(lambda_light)
        self.current_lambda_charm = float(lambda_charm)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_wp: Optional[float] = None,
        lambda_light: Optional[float] = None,
        lambda_charm: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return super().forward(
            logits,
            labels,
            target_wp=target_wp,
            lambda_light=(self.current_lambda_light if lambda_light is None else lambda_light),
            lambda_charm=(self.current_lambda_charm if lambda_charm is None else lambda_charm),
        )


class ConfTrLoss(nn.Module):
    """Differentiable Conformal Training Loss (Stutz et al., ICLR 2022).

    Optimizes a smooth surrogate of coverage and set-size efficiency.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        temperature: float = 0.1,
        coverage_penalty_weight: float = 1.0,
        size_penalty_weight: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.target_coverage = 1.0 - alpha
        self.temperature = temperature
        self.coverage_weight = coverage_penalty_weight
        self.size_weight = size_penalty_weight
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce_loss = self.ce_criterion(logits, labels)
        probs = F.softmax(logits, dim=-1)  # [B, K]
        B, K = probs.shape

        # Non-conformity score: 1 - p(y|x)
        true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        non_conf = 1.0 - true_class_probs                                    # [B]

        # Target quantile estimated on batch
        tau_val = float(np_quantile_safe(non_conf.detach().cpu().numpy(), self.target_coverage))
        tau = torch.tensor(tau_val, device=logits.device, dtype=logits.dtype)

        # Soft inclusion: sigma((tau - (1 - p)) / T)
        # For true labels: soft coverage
        soft_covered = torch.sigmoid((tau - non_conf) / self.temperature)
        empirical_coverage = soft_covered.mean()
        coverage_loss = (empirical_coverage - self.target_coverage) ** 2

        # Soft set size: sum of soft inclusion across all classes
        all_non_conf = 1.0 - probs  # [B, K]
        soft_set_size = torch.sigmoid((tau - all_non_conf) / self.temperature).sum(dim=-1).mean()

        total_loss = ce_loss + self.coverage_weight * coverage_loss + self.size_weight * soft_set_size

        metrics = {
            "loss": float(total_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "conftr_cov_loss": float(coverage_loss.item()),
            "soft_coverage": float(empirical_coverage.item()),
            "soft_set_size": float(soft_set_size.item()),
        }
        return total_loss, metrics


def np_quantile_safe(arr: np.ndarray, q: float) -> float:
    """Compute quantile with clipping to [0, 1]."""
    if len(arr) == 0:
        return 0.5
    q_clamped = max(0.0, min(1.0, float(q)))
    return float(np.quantile(arr, q_clamped))
