"""Periodic, non-differentiable physics-impact risk allocation for PI-WPCT.

Recomputes scalar loss weights lambda_light(w) and lambda_charm(w) based on
downstream b-jet purity sensitivity and validation mistag rate variance.

Runs strictly under torch.no_grad() on held-out validation data only.
Calibration and test splits are strictly forbidden.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from piwpct.utils.split_integrity import SplitIntegrityViolation


@dataclass
class RiskAllocationResult:
    """Complete diagnostic record of a risk allocation refresh."""
    target_wp: float
    b_efficiency: float
    threshold: float
    r_light: float
    r_charm: float
    rejection_light: float
    rejection_charm: float
    var_r_light: float
    var_r_charm: float
    s_yield: float
    b_light_yield: float
    b_charm_yield: float
    purity: float
    dp_dr_light: float
    dp_dr_charm: float
    r_light_risk: float
    r_charm_risk: float
    risk_ratio: float
    lambda_light: float
    lambda_charm: float
    lambda_bg: float
    stabilization_applied: bool
    n_val_total: int
    n_b_val: int
    n_light_val: int
    n_charm_val: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskAllocator:
    """Periodic, validation-only risk allocator for PI-WPCT."""

    def __init__(
        self,
        lambda_bg: float = 1.0,
        variance_stabilization: str = "wilson_floor",
        min_variance_floor: float = 1.0e-7,
        z_score: float = 1.96,
        zero_sum_guard_epsilon: float = 1.0e-12,
        fixed_yields: Optional[Dict[str, float]] = None,
    ):
        """Initialize risk allocator with predeclared config parameters.

        Args:
            lambda_bg: Total weight allocated across light and charm mistags.
            variance_stabilization: Stabilization method ('wilson_floor', 'min_floor', or 'binomial').
            min_variance_floor: Lower bound for variance estimates to prevent zero or unstable risks.
            z_score: Normal quantile for Wilson score interval regularization.
            zero_sum_guard_epsilon: Guard against division by near-zero risk sum.
            fixed_yields: Optional physics cross-section yields for (light, charm, b).
        """
        self.lambda_bg = lambda_bg
        self.variance_stabilization = variance_stabilization
        self.min_variance_floor = min_variance_floor
        self.z_score = z_score
        self.zero_sum_guard_epsilon = zero_sum_guard_epsilon
        self.fixed_yields = fixed_yields or {"light": 100000.0, "charm": 25000.0, "b": 25000.0}

    def compute_variance(
        self, k: int, n: int
    ) -> Tuple[float, bool]:
        """Estimate variance of mistag rate r = k / n with documented stabilization.

        Args:
            k: number of passing background jets.
            n: total number of background jets.

        Returns:
            var_r: estimated and stabilized variance.
            stabilization_applied: whether floor/Wilson adjustment modified raw variance.
        """
        if n <= 0:
            return self.min_variance_floor, True

        raw_r = k / n
        raw_var = (raw_r * (1.0 - raw_r)) / n

        stabilization_applied = False

        if self.variance_stabilization == "wilson_floor":
            # Wilson-interval adjusted variance
            z2 = self.z_score ** 2
            n_tilde = n + z2
            r_tilde = (k + 0.5 * z2) / n_tilde
            wilson_var = (r_tilde * (1.0 - r_tilde)) / n_tilde
            final_var = max(wilson_var, self.min_variance_floor)
            stabilization_applied = (final_var != raw_var)
        elif self.variance_stabilization == "min_floor":
            final_var = max(raw_var, self.min_variance_floor)
            stabilization_applied = (final_var != raw_var)
        else:
            final_var = max(raw_var, self.min_variance_floor)
            stabilization_applied = (final_var != raw_var)

        return float(final_var), stabilization_applied

    @torch.no_grad()
    def evaluate_validation_logits(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run validation split forward to obtain logits and true labels.

        CRITICAL LEAKAGE ASSERTION: Validates split identity.
        """
        # Audit dataset split name if available
        if hasattr(val_loader.dataset, "split_name"):
            split_name = val_loader.dataset.split_name.lower()
            if split_name in {"calibration", "cal"}:
                raise SplitIntegrityViolation(
                    "FATAL ERROR: RiskAllocator was passed the calibration split! "
                    "Calibration split must remain completely isolated."
                )
            if split_name in {"test", "testing"}:
                raise SplitIntegrityViolation(
                    "FATAL ERROR: RiskAllocator was passed the test split! "
                    "Test split cannot be used for risk updates."
                )

        model.eval()
        all_logits = []
        all_labels = []

        for batch in val_loader:
            features = batch["features"].to(device)
            points = batch["points"].to(device) if "points" in batch else None
            mask = batch["mask"].to(device) if "mask" in batch else None
            vectors = batch["vectors"].to(device) if "vectors" in batch else None
            high_level = batch["high_level"].to(device) if "high_level" in batch else None

            # Handle GBT or PyTorch models uniformly
            logits = model(
                features=features,
                points=points,
                mask=mask,
                vectors=vectors,
                high_level=high_level,
            )
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        return logits_arr, labels_arr

    def compute_risk_allocation(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        target_wp: float,
        yields: Optional[Dict[str, float]] = None,
    ) -> RiskAllocationResult:
        """Core mathematical risk allocation calculation from validation logits and labels.

        Classes: 0: light, 1: charm, 2: b.
        """
        assert 0.0 < target_wp < 1.0, f"Target WP must be in (0, 1), got {target_wp}"

        y_priors = yields or self.fixed_yields
        y_light = float(y_priors.get("light", 100000.0))
        y_charm = float(y_priors.get("charm", 25000.0))
        y_b = float(y_priors.get("b", 25000.0))

        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        b_scores = probs[:, 2]

        is_b = (labels == 2)
        is_light = (labels == 0)
        is_charm = (labels == 1)

        n_b = int(np.sum(is_b))
        n_light = int(np.sum(is_light))
        n_charm = int(np.sum(is_charm))
        n_total = len(labels)

        if n_b == 0 or n_light == 0 or n_charm == 0:
            raise ValueError(
                f"Validation split missing classes: n_b={n_b}, n_light={n_light}, n_charm={n_charm}"
            )

        # 1. Determine threshold tau_w for target b-efficiency w
        # Quantile (1 - w) of true b scores gives threshold where fraction w passes
        b_scores_true = b_scores[is_b]
        threshold = float(np.quantile(b_scores_true, 1.0 - target_wp))

        # Passing mask
        passed = (b_scores >= threshold)

        # 2. Compute b-efficiency and mistag rates
        b_pass = int(np.sum(passed & is_b))
        light_pass = int(np.sum(passed & is_light))
        charm_pass = int(np.sum(passed & is_charm))

        eps_b = b_pass / n_b
        r_light = light_pass / n_light
        r_charm = charm_pass / n_charm

        rej_light = 1.0 / r_light if r_light > 0 else float("inf")
        rej_charm = 1.0 / r_charm if r_charm > 0 else float("inf")

        # 3. Variance estimation with stabilization
        var_r_light, stab_l = self.compute_variance(light_pass, n_light)
        var_r_charm, stab_c = self.compute_variance(charm_pass, n_charm)
        stabilization_applied = stab_l or stab_c

        # 4. Yields and Purity proxy U = P(w)
        s_yield = y_b * eps_b
        b_light_yield = y_light * r_light
        b_charm_yield = y_charm * r_charm
        total_selected = s_yield + b_light_yield + b_charm_yield

        purity = s_yield / max(total_selected, 1e-12)

        # 5. Closed-form analytic sensitivities dP/dr_i
        # dP/dr_i = - S * Y_i / [S + Y_l*r_l + Y_c*r_c]^2
        denom_sq = max(total_selected ** 2, 1e-12)
        dp_dr_light = - (s_yield * y_light) / denom_sq
        dp_dr_charm = - (s_yield * y_charm) / denom_sq

        # 6. Risk contributions: R_i(w) = [dP/dr_i]^2 * Var[r_i(w)]
        r_light_risk = (dp_dr_light ** 2) * var_r_light
        r_charm_risk = (dp_dr_charm ** 2) * var_r_charm
        total_risk = r_light_risk + r_charm_risk

        # 7. Normalized loss weights with zero-sum division guard
        if total_risk < self.zero_sum_guard_epsilon:
            # Fallback to equal allocation if risk sums to zero
            lambda_l = 0.5 * self.lambda_bg
            lambda_c = 0.5 * self.lambda_bg
            risk_ratio = 1.0
        else:
            lambda_l = self.lambda_bg * (r_light_risk / total_risk)
            lambda_c = self.lambda_bg * (r_charm_risk / total_risk)
            risk_ratio = r_light_risk / max(r_charm_risk, self.zero_sum_guard_epsilon)

        return RiskAllocationResult(
            target_wp=target_wp,
            b_efficiency=eps_b,
            threshold=threshold,
            r_light=r_light,
            r_charm=r_charm,
            rejection_light=rej_light,
            rejection_charm=rej_charm,
            var_r_light=var_r_light,
            var_r_charm=var_r_charm,
            s_yield=s_yield,
            b_light_yield=b_light_yield,
            b_charm_yield=b_charm_yield,
            purity=purity,
            dp_dr_light=dp_dr_light,
            dp_dr_charm=dp_dr_charm,
            r_light_risk=r_light_risk,
            r_charm_risk=r_charm_risk,
            risk_ratio=risk_ratio,
            lambda_light=lambda_l,
            lambda_charm=lambda_c,
            lambda_bg=self.lambda_bg,
            stabilization_applied=stabilization_applied,
            n_val_total=n_total,
            n_b_val=n_b,
            n_light_val=n_light,
            n_charm_val=n_charm,
        )

    def refresh(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        target_wp: float,
        device: torch.device,
        yields: Optional[Dict[str, float]] = None,
    ) -> RiskAllocationResult:
        """Public refresh method called at epoch boundaries during PI-WPCT training.

        Freezes model, runs forward on validation only, computes risk, and returns weights.
        """
        logits, labels = self.evaluate_validation_logits(model, val_loader, device)
        return self.compute_risk_allocation(logits, labels, target_wp, yields)
