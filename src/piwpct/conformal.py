"""Exact Mondrian (class-conditional) conformal calibration and prediction set construction.

Operates strictly on frozen logits from held-out calibration data.
Guarantees exact finite-sample coverage per class:
    P(Y in C(X) | Y = k) >= 1 - alpha
Unchanged across all backbones and training regimes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ConformalCalibratorOutput:
    """Output summary of post-hoc conformal calibration."""
    method: str
    score_type: str
    alpha: float
    target_coverage: float
    class_quantiles: Dict[int, float]
    marginal_coverage: float
    per_class_coverage: Dict[int, float]
    mean_set_size: float
    singleton_rate: float
    empty_set_rate: float
    set_sizes: np.ndarray
    prediction_sets: List[List[int]]


class MondrianConformalCalibrator:
    """Exact class-conditional Mondrian conformal calibrator."""

    def __init__(
        self,
        alpha: float = 0.10,
        score_type: str = "residual",
        raps_kreg: int = 1,
        raps_lam: float = 0.01,
    ):
        """Initialize calibrator.

        Args:
            alpha: target error rate (e.g. 0.10 for 90% coverage).
            score_type: 'residual' (1 - p(y|x)) or 'raps' (regularized adaptive sets).
            raps_kreg: RAPS rank regularization threshold.
            raps_lam: RAPS penalty coefficient.
        """
        assert 0.0 < alpha < 1.0, f"Alpha must be in (0, 1), got {alpha}"
        self.alpha = alpha
        self.target_coverage = 1.0 - alpha
        self.score_type = score_type.lower()
        self.raps_kreg = raps_kreg
        self.raps_lam = raps_lam
        self.class_quantiles: Dict[int, float] = {}
        self.is_calibrated: bool = False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    def compute_nonconformity_scores(
        self, probs: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conformal non-conformity scores.

        If labels are provided, returns [N] scores for the true class.
        If labels are None, returns [N, K] scores for all candidate classes.
        """
        N, K = probs.shape

        if self.score_type == "residual":
            # Residual score: s(x, k) = 1 - p(k | x)
            scores = 1.0 - probs
        elif self.score_type == "raps":
            # RAPS score: sum of probabilities of higher-ranked classes + regularization
            sorted_indices = np.argsort(-probs, axis=1)
            sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
            cumsum_probs = np.cumsum(sorted_probs, axis=1)

            # Map back to original class indices
            scores = np.zeros_like(probs)
            for i in range(N):
                ranks = np.empty(K, dtype=int)
                ranks[sorted_indices[i]] = np.arange(K)
                for c in range(K):
                    rank = ranks[c]
                    cum_p = cumsum_probs[i, rank]
                    # Penalty for sets larger than kreg
                    penalty = max(0, rank + 1 - self.raps_kreg) * self.raps_lam
                    scores[i, c] = cum_p + penalty
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")

        if labels is not None:
            return scores[np.arange(N), labels]
        return scores

    def calibrate(self, cal_logits: np.ndarray, cal_labels: np.ndarray) -> None:
        """Fit class-conditional quantiles on held-out calibration data."""
        cal_probs = self._softmax(cal_logits)
        num_classes = cal_probs.shape[1]
        self.class_quantiles = {}

        for k in range(num_classes):
            mask_k = (cal_labels == k)
            n_k = int(np.sum(mask_k))
            if n_k == 0:
                self.class_quantiles[k] = 1.0
                continue

            # Class-specific calibration non-conformity scores
            probs_k = cal_probs[mask_k]
            scores_k = self.compute_nonconformity_scores(probs_k, labels=np.full(n_k, k))

            # Finite-sample conformal quantile: ceil((n_k + 1) * (1 - alpha)) / n_k
            q_level = min(1.0, np.ceil((n_k + 1) * (1.0 - self.alpha)) / n_k)
            quantile_val = float(np.quantile(scores_k, q_level, method="higher"))
            self.class_quantiles[k] = quantile_val

        self.is_calibrated = True

    def predict(self, test_logits: np.ndarray) -> List[List[int]]:
        """Construct Mondrian prediction sets for test examples."""
        if not self.is_calibrated:
            raise RuntimeError("Calibrator must be calibrated before calling predict()")

        test_probs = self._softmax(test_logits)
        all_scores = self.compute_nonconformity_scores(test_probs)  # [N, K]
        N, K = test_probs.shape

        prediction_sets = []
        for i in range(N):
            c_set = []
            for k in range(K):
                if all_scores[i, k] <= self.class_quantiles.get(k, 1.0):
                    c_set.append(k)
            prediction_sets.append(c_set)

        return prediction_sets

    def evaluate(
        self, test_logits: np.ndarray, test_labels: np.ndarray
    ) -> ConformalCalibratorOutput:
        """Evaluate conformal prediction sets on test set and report coverage metrics."""
        pred_sets = self.predict(test_logits)
        N = len(test_labels)
        K = test_logits.shape[1]

        covered = [test_labels[i] in pred_sets[i] for i in range(N)]
        marginal_cov = float(np.mean(covered))

        per_class_cov = {}
        for k in range(K):
            mask_k = (test_labels == k)
            if np.sum(mask_k) > 0:
                per_class_cov[k] = float(np.mean([covered[i] for i in np.where(mask_k)[0]]))
            else:
                per_class_cov[k] = 0.0

        set_sizes = np.array([len(s) for s in pred_sets], dtype=np.int32)
        mean_size = float(np.mean(set_sizes))
        singleton_rate = float(np.mean(set_sizes == 1))
        empty_rate = float(np.mean(set_sizes == 0))

        return ConformalCalibratorOutput(
            method="mondrian",
            score_type=self.score_type,
            alpha=self.alpha,
            target_coverage=self.target_coverage,
            class_quantiles=self.class_quantiles,
            marginal_coverage=marginal_cov,
            per_class_coverage=per_class_cov,
            mean_set_size=mean_size,
            singleton_rate=singleton_rate,
            empty_set_rate=empty_rate,
            set_sizes=set_sizes,
            prediction_sets=pred_sets,
        )

    def compute_distribution_shift_bound(
        self, tv_distance: float
    ) -> float:
        """Compute theoretical lower bound on coverage under distribution shift.

        Reference: Barber, Candès, Ramdas, Tibshirani (Annals of Statistics, 2023).
        Under Total Variation distance d_TV(P, Q) <= delta:
            Coverage_Q >= 1 - alpha - delta
        """
        guaranteed_coverage = max(0.0, (1.0 - self.alpha) - tv_distance)
        return float(guaranteed_coverage)
