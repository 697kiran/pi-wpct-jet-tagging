"""Evaluation metrics and statistical inference utilities for jet flavour tagging.

Includes:
- Working-point metrics (threshold search, mistag rates, rejection).
- Tagging performance (Accuracy, Macro-AUC, ROC curves).
- Calibration (Expected Calibration Error - ECE).
- Statistical uncertainty (non-parametric and paired bootstrap CIs, McNemar test).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_working_point_metrics(
    logits_or_probs: np.ndarray,
    labels: np.ndarray,
    target_wp: float = 0.70,
    is_logits: bool = True,
) -> Dict[str, float]:
    """Compute b-efficiency, mistag rates, and background rejections at target WP."""
    if is_logits:
        exp_l = np.exp(logits_or_probs - np.max(logits_or_probs, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    else:
        probs = logits_or_probs

    b_scores = probs[:, 2]

    is_b = (labels == 2)
    is_light = (labels == 0)
    is_charm = (labels == 1)

    n_b = int(np.sum(is_b))
    n_light = int(np.sum(is_light))
    n_charm = int(np.sum(is_charm))

    if n_b == 0 or n_light == 0 or n_charm == 0:
        return {
            "target_wp": target_wp,
            "threshold": 0.5,
            "eps_b": 0.0,
            "r_light": 1.0,
            "r_charm": 1.0,
            "rej_light": 1.0,
            "rej_charm": 1.0,
        }

    # Threshold where fraction target_wp of true b-jets pass
    threshold = float(np.quantile(b_scores[is_b], 1.0 - target_wp))

    passed = (b_scores >= threshold)
    eps_b = float(np.sum(passed & is_b) / n_b)
    r_light = float(np.sum(passed & is_light) / n_light)
    r_charm = float(np.sum(passed & is_charm) / n_charm)

    rej_light = float(1.0 / r_light) if r_light > 0 else 1e6
    rej_charm = float(1.0 / r_charm) if r_charm > 0 else 1e6

    return {
        "target_wp": target_wp,
        "threshold": threshold,
        "eps_b": eps_b,
        "r_light": r_light,
        "r_charm": r_charm,
        "rej_light": rej_light,
        "rej_charm": rej_charm,
    }


def compute_classification_metrics(
    logits: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    """Compute overall classification metrics: accuracy, macro-AUC, ovr-AUC."""
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    acc = float(accuracy_score(labels, preds))

    # Multi-class One-vs-Rest AUC
    try:
        macro_auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = 0.5

    return {
        "accuracy": acc,
        "macro_auc": macro_auc,
        "predictions": preds,
        "probabilities": probs,
    }


def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_bootstrap_ci(
    logits: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute non-parametric bootstrap confidence intervals for a metric function.

    Returns:
        dict of metric_name -> (mean_val, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(labels)
    boot_results: Dict[str, List[float]] = {}

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, size=n_samples)
        boot_logits = logits[boot_idx]
        boot_labels = labels[boot_idx]

        metrics = metric_fn(boot_logits, boot_labels)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                if k not in boot_results:
                    boot_results[k] = []
                boot_results[k].append(float(v))

    alpha = 1.0 - ci_level
    summary = {}
    for k, values in boot_results.items():
        if len(values) > 0:
            arr = np.array(values)
            mean_val = float(np.mean(arr))
            low_ci = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
            high_ci = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
            summary[k] = (mean_val, low_ci, high_ci)

    return summary


def paired_bootstrap_comparison(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Paired bootstrap comparison between two regimes (e.g. PI-WPCT vs WPCT).

    Evaluates whether Delta = Metric_B - Metric_A is statistically significant
    (i.e. whether the (1 - alpha)% CI excludes 0).
    """
    rng = np.random.RandomState(seed)
    n_samples = len(labels)
    diffs: Dict[str, List[float]] = {}

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, size=n_samples)
        l_a = logits_a[boot_idx]
        l_b = logits_b[boot_idx]
        y_b = labels[boot_idx]

        m_a = metric_fn(l_a, y_b)
        m_b = metric_fn(l_b, y_b)

        for k in m_a.keys():
            if k in m_b and isinstance(m_a[k], (int, float)) and isinstance(m_b[k], (int, float)):
                diff = float(m_b[k] - m_a[k])
                if k not in diffs:
                    diffs[k] = []
                diffs[k].append(diff)

    alpha = 1.0 - ci_level
    summary = {}
    for k, d_vals in diffs.items():
        arr = np.array(d_vals)
        mean_diff = float(np.mean(arr))
        low = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
        high = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
        # Significant if CI does not contain zero
        is_significant = (low > 0.0) or (high < 0.0)
        summary[k] = {
            "mean_diff": mean_diff,
            "ci_lower": low,
            "ci_upper": high,
            "is_significant": is_significant,
        }

    return summary


def mcnemar_test(
    preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute McNemar's test for paired classification predictions."""
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # Contingency table
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    total_discordant = b + c
    if total_discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # Edwards' continuity correction
        stat = float(((abs(b - c) - 1.0) ** 2) / total_discordant)
        p_val = float(stats.chi2.sf(stat, df=1))

    return {
        "mcnemar_statistic": stat,
        "p_value": p_val,
        "n_a_only_correct": b,
        "n_b_only_correct": c,
    }
