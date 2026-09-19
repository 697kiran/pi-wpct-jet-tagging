"""Detector-mismodelling perturbations and robustness analysis.

Applies standard High-Energy Physics detector mismodelling perturbations:
- Jet Energy Scale (JES) shift: pT -> (1 + delta) * pT
- Jet Energy Resolution (JER) smearing: constituent momenta perturbed by Gaussian noise
- Impact parameter resolution smearing: d0, dz perturbed by Gaussian smearing
- Tracking inefficiency: random constituent track dropping
"""

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from piwpct.conformal import MondrianConformalCalibrator
from piwpct.data import extract_high_level_features, transform_impact_parameters
from piwpct.metrics import compute_working_point_metrics


def apply_jes_shift(
    features: np.ndarray, vectors: np.ndarray, shift_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Jet Energy Scale (JES) systematic shift."""
    feat_shifted = features.copy()
    vec_shifted = vectors.copy()

    # In constituent features: channel 2 is log_pt, channel 3 is log_e
    feat_shifted[:, :, 2] += np.log(shift_factor)
    feat_shifted[:, :, 3] += np.log(shift_factor)

    # In 4-vectors: px, py, pz, E all scale linearly with energy/momentum
    vec_shifted *= shift_factor
    return feat_shifted, vec_shifted


def apply_jer_smearing(
    features: np.ndarray, vectors: np.ndarray, smear_sigma: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Jet Energy Resolution (JER) Gaussian smearing."""
    if smear_sigma <= 0.0:
        return features.copy(), vectors.copy()

    rng = np.random.RandomState(seed)
    noise = rng.normal(1.0, smear_sigma, size=features.shape[:2])
    noise = np.clip(noise, 0.5, 2.0)

    feat_smeared = features.copy()
    vec_smeared = vectors.copy()

    feat_smeared[:, :, 2] += np.log(noise)
    feat_smeared[:, :, 3] += np.log(noise)
    vec_smeared *= noise[:, :, np.newaxis]

    return feat_smeared, vec_smeared


def apply_ip_smearing(
    features: np.ndarray, smear_sigma: float, seed: int = 42
) -> np.ndarray:
    """Apply impact parameter measurement resolution smearing."""
    if smear_sigma <= 0.0:
        return features.copy()

    rng = np.random.RandomState(seed)
    feat_smeared = features.copy()

    # Inverse tanh to recover uncompressed values, add noise, re-apply tanh
    tanh_d0 = feat_smeared[:, :, 4]
    tanh_dz = feat_smeared[:, :, 5]

    d0_raw = np.arctanh(np.clip(tanh_d0, -0.999, 0.999))
    dz_raw = np.arctanh(np.clip(tanh_dz, -0.999, 0.999))

    d0_smeared = d0_raw + rng.normal(0.0, smear_sigma, size=d0_raw.shape)
    dz_smeared = dz_raw + rng.normal(0.0, smear_sigma, size=dz_raw.shape)

    new_tanh_d0, new_tanh_dz = transform_impact_parameters(d0_smeared, dz_smeared)
    feat_smeared[:, :, 4] = new_tanh_d0
    feat_smeared[:, :, 5] = new_tanh_dz

    return feat_smeared


def apply_track_dropping(
    features: np.ndarray, mask: np.ndarray, drop_rate: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate tracking inefficiency by randomly dropping constituent tracks."""
    if drop_rate <= 0.0:
        return features.copy(), mask.copy()

    rng = np.random.RandomState(seed)
    feat_dropped = features.copy()
    mask_dropped = mask.copy()

    drop_mask = rng.binomial(1, 1.0 - drop_rate, size=mask.shape).astype(np.float32)
    new_mask = mask_dropped * drop_mask

    feat_dropped *= new_mask[:, :, np.newaxis]
    return feat_dropped, new_mask


def evaluate_robustness_grid(
    model: torch.nn.Module,
    calibrator: MondrianConformalCalibrator,
    test_data: Dict[str, np.ndarray],
    target_wp: float = 0.70,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate conformal coverage and WP rejection under perturbation sweeps."""
    dev = device or torch.device("cpu")
    model.eval()

    labels = test_data["labels"]
    results: Dict[str, List[Dict[str, float]]] = {
        "jes": [],
        "jer": [],
        "ip_smear": [],
        "track_drop": [],
    }

    # Helper function to run model and conformal evaluator on modified test data
    def _run_perturbed(f, m, v):
        pts = f[:, :, :2].copy()
        hl = extract_high_level_features(f, m)
        with torch.no_grad():
            f_t = torch.from_numpy(f).float().to(dev)
            m_t = torch.from_numpy(m).float().to(dev)
            p_t = torch.from_numpy(pts).float().to(dev)
            v_t = torch.from_numpy(v).float().to(dev)
            hl_t = torch.from_numpy(hl).float().to(dev)
            logits = model(
                features=f_t, points=p_t, mask=m_t, vectors=v_t, high_level=hl_t
            ).cpu().numpy()

        wp_m = compute_working_point_metrics(logits, labels, target_wp=target_wp)
        conf_m = calibrator.evaluate(logits, labels)
        return {
            "marginal_coverage": conf_m.marginal_coverage,
            "mean_set_size": conf_m.mean_set_size,
            "rej_light": wp_m["rej_light"],
            "rej_charm": wp_m["rej_charm"],
            "eps_b": wp_m["eps_b"],
        }

    # 1. JES sweep
    for factor in [0.95, 0.98, 1.00, 1.02, 1.05]:
        f_s, v_s = apply_jes_shift(test_data["features"], test_data["vectors"], factor)
        res = _run_perturbed(f_s, test_data["mask"], v_s)
        res["factor"] = factor
        results["jes"].append(res)

    # 2. JER sweep
    for sigma in [0.0, 0.05, 0.10, 0.15]:
        f_s, v_s = apply_jer_smearing(test_data["features"], test_data["vectors"], sigma)
        res = _run_perturbed(f_s, test_data["mask"], v_s)
        res["sigma"] = sigma
        results["jer"].append(res)

    # 3. IP Smear sweep
    for sigma in [0.0, 0.02, 0.05, 0.10]:
        f_s = apply_ip_smearing(test_data["features"], sigma)
        res = _run_perturbed(f_s, test_data["mask"], test_data["vectors"])
        res["sigma"] = sigma
        results["ip_smear"].append(res)

    # 4. Track drop sweep
    for drop in [0.0, 0.05, 0.10, 0.15]:
        f_s, m_s = apply_track_dropping(test_data["features"], test_data["mask"], drop)
        res = _run_perturbed(f_s, m_s, test_data["vectors"])
        res["drop_rate"] = drop
        results["track_drop"].append(res)

    return results
