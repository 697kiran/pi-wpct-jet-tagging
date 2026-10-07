"""Detector-mismodelling perturbations and robustness analysis.

Applies standard High-Energy Physics detector mismodelling perturbations:
- Jet Energy Scale (JES) shift: pT -> (1 + delta) * pT
- Jet Energy Resolution (JER) smearing: constituent momenta perturbed by Gaussian noise
- Impact parameter resolution smearing: d0, dz perturbed by Gaussian smearing
- Tracking inefficiency: random constituent track dropping
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from piwpct.conformal import MondrianConformalCalibrator
from piwpct.data import extract_high_level_features, transform_impact_parameters
from piwpct.metrics import compute_working_point_metrics
from piwpct.schema import FEATURE_INDEX, feature_index
from piwpct.train import apply_temperature


def apply_jes_shift(
    features: np.ndarray, vectors: np.ndarray, shift_factor: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Jet Energy Scale (JES) systematic shift."""
    if shift_factor <= 0.0:
        raise ValueError("JES shift_factor must be positive.")
    feat_shifted = features.copy()
    vec_shifted = vectors.copy()
    valid = mask if mask is not None else np.ones(features.shape[:2], dtype=np.float32)
    log_shift = np.log(shift_factor) * 0.7 * valid
    feat_shifted[:, :, feature_index("part_pt_log")] += log_shift
    feat_shifted[:, :, feature_index("part_e_log")] += log_shift
    # Relative pT/energy features are invariant under a common jet/constituent scale.
    vec_shifted *= shift_factor
    return feat_shifted, vec_shifted


def apply_jer_smearing(
    features: np.ndarray, vectors: np.ndarray, smear_sigma: float, seed: int = 42,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Jet Energy Resolution (JER) Gaussian smearing."""
    if smear_sigma <= 0.0:
        return features.copy(), vectors.copy()

    rng = np.random.RandomState(seed)
    noise = rng.normal(1.0, smear_sigma, size=features.shape[:2])
    noise = np.clip(noise, 0.5, 2.0)
    valid = mask if mask is not None else np.ones(features.shape[:2], dtype=np.float32)
    log_noise = np.log(noise) * 0.7 * valid

    feat_smeared = features.copy()
    vec_smeared = vectors.copy()
    feat_smeared[:, :, feature_index("part_pt_log")] += log_noise
    feat_smeared[:, :, feature_index("part_e_log")] += log_noise
    feat_smeared[:, :, feature_index("part_logptrel")] += log_noise
    feat_smeared[:, :, feature_index("part_logerel")] += log_noise
    vec_smeared *= noise[:, :, np.newaxis]

    return feat_smeared, vec_smeared


def apply_ip_smearing(
    features: np.ndarray, smear_sigma: float, seed: int = 42,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply impact parameter measurement resolution smearing."""
    if smear_sigma <= 0.0:
        return features.copy()

    rng = np.random.RandomState(seed)
    feat_smeared = features.copy()
    valid = mask if mask is not None else np.ones(features.shape[:2], dtype=np.float32)

    # Inverse tanh to recover uncompressed values, add noise, re-apply tanh
    d0_index = feature_index("part_d0")
    dz_index = feature_index("part_dz")
    tanh_d0 = feat_smeared[:, :, d0_index]
    tanh_dz = feat_smeared[:, :, dz_index]

    d0_raw = np.arctanh(np.clip(tanh_d0, -0.999, 0.999))
    dz_raw = np.arctanh(np.clip(tanh_dz, -0.999, 0.999))

    d0_smeared = d0_raw + rng.normal(0.0, smear_sigma, size=d0_raw.shape)
    dz_smeared = dz_raw + rng.normal(0.0, smear_sigma, size=dz_raw.shape)

    new_tanh_d0, new_tanh_dz = transform_impact_parameters(d0_smeared, dz_smeared)
    feat_smeared[:, :, d0_index] = new_tanh_d0 * valid
    feat_smeared[:, :, dz_index] = new_tanh_dz * valid

    return feat_smeared


def apply_track_dropping(
    features: np.ndarray, mask: np.ndarray, drop_rate: float, seed: int = 42,
    vectors: Optional[np.ndarray] = None,
) -> Any:
    """Simulate tracking inefficiency by randomly dropping constituent tracks."""
    if drop_rate <= 0.0:
        if vectors is not None:
            return features.copy(), mask.copy(), vectors.copy()
        return features.copy(), mask.copy()

    rng = np.random.RandomState(seed)
    feat_dropped = features.copy()
    mask_dropped = mask.copy()

    drop_mask = rng.binomial(1, 1.0 - drop_rate, size=mask.shape).astype(np.float32)
    new_mask = mask_dropped * drop_mask

    feat_dropped *= new_mask[:, :, np.newaxis]
    if vectors is not None:
        vectors_dropped = vectors.copy() * new_mask[:, :, np.newaxis]
        return feat_dropped, new_mask, vectors_dropped
    return feat_dropped, new_mask


def evaluate_robustness_grid(
    model: torch.nn.Module,
    calibrator: MondrianConformalCalibrator,
    test_data: Dict[str, np.ndarray],
    target_wp: float = 0.70,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
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
        pts = f[:, :, [FEATURE_INDEX["part_deta"], FEATURE_INDEX["part_dphi"]]].copy()
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
        logits = apply_temperature(logits, temperature)

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
        f_s, v_s = apply_jes_shift(test_data["features"], test_data["vectors"], factor, test_data["mask"])
        res = _run_perturbed(f_s, test_data["mask"], v_s)
        res["factor"] = factor
        results["jes"].append(res)

    # 2. JER sweep
    for sigma in [0.0, 0.05, 0.10, 0.15]:
        f_s, v_s = apply_jer_smearing(test_data["features"], test_data["vectors"], sigma, mask=test_data["mask"])
        res = _run_perturbed(f_s, test_data["mask"], v_s)
        res["sigma"] = sigma
        results["jer"].append(res)

    # 3. IP Smear sweep
    for sigma in [0.0, 0.02, 0.05, 0.10]:
        f_s = apply_ip_smearing(test_data["features"], sigma, mask=test_data["mask"])
        res = _run_perturbed(f_s, test_data["mask"], test_data["vectors"])
        res["sigma"] = sigma
        results["ip_smear"].append(res)

    # 4. Track drop sweep
    for drop in [0.0, 0.05, 0.10, 0.15]:
        f_s, m_s, v_s = apply_track_dropping(
            test_data["features"], test_data["mask"], drop, vectors=test_data["vectors"]
        )
        res = _run_perturbed(f_s, m_s, v_s)
        res["drop_rate"] = drop
        results["track_drop"].append(res)

    return results


def _streaming_metrics(
    model: torch.nn.Module,
    calibrator: MondrianConformalCalibrator,
    data_loader: Any,
    device: torch.device,
    *,
    target_wp: float,
    temperature: float,
    perturbation: str,
    amount: float,
    seed: int,
) -> Dict[str, float]:
    """Score one perturbation in batches without materializing the test jets."""
    b_scores: List[np.ndarray] = []
    light_scores: List[np.ndarray] = []
    charm_scores: List[np.ndarray] = []
    correct_count = 0
    total_count = 0
    set_size_total = 0
    singleton_count = 0
    empty_count = 0

    model.eval()
    for batch_index, batch in enumerate(data_loader):
        features = batch["features"].cpu().numpy()
        vectors = batch["vectors"].cpu().numpy()
        mask = batch["mask"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        batch_seed = seed + batch_index
        if perturbation == "jes":
            features, vectors = apply_jes_shift(features, vectors, amount, mask)
        elif perturbation == "jer":
            features, vectors = apply_jer_smearing(features, vectors, amount, batch_seed, mask)
        elif perturbation == "ip_smear":
            features = apply_ip_smearing(features, amount, batch_seed, mask)
        elif perturbation == "track_drop":
            features, mask, vectors = apply_track_dropping(
                features, mask, amount, batch_seed, vectors=vectors
            )

        points = features[:, :, [feature_index("part_deta"), feature_index("part_dphi")]].copy()
        high_level = extract_high_level_features(features, mask)
        with torch.no_grad():
            logits = model(
                features=torch.from_numpy(features).float().to(device),
                points=torch.from_numpy(points).float().to(device),
                mask=torch.from_numpy(mask).float().to(device),
                vectors=torch.from_numpy(vectors).float().to(device),
                high_level=torch.from_numpy(high_level).float().to(device),
            ).detach().cpu().numpy()
        logits = apply_temperature(logits, temperature)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        class2_scores = probabilities[:, 2]
        b_scores.append(class2_scores[labels == 2].astype(np.float32, copy=False))
        light_scores.append(class2_scores[labels == 0].astype(np.float32, copy=False))
        charm_scores.append(class2_scores[labels == 1].astype(np.float32, copy=False))

        prediction_sets = calibrator.predict(logits)
        total_count += len(labels)
        correct_count += sum(int(label in prediction_sets[i]) for i, label in enumerate(labels))
        set_sizes = [len(values) for values in prediction_sets]
        set_size_total += sum(set_sizes)
        singleton_count += sum(size == 1 for size in set_sizes)
        empty_count += sum(size == 0 for size in set_sizes)

    b = np.concatenate(b_scores) if b_scores else np.empty(0, dtype=np.float32)
    light = np.concatenate(light_scores) if light_scores else np.empty(0, dtype=np.float32)
    charm = np.concatenate(charm_scores) if charm_scores else np.empty(0, dtype=np.float32)
    if not len(b) or not len(light) or not len(charm):
        raise ValueError("Flavour robustness evaluation requires non-empty light, charm, and b test classes.")
    threshold = float(np.quantile(b, 1.0 - target_wp))
    eps_b = float(np.mean(b >= threshold))
    r_light = float(np.mean(light >= threshold))
    r_charm = float(np.mean(charm >= threshold))
    return {
        "marginal_coverage": correct_count / max(total_count, 1),
        "mean_set_size": set_size_total / max(total_count, 1),
        "singleton_rate": singleton_count / max(total_count, 1),
        "empty_set_rate": empty_count / max(total_count, 1),
        "accuracy": float("nan"),
        "rej_light": 1.0 / r_light if r_light > 0 else 1e6,
        "rej_charm": 1.0 / r_charm if r_charm > 0 else 1e6,
        "eps_b": eps_b,
    }


def evaluate_robustness_loader(
    model: torch.nn.Module,
    calibrator: MondrianConformalCalibrator,
    data_loader: Any,
    target_wp: float = 0.70,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Run the standard perturbation grid over a lazy prepared-data loader."""
    dev = device or torch.device("cpu")
    grid = {
        "jes": ("factor", [0.95, 0.98, 1.00, 1.02, 1.05]),
        "jer": ("sigma", [0.0, 0.05, 0.10, 0.15]),
        "ip_smear": ("sigma", [0.0, 0.02, 0.05, 0.10]),
        "track_drop": ("drop_rate", [0.0, 0.05, 0.10, 0.15]),
    }
    results: Dict[str, List[Dict[str, float]]] = {name: [] for name in grid}
    for name, (field, values) in grid.items():
        for index, amount in enumerate(values):
            res = _streaming_metrics(
                model, calibrator, data_loader, dev,
                target_wp=target_wp,
                temperature=temperature,
                perturbation=name,
                amount=amount,
                seed=42 + index * 1000,
            )
            res[field] = amount
            results[name].append(res)
    return results
