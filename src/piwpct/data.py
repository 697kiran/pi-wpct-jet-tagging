"""Data ingestion, preprocessing, and canonical 4-way split management.

Handles:
- JetClass 3-class filtering (light=0, charm=1, b=2).
- tanh(d0) and tanh(dz) impact parameter transformations.
- Generation and hashing of the canonical 4-way split (train/val/cal/test).
- Pilot mode with provisional 4:1:1 class ratio explicitly flagged.
- Sharded and in-memory PyTorch Dataset implementations.
"""

import hashlib
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from piwpct.utils.split_integrity import SplitAuditor, check_splits_disjoint

CLASS_NAMES = ["light", "charm", "b"]
CLASS_TO_IDX = {"light": 0, "charm": 1, "b": 2}
IDX_TO_CLASS = {0: "light", 1: "charm", 2: "b"}


def transform_impact_parameters(
    d0: np.ndarray, dz: np.ndarray, clip_val: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply tanh(d0) and tanh(dz) non-linear transformations for numerical stability.

    High-energy particles at colliders exhibit long tails in impact parameters.
    Applying tanh compresses extreme outliers while maintaining linearity near zero.
    """
    d0_clipped = np.clip(d0, -clip_val, clip_val)
    dz_clipped = np.clip(dz, -clip_val, clip_val)
    return np.tanh(d0_clipped), np.tanh(dz_clipped)


def extract_high_level_features(
    features: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Compute high-level summary features per jet from constituent arrays for GBT.

    Args:
        features: [N_jets, N_particles, C] array of constituent features.
            Expected channels (pilot / JetClass mapping):
            0: d_eta, 1: d_phi, 2: log_pt, 3: log_E, 4: tanh_d0, 5: tanh_dz, ...
        mask: [N_jets, N_particles] boolean/float mask.

    Returns:
        high_level: [N_jets, D] high-level jet features.
    """
    n_jets = features.shape[0]
    mask_bool = mask.astype(bool)

    high_level = []
    for i in range(n_jets):
        valid = mask_bool[i]
        n_constituents = np.sum(valid)
        if n_constituents == 0:
            high_level.append(np.zeros(12, dtype=np.float32))
            continue

        f_valid = features[i, valid]
        d_eta = f_valid[:, 0]
        d_phi = f_valid[:, 1]
        log_pt = f_valid[:, 2]
        tanh_d0 = f_valid[:, 4] if f_valid.shape[1] > 4 else np.zeros(n_constituents)
        tanh_dz = f_valid[:, 5] if f_valid.shape[1] > 5 else np.zeros(n_constituents)

        pt_weights = np.exp(log_pt)
        sum_pt = np.sum(pt_weights) + 1e-8
        norm_pt = pt_weights / sum_pt

        # Summary observables: constituent count, pT dispersion, impact parameter moments
        jet_width_eta = np.sqrt(np.sum(norm_pt * (d_eta ** 2)))
        jet_width_phi = np.sqrt(np.sum(norm_pt * (d_phi ** 2)))
        pt_lead_frac = np.max(pt_weights) / sum_pt
        mean_tanh_d0 = np.mean(tanh_d0)
        std_tanh_d0 = np.std(tanh_d0)
        max_tanh_d0 = np.max(np.abs(tanh_d0))
        mean_tanh_dz = np.mean(tanh_dz)
        std_tanh_dz = np.std(tanh_dz)
        max_tanh_dz = np.max(np.abs(tanh_dz))

        high_level.append([
            float(n_constituents),
            float(sum_pt),
            float(jet_width_eta),
            float(jet_width_phi),
            float(pt_lead_frac),
            float(mean_tanh_d0),
            float(std_tanh_d0),
            float(max_tanh_d0),
            float(mean_tanh_dz),
            float(std_tanh_dz),
            float(max_tanh_dz),
            float(np.mean(log_pt)),
        ])

    return np.array(high_level, dtype=np.float32)


def generate_pilot_dataset(
    n_samples: int = 12000,
    num_particles: int = 32,
    feature_dim: int = 16,
    seed: int = 42,
    class_ratio: Tuple[int, int, int] = (4, 1, 1),
) -> Dict[str, np.ndarray]:
    """Generate a reproducible, physics-inspired pilot dataset.

    Provisional class ratio: ~4:1:1 (light:charm:b).
    Explicitly labeled as 'pilot, not final LHC composition'.

    Physics parameters emulated:
    - b-jets: displaced secondary vertices (larger d0, dz), higher track multiplicity.
    - c-jets: moderate decay length (intermediate d0, dz).
    - light-jets: prompt tracks centered at 0 with small measurement resolution.
    """
    rng = np.random.RandomState(seed)

    ratio_sum = sum(class_ratio)
    n_light = int(n_samples * class_ratio[0] / ratio_sum)
    n_charm = int(n_samples * class_ratio[1] / ratio_sum)
    n_b = n_samples - n_light - n_charm

    labels = np.array([0] * n_light + [1] * n_charm + [2] * n_b, dtype=np.int64)
    # Shuffle order while keeping track of indices
    perm = rng.permutation(n_samples)
    labels = labels[perm]

    features = np.zeros((n_samples, num_particles, feature_dim), dtype=np.float32)
    vectors = np.zeros((n_samples, num_particles, 4), dtype=np.float32)
    mask = np.zeros((n_samples, num_particles), dtype=np.float32)

    for i in range(n_samples):
        cls = labels[i]
        # Multiplicity depends on flavour
        if cls == 2:    # b-jet
            n_const = rng.randint(16, num_particles + 1)
            ip_scale = 1.2
        elif cls == 1:  # charm-jet
            n_const = rng.randint(12, num_particles - 2)
            ip_scale = 0.5
        else:           # light-jet
            n_const = rng.randint(8, num_particles - 6)
            ip_scale = 0.05

        mask[i, :n_const] = 1.0

        # Coordinates Delta eta, Delta phi
        d_eta = rng.normal(0.0, 0.15, size=n_const)
        d_phi = rng.normal(0.0, 0.15, size=n_const)
        features[i, :n_const, 0] = d_eta
        features[i, :n_const, 1] = d_phi

        # Transverse momentum and energy
        pt = rng.exponential(scale=15.0, size=n_const) + 1.0
        log_pt = np.log(pt)
        e = pt * np.cosh(d_eta)
        log_e = np.log(e)
        features[i, :n_const, 2] = log_pt
        features[i, :n_const, 3] = log_e

        # Impact parameters d0 and dz with flavour-dependent displacement
        raw_d0 = rng.laplace(0.0, ip_scale, size=n_const)
        raw_dz = rng.laplace(0.0, ip_scale * 1.5, size=n_const)
        tanh_d0, tanh_dz = transform_impact_parameters(raw_d0, raw_dz)

        features[i, :n_const, 4] = tanh_d0
        features[i, :n_const, 5] = tanh_dz

        # Extra constituent features (charge, PID-like dummy embeddings)
        features[i, :n_const, 6:] = rng.normal(0.0, 1.0, size=(n_const, feature_dim - 6))

        # 4-vectors: (px, py, pz, E)
        px = pt * np.cos(d_phi)
        py = pt * np.sin(d_phi)
        pz = pt * np.sinh(d_eta)
        vectors[i, :n_const, 0] = px
        vectors[i, :n_const, 1] = py
        vectors[i, :n_const, 2] = pz
        vectors[i, :n_const, 3] = e

    points = features[:, :, :2].copy()  # (d_eta, d_phi)
    high_level = extract_high_level_features(features, mask)

    return {
        "features": features,
        "points": points,
        "mask": mask,
        "vectors": vectors,
        "high_level": high_level,
        "labels": labels,
    }


def create_canonical_splits(
    data: Dict[str, np.ndarray],
    output_dir: str,
    fractions: Tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
    seed: int = 42,
    is_pilot: bool = True,
) -> Dict[str, Any]:
    """Create, verify disjointness, and save the canonical 4-way split artifact with SHA-256 manifest.

    Splits:
        train (gradient optimization)
        validation (model selection & periodic risk refresh)
        calibration (exact post-training Mondrian conformal calibration only)
        test (final evaluation only)
    """
    assert len(fractions) == 4, "Must provide 4 fractions (train, val, cal, test)"
    assert np.isclose(sum(fractions), 1.0), "Split fractions must sum to 1.0"

    os.makedirs(output_dir, exist_ok=True)
    n_total = len(data["labels"])
    rng = np.random.RandomState(seed)

    # Stratified split to maintain class proportions exactly in every partition
    labels = data["labels"]
    unique_classes = np.unique(labels)

    train_idx, val_idx, cal_idx, test_idx = [], [], [], []

    for c in unique_classes:
        c_indices = np.where(labels == c)[0]
        rng.shuffle(c_indices)
        n_c = len(c_indices)

        n_train = int(n_c * fractions[0])
        n_val = int(n_c * fractions[1])
        n_cal = int(n_c * fractions[2])

        train_idx.extend(c_indices[:n_train])
        val_idx.extend(c_indices[n_train:n_train + n_val])
        cal_idx.extend(c_indices[n_train + n_val:n_train + n_val + n_cal])
        test_idx.extend(c_indices[n_train + n_val + n_cal:])

    split_indices = {
        "train": np.array(train_idx, dtype=np.int64),
        "validation": np.array(val_idx, dtype=np.int64),
        "calibration": np.array(cal_idx, dtype=np.int64),
        "test": np.array(test_idx, dtype=np.int64),
    }

    # Verify strict disjointness
    check_splits_disjoint(split_indices)

    # Save arrays for each partition
    files_saved = {}
    array_hashes = {}

    for split_name, idxs in split_indices.items():
        split_data = {
            "features": data["features"][idxs],
            "points": data["points"][idxs],
            "mask": data["mask"][idxs],
            "vectors": data["vectors"][idxs],
            "high_level": data["high_level"][idxs],
            "labels": data["labels"][idxs],
            "indices": idxs,
        }
        filepath = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(filepath, **split_data)
        files_saved[split_name] = filepath
        array_hashes[split_name] = SplitAuditor.compute_sha256(filepath)

    # Build manifest
    manifest = {
        "dataset_name": "JetClass_3class_pilot" if is_pilot else "JetClass_3class_full",
        "is_pilot": is_pilot,
        "pilot_composition_warning": (
            "PILOT SUBSET: Provisional light:b:c ratio ~4:1:1. NOT final LHC composition."
            if is_pilot else None
        ),
        "created_seed": seed,
        "fractions": {
            "train": fractions[0],
            "validation": fractions[1],
            "calibration": fractions[2],
            "test": fractions[3],
        },
        "counts": {k: len(v) for k, v in split_indices.items()},
        "files": files_saved,
        "sha256_checksums": array_hashes,
        "splits": {
            "indices": {k: v.tolist() for k, v in split_indices.items()}
        }
    }

    manifest_path = os.path.join(output_dir, "canonical_split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


class JetDataset(Dataset):
    """PyTorch Dataset wrapper for jet constituent and high-level features."""

    def __init__(
        self,
        npz_path_or_dict: Union[str, Dict[str, np.ndarray]],
        split_name: str,
        auditor: Optional[SplitAuditor] = None,
    ):
        self.split_name = split_name
        self.auditor = auditor

        if isinstance(npz_path_or_dict, str):
            loaded = np.load(npz_path_or_dict)
            self.data = {k: loaded[k] for k in loaded.files}
        else:
            self.data = npz_path_or_dict

        # Audit split indices if auditor is provided
        if self.auditor and "indices" in self.data:
            self.auditor.assert_indices_belong_strictly_to_split(
                self.data["indices"], self.split_name
            )

    def __len__(self) -> int:
        return len(self.data["labels"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.data["features"][idx]).float(),
            "points": torch.from_numpy(self.data["points"][idx]).float(),
            "mask": torch.from_numpy(self.data["mask"][idx]).float(),
            "vectors": torch.from_numpy(self.data["vectors"][idx]).float(),
            "high_level": torch.from_numpy(self.data["high_level"][idx]).float(),
            "labels": torch.tensor(self.data["labels"][idx], dtype=torch.long),
            "index": torch.tensor(
                self.data["indices"][idx] if "indices" in self.data else idx,
                dtype=torch.long
            ),
        }


def get_dataloaders(
    manifest_path: str,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Instantiate DataLoaders for all 4 canonical splits from a manifest."""
    auditor = SplitAuditor(manifest_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dataloaders = {}
    for split_name in ["train", "validation", "calibration", "test"]:
        filepath = manifest["files"][split_name]
        dataset = JetDataset(filepath, split_name=split_name, auditor=auditor)
        shuffle = (split_name == "train")
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=(split_name == "train"),
        )

    return dataloaders
