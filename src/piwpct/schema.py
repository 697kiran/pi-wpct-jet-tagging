"""Authoritative JetClass constituent input and storage schemas.

The stored jagged columns retain the ROOT values.  The model feature matrix is
constructed at batch time using the transforms documented below, so a shard
does not duplicate the raw four-vector representation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


FEATURE_SCHEMA_VERSION = "jetclass-particle-features-v1"
PARTICLE_CAP_DEFAULT = 128
FEATURE_DTYPE = "float32"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source_branches: Tuple[str, ...]
    transformation: str
    dtype: str = FEATURE_DTYPE


# This is the sole ordering/index authority for all 17 neural scalar inputs.
FEATURE_SPECS: Tuple[FeatureSpec, ...] = (
    FeatureSpec("part_deta", ("part_deta",), "identity; signed jet-relative eta"),
    FeatureSpec("part_dphi", ("part_dphi",), "identity; signed jet-relative phi"),
    FeatureSpec("part_pt_log", ("part_px", "part_py"), "(log(hypot(px, py)) - 1.7) * 0.7"),
    FeatureSpec("part_e_log", ("part_energy",), "(log(energy) - 2.0) * 0.7"),
    FeatureSpec("part_d0", ("part_d0val",), "tanh(part_d0val)"),
    FeatureSpec("part_dz", ("part_dzval",), "tanh(part_dzval)"),
    FeatureSpec("part_logptrel", ("part_px", "part_py", "jet_pt"), "(log(pt / jet_pt) + 4.7) * 0.7"),
    FeatureSpec("part_logerel", ("part_energy", "jet_energy"), "(log(energy / jet_energy) + 4.7) * 0.7"),
    FeatureSpec("part_deltaR", ("part_deta", "part_dphi"), "(hypot(deta, dphi) - 0.2) * 4.0"),
    FeatureSpec("part_charge", ("part_charge",), "identity"),
    FeatureSpec("part_isChargedHadron", ("part_isChargedHadron",), "identity; binary"),
    FeatureSpec("part_isNeutralHadron", ("part_isNeutralHadron",), "identity; binary"),
    FeatureSpec("part_isPhoton", ("part_isPhoton",), "identity; binary"),
    FeatureSpec("part_isElectron", ("part_isElectron",), "identity; binary"),
    FeatureSpec("part_isMuon", ("part_isMuon",), "identity; binary"),
    FeatureSpec("part_d0err", ("part_d0err",), "clip(value, 0.0, 1.0)"),
    FeatureSpec("part_dzerr", ("part_dzerr",), "clip(value, 0.0, 1.0)"),
)

FEATURE_NAMES: Tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_INDEX: Dict[str, int] = {name: index for index, name in enumerate(FEATURE_NAMES)}

# A compact, named jagged storage matrix. Four-vectors are retained separately
# from transformed model inputs and reconstructed using these stored columns.
RAW_PARTICLE_BRANCHES: Tuple[str, ...] = (
    "part_deta",
    "part_dphi",
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)
RAW_PARTICLE_INDEX: Dict[str, int] = {
    name: index for index, name in enumerate(RAW_PARTICLE_BRANCHES)
}
FOUR_VECTOR_BRANCHES: Tuple[str, ...] = (
    "part_px", "part_py", "part_pz", "part_energy"
)
JET_METADATA_BRANCHES: Tuple[str, ...] = (
    "jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles",
    "jet_sdmass", "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4",
)
POINT_FEATURE_NAMES: Tuple[str, ...] = ("part_deta", "part_dphi")


def feature_schema_document() -> Dict[str, object]:
    """Return JSON-safe schema/provenance metadata."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "dtype": FEATURE_DTYPE,
        "ordering": list(FEATURE_NAMES),
        "features": [
            {
                "index": index,
                "name": spec.name,
                "source_branches": list(spec.source_branches),
                "transformation": spec.transformation,
                "dtype": spec.dtype,
            }
            for index, spec in enumerate(FEATURE_SPECS)
        ],
        "reference": {
            "url": "https://github.com/jet-universe/particle_transformer/blob/main/data/JetClass/JetClass_full.yaml",
            "note": "JetClass variables and manual input scaling; this project preserves its explicit candidate order.",
        },
        "points": list(POINT_FEATURE_NAMES),
        "four_vectors": list(FOUR_VECTOR_BRANCHES),
        "mask": "derived from jagged offsets: 1 for stored constituents, 0 for batch padding",
        "storage": {
            "raw_particle_branches": list(RAW_PARTICLE_BRANCHES),
            "raw_dtype": "float32",
            "layout": "flat [total_constituents, 16] plus int64 offsets [num_jets + 1]",
        },
    }


def _as_2d_float32(values: np.ndarray, columns: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f"{name} must have shape [N, {columns}], got {result.shape}.")
    return result


def build_features(
    raw_particles: np.ndarray,
    jet_pt: np.ndarray,
    jet_energy: np.ndarray,
) -> np.ndarray:
    """Build the 17 model features from named raw ROOT columns.

    `jet_pt` and `jet_energy` contain the owning jet's values repeated once per
    constituent. Input validation for finite, positive values belongs to the
    ROOT reader; this function still rejects invalid values for direct callers.
    """
    raw = _as_2d_float32(raw_particles, len(RAW_PARTICLE_BRANCHES), "raw_particles")
    pt_jet = np.asarray(jet_pt, dtype=np.float32).reshape(-1)
    energy_jet = np.asarray(jet_energy, dtype=np.float32).reshape(-1)
    if len(pt_jet) != len(raw) or len(energy_jet) != len(raw):
        raise ValueError("jet_pt and jet_energy must have one value per constituent.")

    def col(name: str) -> np.ndarray:
        return raw[:, RAW_PARTICLE_INDEX[name]]
    pt = np.hypot(col("part_px"), col("part_py"))
    energy = col("part_energy")
    if not np.isfinite(raw).all() or not np.isfinite(pt_jet).all() or not np.isfinite(energy_jet).all():
        raise ValueError("Feature inputs must be finite.")
    if np.any(pt <= 0) or np.any(energy <= 0) or np.any(pt_jet <= 0) or np.any(energy_jet <= 0):
        raise ValueError("Particle and jet pT/energy inputs must be strictly positive.")

    deta = col("part_deta")
    dphi = col("part_dphi")
    features = np.column_stack((
        deta,
        dphi,
        (np.log(pt) - 1.7) * 0.7,
        (np.log(energy) - 2.0) * 0.7,
        np.tanh(col("part_d0val")),
        np.tanh(col("part_dzval")),
        (np.log(pt / pt_jet) + 4.7) * 0.7,
        (np.log(energy / energy_jet) + 4.7) * 0.7,
        (np.hypot(deta, dphi) - 0.2) * 4.0,
        col("part_charge"),
        col("part_isChargedHadron"),
        col("part_isNeutralHadron"),
        col("part_isPhoton"),
        col("part_isElectron"),
        col("part_isMuon"),
        np.clip(col("part_d0err"), 0.0, 1.0),
        np.clip(col("part_dzerr"), 0.0, 1.0),
    )).astype(np.float32, copy=False)
    if features.shape[1] != len(FEATURE_SPECS):
        raise AssertionError("Feature builder and authoritative schema have diverged.")
    return features


def build_points(raw_particles: np.ndarray) -> np.ndarray:
    raw = _as_2d_float32(raw_particles, len(RAW_PARTICLE_BRANCHES), "raw_particles")
    return raw[:, [RAW_PARTICLE_INDEX[name] for name in POINT_FEATURE_NAMES]].copy()


def build_four_vectors(raw_particles: np.ndarray) -> np.ndarray:
    raw = _as_2d_float32(raw_particles, len(RAW_PARTICLE_BRANCHES), "raw_particles")
    return raw[:, [RAW_PARTICLE_INDEX[name] for name in FOUR_VECTOR_BRANCHES]].copy()


def feature_index(name: str) -> int:
    try:
        return FEATURE_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"Unknown feature name {name!r}; valid names: {FEATURE_NAMES}.") from exc
