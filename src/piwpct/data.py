"""Data ingestion, preprocessing, and canonical 4-way split management.

Handles named feature construction, explicitly synthetic pilot data,
canonical pilot splits, and memory-mapped real-data shards. Real ROOT
conversion and target definitions live in the dedicated ingestion modules.
"""

import json
import os
from bisect import bisect_right
from collections import OrderedDict
from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from piwpct.data_paths import PROJECT_ROOT
from piwpct.utils.split_integrity import SplitAuditor, check_splits_disjoint
from piwpct.schema import (
    FEATURE_NAMES,
    JET_METADATA_BRANCHES,
    PARTICLE_CAP_DEFAULT,
    RAW_PARTICLE_BRANCHES,
    RAW_PARTICLE_INDEX,
    build_features,
    build_four_vectors,
    build_points,
    feature_index,
    feature_schema_document,
)

CLASS_NAMES = ["light", "charm", "b"]
CLASS_TO_IDX = {"light": 0, "charm": 1, "b": 2}
IDX_TO_CLASS = {0: "light", 1: "charm", 2: "b"}


def transform_impact_parameters(
    d0: np.ndarray, dz: np.ndarray, clip_val: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the reference transforms ``tanh(part_d0val)`` and ``tanh(part_dzval)``."""
    del clip_val  # retained for source compatibility; reference transform has no extra clipping.
    return np.tanh(d0), np.tanh(dz)


def extract_high_level_features(
    features: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Compute high-level summary features per jet from constituent arrays for GBT.

    Args:
        features: [N_jets, N_particles, C] array of constituent features.
        Feature channels follow piwpct.schema.FEATURE_NAMES.
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
        d_eta = f_valid[:, feature_index("part_deta")]
        d_phi = f_valid[:, feature_index("part_dphi")]
        log_pt_scaled = f_valid[:, feature_index("part_pt_log")]
        tanh_d0 = f_valid[:, feature_index("part_d0")]
        tanh_dz = f_valid[:, feature_index("part_dz")]

        # Undo the documented JetClass manual normalization for pT weighting.
        pt_weights = np.exp(log_pt_scaled / 0.7 + 1.7)
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

        high_level.append(np.asarray([
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
            float(np.mean(log_pt_scaled)),
        ], dtype=np.float32))

    return np.array(high_level, dtype=np.float32)


def generate_pilot_dataset(
    n_samples: int = 12000,
    num_particles: int = 32,
    feature_dim: Optional[int] = None,
    seed: int = 42,
    class_ratio: Tuple[int, int, int] = (4, 1, 1),
) -> Dict[str, np.ndarray]:
    """Generate a reproducible synthetic pilot with the named 17-feature schema.

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

    if feature_dim is not None and feature_dim != len(FEATURE_NAMES):
        raise ValueError(
            f"feature_dim is fixed by the authoritative schema at {len(FEATURE_NAMES)}; "
            f"received {feature_dim}."
        )
    raw_particles = np.zeros(
        (n_samples, num_particles, len(RAW_PARTICLE_BRANCHES)), dtype=np.float32
    )
    vectors = np.zeros((n_samples, num_particles, 4), dtype=np.float32)
    mask = np.zeros((n_samples, num_particles), dtype=np.float32)
    jet_pt = np.ones(n_samples, dtype=np.float32)
    jet_energy = np.ones(n_samples, dtype=np.float32)

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

        # Jet-relative coordinates.
        d_eta = rng.normal(0.0, 0.15, size=n_const)
        d_phi = rng.normal(0.0, 0.15, size=n_const)
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_deta"]] = d_eta
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_dphi"]] = d_phi

        # Positive constituent four-vectors and their owning jet scales.
        pt = rng.exponential(scale=15.0, size=n_const) + 1.0
        e = pt * np.cosh(d_eta)
        jet_pt[i] = max(float(pt.sum()), 1e-6)
        jet_energy[i] = max(float(e.sum()), 1e-6)
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_px"]] = pt * np.cos(d_phi)
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_py"]] = pt * np.sin(d_phi)
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_pz"]] = pt * np.sinh(d_eta)
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_energy"]] = e

        # Impact parameters d0 and dz with flavour-dependent displacement
        raw_d0 = rng.laplace(0.0, ip_scale, size=n_const)
        raw_dz = rng.laplace(0.0, ip_scale * 1.5, size=n_const)
        tanh_d0, tanh_dz = transform_impact_parameters(raw_d0, raw_dz)

        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_d0val"]] = raw_d0
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_dzval"]] = raw_dz
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_d0err"]] = rng.uniform(0.0, 0.2, n_const)
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_dzerr"]] = rng.uniform(0.0, 0.2, n_const)
        raw_particles[i, :n_const, RAW_PARTICLE_INDEX["part_charge"]] = rng.choice([-1, 0, 1], n_const)
        # Explicit categorical flags replace unnamed synthetic channels.
        particle_types = rng.randint(0, 5, n_const)
        for type_id, branch in enumerate((
            "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
            "part_isElectron", "part_isMuon",
        )):
            raw_particles[i, :n_const, RAW_PARTICLE_INDEX[branch]] = (particle_types == type_id)

        # 4-vectors: (px, py, pz, E)
        vectors[i, :n_const] = build_four_vectors(raw_particles[i, :n_const])

    features = np.zeros((n_samples, num_particles, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(n_samples):
        n_const = int(mask[i].sum())
        features[i, :n_const] = build_features(
            raw_particles[i, :n_const],
            np.full(n_const, jet_pt[i], dtype=np.float32),
            np.full(n_const, jet_energy[i], dtype=np.float32),
        )
    points = np.zeros((n_samples, num_particles, 2), dtype=np.float32)
    points[:, :, :] = features[:, :, :2]
    high_level = extract_high_level_features(features, mask)

    return {
        "features": features,
        "raw_particles": raw_particles,
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
    rng = np.random.RandomState(seed)

    # Stratified split to maintain class proportions exactly in every partition
    labels = data["labels"]
    unique_classes = np.unique(labels)

    train_idx: List[int] = []
    val_idx: List[int] = []
    cal_idx: List[int] = []
    test_idx: List[int] = []

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
        # Paths in the manifest are relative to the manifest, not the process CWD.
        files_saved[split_name] = Path(filepath).name
        array_hashes[split_name] = SplitAuditor.compute_sha256(filepath)

    # Build manifest
    manifest = {
        "dataset_name": "JetClass_3class_pilot" if is_pilot else "JetClass_3class_full",
        "is_pilot": is_pilot,
        "format": "piwpct.pilot_npz.v1" if is_pilot else "piwpct.legacy_npz.v1",
        "target_definition": {
            "mode": "flavour_tagging",
            "semantics": "synthetic pilot labels only; not a measured JetClass flavour target",
            "main_experiment": False,
            "classes": [
                {"id": 0, "name": "light"},
                {"id": 1, "name": "charm"},
                {"id": 2, "name": "b"},
            ],
            "class_names": ["light", "charm", "b"],
            "warning": "Synthetic pilot labels are provisional and are not real detector or truth annotations.",
        },
        "feature_schema": feature_schema_document(),
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


class BoundedShuffleSampler(Sampler):
    """Shuffle with a bounded index window instead of randomizing the full shard set."""

    def __init__(self, data_source: Any, buffer_size: int = 8192, seed: int = 42):
        self.data_source = data_source
        self.buffer_size = max(1, int(buffer_size))
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        n_items = len(self.data_source)
        buffer = list(range(min(n_items, self.buffer_size)))
        next_index = len(buffer)
        while buffer:
            position = int(rng.randint(len(buffer)))
            value = buffer[position]
            if next_index < n_items:
                buffer[position] = next_index
                next_index += 1
            else:
                buffer.pop(position)
            yield value


class PreparedShardedJetDataset(Dataset):
    """Lazy memory-mapped Dataset for portable flat-particle NPY shards."""

    def __init__(self, manifest_path: Union[str, Path], split_name: str, cache_shards: int = 3):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.split_name = split_name
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "piwpct.prepared_jagged.v1":
            raise ValueError(f"Unsupported prepared manifest format in {self.manifest_path}.")
        split = self.manifest.get("splits", {}).get(split_name)
        if not isinstance(split, dict):
            raise ValueError(f"Prepared manifest has no split {split_name!r}.")
        self.shards = list(split.get("shards", []))
        if not self.shards:
            raise ValueError(f"Prepared split {split_name!r} has no shards.")
        self.cumulative_ends: List[int] = []
        total = 0
        for shard in self.shards:
            count = int(shard["n_jets"])
            if count < 1:
                raise ValueError("Prepared shard metadata must contain positive n_jets.")
            total += count
            self.cumulative_ends.append(total)
        if total != int(split.get("n_jets", total)):
            raise ValueError(f"Prepared split {split_name!r} jet count does not match shard metadata.")
        self.n_jets = total
        self.target_definition = self.manifest.get("target_definition", {})
        self.target_mode = self.target_definition.get("mode")
        self.class_names = list(self.target_definition.get("class_names", []))
        self.complete = bool(self.manifest.get("complete", False))
        self.cache_shards = max(1, int(cache_shards))
        self._cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return self.n_jets

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _open_shard(self, shard_index: int) -> Dict[str, np.ndarray]:
        if shard_index in self._cache:
            self._cache.move_to_end(shard_index)
            return self._cache[shard_index]
        shard_dir = self.root / self.shards[shard_index]["path"]
        arrays = {
            name: np.load(shard_dir / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in ("particles", "offsets", "jets", "labels", "sample_ids")
        }
        expected = int(self.shards[shard_index]["n_jets"])
        if len(arrays["labels"]) != expected or len(arrays["jets"]) != expected:
            raise ValueError(f"Shard {shard_dir} array lengths disagree with its manifest.")
        if arrays["offsets"].shape != (expected + 1,):
            raise ValueError(f"Shard {shard_dir} offsets must have length n_jets + 1.")
        if arrays["sample_ids"].shape != (expected, 2):
            raise ValueError(f"Shard {shard_dir} sample_ids must have shape [n_jets, 2].")
        self._cache[shard_index] = arrays
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_ends, index)
        start_global = self.cumulative_ends[shard_index - 1] if shard_index else 0
        local_index = index - start_global
        arrays = self._open_shard(shard_index)
        start = int(arrays["offsets"][local_index])
        stop = int(arrays["offsets"][local_index + 1])
        return {
            "raw_particles": arrays["particles"][start:stop],
            "jets": np.asarray(arrays["jets"][local_index], dtype=np.float32),
            "labels": int(arrays["labels"][local_index]),
            "sample_id": np.asarray(arrays["sample_ids"][local_index], dtype=np.uint64),
        }


def collate_prepared_jets(
    items: Sequence[Dict[str, Any]], particle_cap: int = PARTICLE_CAP_DEFAULT
) -> Dict[str, torch.Tensor]:
    """Construct named features and explicit masks while padding each batch."""
    if not items:
        raise ValueError("Cannot collate an empty jet batch.")
    if particle_cap < 1:
        raise ValueError("particle_cap must be positive.")
    batch_size = len(items)
    features = np.zeros((batch_size, particle_cap, len(FEATURE_NAMES)), dtype=np.float32)
    points = np.zeros((batch_size, particle_cap, 2), dtype=np.float32)
    vectors = np.zeros((batch_size, particle_cap, 4), dtype=np.float32)
    mask = np.zeros((batch_size, particle_cap), dtype=np.float32)
    jets_batch = np.zeros((batch_size, len(JET_METADATA_BRANCHES)), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int64)
    sample_ids = np.zeros((batch_size, 2), dtype=np.uint64)
    jet_pt_index = JET_METADATA_BRANCHES.index("jet_pt")
    jet_energy_index = JET_METADATA_BRANCHES.index("jet_energy")
    truncated = 0
    for row, item in enumerate(items):
        raw = np.asarray(item["raw_particles"], dtype=np.float32)
        jets = np.asarray(item["jets"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != len(RAW_PARTICLE_BRANCHES):
            raise ValueError("Prepared raw particle fields do not match the named storage schema.")
        if jets.shape != (len(JET_METADATA_BRANCHES),):
            raise ValueError("Prepared jet metadata does not match the named storage schema.")
        n_original = len(raw)
        n_particles = min(n_original, particle_cap)
        truncated += int(n_original > particle_cap)
        if n_particles:
            raw = raw[:n_particles]
            repeated_pt = np.full(n_particles, jets[jet_pt_index], dtype=np.float32)
            repeated_energy = np.full(n_particles, jets[jet_energy_index], dtype=np.float32)
            features[row, :n_particles] = build_features(raw, repeated_pt, repeated_energy)
            points[row, :n_particles] = build_points(raw)
            vectors[row, :n_particles] = build_four_vectors(raw)
            mask[row, :n_particles] = 1.0
        jets_batch[row] = jets
        labels[row] = int(item["labels"])
        sample_ids[row] = item["sample_id"]
    high_level = extract_high_level_features(features, mask)
    return {
        "features": torch.from_numpy(features),
        "points": torch.from_numpy(points),
        "mask": torch.from_numpy(mask),
        "vectors": torch.from_numpy(vectors),
        "high_level": torch.from_numpy(high_level),
        "jets": torch.from_numpy(jets_batch),
        "labels": torch.from_numpy(labels),
        "sample_ids": torch.from_numpy(sample_ids.astype(np.int64)),
        "batch_truncated_jets": torch.tensor(truncated, dtype=torch.long),
    }


def _resolve_pilot_file(manifest_path: str, manifest: Dict[str, Any], split_name: str) -> str:
    """Find a pilot NPZ beside its manifest, retaining checked legacy references."""
    reference = manifest.get("files", {}).get(split_name)
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Pilot manifest {manifest_path} has no file reference for {split_name!r}.")
    native = Path(reference).expanduser()
    if PureWindowsPath(reference).is_absolute() and not native.is_absolute():
        raise ValueError(
            f"Pilot split {split_name!r} uses a Windows absolute path that is unavailable here: {reference!r}."
        )
    normalized = Path(reference.replace("\\", "/"))
    manifest_dir = Path(manifest_path).resolve().parent
    if native.is_absolute():
        candidates = [native.resolve()]
    else:
        candidates = [manifest_dir / normalized]
        if len(normalized.parts) > 1:
            # Older manifests stored paths relative to the launch CWD. A file
            # moved beside the manifest is accepted only when its hash matches.
            candidates.extend((manifest_dir / normalized.name, PROJECT_ROOT / normalized))
    existing = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if not existing:
        raise FileNotFoundError(
            f"Missing pilot split {split_name!r} NPZ {reference!r} referenced by {manifest_path}; "
            f"checked: {', '.join(map(str, candidates))}. No data was generated."
        )
    expected_hash = manifest.get("sha256_checksums", {}).get(split_name)
    if not expected_hash and not native.is_absolute() and len(normalized.parts) > 1 and candidates[0].resolve() not in existing:
        raise ValueError(
            f"Pilot split {split_name!r} in {manifest_path} uses a legacy CWD-relative path "
            "but has no SHA-256 checksum to verify a fallback file."
        )
    if expected_hash:
        matching = [path for path in existing if SplitAuditor.compute_sha256(str(path)) == expected_hash]
        if not matching:
            raise ValueError(
                f"Pilot split {split_name!r} referenced by {manifest_path} exists, "
                "but no candidate matches its recorded SHA-256 checksum."
            )
        return str(matching[0])
    if len(existing) > 1:
        raise ValueError(
            f"Pilot split {split_name!r} in {manifest_path} has ambiguous legacy paths and no checksum."
        )
    return str(existing[0])


def get_dataloaders(
    manifest_path: str,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_buffer_size: int = 8192,
    seed: int = 42,
    particle_cap: Optional[int] = None,
    drop_last: bool = True,
    allow_incomplete: bool = False,
) -> Dict[str, DataLoader]:
    """Load legacy pilot NPZ or memory-mapped prepared jagged shards."""
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest.get("is_pilot", False) and manifest.get("format") != "piwpct.prepared_jagged.v1":
        raise ValueError(
            "Real-data mode requires the portable piwpct.prepared_jagged.v1 manifest; "
            "legacy real-data NPZ manifests are not supported."
        )
    prepared = manifest.get("format") == "piwpct.prepared_jagged.v1"
    if prepared and not manifest.get("complete", False) and not allow_incomplete:
        raise ValueError(
            "This manifest describes a bounded smoke conversion, not the complete official partitions. "
            "Use --allow-incomplete-data only for an explicitly requested pipeline smoke test."
        )
    if not prepared:
        auditor = SplitAuditor(manifest_path)
    else:
        auditor = None
    cap = int(particle_cap or manifest.get("particle_cap", PARTICLE_CAP_DEFAULT))
    dataloaders: Dict[str, DataLoader] = {}
    for split_name in ("train", "validation", "calibration", "test"):
        dataset: Any
        if prepared:
            dataset = PreparedShardedJetDataset(manifest_path, split_name)
            collate_fn = partial(collate_prepared_jets, particle_cap=cap)
        else:
            filepath = _resolve_pilot_file(manifest_path, manifest, split_name)
            dataset = JetDataset(filepath, split_name=split_name, auditor=auditor)
            target_definition = manifest.get("target_definition", {})
            dataset.target_mode = target_definition.get("mode", "flavour_tagging")
            dataset.class_names = list(target_definition.get("class_names", CLASS_NAMES))
            collate_fn = None
        if split_name == "train":
            sampler = BoundedShuffleSampler(dataset, shuffle_buffer_size, seed)
            shuffle = False
            use_drop_last = drop_last
        else:
            sampler = None
            shuffle = False
            use_drop_last = False
        options: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "sampler": sampler,
            "shuffle": shuffle,
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
            "drop_last": use_drop_last,
            "collate_fn": collate_fn,
        }
        dataloaders[split_name] = DataLoader(**options)
    return dataloaders
