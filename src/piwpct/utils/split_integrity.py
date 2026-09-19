"""Split integrity verification and leakage prevention utilities.

Enforces strict separation across the 4-way canonical split:
    train / validation / test / calibration
Ensuring that calibration and test data are never touched during training or risk allocation,
and validation data is never used for headline test intervals.
"""

import hashlib
import json
import os
from typing import Dict, List, Set, Union, Optional
import numpy as np


class SplitIntegrityViolation(Exception):
    """Raised when any split leakage or disjointness violation occurs."""
    pass


class SplitAuditor:
    """Audits and guarantees strict isolation across train, val, cal, and test splits."""

    ALLOWED_ROLES = {"train", "validation", "calibration", "test"}

    def __init__(self, manifest_path: Optional[str] = None):
        self.manifest_path = manifest_path
        self.manifest: Optional[Dict] = None
        self.split_indices: Dict[str, Set[int]] = {}
        if manifest_path and os.path.exists(manifest_path):
            self.load_manifest(manifest_path)

    def load_manifest(self, manifest_path: str) -> None:
        """Load and verify a canonical split manifest."""
        self.manifest_path = manifest_path
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        # Build index sets for fast membership checks if provided
        if "splits" in self.manifest and "indices" in self.manifest["splits"]:
            for split_name, idxs in self.manifest["splits"]["indices"].items():
                self.split_indices[split_name] = set(idxs)
            self.verify_disjointness()

    @staticmethod
    def compute_sha256(data: Union[np.ndarray, bytes, str]) -> str:
        """Compute SHA-256 hash of a numpy array, byte sequence, or file path."""
        hasher = hashlib.sha256()
        if isinstance(data, str) and os.path.isfile(data):
            with open(data, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
        elif isinstance(data, np.ndarray):
            hasher.update(np.ascontiguousarray(data).tobytes())
        elif isinstance(data, bytes):
            hasher.update(data)
        elif isinstance(data, str):
            hasher.update(data.encode("utf-8"))
        else:
            raise TypeError(f"Unsupported data type for hashing: {type(data)}")
        return hasher.hexdigest()

    def verify_disjointness(self) -> None:
        """Verify that all splits in the manifest are strictly pairwise disjoint."""
        split_names = list(self.split_indices.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                name_a, name_b = split_names[i], split_names[j]
                overlap = self.split_indices[name_a].intersection(self.split_indices[name_b])
                if overlap:
                    raise SplitIntegrityViolation(
                        f"FATAL SPLIT LEAKAGE: {name_a} and {name_b} share {len(overlap)} samples! "
                        f"First overlapping index: {next(iter(overlap))}"
                    )

    def assert_no_calibration_leakage(self, active_split_name: str) -> None:
        """Assert that calibration split is NEVER accessed in training or risk allocation."""
        if active_split_name.lower() in {"calibration", "cal"}:
            raise SplitIntegrityViolation(
                "CRITICAL ERROR: Calibration split was invoked in a training/risk-allocation path! "
                "Calibration split must be held out strictly for post-training Mondrian calibration."
            )

    def assert_no_test_leakage(self, active_split_name: str) -> None:
        """Assert that test split is NEVER accessed in training, risk allocation, or tuning."""
        if active_split_name.lower() in {"test", "testing"}:
            raise SplitIntegrityViolation(
                "CRITICAL ERROR: Test split was invoked in a model-tuning/training path! "
                "Test split is reserved strictly for final headline evaluation."
            )

    def assert_indices_belong_strictly_to_split(
        self, indices: Union[List[int], np.ndarray], expected_split: str
    ) -> None:
        """Assert that a given array of sample indices belongs ONLY to expected_split."""
        if not self.split_indices:
            return  # Manifest not loaded with raw indices

        if expected_split not in self.split_indices:
            raise ValueError(f"Unknown split name: {expected_split}")

        idx_set = set(int(i) for i in indices)
        expected_set = self.split_indices[expected_split]
        # Check if any leaked from other splits
        for other_split, other_set in self.split_indices.items():
            if other_split != expected_split:
                leaked = idx_set.intersection(other_set)
                if leaked:
                    raise SplitIntegrityViolation(
                        f"SPLIT LEAKAGE: Found {len(leaked)} indices from '{other_split}' in '{expected_split}'!"
                    )

        not_in_expected = idx_set - expected_set
        if not_in_expected:
            raise SplitIntegrityViolation(
                f"{len(not_in_expected)} indices do not belong to declared split '{expected_split}'!"
            )


def check_splits_disjoint(splits_dict: Dict[str, np.ndarray]) -> bool:
    """Standalone functional check for pairwise disjoint index arrays."""
    names = list(splits_dict.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = splits_dict[names[i]], splits_dict[names[j]]
            overlap = np.intersect1d(a, b)
            if len(overlap) > 0:
                raise SplitIntegrityViolation(
                    f"Splits '{names[i]}' and '{names[j]}' overlap on {len(overlap)} indices!"
                )
    return True
