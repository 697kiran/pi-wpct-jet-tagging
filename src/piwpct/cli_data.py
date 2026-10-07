"""Shared CLI safeguards for synthetic pilot and prepared real-data splits."""

import json
import os
from typing import Any, Dict, Optional


class SplitManifestError(ValueError):
    """Raised when a requested data mode does not match its split manifest."""


def add_data_mode_arguments(parser: Any) -> None:
    """Require callers to explicitly select synthetic pilot or real data."""
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pilot",
        dest="data_mode",
        action="store_const",
        const="pilot",
        help="Use the synthetic pilot dataset (never real JetClass data).",
    )
    modes.add_argument(
        "--real-data",
        dest="data_mode",
        action="store_const",
        const="real",
        help="Use an existing prepared real-data manifest; never fall back to synthetic pilot data.",
    )


def ensure_split_manifest(
    manifest_path: str,
    *,
    data_mode: str,
    generate_pilot_if_missing: bool = False,
    pilot_samples: int = 12000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Require a manifest matching the requested mode; generate only explicit pilots."""
    if data_mode not in {"pilot", "real"}:
        raise SplitManifestError(f"Unknown data mode: {data_mode!r}")

    if not os.path.isfile(manifest_path):
        if data_mode != "pilot":
            raise SplitManifestError(
                f"Real-data mode requires an existing split manifest at '{manifest_path}'. "
                "Prepare the ROOT data with scripts/convert_root.py; no synthetic pilot was generated."
            )
        if not generate_pilot_if_missing:
            raise SplitManifestError(
                f"Pilot mode requires an existing split manifest at '{manifest_path}'. "
                "Create it explicitly with scripts/build_splits.py --pilot."
            )

        output_dir = os.path.dirname(manifest_path) or "."
        ensure_pilot_output_is_safe(output_dir)
        from piwpct.data import create_canonical_splits, generate_pilot_dataset

        os.makedirs(output_dir, exist_ok=True)
        data = generate_pilot_dataset(n_samples=pilot_samples, seed=seed)
        create_canonical_splits(
            data=data,
            output_dir=output_dir,
            seed=seed,
            is_pilot=True,
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitManifestError(f"Could not read split manifest '{manifest_path}': {exc}") from exc

    is_pilot: Optional[bool] = manifest.get("is_pilot")
    if not isinstance(is_pilot, bool):
        raise SplitManifestError(
            f"Split manifest '{manifest_path}' must declare boolean 'is_pilot' metadata."
        )
    expected_is_pilot = data_mode == "pilot"
    if is_pilot != expected_is_pilot:
        expected = "synthetic pilot" if expected_is_pilot else "real data"
        actual = "synthetic pilot" if is_pilot else "real data"
        raise SplitManifestError(
            f"Requested {expected} mode, but '{manifest_path}' is marked as {actual}."
        )
    return manifest


def ensure_pilot_output_is_safe(output_dir: str) -> None:
    """Avoid replacing real or unclassified split artifacts with synthetic data."""
    manifest_path = os.path.join(output_dir, "canonical_split_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitManifestError(
                f"Refusing to replace unreadable manifest '{manifest_path}': {exc}"
            ) from exc
        if manifest.get("is_pilot") is not True:
            raise SplitManifestError(
                f"Refusing to replace non-pilot or unclassified split manifest '{manifest_path}' with synthetic data."
            )
        return

    existing_splits = [
        name for name in ("train", "validation", "calibration", "test")
        if os.path.exists(os.path.join(output_dir, f"{name}.npz"))
    ]
    if existing_splits:
        raise SplitManifestError(
            f"Refusing to replace split arrays without a manifest in '{output_dir}': "
            f"{', '.join(existing_splits)}.npz."
        )
