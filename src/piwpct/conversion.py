"""Portable, bounded conversion from official JetClass TAR partitions."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from piwpct.data_paths import ensure_path_outside_raw
from piwpct.root_data import (
    assign_source_member_splits,
    convert_root_member,
    discover_root_members,
)
from piwpct.schema import PARTICLE_CAP_DEFAULT, feature_schema_document
from piwpct.targets import load_target_definition, target_manifest_metadata


class ConversionError(ValueError):
    """Raised when conversion settings violate data safety or provenance rules."""


EXPECTED_JET_COUNT = 125_000_000


def estimate_prepared_bytes(manifest: Mapping[str, Any], expected_jets: int = EXPECTED_JET_COUNT) -> int:
    """Linear sample-based storage estimate; explicitly approximate."""
    conversion = manifest.get("conversion", {})
    sample_jets = int(conversion.get("processed_jets", 0))
    sample_bytes = int(conversion.get("output_bytes", 0))
    if sample_jets <= 0 or sample_bytes <= 0:
        raise ConversionError("Size estimate manifest must contain processed_jets and output_bytes > 0.")
    return int(np.ceil(sample_bytes * float(expected_jets) / sample_jets))


def convert_jetclass_archives(
    *,
    raw_data_root: Path,
    prepared_data_root: Path,
    target_mode: str,
    target_config: Optional[str] = None,
    chunk_size: int = 2048,
    particle_cap: int = PARTICLE_CAP_DEFAULT,
    calibration_fraction: float = 0.10,
    seed: int = 42,
    max_files_per_partition: Optional[int] = None,
    max_jets_per_file: Optional[int] = None,
    malformed_policy: str = "error",
    max_malformed_fraction: float = 0.0,
    full_conversion: bool = False,
    size_estimate_manifest: Optional[Path] = None,
    confirm_full_conversion: bool = False,
) -> Dict[str, Any]:
    """Convert one bounded ROOT member at a time into portable jagged shards."""
    raw_data_root = raw_data_root.resolve()
    prepared_data_root = prepared_data_root.resolve()
    ensure_path_outside_raw(prepared_data_root, raw_data_root, name="PREPARED_DATA_ROOT")
    if prepared_data_root.exists():
        raise ConversionError(
            f"Prepared destination already exists: {prepared_data_root}. Choose a new empty destination; "
            "conversion never overwrites an existing dataset."
        )
    if not full_conversion and max_files_per_partition is None and max_jets_per_file is None:
        raise ConversionError(
            "Conversion is capped by default. Supply --max-files-per-partition and/or "
            "--max-jets-per-file for a small conversion, or the guarded --full option."
        )
    if full_conversion:
        if max_files_per_partition is not None or max_jets_per_file is not None:
            raise ConversionError("--full cannot be combined with sample-size caps.")
        if not confirm_full_conversion:
            raise ConversionError("Full conversion requires explicit --confirm-full-conversion.")
        if size_estimate_manifest is None:
            raise ConversionError("Full conversion requires --size-estimate-manifest from a prior tiny conversion.")
        try:
            sample_manifest = json.loads(size_estimate_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(f"Could not read size-estimate manifest {size_estimate_manifest}: {exc}") from exc
        estimate = estimate_prepared_bytes(sample_manifest)
        if not prepared_data_root.parent.exists():
            raise ConversionError(f"Prepared destination parent does not exist: {prepared_data_root.parent}")
        free_bytes = shutil.disk_usage(prepared_data_root.parent).free
        print(
            f"Full conversion estimate: {estimate / (1024 ** 3):.1f} GiB prepared "
            f"(sample-scaled to {EXPECTED_JET_COUNT:,} jets); destination has "
            f"{free_bytes / (1024 ** 3):.1f} GiB free. Requires 25% headroom."
        )
        if free_bytes < int(estimate * 1.25):
            raise ConversionError("Destination free space is below the estimated size plus 25% headroom.")

    if chunk_size < 1 or particle_cap < 1:
        raise ConversionError("chunk_size and particle_cap must be positive.")
    if max_files_per_partition is not None and max_files_per_partition < 1:
        raise ConversionError("max_files_per_partition must be positive.")
    target = load_target_definition(target_mode, target_config)
    members = discover_root_members(raw_data_root, max_files_per_partition)
    assignments = assign_source_member_splits(members, calibration_fraction, seed)
    ordered_members = sorted(members, key=lambda row: (row["archive_relative_path"], row["member_name"]))
    file_ids = {
        (row["archive_relative_path"], row["member_name"]): index
        for index, row in enumerate(ordered_members)
    }
    file_lookup: Dict[str, Dict[str, Any]] = {}
    split_shards: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in ("train", "calibration", "validation", "test")
    }
    stats_by_split: Dict[str, Dict[str, Any]] = {
        name: {
            "processed_jets": 0,
            "constituents": 0,
            "truncated_jets": 0,
            "malformed_jets": 0,
            "read_entries": 0,
            "output_bytes": 0,
            "class_counts": {name: 0 for name in target["class_names"]},
        }
        for name in split_shards
    }
    parent = prepared_data_root.parent
    if not parent.exists():
        raise ConversionError(f"Prepared destination parent does not exist: {parent}")
    build_root = Path(tempfile.mkdtemp(prefix="." + prepared_data_root.name + ".building-", dir=str(parent)))
    staging_root = Path(tempfile.mkdtemp(prefix=".piwpct-root-stage-", dir=str(parent)))
    ensure_path_outside_raw(staging_root, raw_data_root, name="ROOT staging directory")
    started = time.monotonic()
    try:
        for row in ordered_members:
            key = (row["archive_relative_path"], row["member_name"])
            split_name = assignments[key]
            source_id = file_ids[key]
            shard_name = "file-%06d" % source_id
            shard_dir = build_root / "shards" / split_name / shard_name
            shard_dir.parent.mkdir(parents=True, exist_ok=True)
            member_stats = convert_root_member(
                row,
                shard_dir,
                source_id,
                target,
                staging_dir=staging_root,
                chunk_size=chunk_size,
                particle_cap=particle_cap,
                max_jets=max_jets_per_file,
                malformed_policy=malformed_policy,
                max_malformed_fraction=max_malformed_fraction,
            )
            rel_shard = PurePosixPath(shard_dir.relative_to(build_root).as_posix()).as_posix()
            split_shards[split_name].append({
                "path": rel_shard,
                "source_file_id": source_id,
                "n_jets": member_stats["n_jets"],
                "n_constituents": member_stats["n_constituents"],
                "class_counts": member_stats["class_counts"],
            })
            file_lookup[str(source_id)] = {
                "archive_relative_path": row["archive_relative_path"],
                "root_member_name": row["member_name"],
                "official_partition": row["partition"],
                "assigned_split": split_name,
            }
            stat = stats_by_split[split_name]
            stat["processed_jets"] += member_stats["n_jets"]
            stat["constituents"] += member_stats["n_constituents"]
            stat["truncated_jets"] += member_stats["truncated_jets"]
            stat["malformed_jets"] += member_stats["malformed_jets"]
            stat["read_entries"] += member_stats["read_entries"]
            stat["output_bytes"] += member_stats["output_bytes"]
            for name, count in member_stats["class_counts"].items():
                stat["class_counts"][name] += count

        processed_jets = sum(value["processed_jets"] for value in stats_by_split.values())
        output_bytes = sum(value["output_bytes"] for value in stats_by_split.values())
        if processed_jets <= 0:
            raise ConversionError("Conversion produced no jets.")
        complete = bool(full_conversion)
        elapsed = time.monotonic() - started
        for stat in stats_by_split.values():
            jets = stat["processed_jets"]
            stat["average_constituents"] = stat["constituents"] / jets if jets else 0.0
            stat["percent_truncated"] = 100.0 * stat["truncated_jets"] / jets if jets else 0.0
            stat["malformed_fraction"] = stat["malformed_jets"] / max(stat["read_entries"], 1)
        manifest = {
            "format": "piwpct.prepared_jagged.v1",
            "format_version": 1,
            "dataset_name": "JetClass-prepared",
            "is_pilot": False,
            "complete": complete,
            "conversion_scope": "full" if complete else "bounded-smoke-sample",
            "target_definition": target_manifest_metadata(target),
            "feature_schema": feature_schema_document(),
            "particle_cap": int(particle_cap),
            "storage_dtype": "float32",
            "label_dtype": "int64 class IDs ordered as target_definition.classes",
            "sample_id_format": ["file_id", "entry_index"],
            "file_id_lookup": file_lookup,
            "splits": {
                name: {
                    "n_jets": stats_by_split[name]["processed_jets"],
                    "shards": split_shards[name],
                }
                for name in ("train", "validation", "calibration", "test")
            },
            "split_provenance": {
                "train": "official training archive; source members not reserved for calibration",
                "calibration": "whole ROOT members reserved from official training archives by seeded assignment",
                "validation": "official validation archive",
                "test": "official test archive; never included in train or calibration",
                "seed": int(seed),
                "calibration_member_fraction_requested": float(calibration_fraction),
                "source_group": "(archive_relative_path, root_member_name); no event/group ID exists in the inspected schema",
            },
            "conversion": {
                "processed_root_members": len(file_lookup),
                "processed_jets": processed_jets,
                "output_bytes": output_bytes,
                "duration_seconds": elapsed,
                "chunk_size_entries": int(chunk_size),
                "max_jets_per_file": max_jets_per_file,
                "max_files_per_partition": max_files_per_partition,
                "malformed_policy": malformed_policy,
                "max_malformed_fraction": float(max_malformed_fraction),
                "split_statistics": stats_by_split,
                "classes": {
                    name: sum(stat["class_counts"][name] for stat in stats_by_split.values())
                    for name in target["class_names"]
                },
                "prepared_bytes_per_jet": output_bytes / processed_jets,
                "full_size_estimate_bytes_for_125m_jets": int(
                    np.ceil(output_bytes * EXPECTED_JET_COUNT / processed_jets)
                ),
                "full_size_estimate_assumption": "linear projection from this conversion sample; class mix, compression and constituent multiplicity can change the result",
            },
            "paths": {
                "shards_root": "shards",
                "manifest_paths_are_relative": True,
            },
        }
        manifest_path = build_root / "canonical_split_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(str(build_root), str(prepared_data_root))
        return manifest
    except Exception:
        shutil.rmtree(build_root, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
