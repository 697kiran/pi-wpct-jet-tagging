"""Read-only inventory and structural readiness checks for local/Astra data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_SPLITS = ("train", "validation", "calibration", "test")
SHARD_ARRAYS = ("particles", "offsets", "jets", "labels", "sample_ids")
PREPARED_FORMAT = "piwpct.prepared_jagged.v1"
# The Astra guide says "5 GB" (decimal unit); do not silently treat it as GiB.
LARGE_FILE_THRESHOLD = 5_000_000_000


def transfer_method(size_bytes: int) -> str:
    """Return the Astra guide's method for one file based on its size."""
    if size_bytes > LARGE_FILE_THRESHOLD:
        return "ICTS external-drive procedure (>5 GB)"
    if size_bytes == LARGE_FILE_THRESHOLD:
        return "Confirm with ICTS; the guide specifies files smaller or larger than 5 GB"
    return "Astra upload interface (under 5 GB)"


def inspect_manifest(
    manifest_path: Path,
    *,
    stage: str,
    target_mode: Optional[str],
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect metadata and referenced files without loading dataset arrays."""
    result: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "status": "PRESENT_AND_STRUCTURALLY_VALID",
        "errors": [],
        "warnings": [],
    }
    if not manifest_path.is_file():
        result.update(status="MISSING", errors=[f"Manifest not found: {manifest_path}"])
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.update(status="MISSING", errors=[f"Cannot read manifest: {exc}"])
        return result

    result["format"] = manifest.get("format", "legacy-pilot-npz" if manifest.get("is_pilot") else None)
    result["complete"] = bool(manifest.get("complete", False))
    result["is_pilot"] = bool(manifest.get("is_pilot", False))
    definition = manifest.get("target_definition") or {}
    observed_mode = definition.get("mode")
    result["target_mode"] = observed_mode or ("synthetic_pilot" if result["is_pilot"] else None)

    if not result["is_pilot"] and manifest.get("format") == PREPARED_FORMAT and observed_mode not in {
        "process_classification", "flavour_tagging"
    }:
        result["errors"].append("Prepared manifest is missing a recognized target_definition.mode.")
        result["status"] = "TARGET_NOT_VERIFIED"

    expected = PREPARED_FORMAT if manifest.get("format") == PREPARED_FORMAT else None
    if manifest.get("format") and not expected:
        result["errors"].append(f"Unsupported prepared manifest format: {manifest.get('format')!r}")
    if expected and not isinstance(manifest.get("splits"), dict):
        result["errors"].append("Prepared manifest has no splits mapping.")

    refs: List[Path] = []
    root = manifest_path.resolve().parent
    if expected:
        splits = manifest.get("splits", {})
        for split in REQUIRED_SPLITS:
            split_info = splits.get(split)
            if not isinstance(split_info, dict) or not isinstance(split_info.get("shards"), list):
                result["errors"].append(f"Required split {split!r} has no shard list.")
                continue
            if not split_info["shards"]:
                result["errors"].append(f"Required split {split!r} has no shards.")
            for shard in split_info["shards"]:
                rel = shard.get("path") if isinstance(shard, dict) else None
                if not isinstance(rel, str) or not rel:
                    result["errors"].append(f"Split {split!r} has a shard without a path.")
                    continue
                shard_dir = (root / rel).resolve()
                try:
                    shard_dir.relative_to(root)
                except ValueError:
                    result["errors"].append(f"Shard path escapes the manifest root: {rel}")
                    continue
                refs.append(shard_dir)
                if not shard_dir.is_dir():
                    result["errors"].append(f"Missing shard directory: {rel}")
                    continue
                for name in SHARD_ARRAYS:
                    if not (shard_dir / f"{name}.npy").is_file():
                        result["errors"].append(f"Missing {name}.npy in shard {rel}")
    elif result["is_pilot"]:
        files = manifest.get("files")
        if not isinstance(files, dict):
            result["errors"].append("Pilot manifest has no files mapping.")
        else:
            for split in REQUIRED_SPLITS:
                rel = files.get(split)
                if not isinstance(rel, str):
                    result["errors"].append(f"Pilot manifest has no {split!r} NPZ reference.")
                    continue
                # Historical manifests may contain Windows separators; normalize for relocation.
                candidate = root.joinpath(*rel.replace("\\", "/").split("/" )).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    result["errors"].append(f"Pilot path escapes the manifest root: {rel}")
                    continue
                refs.append(candidate)
                if not candidate.is_file():
                    result["errors"].append(f"Missing pilot split file: {rel}")
    else:
        result["errors"].append("Manifest is neither piwpct.prepared_jagged.v1 nor an explicit pilot manifest.")

    result["resolved_references"] = len(refs)
    if result["errors"]:
        result["status"] = "MISSING"
    elif expected and not result["complete"]:
        result["status"] = "PRESENT_BUT_INCOMPLETE"

    if target_mode and result["is_pilot"]:
        result["errors"].append("Synthetic pilot data cannot satisfy a real target-mode request.")
        result["status"] = "TARGET_NOT_VERIFIED"
    if data_mode == "real" and result["is_pilot"]:
        result["errors"].append("Synthetic pilot manifest cannot satisfy real-data readiness.")
        result["status"] = "TARGET_NOT_VERIFIED"
    elif data_mode == "pilot" and not result["is_pilot"]:
        result["errors"].append("Real prepared data cannot be silently selected as a synthetic pilot.")
        result["status"] = "TARGET_NOT_VERIFIED"
    elif target_mode and observed_mode != target_mode:
        result["errors"].append(
            f"Requested target mode {target_mode!r} does not match manifest mode {observed_mode!r}."
        )
        result["status"] = "TARGET_NOT_VERIFIED"
    if target_mode == "flavour_tagging" and observed_mode != "flavour_tagging":
        result["status"] = "TARGET_NOT_VERIFIED"
    if stage == "train-full" and not result["complete"]:
        result["errors"].append("Full training requires complete: true.")
        if result["status"] in {"PRESENT_AND_STRUCTURALLY_VALID", "PRESENT_BUT_INCOMPLETE"}:
            result["status"] = "PRESENT_BUT_INCOMPLETE"
    result["warnings"].append("Array contents and scientific labels are not evaluated by this structural check.")
    return result


def inspect_raw_root(raw_root: Path) -> Dict[str, Any]:
    """Count and stat archive files; check only the first 512-byte TAR header."""
    archives = sorted(raw_root.rglob("*.tar")) if raw_root.is_dir() else []
    details = []
    for path in archives:
        row: Dict[str, Any] = {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "regular_file": path.is_file(), "transfer_method": transfer_method(path.stat().st_size),
        }
        try:
            with path.open("rb") as stream:
                header = stream.read(512)
            checksum = int(header[148:156].rstrip(b"\0 ") or b"0", 8)
            calculated = sum(32 if 148 <= index < 156 else value for index, value in enumerate(header))
            row["header_checksum_valid"] = len(header) == 512 and checksum == calculated
        except (OSError, ValueError):
            row["header_checksum_valid"] = False
        details.append(row)
    if not details:
        status = "MISSING"
    elif any(not row["header_checksum_valid"] for row in details):
        status = "PRESENT_BUT_INCOMPLETE"
    else:
        status = "PRESENT_AND_STRUCTURALLY_VALID"
    return {"root": str(raw_root.resolve()), "status": status, "archives": details}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stage", choices=("inventory", "convert-smoke", "convert-full", "train-smoke", "train-full"), default="inventory")
    parser.add_argument("--data-mode", choices=("real", "pilot"))
    parser.add_argument("--target-mode", choices=("process_classification", "flavour_tagging"))
    args = parser.parse_args()
    report: Dict[str, Any] = {"stage": args.stage, "checks": []}
    if args.raw_root:
        report["checks"].append({"kind": "raw_archives", **inspect_raw_root(args.raw_root)})
    if args.manifest:
        report["checks"].append({
            "kind": "prepared_manifest",
            **inspect_manifest(args.manifest, stage=args.stage, target_mode=args.target_mode, data_mode=args.data_mode),
        })
    if not args.raw_root and not args.manifest:
        parser.error("provide --raw-root and/or --manifest")
    report["astra_status"] = "REQUIRES_ASTRA_CHECK"
    print(json.dumps(report, indent=2))
    blocked = any(check["status"] in {"MISSING", "PRESENT_BUT_INCOMPLETE", "TARGET_NOT_VERIFIED"} for check in report["checks"])
    return 1 if blocked and args.stage == "train-full" else 0


if __name__ == "__main__":
    raise SystemExit(main())
