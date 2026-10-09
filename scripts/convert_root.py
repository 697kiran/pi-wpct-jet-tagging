"""Convert JetClass TAR/ROOT partitions into portable jagged NPY shards."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.conversion import ConversionError, convert_jetclass_archives
from piwpct.data_paths import load_data_config, resolve_data_root, resolve_project_path
from piwpct.root_data import RootDataError
from piwpct.schema import PARTICLE_CAP_DEFAULT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert official JetClass TAR/ROOT data into bounded jagged shards. "
            "This command only reads RAW_DATA_ROOT and never creates pilot data."
        )
    )
    parser.add_argument("--real-data", action="store_true", required=True,
                        help="Required explicit confirmation that this conversion reads real ROOT data.")
    parser.add_argument("--config", default="configs/data.yaml", help="YAML paths and conversion settings.")
    parser.add_argument("--raw-data-root", help="Read-only root containing official TAR archives.")
    parser.add_argument("--prepared-data-root", help="New destination for jagged prepared shards (required via CLI/config/env).")
    parser.add_argument("--output-root", help="Root for later training outputs; recorded in run guidance, not used for shards.")
    parser.add_argument("--target-mode", required=True, choices=("process_classification", "flavour_tagging"),
                        help="Explicit target semantics. Process classes are not flavour labels.")
    parser.add_argument("--target-config", help="Optional YAML class-to-label-branch mapping.")
    parser.add_argument("--particle-cap", type=int, default=PARTICLE_CAP_DEFAULT,
                        help="Batch-time padding/truncation cap; default 128.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="ROOT entries read per bounded chunk.")
    parser.add_argument("--calibration-fraction", type=float, default=0.10,
                        help="Fraction of training ROOT members reserved whole for calibration.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files-per-partition", type=int,
                        help="Small-conversion cap on ROOT members in each official partition (train/validation/test).")
    parser.add_argument("--max-jets-per-file", type=int,
                        help="Small-conversion cap per ROOT member; never implies a full conversion.")
    parser.add_argument("--full", action="store_true", help="Opt into the guarded full-dataset conversion path.")
    parser.add_argument("--size-estimate-manifest", type=Path,
                        help="Prior tiny-conversion manifest required by --full.")
    parser.add_argument("--confirm-full-conversion", action="store_true",
                        help="Explicit confirmation after reviewing the sample-based storage estimate.")
    parser.add_argument("--malformed-policy", choices=("error", "drop"), default="error")
    parser.add_argument("--max-malformed-fraction", type=float, default=0.0,
                        help="Required positive ceiling when --malformed-policy=drop.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_data_config(args.config)
        raw_root, _ = resolve_data_root("raw_data_root", args.raw_data_root, config)
        prepared_root, prepared_source = resolve_data_root("prepared_data_root", args.prepared_data_root, config)
        if prepared_source == "default":
            parser.error(
                "Prepared output destination is required explicitly via --prepared-data-root, "
                "PREPARED_DATA_ROOT, or a data.prepared_data_root config value."
            )
        manifest = convert_jetclass_archives(
            raw_data_root=raw_root,
            prepared_data_root=prepared_root,
            target_mode=args.target_mode,
            target_config=str(resolve_project_path(args.target_config)) if args.target_config else None,
            chunk_size=args.chunk_size,
            particle_cap=args.particle_cap,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
            max_files_per_partition=args.max_files_per_partition,
            max_jets_per_file=args.max_jets_per_file,
            malformed_policy=args.malformed_policy,
            max_malformed_fraction=args.max_malformed_fraction,
            full_conversion=args.full,
            size_estimate_manifest=resolve_project_path(args.size_estimate_manifest) if args.size_estimate_manifest else None,
            confirm_full_conversion=args.confirm_full_conversion,
        )
    except (ValueError, RootDataError, ConversionError) as exc:
        parser.error(str(exc))
    print(
        "Conversion complete: "
        f"{manifest['conversion']['processed_root_members']} ROOT members, "
        f"{manifest['conversion']['processed_jets']} jets, "
        f"{manifest['conversion']['output_bytes'] / (1024 ** 2):.2f} MiB prepared, "
        f"{manifest['conversion']['duration_seconds']:.1f}s."
    )
    print("Classes:", json.dumps(manifest["conversion"]["classes"], sort_keys=True))
    print("Manifest:", prepared_root / "canonical_split_manifest.json")


if __name__ == "__main__":
    main()
