"""Script to generate, hash, and verify the canonical 4-way data split.

Usage:
    python scripts/build_splits.py --pilot --output-dir data/splits --seed 42
"""

import argparse
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.cli_data import (
    SplitManifestError,
    add_data_mode_arguments,
    ensure_pilot_output_is_safe,
)
from piwpct.data_paths import resolve_project_path


def main():
    parser = argparse.ArgumentParser(
        description="Build a synthetic pilot split. Use convert_root.py to prepare real ROOT data."
    )
    add_data_mode_arguments(parser)
    parser.add_argument("--n-samples", type=int, default=12000, help="Number of synthetic pilot samples")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for split files")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    if args.data_mode == "real":
        parser.error(
            "build_splits.py creates only synthetic pilot splits. Use scripts/convert_root.py "
            "--real-data with an explicit --target-mode to prepare real ROOT data."
        )

    output_dir = str(resolve_project_path(args.output_dir))
    try:
        ensure_pilot_output_is_safe(output_dir)
    except SplitManifestError as exc:
        parser.error(str(exc))

    # Keep heavyweight training dependencies out of the unsupported real-data path.
    from piwpct.data import create_canonical_splits, generate_pilot_dataset
    from piwpct.utils.seeding import seed_everything

    seed_everything(args.seed)
    print(f"[*] Building SYNTHETIC pilot splits in '{output_dir}' (seed={args.seed})...")

    print("[!] Generating synthetic pilot data with provisional light:b:c ratio (~4:1:1)...")
    data = generate_pilot_dataset(n_samples=args.n_samples, seed=args.seed)

    manifest = create_canonical_splits(
        data=data,
        output_dir=output_dir,
        fractions=(0.60, 0.15, 0.10, 0.15),
        seed=args.seed,
        is_pilot=True,
    )

    manifest_path = os.path.join(output_dir, "canonical_split_manifest.json")
    print("[+] Successfully built canonical splits!")
    print(f"[+] Manifest path: {manifest_path}")
    print(f"[+] Sample counts: {manifest['counts']}")
    print("[+] SHA-256 Checksums:")
    for split_name, sha in manifest["sha256_checksums"].items():
        print(f"    - {split_name}: {sha}")


if __name__ == "__main__":
    main()
