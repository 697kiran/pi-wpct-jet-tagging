"""Script to generate, hash, and verify the canonical 4-way data split.

Usage:
    python scripts/build_splits.py --pilot --output-dir data/splits --seed 42
"""

import argparse
import json
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.data import create_canonical_splits, generate_pilot_dataset
from piwpct.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Build canonical 4-way split artifact for JetClass.")
    parser.add_argument("--pilot", action="store_true", default=True, help="Generate pilot dataset")
    parser.add_argument("--n-samples", type=int, default=12000, help="Number of pilot samples")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for split files")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"[*] Building canonical splits in '{args.output_dir}' (pilot={args.pilot}, seed={args.seed})...")

    if args.pilot:
        print("[!] Generating pilot dataset with provisional light:b:c ratio (~4:1:1)...")
        data = generate_pilot_dataset(n_samples=args.n_samples, seed=args.seed)
    else:
        # Placeholder for full JetClass shard loading
        raise NotImplementedError("Full 189.8 GB JetClass raw shard ingestion requires mounted path.")

    manifest = create_canonical_splits(
        data=data,
        output_dir=args.output_dir,
        fractions=(0.60, 0.15, 0.10, 0.15),
        seed=args.seed,
        is_pilot=args.pilot,
    )

    manifest_path = os.path.join(args.output_dir, "canonical_split_manifest.json")
    print(f"[+] Successfully built canonical splits!")
    print(f"[+] Manifest path: {manifest_path}")
    print(f"[+] Sample counts: {manifest['counts']}")
    print(f"[+] SHA-256 Checksums:")
    for split_name, sha in manifest["sha256_checksums"].items():
        print(f"    - {split_name}: {sha}")


if __name__ == "__main__":
    main()
