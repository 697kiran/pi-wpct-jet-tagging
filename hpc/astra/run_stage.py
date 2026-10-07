"""Astra Python wrapper for reproducible PI-WPCT conversion/training stages.

Submit this wrapper through the Astra Jobs > Shell Script interface described
in the supplied guide. Resource selection is performed in Astra's UI because
the guide does not document portable PBS resource-directive syntax.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _required_root(cli_value, env_name):
    value = cli_value or os.environ.get(env_name)
    if not value:
        raise ValueError("Set %s or provide its command-line override." % env_name)
    return str(Path(value).expanduser().resolve())


def build_command(args, project_root, raw_root, prepared_root, output_root):
    python = args.python or os.environ.get("PIWPCT_PYTHON") or sys.executable
    converter = str(project_root / "scripts" / "convert_root.py")
    trainer = str(project_root / "scripts" / "run_regime.py")
    common = ["--target-mode", args.target_mode]
    if args.stage == "convert-smoke":
        if args.max_files_per_partition < 2:
            raise ValueError("Conversion smoke needs at least two train ROOT members for a file-disjoint calibration split.")
        return [
            python, converter, "--real-data",
            "--raw-data-root", raw_root,
            "--prepared-data-root", prepared_root,
            "--output-root", output_root,
        ] + common + [
            "--max-files-per-partition", str(args.max_files_per_partition),
            "--max-jets-per-file", str(args.max_jets_per_file),
            "--chunk-size", str(args.chunk_size),
            "--particle-cap", str(args.particle_cap),
        ]
    if args.stage == "convert-full":
        if not args.size_estimate_manifest:
            raise ValueError("Full conversion requires --size-estimate-manifest from a previous tiny conversion.")
        return [
            python, converter, "--real-data",
            "--raw-data-root", raw_root,
            "--prepared-data-root", prepared_root,
            "--output-root", output_root,
        ] + common + [
            "--full", "--confirm-full-conversion",
            "--size-estimate-manifest", str(Path(args.size_estimate_manifest).resolve()),
            "--chunk-size", str(args.chunk_size),
            "--particle-cap", str(args.particle_cap),
        ]
    regime = args.regime
    if args.stage == "train-smoke":
        regime = "baseline_ce"
    if args.target_mode == "process_classification" and regime in {"wpct", "pi_wpct"}:
        raise ValueError("WPCT/PI-WPCT require flavour labels; process classes are a separate objective.")
    command = [
        python, trainer, "--real-data",
        "--prepared-data-root", prepared_root,
        "--output-root", output_root,
        "--workers", str(args.workers),
        "--particle-cap", str(args.particle_cap),
        "--batch-size", str(args.batch_size),
        "--regime", regime,
        "--backbone", args.backbone,
        "--epochs", str(1 if args.stage == "train-smoke" else args.epochs),
    ] + common
    if args.stage == "train-smoke":
        command.append("--allow-incomplete-data")
    if args.pin_memory:
        command.append("--pin-memory")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("convert-smoke", "convert-full", "train-smoke", "train-full"))
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--python", help="Project virtual-environment interpreter; defaults to PIWPCT_PYTHON or this interpreter")
    parser.add_argument("--raw-data-root", help="Overrides RAW_DATA_ROOT")
    parser.add_argument("--prepared-data-root", help="Overrides PREPARED_DATA_ROOT")
    parser.add_argument("--output-root", help="Overrides OUTPUT_ROOT")
    parser.add_argument("--target-mode", required=True, choices=("process_classification", "flavour_tagging"))
    parser.add_argument("--size-estimate-manifest")
    parser.add_argument("--max-files-per-partition", type=int, default=2)
    parser.add_argument("--max-jets-per-file", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--particle-cap", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--regime", default="pi_wpct",
                        choices=("baseline_ce", "ce_temp_scaling", "ce_mondrian_cp", "conftr", "wpct", "pi_wpct"))
    parser.add_argument("--backbone", default="pn_lite",
                        choices=("pn_lite", "lorentznet_small", "part", "gbt"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    try:
        project_root = Path(args.project_root).expanduser().resolve()
        raw_root = _required_root(args.raw_data_root, "RAW_DATA_ROOT")
        prepared_root = _required_root(args.prepared_data_root, "PREPARED_DATA_ROOT")
        output_root = _required_root(args.output_root, "OUTPUT_ROOT")
        command = build_command(args, project_root, raw_root, prepared_root, output_root)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ["RAW_DATA_ROOT"] = raw_root
    os.environ["PREPARED_DATA_ROOT"] = prepared_root
    os.environ["OUTPUT_ROOT"] = output_root
    print("Running on one allocated accelerator; command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(project_root), env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
