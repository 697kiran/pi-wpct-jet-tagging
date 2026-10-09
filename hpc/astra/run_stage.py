"""CLI orchestrator for the scientific Astra conversion/training stages.

The GUI-facing, no-argument entry point is run_on_astra/submit_stage.py.
Resource selection remains in Astra's UI; no scheduler directives are assumed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value, project_root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _required_root(cli_value, env_name, project_root):
    value = cli_value or os.environ.get(env_name)
    if not value:
        raise ValueError("Set %s or provide its command-line override." % env_name)
    return str(_project_path(value, project_root))


def build_command(args, project_root, raw_root, prepared_root, output_root):
    python = str(_project_path(args.python or os.environ.get("PIWPCT_PYTHON") or sys.executable, project_root))
    converter = str(project_root / "scripts" / "convert_root.py")
    trainer = str(project_root / "scripts" / "run_regime.py")
    common = ["--target-mode", args.target_mode]
    if args.stage == "convert-smoke":
        if args.max_files_per_partition < 2:
            raise ValueError("Conversion smoke needs at least two train ROOT members for a file-disjoint calibration split.")
        command = [
            python, converter, "--real-data",
            "--raw-data-root", raw_root,
            "--prepared-data-root", prepared_root,
        ] + common + [
            "--max-files-per-partition", str(args.max_files_per_partition),
            "--max-jets-per-file", str(args.max_jets_per_file),
            "--chunk-size", str(args.chunk_size),
            "--particle-cap", str(args.particle_cap),
            "--calibration-fraction", str(getattr(args, "calibration_fraction", 0.10)),
            "--seed", str(getattr(args, "seed", 42)),
        ]
        if getattr(args, "target_config", None):
            command.extend(["--target-config", str(_project_path(args.target_config, project_root))])
        return command
    if args.stage == "convert-full":
        if not args.size_estimate_manifest:
            raise ValueError("Full conversion requires --size-estimate-manifest from a previous tiny conversion.")
        command = [
            python, converter, "--real-data",
            "--raw-data-root", raw_root,
            "--prepared-data-root", prepared_root,
        ] + common + [
            "--full", "--confirm-full-conversion",
            "--size-estimate-manifest", str(_project_path(args.size_estimate_manifest, project_root)),
            "--chunk-size", str(args.chunk_size),
            "--particle-cap", str(args.particle_cap),
            "--calibration-fraction", str(getattr(args, "calibration_fraction", 0.10)),
            "--seed", str(getattr(args, "seed", 42)),
        ]
        if getattr(args, "target_config", None):
            command.extend(["--target-config", str(_project_path(args.target_config, project_root))])
        return command
    data_mode = getattr(args, "data_mode", "real")
    if data_mode not in {"real", "pilot"}:
        raise ValueError("Training data mode must be 'real' or 'pilot'.")
    if args.stage.startswith("convert-") and data_mode != "real":
        raise ValueError("ROOT conversion requires data mode 'real'.")
    regime = args.regime
    if args.stage == "train-smoke":
        regime = "baseline_ce"
    if args.target_mode == "process_classification" and regime in {"wpct", "pi_wpct"}:
        raise ValueError("WPCT/PI-WPCT require flavour labels; process classes are a separate objective.")
    command = [python, trainer]
    command.append("--real-data" if data_mode == "real" else "--pilot")
    if data_mode == "real":
        if not getattr(args, "target_mode", None):
            raise ValueError("Real-data training requires an explicit --target-mode.")
        command.extend(["--prepared-data-root", prepared_root, "--target-mode", args.target_mode])
    if getattr(args, "split_manifest", None):
        command.extend(["--split-manifest", str(_project_path(args.split_manifest, project_root))])
    command.extend([
        "--output-root", output_root,
        "--workers", str(args.workers),
        "--particle-cap", str(args.particle_cap),
        "--batch-size", str(args.batch_size),
        "--regime", regime,
        "--backbone", args.backbone,
        "--epochs", str(1 if args.stage == "train-smoke" else args.epochs),
        "--wp", str(getattr(args, "wp", 0.70)),
        "--seed", str(getattr(args, "seed", 42)),
    ])
    if getattr(args, "run_id", None):
        command.extend(["--run-id", args.run_id])
    if args.stage == "train-smoke":
        command.append("--allow-incomplete-data")
    if args.pin_memory:
        command.append("--pin-memory")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("convert-smoke", "convert-full", "train-smoke", "train-full"))
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--python", help="Project virtual-environment interpreter; defaults to PIWPCT_PYTHON or this interpreter")
    parser.add_argument("--raw-data-root", help="Overrides RAW_DATA_ROOT")
    parser.add_argument("--prepared-data-root", help="Overrides PREPARED_DATA_ROOT")
    parser.add_argument("--output-root", help="Overrides OUTPUT_ROOT")
    parser.add_argument("--run-id", help="Optional explicit training run ID; existing IDs are refused")
    parser.add_argument("--data-mode", choices=("real", "pilot"), default="real",
                        help="Training input mode; conversion always requires real ROOT data.")
    parser.add_argument("--target-mode", choices=("process_classification", "flavour_tagging"))
    parser.add_argument("--size-estimate-manifest")
    parser.add_argument("--split-manifest")
    parser.add_argument("--target-config")
    parser.add_argument("--max-files-per-partition", type=int, default=2)
    parser.add_argument("--max-jets-per-file", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--particle-cap", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--regime", default="pi_wpct",
                        choices=("baseline_ce", "ce_temp_scaling", "ce_mondrian_cp", "conftr", "wpct", "pi_wpct"))
    parser.add_argument("--backbone", default="pn_lite",
                        choices=("pn_lite", "lorentznet_small", "part", "gbt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--wp", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        project_root = _project_path(args.project_root, DEFAULT_PROJECT_ROOT)
        if args.output_root or os.environ.get("OUTPUT_ROOT"):
            output_root = _required_root(args.output_root, "OUTPUT_ROOT", project_root)
        elif args.stage.startswith("train-"):
            raise ValueError("Training requires OUTPUT_ROOT or --output-root.")
        else:
            output_root = None
        raw_root = None
        if args.stage.startswith("convert-"):
            raw_root = _required_root(args.raw_data_root, "RAW_DATA_ROOT", project_root)
        prepared_root = None
        if args.stage.startswith("convert-") or (args.stage.startswith("train-") and args.data_mode == "real"):
            prepared_root = _required_root(args.prepared_data_root, "PREPARED_DATA_ROOT", project_root)
        if args.stage.startswith("train-") and args.data_mode == "real" and not args.target_mode:
            raise ValueError("Real-data training requires an explicit --target-mode.")
        if args.stage.startswith("convert-") and not args.target_mode:
            raise ValueError("ROOT conversion requires an explicit --target-mode.")
        if args.stage == "convert-full" and not args.size_estimate_manifest:
            raise ValueError("Full conversion requires --size-estimate-manifest from a previous tiny conversion.")
        if args.stage == "train-full" and args.epochs < 1:
            raise ValueError("Full training requires a positive --epochs value.")
        if args.stage.startswith("convert-") and not 0.0 <= args.calibration_fraction < 1.0:
            raise ValueError("--calibration-fraction must be in [0, 1).")
        if args.stage.startswith("convert-") and args.data_mode != "real":
            raise ValueError("ROOT conversion requires --data-mode real.")
        if args.data_mode == "pilot" and args.stage != "train-smoke":
            raise ValueError("Pilot data is supported only for train-smoke.")
        if args.data_mode == "pilot" and args.target_mode is not None:
            raise ValueError("Pilot mode must not be assigned a real-data --target-mode.")
        command = build_command(args, project_root, raw_root, prepared_root, output_root)
    except ValueError as exc:
        parser.error(str(exc))
    if raw_root is not None:
        os.environ["RAW_DATA_ROOT"] = raw_root
    else:
        os.environ.pop("RAW_DATA_ROOT", None)
    if prepared_root is not None:
        os.environ["PREPARED_DATA_ROOT"] = prepared_root
    elif args.data_mode == "pilot":
        os.environ.pop("PREPARED_DATA_ROOT", None)
    if output_root is not None and args.stage.startswith("train-"):
        os.environ["OUTPUT_ROOT"] = output_root
    else:
        os.environ.pop("OUTPUT_ROOT", None)
    print("Executing stage command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(project_root), env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
