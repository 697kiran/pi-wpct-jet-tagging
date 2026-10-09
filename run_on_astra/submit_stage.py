#!/usr/bin/env python3
"""No-argument Astra GUI entry point; configuration is read beside this file."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


WRAPPER_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = WRAPPER_DIR.parent
DEFAULT_CONFIG = WRAPPER_DIR / "job_config.json"
STAGES = {"convert-smoke", "convert-full", "train-smoke", "train-full"}
TARGET_MODES = {"process_classification", "flavour_tagging"}
REGIMES = {"baseline_ce", "ce_temp_scaling", "ce_mondrian_cp", "conftr", "wpct", "pi_wpct"}
BACKBONES = {"pn_lite", "lorentznet_small", "part", "gbt"}
CONFIG_KEYS = {
    "enabled", "project_root", "python", "stage", "data_mode", "target_mode",
    "raw_data_root", "prepared_data_root", "output_root", "split_manifest",
    "target_config", "size_estimate_manifest", "confirm_full_conversion",
    "working_point", "seed", "run_id", "regime", "backbone", "epochs", "workers",
    "batch_size", "pin_memory", "conversion",
}


class ConfigurationError(ValueError):
    """Raised when deployment settings are missing or unsafe."""


def _configured_path(value: Any, *, name: str, project_root: Path, required: bool = False) -> Optional[Path]:
    if value is None:
        if required:
            raise ConfigurationError(f"Set {name} in {DEFAULT_CONFIG.name} before selecting the Astra wrapper.")
        return None
    if not isinstance(value, str) or not value.strip() or "<" in value or ">" in value:
        raise ConfigurationError(f"{name} must be a real path string, not an empty value or placeholder.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load JSON without depending on the process working directory."""
    path = path if path.is_absolute() else (WRAPPER_DIR / path).resolve()
    if not path.is_file():
        raise ConfigurationError(
            f"Deployment config is missing: {path}. Copy job_config.example.json to job_config.json, "
            "complete its settings, and set enabled to true."
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read deployment config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigurationError("Deployment config must be a JSON object.")
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise ConfigurationError(f"Unknown deployment config key(s): {', '.join(unknown)}.")
    return config


def resolve_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve config paths relative to the project, never the scheduler CWD."""
    if config.get("enabled") is not True:
        raise ConfigurationError("Deployment is disabled. Review the settings and deliberately set enabled to true.")
    project_root = _configured_path(
        config.get("project_root"), name="project_root", project_root=DEFAULT_PROJECT_ROOT
    ) or DEFAULT_PROJECT_ROOT.resolve()
    if not project_root.is_dir():
        raise ConfigurationError(f"Project root does not exist: {project_root}")
    stage = config.get("stage")
    if not isinstance(stage, str) or stage not in STAGES:
        raise ConfigurationError(f"stage must be one of: {', '.join(sorted(STAGES))}.")
    data_mode = config.get("data_mode")
    if not isinstance(data_mode, str) or data_mode not in {"real", "pilot"}:
        raise ConfigurationError("data_mode must explicitly be 'real' or 'pilot'.")
    target_mode = config.get("target_mode")
    if data_mode == "real" and (not isinstance(target_mode, str) or target_mode not in TARGET_MODES):
        raise ConfigurationError("Real data requires target_mode 'process_classification' or 'flavour_tagging'.")
    if data_mode == "pilot" and (stage != "train-smoke" or target_mode is not None):
        raise ConfigurationError("Pilot data is supported only for train-smoke and must not claim a real target_mode.")
    if stage.startswith("convert-") and data_mode != "real":
        raise ConfigurationError("ROOT conversion requires data_mode 'real'.")
    training = stage.startswith("train-")
    run_id = config.get("run_id")
    if run_id is not None:
        if not training:
            raise ConfigurationError("run_id is only valid for a training stage.")
        reserved = {"CON", "PRN", "AUX", "NUL"} | {
            f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
        }
        if (not isinstance(run_id, str) or not run_id or not run_id.isascii() or len(run_id) > 96
                or not run_id[0].isalnum()
                or run_id.endswith(".") or run_id.split(".", 1)[0].upper() in reserved
                or any(not (char.isalnum() or char in "._-") for char in run_id)):
            raise ConfigurationError("run_id must be a path-free 1-96 character ID using letters, digits, dots, underscores, or hyphens.")
    regime = config.get("regime")
    if training and stage == "train-full" and (not isinstance(regime, str) or regime not in REGIMES):
        raise ConfigurationError("train-full requires a supported regime.")
    if training and stage == "train-smoke" and regime is not None and regime != "baseline_ce":
        raise ConfigurationError("train-smoke uses baseline_ce; choose train-full for other regimes.")
    selected_regime = ("baseline_ce" if stage == "train-smoke" else regime) if training else None
    if stage.startswith("train-") and data_mode == "real" and target_mode == "process_classification" and selected_regime in {"wpct", "pi_wpct"}:
        raise ConfigurationError("WPCT/PI-WPCT require flavour labels; process classes are a different objective.")
    backbone = config.get("backbone", "pn_lite")
    if training and (not isinstance(backbone, str) or backbone not in BACKBONES):
        raise ConfigurationError(f"backbone must be one of: {', '.join(sorted(BACKBONES))}.")
    working_point = config.get("working_point", 0.70)
    if training and (not isinstance(working_point, (int, float)) or isinstance(working_point, bool) or not 0 < working_point < 1):
        raise ConfigurationError("working_point must be a number strictly between 0 and 1.")
    if not training:
        working_point = 0.70
    seed = config.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigurationError("seed must be a non-negative integer.")
    epochs = config.get("epochs", 10)
    if training and stage == "train-full" and (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1):
        raise ConfigurationError("train-full requires epochs to be a positive integer.")
    workers = config.get("workers", 2)
    batch_size = config.get("batch_size", 64)
    if training and (not isinstance(workers, int) or isinstance(workers, bool) or workers < 0):
        raise ConfigurationError("workers must be a non-negative integer.")
    if training and (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ConfigurationError("batch_size must be a positive integer.")
    if training and not isinstance(config.get("pin_memory", False), bool):
        raise ConfigurationError("pin_memory must be true or false.")
    conversion = config.get("conversion", {})
    if not isinstance(conversion, dict):
        raise ConfigurationError("conversion must be a JSON object.")
    particle_cap = conversion.get("particle_cap", 128) if stage.startswith("convert-") else config.get("particle_cap", 128)
    if not isinstance(particle_cap, int) or isinstance(particle_cap, bool) or particle_cap < 1:
        raise ConfigurationError("particle_cap must be a positive integer.")

    paths = {
        "raw_data_root": _configured_path(config.get("raw_data_root"), name="raw_data_root",
                                         project_root=project_root, required=stage.startswith("convert-")),
        "prepared_data_root": _configured_path(
            config.get("prepared_data_root"), name="prepared_data_root", project_root=project_root,
            required=(stage.startswith("convert-") or (training and data_mode == "real")),
        ),
        "output_root": _configured_path(config.get("output_root"), name="output_root",
                                       project_root=project_root, required=training),
        "split_manifest": _configured_path(config.get("split_manifest"), name="split_manifest",
                                           project_root=project_root,
                                           required=(data_mode == "pilot")),
        "target_config": _configured_path(config.get("target_config"), name="target_config",
                                         project_root=project_root),
        "size_estimate_manifest": _configured_path(
            config.get("size_estimate_manifest"), name="size_estimate_manifest", project_root=project_root,
            required=(stage == "convert-full"),
        ),
    }
    if stage == "convert-full" and config.get("confirm_full_conversion") is not True:
        raise ConfigurationError("convert-full requires confirm_full_conversion: true after reviewing the size estimate.")
    if training and data_mode == "real" and paths["split_manifest"] is None:
        paths["split_manifest"] = paths["prepared_data_root"] / "canonical_split_manifest.json"
    python = _configured_path(config.get("python"), name="python", project_root=project_root)
    if python is None:
        python = project_root / "env" / "bin" / "python"
    return {
        **config, **paths,
        "project_root": project_root,
        "python": python,
        "stage": stage,
        "data_mode": data_mode,
        "target_mode": target_mode,
        "regime": selected_regime,
        "backbone": backbone if training else "pn_lite",
        "working_point": float(working_point),
        "run_id": run_id,
        "seed": seed,
        "epochs": 1 if stage == "train-smoke" else epochs,
        "workers": workers,
        "batch_size": batch_size,
        "particle_cap": particle_cap,
    }


def build_command(settings: Dict[str, Any]) -> List[str]:
    """Build a fully explicit command for the existing stage runner."""
    runner = settings["project_root"] / "hpc" / "astra" / "run_stage.py"
    command = [
        str(settings["python"]), str(runner), settings["stage"],
        "--project-root", str(settings["project_root"]),
        "--python", str(settings["python"]),
        "--data-mode", settings["data_mode"],
        "--particle-cap", str(settings["particle_cap"]),
    ]
    if settings["stage"].startswith("train-"):
        command.extend(["--output-root", str(settings["output_root"])])
        if settings.get("run_id") is not None:
            command.extend(["--run-id", settings["run_id"]])
    if settings["target_mode"] is not None:
        command.extend(["--target-mode", settings["target_mode"]])
    stage_paths = []
    if settings["stage"].startswith("convert-"):
        stage_paths.extend((("raw_data_root", "--raw-data-root"),
                            ("prepared_data_root", "--prepared-data-root"),
                            ("target_config", "--target-config")))
    if settings["stage"] == "convert-full":
        stage_paths.append(("size_estimate_manifest", "--size-estimate-manifest"))
    for key, flag in stage_paths:
        if settings[key] is not None:
            command.extend([flag, str(settings[key])])
    if settings["stage"].startswith("convert-"):
        conversion = settings.get("conversion", {})
        if settings["stage"] == "convert-smoke":
            for name, flag, default in (
                ("max_files_per_partition", "--max-files-per-partition", 2),
                ("max_jets_per_file", "--max-jets-per-file", 8),
            ):
                value = conversion.get(name, default)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ConfigurationError(f"conversion.{name} must be a positive integer.")
                if name == "max_files_per_partition" and value < 2:
                    raise ConfigurationError("conversion.max_files_per_partition must be at least 2 for a disjoint calibration split.")
                command.extend([flag, str(value)])
        chunk_size = conversion.get("chunk_size", 256)
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ConfigurationError("conversion.chunk_size must be a positive integer.")
        calibration_fraction = conversion.get("calibration_fraction", 0.10)
        if (not isinstance(calibration_fraction, (int, float)) or isinstance(calibration_fraction, bool)
                or not 0.0 <= calibration_fraction < 1.0):
            raise ConfigurationError("conversion.calibration_fraction must be in [0, 1).")
        command.extend([
            "--chunk-size", str(chunk_size),
            "--calibration-fraction", str(calibration_fraction),
            "--seed", str(settings["seed"]),
        ])
    else:
        command.extend([
            "--workers", str(settings["workers"]),
            "--batch-size", str(settings["batch_size"]),
            "--wp", str(settings["working_point"]),
            "--seed", str(settings["seed"]),
            "--backbone", settings["backbone"],
            "--epochs", str(settings["epochs"]),
            "--regime", settings["regime"],
            "--split-manifest", str(settings["split_manifest"]),
        ])
        if settings["data_mode"] == "real":
            command.extend(["--prepared-data-root", str(settings["prepared_data_root"])])
        if settings.get("pin_memory") is True:
            command.append("--pin-memory")
    return command


def validate_project_sources(settings: Dict[str, Any]) -> None:
    """Check source files that must exist independently of future Astra data."""
    project_root = settings["project_root"]
    runner = project_root / "hpc" / "astra" / "run_stage.py"
    scientific_script = project_root / "scripts" / (
        "convert_root.py" if settings["stage"].startswith("convert-") else "run_regime.py"
    )
    for source in (runner, scientific_script):
        if not source.is_file():
            raise ConfigurationError(f"Required project source file is missing: {source}")


def validate_runtime(settings: Dict[str, Any]) -> None:
    """Check the selected Astra environment and input data before launch."""
    validate_project_sources(settings)
    project_root = settings["project_root"]
    if not settings["python"].is_file():
        raise ConfigurationError(
            f"Project interpreter is missing: {settings['python']}. Create the Astra env after login or configure python."
        )
    if settings["stage"].startswith("convert-"):
        if not settings["raw_data_root"].is_dir():
            raise ConfigurationError(f"Configured raw-data directory is missing: {settings['raw_data_root']}")
        if settings["prepared_data_root"].exists():
            raise ConfigurationError(
                f"Prepared destination already exists: {settings['prepared_data_root']}. Choose a new destination; conversion never overwrites."
            )
        if not settings["prepared_data_root"].parent.is_dir():
            raise ConfigurationError(
                f"Prepared destination parent is missing: {settings['prepared_data_root'].parent}. Create/verify it before conversion."
            )
        if settings["stage"] == "convert-full" and not settings["size_estimate_manifest"].is_file():
            raise ConfigurationError(f"Full conversion size-estimate manifest is missing: {settings['size_estimate_manifest']}")
        return
    if settings["data_mode"] == "real" and not settings["prepared_data_root"].is_dir():
        raise ConfigurationError(f"Configured prepared-data directory is missing: {settings['prepared_data_root']}")
    manifest = settings["split_manifest"]
    if manifest is None and settings["data_mode"] == "real":
        manifest = settings["prepared_data_root"] / "canonical_split_manifest.json"
    if not manifest.is_file():
        raise ConfigurationError(f"Required {settings['data_mode']} split manifest is missing: {manifest}")
    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read split manifest {manifest}: {exc}") from exc
    expected_pilot = settings["data_mode"] == "pilot"
    if not isinstance(manifest_data, dict):
        raise ConfigurationError(f"Split manifest must contain a JSON object: {manifest}")
    if manifest_data.get("is_pilot") is not expected_pilot:
        raise ConfigurationError(
            f"Configured data_mode {settings['data_mode']!r} does not match manifest {manifest}."
        )
    if not expected_pilot:
        target_definition = manifest_data.get("target_definition", {})
        manifest_target = target_definition.get("mode") if isinstance(target_definition, dict) else None
        if manifest_target != settings["target_mode"]:
            raise ConfigurationError(
                f"Requested target_mode {settings['target_mode']!r} does not match manifest target {manifest_target!r}."
            )
        if manifest_target == "process_classification" and settings["regime"] in {"wpct", "pi_wpct"}:
            raise ConfigurationError("WPCT/PI-WPCT require flavour labels; process classes are a different objective.")
        if settings["stage"] == "train-full" and manifest_data.get("complete") is not True:
            raise ConfigurationError("train-full requires a complete prepared manifest; use train-smoke for bounded/incomplete data.")


def run(settings: Dict[str, Any], *, dry_run: bool = False, runner=subprocess.run) -> int:
    command = build_command(settings)
    print("Astra project:", settings["project_root"], flush=True)
    print("Stage:", settings["stage"], flush=True)
    print("Interpreter:", settings["python"], flush=True)
    print("Command:", json.dumps(command), flush=True)
    if dry_run:
        validate_project_sources(settings)
        return 0
    validate_runtime(settings)
    environment = os.environ.copy()
    for key, env_name in (("raw_data_root", "RAW_DATA_ROOT"),
                          ("prepared_data_root", "PREPARED_DATA_ROOT"), ("output_root", "OUTPUT_ROOT")):
        value = settings.get(key)
        if value is not None:
            environment[env_name] = str(value)
        else:
            environment.pop(env_name, None)
    try:
        runner(command, cwd=str(settings["project_root"]), env=environment, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Astra stage {settings['stage']!r} failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1
    except OSError as exc:
        print(f"Could not start the configured project interpreter or stage runner: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON deployment file; defaults to job_config.json beside this wrapper.")
    parser.add_argument("--dry-run", action="store_true", help="Validate settings and print the command without launching it.")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (WRAPPER_DIR / config_path).resolve()
    try:
        settings = resolve_settings(load_config(config_path))
        return run(settings, dry_run=args.dry_run)
    except ConfigurationError as exc:
        print(f"Astra wrapper configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
