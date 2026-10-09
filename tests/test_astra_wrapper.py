"""Safe command-construction and fail-closed checks for the Astra GUI wrapper."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = PROJECT_ROOT / "run_on_astra" / "submit_stage.py"
SPEC = importlib.util.spec_from_file_location("astra_submit_stage", WRAPPER_PATH)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def _config(root, **overrides):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "enabled": True,
        "project_root": str(root),
        "python": None,
        "stage": "train-full",
        "data_mode": "real",
        "target_mode": "flavour_tagging",
        "raw_data_root": None,
        "prepared_data_root": "prepared",
        "output_root": "outputs",
        "split_manifest": "prepared/canonical_split_manifest.json",
        "target_config": None,
        "size_estimate_manifest": None,
        "confirm_full_conversion": False,
        "working_point": 0.77,
        "seed": 2024,
        "run_id": None,
        "regime": "pi_wpct",
        "backbone": "pn_lite",
        "epochs": 12,
        "workers": 3,
        "batch_size": 32,
        "particle_cap": 96,
        "pin_memory": True,
        "conversion": {},
    }
    config.update(overrides)
    return config


def _write_runtime_tree(root, manifest):
    (root / "env" / "bin").mkdir(parents=True, exist_ok=True)
    (root / "env" / "bin" / "python").write_text("fake interpreter", encoding="utf-8")
    for relative in ("hpc/astra/run_stage.py", "scripts/run_regime.py", "scripts/convert_root.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake source", encoding="utf-8")
    (root / "prepared").mkdir(exist_ok=True)
    (root / "prepared" / "canonical_split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_arbitrary_working_directory_keeps_project_and_runner_paths(tmp_path, monkeypatch):
    project = tmp_path / "astra-project"
    settings = wrapper.resolve_settings(_config(project))
    first = wrapper.build_command(settings)
    monkeypatch.chdir(tmp_path)
    second = wrapper.build_command(wrapper.resolve_settings(_config(project)))
    assert first == second
    assert first[0] == str(project / "env" / "bin" / "python")
    assert first[1] == str(project / "hpc" / "astra" / "run_stage.py")
    assert first[first.index("--prepared-data-root") + 1] == str(project / "prepared")


@pytest.mark.parametrize(
    ("stage", "data_mode", "target_mode", "expected_script", "required_flags"),
    [
        ("convert-smoke", "real", "process_classification", "convert_root.py", ["--raw-data-root", "--prepared-data-root", "--target-mode"]),
        ("convert-full", "real", "flavour_tagging", "convert_root.py", ["--size-estimate-manifest", "--target-config", "--target-mode"]),
        ("train-smoke", "real", "flavour_tagging", "run_regime.py", ["--prepared-data-root", "--split-manifest", "--target-mode", "--regime"]),
        ("train-full", "real", "flavour_tagging", "run_regime.py", ["--prepared-data-root", "--split-manifest", "--target-mode", "--regime"]),
        ("train-smoke", "pilot", None, "run_regime.py", ["--data-mode", "--split-manifest"]),
    ],
)
def test_supported_stages_build_explicit_commands(tmp_path, stage, data_mode, target_mode, expected_script, required_flags):
    options = {"stage": stage, "data_mode": data_mode, "target_mode": target_mode}
    if stage == "train-smoke":
        options["regime"] = "baseline_ce"
    if stage.startswith("convert-"):
        options.update(raw_data_root="raw", prepared_data_root="prepared", output_root=None)
    if stage == "convert-full":
        options.update(size_estimate_manifest="sample/manifest.json", confirm_full_conversion=True,
                       target_config="configs/targets/flavour_tagging.yaml")
    if stage.startswith("train-"):
        options["split_manifest"] = "prepared/canonical_split_manifest.json" if data_mode == "real" else "pilot/canonical_split_manifest.json"
    settings = wrapper.resolve_settings(_config(tmp_path / "project", **options))
    command = wrapper.build_command(settings)
    if stage.startswith("convert-"):
        assert "--output-root" not in command
    assert command[1].endswith("run_stage.py")
    assert command[2] == stage
    for flag in required_flags:
        assert flag in command

    stage_spec = importlib.util.spec_from_file_location(
        "astra_stage_for_" + stage.replace("-", "_"), PROJECT_ROOT / "hpc" / "astra" / "run_stage.py"
    )
    stage_runner = importlib.util.module_from_spec(stage_spec)
    stage_spec.loader.exec_module(stage_runner)
    conversion = settings["conversion"]
    stage_args = SimpleNamespace(
        stage=stage,
        python=str(settings["python"]),
        data_mode=data_mode,
        target_mode=target_mode,
        regime=settings["regime"],
        backbone=settings["backbone"],
        epochs=settings["epochs"],
        workers=settings["workers"],
        batch_size=settings["batch_size"],
        particle_cap=settings["particle_cap"],
        pin_memory=settings["pin_memory"],
        wp=settings["working_point"],
        seed=settings["seed"],
        split_manifest=str(settings["split_manifest"]) if settings["split_manifest"] else None,
        target_config=str(settings["target_config"]) if settings["target_config"] else None,
        size_estimate_manifest=str(settings["size_estimate_manifest"]) if settings["size_estimate_manifest"] else None,
        max_files_per_partition=conversion.get("max_files_per_partition", 2),
        max_jets_per_file=conversion.get("max_jets_per_file", 8),
        chunk_size=conversion.get("chunk_size", 256),
    )
    downstream = stage_runner.build_command(
        stage_args,
        settings["project_root"],
        str(settings["raw_data_root"]) if settings["raw_data_root"] else None,
        str(settings["prepared_data_root"]) if settings["prepared_data_root"] else None,
        str(settings["output_root"]),
    )
    assert downstream[1].endswith(expected_script)
    if stage == "convert-full":
        assert "--max-files-per-partition" not in command
        assert "--max-jets-per-file" not in command


def test_training_choices_and_paths_are_forwarded_unchanged(tmp_path):
    project = tmp_path / "project"
    settings = wrapper.resolve_settings(_config(project, run_id="astra-run-01"))
    command = wrapper.build_command(settings)
    expected = {
        "--wp": "0.77", "--seed": "2024", "--regime": "pi_wpct", "--backbone": "pn_lite",
        "--epochs": "12", "--workers": "3", "--batch-size": "32", "--particle-cap": "96",
        "--target-mode": "flavour_tagging",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert "--pin-memory" in command
    assert command[command.index("--output-root") + 1] == str(project / "outputs")
    assert command[command.index("--run-id") + 1] == "astra-run-01"


def test_missing_or_disabled_config_fails_before_any_launch(tmp_path, capsys):
    with pytest.raises(wrapper.ConfigurationError, match="enabled to true"):
        wrapper.resolve_settings(_config(tmp_path / "project", enabled=False))
    result = wrapper.main(["--config", str(tmp_path / "missing.json")])
    assert result == 2
    assert "config is missing" in capsys.readouterr().err


def test_full_training_dry_run_does_not_require_astra_interpreter_or_data(tmp_path, monkeypatch):
    project = tmp_path / "project"
    for relative in ("hpc/astra/run_stage.py", "scripts/run_regime.py"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source", encoding="utf-8")
    settings = wrapper.resolve_settings(_config(project))
    launched = []
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *args, **kwargs: launched.append(args))
    assert wrapper.run(settings, dry_run=True) == 0
    assert launched == []
    assert not (project / "outputs").exists()


def test_execution_fails_clearly_when_project_interpreter_is_missing(tmp_path):
    project = tmp_path / "project"
    for relative in ("hpc/astra/run_stage.py", "scripts/run_regime.py"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source", encoding="utf-8")
    settings = wrapper.resolve_settings(_config(project))
    with pytest.raises(wrapper.ConfigurationError, match="Create the Astra env after login"):
        wrapper.validate_runtime(settings)


def test_child_failure_is_returned_and_manifest_target_is_checked(tmp_path, capsys):
    project = tmp_path / "project"
    process_manifest = {"is_pilot": False, "complete": True, "target_definition": {"mode": "process_classification"}}
    _write_runtime_tree(project, process_manifest)
    settings = wrapper.resolve_settings(_config(project, target_mode="flavour_tagging", regime="baseline_ce"))
    with pytest.raises(wrapper.ConfigurationError, match="does not match manifest target"):
        wrapper.validate_runtime(settings)

    flavour_manifest = {"is_pilot": False, "complete": True, "target_definition": {"mode": "flavour_tagging"}}
    _write_runtime_tree(project, flavour_manifest)
    settings = wrapper.resolve_settings(_config(project, regime="baseline_ce"))

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(17, args[0])

    assert wrapper.run(settings, runner=fail) == 17
    assert "failed with exit code 17" in capsys.readouterr().err


def test_process_classes_cannot_select_wpct_and_pilot_cannot_be_full_training(tmp_path):
    with pytest.raises(wrapper.ConfigurationError, match="WPCT/PI-WPCT require flavour labels"):
        wrapper.resolve_settings(_config(tmp_path / "project", target_mode="process_classification"))
    with pytest.raises(wrapper.ConfigurationError, match="Pilot data is supported only"):
        wrapper.resolve_settings(_config(tmp_path / "project", data_mode="pilot", target_mode=None,
                                         split_manifest="pilot/manifest.json"))


def test_conversion_smoke_requires_two_files_and_full_conversion_confirmation(tmp_path):
    settings = wrapper.resolve_settings(_config(
        tmp_path / "project", stage="convert-smoke", data_mode="real", target_mode="flavour_tagging",
        raw_data_root="raw", prepared_data_root="prepared", regime=None,
        conversion={"max_files_per_partition": 1},
    ))
    with pytest.raises(wrapper.ConfigurationError, match="at least 2"):
        wrapper.build_command(settings)
    with pytest.raises(wrapper.ConfigurationError, match="confirm_full_conversion"):
        wrapper.resolve_settings(_config(
            tmp_path / "full-project", stage="convert-full", data_mode="real", target_mode="flavour_tagging",
            raw_data_root="raw", prepared_data_root="prepared", regime=None,
            size_estimate_manifest="sample/manifest.json", confirm_full_conversion=False,
        ))


def test_existing_stage_runner_command_contract_remains_available():
    args = type("Args", (), {
        "stage": "convert-full", "python": "env/bin/python", "target_mode": "process_classification",
        "size_estimate_manifest": "samples/manifest.json", "chunk_size": 256, "particle_cap": 128,
        "target_config": None,
    })()
    stage_spec = importlib.util.spec_from_file_location("astra_run_stage", PROJECT_ROOT / "hpc" / "astra" / "run_stage.py")
    stage = importlib.util.module_from_spec(stage_spec)
    stage_spec.loader.exec_module(stage)
    command = stage.build_command(args, PROJECT_ROOT, "raw", "prepared", "outputs")
    assert command[0] == str(PROJECT_ROOT / "env" / "bin" / "python")
    assert "--confirm-full-conversion" in command
    assert command[command.index("--size-estimate-manifest") + 1] == str(PROJECT_ROOT / "samples" / "manifest.json")


def test_stage_runner_maps_wrapper_mode_and_training_selections_to_scientific_cli():
    stage_spec = importlib.util.spec_from_file_location("astra_run_stage_modes", PROJECT_ROOT / "hpc" / "astra" / "run_stage.py")
    stage = importlib.util.module_from_spec(stage_spec)
    stage_spec.loader.exec_module(stage)
    project = PROJECT_ROOT
    args = SimpleNamespace(
        stage="train-smoke", python=str(project / "env" / "bin" / "python"), data_mode="pilot",
        target_mode=None, regime="baseline_ce", workers=3, particle_cap=96, batch_size=32,
        backbone="pn_lite", epochs=7, pin_memory=True,
        split_manifest=str(project / "data" / "pilot" / "manifest.json"), wp=0.77, seed=2024,
        run_id="wrapper-trial-01",
    )
    command = stage.build_command(args, project, None, None, str(project / "outputs"))
    assert command[1] == str(project / "scripts" / "run_regime.py")
    assert "--pilot" in command
    assert command[command.index("--split-manifest") + 1] == args.split_manifest
    for flag, value in (("--wp", "0.77"), ("--seed", "2024"), ("--workers", "3"),
                        ("--batch-size", "32"), ("--particle-cap", "96"), ("--epochs", "1")):
        assert command[command.index(flag) + 1] == value
    assert "--pin-memory" in command
    assert command[command.index("--run-id") + 1] == "wrapper-trial-01"
