import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from piwpct.train import train_pipeline
from piwpct.utils.logging_utils import EpochMetricsLogger, TrajectoryLogger
from piwpct.utils.run_directory import create_run_directory


def test_identical_science_parameters_get_distinct_run_directories(tmp_path):
    identity = {"run_type": "training", "regime": "pi_wpct", "backbone": "pn_lite", "target_wp": 0.7, "seed": 42}
    first, first_id = create_run_directory(str(tmp_path / "outputs"), identity=identity, project_root=tmp_path)
    second, second_id = create_run_directory(str(tmp_path / "outputs"), identity=identity, project_root=tmp_path)
    assert first != second
    assert first_id != second_id
    assert first.parent == second.parent == tmp_path / "outputs" / "runs"


def test_explicit_run_id_collision_is_refused(tmp_path):
    identity = {"run_type": "training", "regime": "baseline_ce", "seed": 1}
    create_run_directory(str(tmp_path / "outputs"), identity=identity, run_id="repeat-01", project_root=tmp_path)
    with pytest.raises(FileExistsError):
        create_run_directory(str(tmp_path / "outputs"), identity=identity, run_id="repeat-01", project_root=tmp_path)


@pytest.mark.parametrize("run_id", ["../escape", "has space", "CON", "name."])
def test_run_id_must_be_a_cross_platform_path_component(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        create_run_directory(str(tmp_path / "outputs"), identity={"run_type": "training"}, run_id=run_id, project_root=tmp_path)


def test_relative_output_root_is_project_anchored_from_arbitrary_cwd(tmp_path, monkeypatch):
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run_dir, _ = create_run_directory("outputs", identity={"run_type": "training"}, project_root=project)
    assert run_dir.parent == project / "outputs" / "runs"


def test_metadata_records_effective_identity_and_config(tmp_path):
    effective = {"epochs": 7, "learning_rate": 0.001}
    run_dir, run_id = create_run_directory(
        str(tmp_path / "outputs"),
        identity={"run_type": "training", "regime": "baseline_ce", "seed": 2024, "effective_config": effective},
        run_id="metadata-check",
        project_root=tmp_path,
    )
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == run_id
    assert metadata["regime"] == "baseline_ce"
    assert metadata["seed"] == 2024
    assert metadata["effective_config"] == effective
    assert metadata["status"] == "running"
    assert metadata["paths"]["run_directory"] == str(run_dir)


def test_trajectory_files_are_exclusive_and_run_tagged(tmp_path):
    first, first_id = create_run_directory(str(tmp_path / "outputs"), identity={"run_type": "training"}, project_root=tmp_path)
    second, second_id = create_run_directory(str(tmp_path / "outputs"), identity={"run_type": "training"}, project_root=tmp_path)
    first_log = first / "trajectories" / "trajectories.jsonl"
    second_log = second / "trajectories" / "trajectories.jsonl"
    TrajectoryLogger(str(first_log), run_id=first_id).log_event("probe", {"epoch": 0})
    TrajectoryLogger(str(second_log), run_id=second_id).log_event("probe", {"epoch": 0})
    assert json.loads(first_log.read_text(encoding="utf-8"))["run_id"] == first_id
    assert json.loads(second_log.read_text(encoding="utf-8"))["run_id"] == second_id
    with pytest.raises(FileExistsError):
        TrajectoryLogger(str(first_log), run_id=first_id)


def test_epoch_metrics_are_flushed_and_tagged_by_epoch_and_run(tmp_path):
    path = tmp_path / "metrics" / "epochs.jsonl"
    logger = EpochMetricsLogger(str(path), run_id="epoch-run")
    logger.log_epoch(3, {"train_loss": 0.4}, {"val_loss": 0.5})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["run_id"] == "epoch-run"
    assert record["epoch"] == 3
    assert record["validation_metrics"] == {"val_loss": 0.5}
    with pytest.raises(FileExistsError):
        EpochMetricsLogger(str(path), run_id="epoch-run")


def test_checkpoint_paths_are_isolated_for_identical_runs(tmp_path):
    identity = {"run_type": "training", "regime": "baseline_ce", "seed": 42}
    first, _ = create_run_directory(str(tmp_path / "outputs"), identity=identity, project_root=tmp_path)
    second, _ = create_run_directory(str(tmp_path / "outputs"), identity=identity, project_root=tmp_path)
    first_checkpoint = first / "checkpoints" / "checkpoint.pt"
    second_checkpoint = second / "checkpoints" / "checkpoint.pt"
    first_checkpoint.parent.mkdir(parents=True)
    second_checkpoint.parent.mkdir(parents=True)
    first_checkpoint.write_bytes(b"first")
    second_checkpoint.write_bytes(b"second")
    assert first_checkpoint.read_bytes() == b"first"
    assert second_checkpoint.read_bytes() == b"second"


def test_training_failure_marks_reserved_run_failed_without_reusing_it(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("mocked child failure")

    monkeypatch.setattr("piwpct.train._train_pipeline_in_run", fail)
    dataloaders = {"train": SimpleNamespace(dataset=SimpleNamespace(target_mode="process_classification"))}
    with pytest.raises(RuntimeError, match="mocked child failure"):
        train_pipeline(
            "baseline_ce", "pn_lite", {"experiment": {"epochs": 1}}, dataloaders,
            output_dir=str(tmp_path / "outputs"), run_id="failed-run", run_metadata={"dataset_mode": "real"},
        )
    failed = tmp_path / "outputs" / "runs" / "failed-run"
    metadata = json.loads((failed / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    with pytest.raises(FileExistsError):
        train_pipeline(
            "baseline_ce", "pn_lite", {"experiment": {"epochs": 1}}, dataloaders,
            output_dir=str(tmp_path / "outputs"), run_id="failed-run",
        )


def test_training_metadata_uses_effective_execution_arguments(tmp_path, monkeypatch):
    def finish(**kwargs):
        return {
            "run_dir": kwargs["output_dir"], "run_id": kwargs["run_id"],
            "checkpoint_path": None, "epoch_metrics_path": None,
            "best_epoch": 4, "best_val_loss": 0.25,
        }

    monkeypatch.setattr("piwpct.train._train_pipeline_in_run", finish)
    dataloaders = {"train": SimpleNamespace(dataset=SimpleNamespace(target_mode="flavour_tagging"))}
    result = train_pipeline(
        "baseline_ce", "pn_lite", {"experiment": {"epochs": 7}}, dataloaders,
        target_wp=0.77, seed=2024, output_dir=str(tmp_path / "outputs"),
        run_id="effective-config",
        run_metadata={"dataset_mode": "real", "dataset_identifier": "sample-a", "target_mode": "flavour_tagging"},
    )
    metadata = json.loads((Path(result["run_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_training_config"] == {
        "regime": "baseline_ce", "backbone": "pn_lite", "target_wp": 0.77, "seed": 2024, "epochs": 7,
    }
    assert metadata["dataset_identifier"] == "sample-a"
    assert metadata["status"] == "training_complete"
    assert metadata["best_epoch"] == 4
