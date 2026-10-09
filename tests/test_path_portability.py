"""Focused Windows-to-Linux path and manifest relocation checks."""

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from piwpct.data import PreparedShardedJetDataset, create_canonical_splits, generate_pilot_dataset, get_dataloaders
from piwpct.data_paths import PROJECT_ROOT, load_data_config, resolve_data_root, resolve_project_path
from piwpct.schema import JET_METADATA_BRANCHES, RAW_PARTICLE_BRANCHES


def test_project_defaults_and_relative_overrides_ignore_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_data_config("configs/data.yaml")["particle_cap"] == 128
    assert resolve_project_path("configs/experiment.yaml") == PROJECT_ROOT / "configs/experiment.yaml"
    assert resolve_data_root("prepared_data_root", None, {}, environ={}) == (
        PROJECT_ROOT / "data/prepared", "default"
    )
    assert resolve_data_root("raw_data_root", None, {}, environ={"RAW_DATA_ROOT": "external/raw"}) == (
        PROJECT_ROOT / "external/raw", "environment"
    )
    assert resolve_data_root("prepared_data_root", "external/prepared", {}, environ={}) == (
        PROJECT_ROOT / "external/prepared", "cli"
    )
    assert resolve_data_root("prepared_data_root", None, {"prepared_data_root": "configured/prepared"}, environ={}) == (
        PROJECT_ROOT / "configured/prepared", "config"
    )


def test_stage_runner_relative_paths_use_selected_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stage = runpy.run_path(str(PROJECT_ROOT / "hpc/astra/run_stage.py"))
    chosen_root = PROJECT_ROOT / "another-project"
    assert stage["_project_path"]("env/bin/python", chosen_root) == chosen_root / "env/bin/python"
    assert stage["_required_root"]("datasets/raw", "RAW_DATA_ROOT", chosen_root) == str(chosen_root / "datasets/raw")
    args = SimpleNamespace(
        stage="convert-full", python="env/bin/python", target_mode="process_classification",
        size_estimate_manifest="samples/manifest.json", chunk_size=256, particle_cap=128,
    )
    command = stage["build_command"](args, chosen_root, "raw", "prepared", "outputs")
    assert command[0] == str(chosen_root / "env/bin/python")
    assert command[command.index("--size-estimate-manifest") + 1] == str(chosen_root / "samples/manifest.json")


def test_absolute_cli_and_config_paths_are_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    assert resolve_data_root("raw_data_root", str(raw), {"raw_data_root": "ignored"}) == (raw, "cli")
    assert resolve_data_root("prepared_data_root", None, {"prepared_data_root": str(prepared)}) == (
        prepared, "config"
    )
    assert resolve_project_path(raw) == raw
    config = tmp_path / "custom.yaml"
    config.write_text("data:\n  particle_cap: 7\n", encoding="utf-8")
    assert load_data_config(str(config))["particle_cap"] == 7


@pytest.mark.parametrize("script", [
    "scripts/run_regime.py",
    "scripts/run_full_ablation_matrix.py",
    "scripts/run_pretraining_diagnostic.py",
    "scripts/convert_root.py",
    "scripts/build_splits.py",
    "hpc/astra/run_stage.py",
    "run_on_astra/submit_stage.py",
])
def test_entrypoint_help_from_other_cwd(script, tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def _small_pilot(directory):
    data = generate_pilot_dataset(n_samples=60, num_particles=16, seed=9)
    return create_canonical_splits(data, str(directory), seed=9)


def test_pilot_manifest_and_npzs_relocate_together(tmp_path):
    original = tmp_path / "original" / "splits"
    manifest = _small_pilot(original)
    assert manifest["files"] == {name: name + ".npz" for name in (
        "train", "validation", "calibration", "test"
    )}
    expected = get_dataloaders(str(original / "canonical_split_manifest.json"), batch_size=4)["train"].dataset[0]["labels"]
    relocated = tmp_path / "relocated"
    shutil.move(str(original), str(relocated))
    loaded = get_dataloaders(str(relocated / "canonical_split_manifest.json"), batch_size=4)
    assert loaded["train"].dataset[0]["labels"] == expected
    assert all(len(loader.dataset) > 0 for loader in loaded.values())


def test_legacy_windows_relative_references_use_checked_adjacent_npzs(tmp_path):
    directory = tmp_path / "old"
    _small_pilot(directory)
    manifest_path = directory / "canonical_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {name: f"data/splits\\{name}.npz" for name in manifest["files"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    moved = tmp_path / "moved"
    shutil.move(str(directory), str(moved))
    loaders = get_dataloaders(str(moved / "canonical_split_manifest.json"), batch_size=4)
    assert len(loaders["train"].dataset) > 0


def test_missing_pilot_npz_is_reported_without_regeneration(tmp_path):
    directory = tmp_path / "splits"
    _small_pilot(directory)
    missing = directory / "train.npz"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="Missing pilot split 'train' NPZ.*No data was generated"):
        get_dataloaders(str(directory / "canonical_split_manifest.json"), batch_size=4)
    assert not missing.exists()


def test_prepared_shard_references_survive_relocation(tmp_path):
    original = tmp_path / "prepared"
    shard = original / "shards" / "train" / "file-000000"
    shard.mkdir(parents=True)
    np.save(shard / "particles.npy", np.zeros((1, len(RAW_PARTICLE_BRANCHES)), dtype=np.float32))
    np.save(shard / "offsets.npy", np.array([0, 1], dtype=np.int64))
    np.save(shard / "jets.npy", np.zeros((1, len(JET_METADATA_BRANCHES)), dtype=np.float32))
    np.save(shard / "labels.npy", np.array([2], dtype=np.int64))
    np.save(shard / "sample_ids.npy", np.array([[1, 2]], dtype=np.uint64))
    manifest = {
        "format": "piwpct.prepared_jagged.v1",
        "splits": {"train": {"n_jets": 1, "shards": [{"path": "shards/train/file-000000", "n_jets": 1}]}},
    }
    (original / "canonical_split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    relocated = tmp_path / "relocated_prepared"
    shutil.move(str(original), str(relocated))
    dataset = PreparedShardedJetDataset(relocated / "canonical_split_manifest.json", "train")
    assert dataset[0]["labels"] == 2
    assert dataset[0]["sample_id"].tolist() == [1, 2]
