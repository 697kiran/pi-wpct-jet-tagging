"""Safety checks for explicit synthetic-pilot and prepared real-data modes."""

import argparse
import json
import os
import subprocess
import sys

import pytest

from piwpct.cli_data import (
    SplitManifestError,
    add_data_mode_arguments,
    ensure_pilot_output_is_safe,
    ensure_split_manifest,
)


def _mode_parser():
    parser = argparse.ArgumentParser()
    add_data_mode_arguments(parser)
    return parser


def test_data_mode_must_be_explicit_and_mutually_exclusive():
    parser = _mode_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--pilot", "--real-data"])
    assert parser.parse_args(["--pilot"]).data_mode == "pilot"
    assert parser.parse_args(["--real-data"]).data_mode == "real"


def test_real_mode_missing_manifest_never_generates_pilot(tmp_path):
    manifest_path = tmp_path / "splits" / "canonical_split_manifest.json"
    with pytest.raises(SplitManifestError, match="no synthetic pilot was generated"):
        ensure_split_manifest(str(manifest_path), data_mode="real")
    assert not manifest_path.parent.exists()


def test_pilot_mode_does_not_generate_without_explicit_generation_permission(tmp_path):
    manifest_path = tmp_path / "splits" / "canonical_split_manifest.json"
    with pytest.raises(SplitManifestError, match="Create it explicitly"):
        ensure_split_manifest(str(manifest_path), data_mode="pilot")
    assert not manifest_path.parent.exists()


def test_real_mode_accepts_only_an_existing_nonpilot_manifest(tmp_path):
    manifest_path = tmp_path / "canonical_split_manifest.json"
    manifest_path.write_text(json.dumps({"is_pilot": False}), encoding="utf-8")
    assert ensure_split_manifest(str(manifest_path), data_mode="real")["is_pilot"] is False


def test_explicit_pilot_mode_may_generate_missing_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "splits" / "canonical_split_manifest.json"

    def generate_pilot_dataset(n_samples, seed):
        return {"n_samples": n_samples, "seed": seed}

    def create_canonical_splits(data, output_dir, seed, is_pilot):
        os.makedirs(output_dir, exist_ok=True)
        output = os.path.join(output_dir, "canonical_split_manifest.json")
        with open(output, "w", encoding="utf-8") as stream:
            json.dump({"is_pilot": is_pilot, "samples": data["n_samples"], "seed": seed}, stream)

    import types

    fake_data = types.ModuleType("piwpct.data")
    fake_data.generate_pilot_dataset = generate_pilot_dataset
    fake_data.create_canonical_splits = create_canonical_splits
    monkeypatch.setitem(sys.modules, "piwpct.data", fake_data)

    manifest = ensure_split_manifest(
        str(manifest_path),
        data_mode="pilot",
        generate_pilot_if_missing=True,
        pilot_samples=60,
        seed=7,
    )
    assert manifest == {"is_pilot": True, "samples": 60, "seed": 7}
    assert manifest_path.is_file()


@pytest.mark.parametrize(
    ("mode", "is_pilot"),
    [("pilot", False), ("real", True)],
)
def test_manifest_mode_mismatch_is_rejected(tmp_path, mode, is_pilot):
    manifest_path = tmp_path / "canonical_split_manifest.json"
    manifest_path.write_text(json.dumps({"is_pilot": is_pilot}), encoding="utf-8")
    with pytest.raises(SplitManifestError, match="Requested .* mode"):
        ensure_split_manifest(str(manifest_path), data_mode=mode)


def test_pilot_builder_refuses_to_overwrite_real_or_unclassified_splits(tmp_path):
    manifest_path = tmp_path / "canonical_split_manifest.json"
    manifest_path.write_text(json.dumps({"is_pilot": False}), encoding="utf-8")
    with pytest.raises(SplitManifestError, match="Refusing to replace"):
        ensure_pilot_output_is_safe(str(tmp_path))

    manifest_path.unlink()
    (tmp_path / "train.npz").write_bytes(b"existing split")
    with pytest.raises(SplitManifestError, match="without a manifest"):
        ensure_pilot_output_is_safe(str(tmp_path))


def test_build_splits_real_mode_fails_before_writing(tmp_path):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "scripts", "build_splits.py"),
            "--real-data",
            "--output-dir",
            str(tmp_path / "splits"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "scripts/convert_root.py" in result.stderr
    assert not (tmp_path / "splits").exists()
