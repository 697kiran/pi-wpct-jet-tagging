import json
from pathlib import Path

from scripts.check_data_readiness import inspect_manifest, inspect_raw_root, transfer_method


def _prepared_manifest(root: Path, *, complete: bool = True, target: str = "process_classification") -> Path:
    splits = {}
    for split in ("train", "validation", "calibration", "test"):
        shard = root / "shards" / split / "part-0"
        shard.mkdir(parents=True)
        for name in ("particles", "offsets", "jets", "labels", "sample_ids"):
            (shard / f"{name}.npy").write_bytes(b"fixture")
        splits[split] = {"n_jets": 1, "shards": [{"path": f"shards/{split}/part-0", "n_jets": 1}]}
    manifest = {
        "format": "piwpct.prepared_jagged.v1",
        "format_version": 1,
        "complete": complete,
        "is_pilot": False,
        "target_definition": {"mode": target},
        "splits": splits,
    }
    path = root / "canonical_split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_shards_resolve_relative_to_manifest_after_relocation(tmp_path):
    relocated = tmp_path / "moved" / "dataset"
    relocated.mkdir(parents=True)
    manifest = _prepared_manifest(relocated)
    result = inspect_manifest(manifest, stage="train-full", target_mode="process_classification")
    assert result["status"] == "PRESENT_AND_STRUCTURALLY_VALID"
    assert result["resolved_references"] == 4


def test_missing_manifest_and_references_are_reported(tmp_path):
    assert inspect_manifest(tmp_path / "absent.json", stage="train-full", target_mode=None)["status"] == "MISSING"
    manifest = _prepared_manifest(tmp_path)
    (tmp_path / "shards" / "train" / "part-0" / "labels.npy").unlink()
    assert inspect_manifest(manifest, stage="train-full", target_mode=None)["status"] == "MISSING"


def test_incomplete_dataset_cannot_pass_full_training_readiness(tmp_path):
    manifest = _prepared_manifest(tmp_path, complete=False)
    result = inspect_manifest(manifest, stage="train-full", target_mode="process_classification")
    assert result["status"] == "PRESENT_BUT_INCOMPLETE"
    assert any("complete: true" in error for error in result["errors"])


def test_process_manifest_cannot_satisfy_flavour_training(tmp_path):
    manifest = _prepared_manifest(tmp_path, target="process_classification")
    result = inspect_manifest(manifest, stage="train-full", target_mode="flavour_tagging")
    assert result["status"] == "TARGET_NOT_VERIFIED"


def test_synthetic_pilot_does_not_pass_real_target_readiness(tmp_path):
    manifest = tmp_path / "pilot.json"
    manifest.write_text(json.dumps({"is_pilot": True, "files": {}}), encoding="utf-8")
    result = inspect_manifest(manifest, stage="train-smoke", target_mode="process_classification")
    assert result["status"] == "TARGET_NOT_VERIFIED"


def test_transfer_classification_uses_five_gib_boundary_without_copying():
    assert transfer_method(5_000_000_000 - 1).startswith("Astra upload interface")
    assert transfer_method(5_000_000_000).startswith("Confirm with ICTS")
    assert transfer_method(5_000_000_000 + 1).startswith("ICTS external-drive")


def test_preflight_is_read_only(tmp_path):
    manifest = _prepared_manifest(tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = inspect_manifest(manifest, stage="train-full", target_mode="process_classification")
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert result["status"] == "PRESENT_AND_STRUCTURALLY_VALID"
    assert after == before


def test_raw_inventory_reads_header_and_classifies_without_changing_archive(tmp_path):
    path = tmp_path / "tiny.tar"
    header = bytearray(512)
    header[148:156] = b"        "
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}".encode() + b"\0 "
    path.write_bytes(header)
    before = path.read_bytes()
    result = inspect_raw_root(tmp_path)
    assert result["archives"][0]["transfer_method"].startswith("Astra upload interface")
    assert path.read_bytes() == before
