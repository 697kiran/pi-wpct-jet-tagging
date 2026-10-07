"""Small ROOT/TAR fixtures for bounded conversion and prepared-shard loading."""

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from piwpct.conversion import convert_jetclass_archives
from piwpct.data import PreparedShardedJetDataset, get_dataloaders
from piwpct.root_data import (
    NpyAppendWriter,
    RootDataError,
    assign_source_member_splits,
)
from piwpct.schema import FEATURE_NAMES, RAW_PARTICLE_BRANCHES
from piwpct.targets import (
    TargetDefinitionError,
    default_target_definition,
    load_target_definition,
    normalize_one_hot_labels,
    validate_label_branches,
)


def test_one_hot_process_labels_normalize_bool_integer_and_float_values():
    definition = default_target_definition("process_classification")
    ids, valid = normalize_one_hot_labels(
        [np.array([True, False, 0.0]), np.array([False, True, 0.0])],
        dict(definition, classes=definition["classes"][:2], class_names=["QCD", "Hbb"],
             label_branches=["label_QCD", "label_Hbb"]),
    )
    np.testing.assert_array_equal(ids, [0, 1, -1])
    np.testing.assert_array_equal(valid, [True, True, False])


def test_flavour_target_fails_clearly_when_only_process_labels_exist():
    definition = default_target_definition("flavour_tagging")
    branches = [branch for _, branch in default_target_definition("process_classification")["classes"]]
    with pytest.raises(TargetDefinitionError, match="do not identify light/charm/b jet flavour"):
        validate_label_branches(definition, branches)


def test_target_config_cannot_rename_process_labels_as_flavour(tmp_path):
    config = tmp_path / "bad-flavour-target.yaml"
    config.write_text(
        "target:\n"
        "  mode: flavour_tagging\n"
        "  semantics: per-jet light/charm/b\n"
        "  classes:\n"
        "    - {name: light, branch: label_QCD}\n"
        "    - {name: charm, branch: label_Hcc}\n"
        "    - {name: b, branch: label_Hbb}\n",
        encoding="utf-8",
    )
    with pytest.raises(TargetDefinitionError, match="cannot define light/charm/b"):
        load_target_definition("flavour_tagging", str(config))


def test_custom_verified_flavour_branch_names_remain_configurable(tmp_path):
    config = tmp_path / "valid-flavour-target.yaml"
    config.write_text(
        "target:\n"
        "  mode: flavour_tagging\n"
        "  semantics: verified per-jet light/charm/b annotation\n"
        "  classes:\n"
        "    - {name: light, branch: truth_light}\n"
        "    - {name: charm, branch: truth_charm}\n"
        "    - {name: b, branch: truth_b}\n",
        encoding="utf-8",
    )
    definition = load_target_definition("flavour_tagging", str(config))
    assert definition["label_branches"] == ["truth_light", "truth_charm", "truth_b"]


def test_process_target_config_cannot_use_flavour_class_names(tmp_path):
    config = tmp_path / "bad-process-target.yaml"
    config.write_text(
        "target:\n"
        "  mode: process_classification\n"
        "  classes:\n"
        "    - {name: light, branch: label_QCD}\n"
        "    - {name: charm, branch: label_Hcc}\n",
        encoding="utf-8",
    )
    with pytest.raises(TargetDefinitionError, match="light/charm/b are flavour labels"):
        load_target_definition("process_classification", str(config))


def test_training_calibration_members_are_disjoint_and_assignment_is_deterministic():
    members = [
        {"archive_relative_path": "train.tar", "member_name": "a.root", "partition": "train"},
        {"archive_relative_path": "train.tar", "member_name": "b.root", "partition": "train"},
        {"archive_relative_path": "val.tar", "member_name": "v.root", "partition": "validation"},
        {"archive_relative_path": "test.tar", "member_name": "t.root", "partition": "test"},
    ]
    first = assign_source_member_splits(members, 0.25, 11)
    second = assign_source_member_splits(members, 0.25, 11)
    assert first == second
    train_ids = {key for key, split in first.items() if split == "train"}
    cal_ids = {key for key, split in first.items() if split == "calibration"}
    assert train_ids.isdisjoint(cal_ids)
    assert len(cal_ids) == 1
    assert first[("val.tar", "v.root")] == "validation"
    assert first[("test.tar", "t.root")] == "test"


def test_appendable_npy_header_is_memmappable(tmp_path):
    path = tmp_path / "array.npy"
    with NpyAppendWriter(path, "<f4", (2,)) as writer:
        writer.append(np.array([[1, 2], [3, 4]], dtype=np.float32))
        writer.append(np.array([[5, 6]], dtype=np.float32))
    result = np.load(path, mmap_mode="r")
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4], [5, 6]])


def _write_root(path: Path, *, missing_branch=None, bad_row=None):
    uproot = pytest.importorskip("uproot")
    awkward = pytest.importorskip("awkward")
    lengths = [3, 2, 1]

    def jagged(value):
        return awkward.Array([[value for _ in range(length)] for length in lengths])

    arrays = {
        "part_deta": jagged(0.1),
        "part_dphi": jagged(-0.2),
        "part_px": awkward.Array([[float(i + 1) for i in range(n)] for n in lengths]),
        "part_py": jagged(0.0),
        "part_pz": jagged(1.0),
        "part_energy": awkward.Array([[float(i + 2) for i in range(n)] for n in lengths]),
        "part_d0val": jagged(0.2),
        "part_d0err": jagged(0.01),
        "part_dzval": jagged(-0.2),
        "part_dzerr": jagged(0.02),
        "part_charge": jagged(1.0),
        "part_isChargedHadron": jagged(1),
        "part_isNeutralHadron": jagged(0),
        "part_isPhoton": jagged(0),
        "part_isElectron": jagged(0),
        "part_isMuon": jagged(0),
        "jet_pt": np.full(3, 100.0, dtype=np.float32),
        "jet_eta": np.zeros(3, dtype=np.float32),
        "jet_phi": np.zeros(3, dtype=np.float32),
        "jet_energy": np.full(3, 200.0, dtype=np.float32),
        "jet_nparticles": np.asarray(lengths, dtype=np.float32),
        "jet_sdmass": np.ones(3, dtype=np.float32),
        "jet_tau1": np.ones(3, dtype=np.float32),
        "jet_tau2": np.ones(3, dtype=np.float32),
        "jet_tau3": np.ones(3, dtype=np.float32),
        "jet_tau4": np.ones(3, dtype=np.float32),
    }
    label_branches = [
        "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
        "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl",
    ]
    for branch in label_branches:
        arrays[branch] = np.array([branch == "label_QCD"] * 3, dtype=np.bool_)
    if missing_branch:
        arrays.pop(missing_branch)
    if bad_row == "multi_label":
        arrays["label_Hbb"][1] = True
    if bad_row == "nonfinite":
        arrays["part_d0val"] = awkward.Array([[0.2, np.nan, 0.2], [0.2, 0.2], [0.2]])
    with uproot.recreate(path) as root_file:
        root_file["tree"] = arrays


def _make_archive(raw_root: Path, name: str, roots):
    archive = raw_root / name
    with tarfile.open(archive, "w") as tar:
        for root_path, member_name in roots:
            tar.add(root_path, arcname=member_name)
    return archive


def _fixture_archives(tmp_path, *, bad_member=None, missing_branch=None):
    raw = tmp_path / "raw"
    root_dir = tmp_path / "root-files"
    raw.mkdir()
    root_dir.mkdir()
    train_members = []
    for index in range(2):
        root_path = root_dir / ("train%d.root" % index)
        _write_root(
            root_path,
            missing_branch=missing_branch if bad_member == "train%d.root" % index else None,
            bad_row="nonfinite" if bad_member == "train%d.root" % index else None,
        )
        train_members.append((root_path, "class/train%d.root" % index))
    val_path = root_dir / "val.root"
    test_path = root_dir / "test.root"
    _write_root(val_path)
    _write_root(
        test_path,
        missing_branch=missing_branch if bad_member == "test.root" else None,
        bad_row="nonfinite" if bad_member == "test.root" else None,
    )
    _make_archive(raw, "JetClass_Pythia_train_100M_part0.tar", train_members)
    _make_archive(raw, "JetClass_Pythia_val_5M.tar", [(val_path, "class/val.root")])
    _make_archive(raw, "JetClass_Pythia_test_20M.tar", [(test_path, "class/test.root")])
    return raw


def test_tiny_tar_conversion_manifest_offsets_ids_and_mmap_loader(tmp_path):
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    raw_root = _fixture_archives(tmp_path)
    prepared = tmp_path / "prepared"
    manifest = convert_jetclass_archives(
        raw_data_root=raw_root,
        prepared_data_root=prepared,
        target_mode="process_classification",
        chunk_size=2,
        particle_cap=1,
        max_files_per_partition=2,
        max_jets_per_file=3,
        seed=3,
    )
    manifest_path = prepared / "canonical_split_manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["is_pilot"] is False
    assert on_disk["complete"] is False
    assert on_disk["target_definition"]["mode"] == "process_classification"
    assert on_disk["target_definition"]["class_names"][0] == "QCD"
    assert "not per-jet flavour" in on_disk["target_definition"]["warning"]
    assert on_disk["feature_schema"]["ordering"] == list(FEATURE_NAMES)
    assert on_disk["conversion"]["processed_root_members"] == 4
    assert on_disk["conversion"]["classes"]["QCD"] == manifest["conversion"]["processed_jets"]
    assert all(stat["truncated_jets"] > 0 for stat in on_disk["conversion"]["split_statistics"].values())
    serialized = manifest_path.read_text(encoding="utf-8")
    assert "D:\\" not in serialized
    assert "/home/" not in serialized

    for split_name in ("train", "calibration", "validation", "test"):
        shard_meta = on_disk["splits"][split_name]["shards"][0]
        shard_dir = prepared / shard_meta["path"]
        particles = np.load(shard_dir / "particles.npy", mmap_mode="r")
        offsets = np.load(shard_dir / "offsets.npy", mmap_mode="r")
        assert particles.shape[1] == len(RAW_PARTICLE_BRANCHES)
        assert offsets.shape == (shard_meta["n_jets"] + 1,)
        assert offsets[0] == 0 and offsets[-1] == len(particles)
        assert np.all(np.diff(offsets) > 0)

    train_keys = {
        (value["archive_relative_path"], value["root_member_name"])
        for value in on_disk["file_id_lookup"].values() if value["assigned_split"] == "train"
    }
    calibration_keys = {
        (value["archive_relative_path"], value["root_member_name"])
        for value in on_disk["file_id_lookup"].values() if value["assigned_split"] == "calibration"
    }
    assert train_keys.isdisjoint(calibration_keys)

    train_dataset = PreparedShardedJetDataset(manifest_path, "train")
    assert isinstance(train_dataset._open_shard(0)["particles"], np.memmap)
    sample_ids = [train_dataset[index]["sample_id"].tolist() for index in range(len(train_dataset))]
    assert len(set(map(tuple, sample_ids))) == len(sample_ids)
    with pytest.raises(ValueError, match="bounded smoke conversion"):
        get_dataloaders(str(manifest_path), batch_size=2)
    loaders = get_dataloaders(
        str(manifest_path), batch_size=2, particle_cap=1, drop_last=False, allow_incomplete=True
    )
    batch = next(iter(loaders["train"]))
    assert batch["features"].shape == (2, 1, 17)
    assert batch["points"].shape == (2, 1, 2)
    assert batch["vectors"].shape == (2, 1, 4)
    assert torch_all(batch["mask"].numpy() == 1.0)
    assert batch["labels"].tolist() == [0, 0]
    # Real-constituent masks come from offsets, including zero-valued coordinates.
    assert batch["features"][:, 0, 0].tolist() == pytest.approx([0.1, 0.1])


def torch_all(values):
    return bool(np.asarray(values).all())


def test_requested_flavour_conversion_fails_and_does_not_publish_outputs(tmp_path):
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    raw_root = _fixture_archives(tmp_path)
    prepared = tmp_path / "should-not-exist"
    with pytest.raises(RootDataError, match="do not identify light/charm/b jet flavour"):
        convert_jetclass_archives(
            raw_data_root=raw_root,
            prepared_data_root=prepared,
            target_mode="flavour_tagging",
            max_files_per_partition=2,
            max_jets_per_file=2,
        )
    assert not prepared.exists()
    assert len(list(raw_root.glob("*.tar"))) == 3


def test_malformed_root_rows_fail_by_default(tmp_path):
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    raw_root = _fixture_archives(tmp_path, bad_member="test.root")
    with pytest.raises(RootDataError, match="Malformed ROOT sample"):
        convert_jetclass_archives(
            raw_data_root=raw_root,
            prepared_data_root=tmp_path / "malformed-output",
            target_mode="process_classification",
            max_files_per_partition=2,
            max_jets_per_file=3,
        )
    assert not (tmp_path / "malformed-output").exists()


def test_missing_required_particle_branch_is_reported(tmp_path):
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    raw_root = _fixture_archives(tmp_path, bad_member="test.root", missing_branch="part_dzerr")
    with pytest.raises(RootDataError, match="part_dzerr"):
        convert_jetclass_archives(
            raw_data_root=raw_root,
            prepared_data_root=tmp_path / "bad-schema-output",
            target_mode="process_classification",
            max_files_per_partition=2,
            max_jets_per_file=2,
        )
    assert not (tmp_path / "bad-schema-output").exists()
