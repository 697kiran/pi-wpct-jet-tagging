"""Bounded ROOT/TAR reading and appendable NPY storage for JetClass."""

import os
import re
import shutil
import struct
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from piwpct.schema import (
    JET_METADATA_BRANCHES,
    RAW_PARTICLE_BRANCHES,
)
from piwpct.targets import (
    TargetDefinitionError,
    normalize_one_hot_labels,
    validate_label_branches,
)


class RootDataError(ValueError):
    """Raised for malformed ROOT schemas, samples, or unsupported inputs."""


_PARTICLE_ID_BRANCHES = (
    "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
    "part_isElectron", "part_isMuon",
)


def archive_partition(name: str) -> Optional[str]:
    lower = name.lower()
    if "_train_" in lower or "_train-" in lower:
        return "train"
    if "_val_" in lower or "_validation_" in lower:
        return "validation"
    if "_test_" in lower:
        return "test"
    return None


def discover_root_members(
    raw_data_root: Path,
    max_files_per_partition: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Enumerate ROOT members without extracting archive contents."""
    raw_data_root = raw_data_root.resolve()
    if not raw_data_root.is_dir():
        raise RootDataError(f"RAW_DATA_ROOT does not exist or is not a directory: {raw_data_root}")
    if max_files_per_partition is not None and max_files_per_partition < 1:
        raise RootDataError("max_files_per_partition must be positive.")

    archives = sorted(
        path for path in raw_data_root.rglob("*")
        if path.is_file() and (path.name.lower().endswith(".tar") or path.name.lower().endswith(".tar.gz"))
    )
    collected: Dict[str, List[Dict[str, str]]] = {
        "train": [], "validation": [], "test": []
    }
    for archive in archives:
        partition = archive_partition(archive.name)
        if partition is None:
            continue
        if max_files_per_partition and len(collected[partition]) >= max_files_per_partition:
            continue
        relative_archive = PurePosixPath(archive.relative_to(raw_data_root).as_posix()).as_posix()
        try:
            with tarfile.open(str(archive), mode="r:*") as tar:
                for member in tar:
                    if member.isfile() and member.name.lower().endswith(".root"):
                        collected[partition].append({
                            "archive_path": str(archive),
                            "archive_relative_path": relative_archive,
                            "member_name": PurePosixPath(member.name).as_posix(),
                            "partition": partition,
                        })
                        if max_files_per_partition and len(collected[partition]) >= max_files_per_partition:
                            break
        except (OSError, tarfile.TarError) as exc:
            raise RootDataError(f"Could not enumerate TAR archive {relative_archive}: {exc}") from exc

    for partition in collected:
        collected[partition].sort(key=lambda row: (row["archive_relative_path"], row["member_name"]))
    missing = [name for name in ("train", "validation", "test") if not collected[name]]
    if missing:
        raise RootDataError(
            f"No ROOT members found for official partition(s) {missing} under {raw_data_root}. "
            "Expected TAR names containing _train_, _val_, and _test_."
        )
    return collected["train"] + collected["validation"] + collected["test"]


def assign_source_member_splits(
    members: Sequence[Mapping[str, str]],
    calibration_fraction: float = 0.10,
    seed: int = 42,
) -> Dict[Tuple[str, str], str]:
    """Reserve complete ROOT members from training archives for calibration."""
    if not 0.0 < calibration_fraction < 1.0:
        raise RootDataError("calibration_fraction must be strictly between 0 and 1.")
    training = sorted(
        (row for row in members if row["partition"] == "train"),
        key=lambda row: (row["archive_relative_path"], row["member_name"]),
    )
    if len(training) < 2:
        raise RootDataError(
            "At least two training ROOT members are required to keep training and calibration "
            "source-file-disjoint. Increase --max-files-per-partition."
        )
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(training))
    n_cal = min(len(training) - 1, max(1, int(round(len(training) * calibration_fraction))))
    calibration_keys = {
        (training[int(index)]["archive_relative_path"], training[int(index)]["member_name"])
        for index in order[:n_cal]
    }
    result: Dict[Tuple[str, str], str] = {}
    for row in members:
        key = (row["archive_relative_path"], row["member_name"])
        if row["partition"] == "train":
            result[key] = "calibration" if key in calibration_keys else "train"
        else:
            result[key] = row["partition"]
    return result


class NpyAppendWriter:
    """Write a row-appendable NPY array and patch its row count on close."""

    _HEADER_ROWS = "12345678901234567890"

    def __init__(self, path: Path, dtype: str, trailing_shape: Tuple[int, ...] = ()):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.trailing_shape = tuple(trailing_shape)
        self.count = 0
        self._stream = open(path, "wb+")
        self._header_row_offset = 0
        self._header_length = 0
        self._write_header(self._HEADER_ROWS, initial=True)

    def _write_header(self, rows: str, initial: bool = False) -> None:
        descr = np.lib.format.dtype_to_descr(self.dtype)
        shape = (rows,) + self.trailing_shape
        shape_text = "(" + ", ".join([str(shape[0])] + [str(dim) for dim in shape[1:]])
        if len(shape) == 1:
            shape_text += ","
        shape_text += ")"
        header_text = "{'descr': %r, 'fortran_order': False, 'shape': %s, }" % (descr, shape_text)
        header = header_text.encode("ascii")
        preamble_size = 10
        padding = (64 - ((preamble_size + len(header) + 1) % 64)) % 64
        header = header + b" " * padding + b"\n"
        if initial:
            self._header_length = len(header)
            match = re.search(b"'shape': \\(", header)
            if not match:
                raise RootDataError("Internal error constructing an appendable NPY header.")
            self._header_row_offset = match.end()
            self._stream.write(b"\x93NUMPY\x01\x00")
            self._stream.write(struct.pack("<H", self._header_length))
            self._stream.write(header)
            return
        row_field = rows.encode("ascii")
        patched = bytearray(header)
        patched[self._header_row_offset:self._header_row_offset + 20] = row_field
        if len(patched) != self._header_length:
            raise RootDataError("Internal error: NPY header length changed while finalizing array.")
        self._stream.seek(10)
        self._stream.write(patched)
        self._stream.flush()

    def append(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=self.dtype)
        expected_shape = (len(array),) + self.trailing_shape
        if array.shape != expected_shape:
            raise RootDataError(f"Array append shape {array.shape} does not match {expected_shape}.")
        array = np.ascontiguousarray(array)
        array.tofile(self._stream)
        self.count += len(array)

    def close(self) -> None:
        if self._stream.closed:
            return
        self._write_header("%-20d" % self.count)
        self._stream.close()

    def __enter__(self) -> "NpyAppendWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._stream.close()


def _stage_member(archive_path: str, member_name: str, staging_dir: Path) -> Path:
    suffix = ".root"
    fd, path_string = tempfile.mkstemp(prefix="jetclass-", suffix=suffix, dir=str(staging_dir))
    os.close(fd)
    staged = Path(path_string)
    try:
        with tarfile.open(archive_path, mode="r:*") as tar:
            member = tar.getmember(member_name)
            if not member.isfile():
                raise RootDataError(f"TAR member {member_name!r} is not a regular file.")
            source = tar.extractfile(member)
            if source is None:
                raise RootDataError(f"Could not read TAR member {member_name!r}.")
            with source, open(staged, "wb") as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _array_to_numpy(awkward: Any, array: Any, branch: str) -> np.ndarray:
    try:
        result = np.asarray(awkward.to_numpy(array))
    except Exception as exc:
        raise RootDataError(f"Branch {branch!r} is not a regular per-jet scalar array: {exc}") from exc
    if result.ndim != 1:
        raise RootDataError(f"Branch {branch!r} must have shape [num_jets], got {result.shape}.")
    return result


def _row_condition(awkward: Any, expression: Any, rows: int) -> np.ndarray:
    try:
        expression = awkward.fill_none(expression, False)
        result = np.asarray(awkward.to_numpy(expression), dtype=bool)
    except Exception as exc:
        raise RootDataError(f"Could not validate jagged ROOT values: {exc}") from exc
    if result.shape != (rows,):
        raise RootDataError(f"Jagged branch condition has shape {result.shape}; expected {(rows,)}.")
    return result


def convert_root_member(
    member: Mapping[str, str],
    output_dir: Path,
    file_id: int,
    target_definition: Mapping[str, Any],
    *,
    staging_dir: Path,
    chunk_size: int = 2048,
    particle_cap: int = 128,
    max_jets: Optional[int] = None,
    malformed_policy: str = "error",
    max_malformed_fraction: float = 0.0,
) -> Dict[str, Any]:
    """Stream one ROOT member in bounded entry chunks into one jagged shard."""
    if chunk_size < 1 or particle_cap < 1:
        raise RootDataError("chunk_size and particle_cap must be positive.")
    if max_jets is not None and max_jets < 1:
        raise RootDataError("max_jets must be positive when provided.")
    if malformed_policy not in {"error", "drop"}:
        raise RootDataError("malformed_policy must be 'error' or 'drop'.")
    if not 0.0 <= max_malformed_fraction < 1.0:
        raise RootDataError("max_malformed_fraction must be in [0, 1).")
    if malformed_policy == "drop" and max_malformed_fraction <= 0.0:
        raise RootDataError("--malformed-policy drop requires an explicit positive maximum malformed fraction.")

    try:
        import awkward as ak
        import uproot
        from uproot.source.file import MemmapSource
    except ImportError as exc:
        raise RootDataError(
            "ROOT ingestion requires uproot and awkward. Install project dependencies with `pip install -e .`."
        ) from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    staged = _stage_member(member["archive_path"], member["member_name"], staging_dir)
    raw_branches = list(RAW_PARTICLE_BRANCHES)
    jet_branches = list(JET_METADATA_BRANCHES)
    label_branches = list(target_definition["label_branches"])
    required = raw_branches + jet_branches + label_branches
    count_total = 0
    malformed_total = 0
    class_counts = {name: 0 for name in target_definition["class_names"]}
    constituent_total = 0
    truncated_jets = 0
    total_entries = 0
    try:
        # An explicit local-file handler avoids fsspec's asynchronous socket
        # machinery for staged local ROOT members and uses direct mmap reads.
        with uproot.open(
            staged,
            handler=MemmapSource,
            object_cache=None,
            array_cache=None,
        ) as root_file:
            if "tree" not in root_file:
                raise RootDataError(f"ROOT member {member['member_name']} has no required tree named 'tree'.")
            tree = root_file["tree"]
            available = [str(key).split(";")[0] for key in tree.keys()]
            missing_particles = [name for name in raw_branches if name not in available]
            missing_jets = [name for name in jet_branches if name not in available]
            if missing_particles or missing_jets:
                raise RootDataError(
                    f"ROOT member {member['member_name']} is missing required branches: "
                    f"particles={missing_particles}, jet metadata={missing_jets}."
                )
            try:
                validate_label_branches(target_definition, available)
            except TargetDefinitionError as exc:
                raise RootDataError(str(exc)) from exc

            entry_stop = min(tree.num_entries, max_jets) if max_jets is not None else tree.num_entries
            with NpyAppendWriter(output_dir / "particles.npy", "<f4", (len(raw_branches),)) as particles_writer, \
                 NpyAppendWriter(output_dir / "offsets.npy", "<i8") as offsets_writer, \
                 NpyAppendWriter(output_dir / "jets.npy", "<f4", (len(jet_branches),)) as jets_writer, \
                 NpyAppendWriter(output_dir / "labels.npy", "<i8") as labels_writer, \
                 NpyAppendWriter(output_dir / "sample_ids.npy", "<u8", (2,)) as sample_ids_writer:
                offsets_writer.append(np.array([0], dtype=np.int64))
                constituent_offset = 0
                entry_cursor = 0
                for arrays in tree.iterate(
                    expressions=required,
                    step_size=chunk_size,
                    library="ak",
                    how=dict,
                    entry_stop=entry_stop,
                ):
                    rows = len(arrays[label_branches[0]])
                    if rows == 0:
                        continue
                    counts_by_branch: Dict[str, np.ndarray] = {}
                    for branch in raw_branches:
                        try:
                            counts_by_branch[branch] = np.asarray(ak.to_numpy(ak.num(arrays[branch], axis=1)), dtype=np.int64)
                        except Exception as exc:
                            raise RootDataError(
                                f"Particle branch {branch!r} must be a jagged array in {member['member_name']}: {exc}"
                            ) from exc
                    base_counts = counts_by_branch[raw_branches[0]]
                    if base_counts.shape != (rows,):
                        raise RootDataError(f"ROOT jagged branch row count mismatch in {member['member_name']}.")
                    valid = np.ones(rows, dtype=bool)
                    for branch, branch_counts in counts_by_branch.items():
                        if branch_counts.shape != (rows,):
                            raise RootDataError(f"Jagged branch {branch!r} has an unexpected outer shape.")
                        valid &= branch_counts == base_counts
                        valid &= _row_condition(
                            ak,
                            ak.all(np.isfinite(arrays[branch]), axis=1),
                            rows,
                        )

                    jets_by_branch = {
                        branch: _array_to_numpy(ak, arrays[branch], branch)
                        for branch in jet_branches
                    }
                    jet_matrix = np.column_stack([jets_by_branch[name] for name in jet_branches]).astype(np.float32)
                    valid &= np.isfinite(jet_matrix).all(axis=1)
                    valid &= jet_matrix[:, jet_branches.index("jet_pt")] > 0
                    valid &= jet_matrix[:, jet_branches.index("jet_energy")] > 0
                    valid &= jet_matrix[:, jet_branches.index("jet_nparticles")] == base_counts
                    valid &= base_counts > 0

                    for branch in _PARTICLE_ID_BRANCHES:
                        is_binary = (arrays[branch] == 0) | (arrays[branch] == 1)
                        valid &= _row_condition(ak, ak.all(is_binary, axis=1), rows)
                    ptx = arrays["part_px"]
                    pty = arrays["part_py"]
                    pt_valid = np.hypot(ptx, pty) > 0
                    energy_valid = arrays["part_energy"] > 0
                    valid &= _row_condition(ak, ak.all(pt_valid, axis=1), rows)
                    valid &= _row_condition(ak, ak.all(energy_valid, axis=1), rows)

                    label_arrays = [_array_to_numpy(ak, arrays[name], name) for name in label_branches]
                    label_ids, labels_valid = normalize_one_hot_labels(label_arrays, target_definition)
                    valid &= labels_valid
                    malformed_here = int(np.count_nonzero(~valid))
                    malformed_total += malformed_here
                    count_total += rows
                    if malformed_here and malformed_policy == "error":
                        first_bad = int(np.flatnonzero(~valid)[0]) + entry_cursor
                        raise RootDataError(
                            f"Malformed ROOT sample at {member['member_name']} entry {first_bad}; "
                            "default policy is to fail. Use --malformed-policy drop with an explicit "
                            "--max-malformed-fraction only after inspecting the cause."
                        )

                    if np.any(valid):
                        good_counts = base_counts[valid]
                        raw_fields = []
                        for branch in raw_branches:
                            field = arrays[branch][valid]
                            raw_fields.append(np.asarray(ak.to_numpy(ak.flatten(field, axis=1)), dtype=np.float32))
                        raw_matrix = np.column_stack(raw_fields).astype(np.float32, copy=False)
                        good_jets = jet_matrix[valid]
                        good_labels = label_ids[valid]
                        entries = np.arange(entry_cursor, entry_cursor + rows, dtype=np.uint64)[valid]
                        sample_ids = np.column_stack((
                            np.full(len(entries), file_id, dtype=np.uint64), entries
                        ))
                        particles_writer.append(raw_matrix)
                        jets_writer.append(good_jets)
                        labels_writer.append(good_labels)
                        sample_ids_writer.append(sample_ids)
                        offsets_writer.append((np.cumsum(good_counts, dtype=np.int64) + constituent_offset).astype(np.int64))
                        constituent_offset += int(good_counts.sum())
                        constituent_total += int(good_counts.sum())
                        truncated_jets += int(np.count_nonzero(good_counts > particle_cap))
                        for class_id, class_name in enumerate(target_definition["class_names"]):
                            class_counts[class_name] += int(np.count_nonzero(good_labels == class_id))
                    entry_cursor += rows
                    total_entries += rows

                malformed_fraction = malformed_total / max(count_total, 1)
                if malformed_fraction > max_malformed_fraction:
                    raise RootDataError(
                        f"Malformed fraction {malformed_fraction:.6%} in {member['member_name']} exceeds "
                        f"the configured limit {max_malformed_fraction:.6%}."
                    )

        n_jets = sum(class_counts.values())
        if n_jets == 0:
            raise RootDataError(f"ROOT member {member['member_name']} contained no valid jets.")
        return {
            "n_jets": n_jets,
            "n_constituents": constituent_total,
            "average_constituents": constituent_total / n_jets,
            "truncated_jets": truncated_jets,
            "malformed_jets": malformed_total,
            "read_entries": count_total,
            "class_counts": class_counts,
            "output_bytes": sum(path.stat().st_size for path in output_dir.glob("*.npy")),
            "source_entry_stop": total_entries,
        }
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    finally:
        staged.unlink(missing_ok=True)
