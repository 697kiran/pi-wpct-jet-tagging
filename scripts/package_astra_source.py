#!/usr/bin/env python3
"""Build a deterministic, source-only Astra ZIP from the current working tree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "dist" / "astra"
ARCHIVE_NAME = "pi-wpct-jet-tagging-source.zip"
MANIFEST_NAME = "source_manifest.json"
CHECKSUM_NAME = "pi-wpct-jet-tagging-source.zip.sha256"
ROOT_FILES = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "DESIGN.md",
    "USER_MANUAL.md",
    "pyproject.toml",
    "requirements.txt",
)


def _iter_files(directory: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in path.parts):
            continue
        if path.suffix.lower() in suffixes:
            yield path


def package_files() -> List[Path]:
    """Return only allowlisted source paths; generated output is never scanned."""
    files = [PROJECT_ROOT / name for name in ROOT_FILES]
    for folder, suffixes in (
        ("src", {".py"}),
        ("scripts", {".py"}),
        ("configs", {".yaml", ".yml"}),
        ("tests", {".py"}),
        ("docs", {".md", ".json"}),
    ):
        files.extend(_iter_files(PROJECT_ROOT / folder, suffixes))
    for folder in ("hpc/astra", "run_on_astra"):
        files.extend(_iter_files(PROJECT_ROOT / folder, {".py", ".md"}))
    files.extend(
        path
        for path in _iter_files(PROJECT_ROOT / "run_on_astra", {".json"})
        if path.name == "job_config.example.json"
    )
    files.append(PROJECT_ROOT / "data" / "README.md")

    unique = {path.relative_to(PROJECT_ROOT).as_posix(): path for path in files}
    return [unique[name] for name in sorted(unique)]


def _validate_package(files: List[Path]) -> None:
    required = {
        ".gitignore",
        "pyproject.toml",
        "requirements.txt",
        "src/piwpct/__init__.py",
        "src/piwpct/utils/run_directory.py",
        "scripts/check_data_readiness.py",
        "scripts/package_astra_source.py",
        "hpc/astra/run_stage.py",
        "run_on_astra/submit_stage.py",
        "run_on_astra/job_config.example.json",
        "data/README.md",
    }
    paths = {path.relative_to(PROJECT_ROOT).as_posix() for path in files}
    missing = sorted(required - paths)
    if missing:
        raise RuntimeError("Required source files are missing: " + ", ".join(missing))
    for path in files:
        relative = path.relative_to(PROJECT_ROOT)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Not a regular source file: {relative}")
        if relative.parts[0] in {".git", "dist", "outputs", "plots", "env", ".venv"}:
            raise RuntimeError(f"Excluded path entered package: {relative}")
        if relative.as_posix() in {
            "run_on_astra/job_config.json",
            "data/transfer_manifest.local.json",
        }:
            raise RuntimeError(f"Private configuration entered package: {relative}")
        if path.stat().st_size > 50 * 1024 * 1024:
            raise RuntimeError(f"Unexpectedly large file entered source package: {relative}")
        if path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}:
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?i)(?<![\w])[A-Z]:[\\/](?![\\/])", text):
                raise RuntimeError(f"Absolute Windows-local path entered source package: {relative}")
            if re.search(r"(?i)/home/(?!<username>)[A-Za-z0-9._-]+(?:/|\b)", text):
                raise RuntimeError(f"Account-specific home path entered source package: {relative}")

    example = PROJECT_ROOT / "run_on_astra" / "job_config.example.json"
    example_config = json.loads(example.read_text(encoding="utf-8"))
    if example_config.get("enabled") is not False:
        raise RuntimeError("The packaged Astra job example must remain disabled")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_package() -> Dict[str, object]:
    files = package_files()
    _validate_package(files)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / ARCHIVE_NAME
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    checksum_path = OUTPUT_DIR / CHECKSUM_NAME

    entries = []
    payloads = []
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        payload = path.read_bytes()
        entries.append({"path": relative, "size_bytes": len(payload), "sha256": _sha256(payload)})
        payloads.append((relative, payload))

    temporary_archive = None
    try:
        with tempfile.NamedTemporaryFile(dir=OUTPUT_DIR, prefix=ARCHIVE_NAME + ".", suffix=".tmp", delete=False) as stream:
            temporary_archive = Path(stream.name)
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, payload in payloads:
                member = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.create_system = 3
                member.external_attr = 0o100644 << 16
                archive.writestr(member, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        archive_bytes = temporary_archive.read_bytes()
        archive_digest = _sha256(archive_bytes)
        os.replace(temporary_archive, archive_path)
        temporary_archive = None
    finally:
        if temporary_archive is not None and temporary_archive.exists():
            temporary_archive.unlink()

    manifest = {
        "schema": "piwpct.astra_source_package.v1",
        "archive": ARCHIVE_NAME,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_digest,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_atomic(manifest_path, manifest_bytes)
    _write_atomic(checksum_path, f"{archive_digest}  {ARCHIVE_NAME}\n".encode("ascii"))
    verify_package(archive_path, manifest)
    return {
        "archive_path": archive_path,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_digest,
        "manifest_path": manifest_path,
        "checksum_path": checksum_path,
        "file_count": len(entries),
    }


def verify_package(archive_path: Path, manifest: Dict[str, object]) -> None:
    expected = {entry["path"]: entry for entry in manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        actual = set(archive.namelist())
        if actual != set(expected):
            raise RuntimeError(
                f"ZIP inventory mismatch: missing={sorted(set(expected) - actual)}, "
                f"extra={sorted(actual - set(expected))}"
            )
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise RuntimeError(f"ZIP integrity failure in {corrupt_member}")
        for relative, entry in expected.items():
            payload = archive.read(relative)
            if len(payload) != entry["size_bytes"] or _sha256(payload) != entry["sha256"]:
                raise RuntimeError(f"Packaged file checksum mismatch: {relative}")

    if _sha256(archive_path.read_bytes()) != manifest["archive_sha256"]:
        raise RuntimeError("Archive SHA-256 does not match the source manifest")
    forbidden = (".git/", "env/", ".venv/", "__pycache__/", ".pytest_cache/", "outputs/", "plots/", "data/raw/", "data/prepared/")
    if any(name.startswith(forbidden) for name in expected):
        raise RuntimeError("An excluded data, environment, cache, or output path entered the ZIP")
    if "run_on_astra/job_config.json" in expected or "data/transfer_manifest.local.json" in expected:
        raise RuntimeError("An operator-specific configuration entered the ZIP")


def main() -> int:
    result = build_package()
    for key, value in result.items():
        if key.endswith("_path"):
            value = Path(value).relative_to(PROJECT_ROOT)
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
