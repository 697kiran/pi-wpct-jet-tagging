"""Small, collision-safe directory and metadata helpers for experiment runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


RUNS_DIRECTORY = "runs"
RUN_METADATA = "metadata.json"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize metadata value of type {type(value).__name__}.")


def _atomic_json_write(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_run_id(run_id: str) -> str:
    """Validate a one-component run ID suitable for Windows and POSIX paths."""
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    reserved_stem = run_id.split(".", 1)[0].upper() if isinstance(run_id, str) else ""
    if (not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id)
            or run_id in {".", ".."} or run_id.endswith(".") or reserved_stem in reserved):
        raise ValueError("run_id must be 1-96 letters, digits, dots, underscores, or hyphens; it cannot be a path.")
    return run_id


def _slug(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return cleaned[:24] or "run"


def _generated_run_id(identity: Dict[str, Any]) -> str:
    parts = [
        _slug(identity.get("run_type", "run"))[:12],
        _slug(identity.get("regime", ""))[:12],
        _slug(identity.get("backbone", ""))[:12],
    ]
    if identity.get("target_wp") is not None:
        parts.append(f"wp{int(round(float(identity['target_wp']) * 100)):02d}")
    if identity.get("seed") is not None:
        parts.append(f"seed{int(identity['seed'])}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    token = uuid.uuid4().hex[:8]
    prefix = "-".join(part for part in parts if part)
    return validate_run_id(f"{prefix}-{stamp}-{token}")


def _git_metadata(project_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"git_commit": "unknown", "uncommitted_changes": None}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(project_root), stderr=subprocess.DEVNULL
        )
        result["uncommitted_changes"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError, UnicodeError):
        pass
    return result


def manifest_identifier(manifest_path: Optional[str]) -> Dict[str, Optional[str]]:
    """Identify a small manifest without hashing dataset files."""
    if not manifest_path:
        return {"manifest_name": None, "manifest_sha256": None}
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        return {"manifest_name": path.name, "manifest_sha256": None}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"manifest_name": path.name, "manifest_sha256": digest}


def create_run_directory(
    output_root: str,
    *,
    identity: Dict[str, Any],
    run_id: Optional[str] = None,
    run_parent_dir: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Atomically reserve a fresh run directory and write its initial metadata."""
    project = (project_root or Path(__file__).resolve().parents[3]).resolve()

    def anchored(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else project / candidate).resolve()

    parent = anchored(run_parent_dir) if run_parent_dir else anchored(output_root) / RUNS_DIRECTORY
    selected_id = validate_run_id(run_id) if run_id is not None else _generated_run_id(identity)
    run_dir = parent / selected_id
    # mkdir(exist_ok=False) is the collision check; the filesystem operation is atomic.
    run_dir.mkdir(parents=True, exist_ok=False)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        version = importlib.metadata.version("piwpct")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    metadata = {
        "schema": "piwpct.run_metadata.v1",
        "run_id": selected_id,
        "created_at_utc": now,
        "updated_at_utc": now,
        "status": "running",
        **identity,
        "project_version": version,
        **_git_metadata(project),
        "paths": {"run_directory": str(run_dir)},
    }
    if identity.get("run_type") in {"training", "ablation_child"}:
        metadata["paths"].update({
            "checkpoint": None,
            "trajectories": str(run_dir / "trajectories" / "trajectories.jsonl"),
            "epoch_metrics": str(run_dir / "metrics" / "epochs.jsonl"),
            "metrics_directory": str(run_dir / "metrics"),
            "plots_directory": str(run_dir / "plots"),
            "logs_directory": str(run_dir / "logs"),
        })
    elif identity.get("run_type") == "ablation_suite":
        metadata["paths"].update({
            "child_runs_directory": str(run_dir / "runs"),
            "reports_directory": str(run_dir / "reports"),
        })
    else:
        metadata["paths"].update({
            "metrics_directory": str(run_dir / "metrics"),
            "plots_directory": str(run_dir / "plots"),
            "logs_directory": str(run_dir / "logs"),
        })
    try:
        _atomic_json_write(run_dir / RUN_METADATA, metadata)
    except Exception:
        # Keep the reserved directory for diagnosis; never reuse it implicitly.
        raise
    return run_dir, selected_id


def update_run_metadata(run_directory: str, **updates: Any) -> Dict[str, Any]:
    """Atomically update status or final artifact fields for one reserved run."""
    path = Path(run_directory).expanduser().resolve() / RUN_METADATA
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(updates.get("paths"), dict):
        metadata["paths"] = {**metadata.get("paths", {}), **updates.pop("paths")}
    metadata.update(updates)
    metadata["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json_write(path, metadata)
    return metadata


def write_run_json(run_directory: str, relative_path: str, value: Dict[str, Any]) -> Path:
    """Atomically write a JSON artifact contained within a reserved run directory."""
    root = Path(run_directory).expanduser().resolve()
    destination = (root / relative_path).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Run artifact paths must remain inside the run directory.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(destination, value)
    return destination
