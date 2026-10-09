"""Project-anchored paths with CLI -> config -> environment precedence."""

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: Union[str, Path], *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve an absolute path unchanged, or a relative path from the project root."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


DEFAULT_ROOTS = {
    "raw_data_root": Path("data/raw"),
    "prepared_data_root": Path("data/prepared"),
    "output_root": Path("outputs"),
}
ENV_NAMES = {
    "raw_data_root": "RAW_DATA_ROOT",
    "prepared_data_root": "PREPARED_DATA_ROOT",
    "output_root": "OUTPUT_ROOT",
}


def load_data_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = resolve_project_path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"Could not read data config {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Data config {config_path} must contain a YAML mapping.")
    data = value.get("data", value)
    return data if isinstance(data, dict) else {}


def resolve_data_root(
    name: str,
    cli_value: Optional[str],
    config: Mapping[str, Any],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[Path, str]:
    """Resolve one root and report which precedence level supplied it."""
    if name not in DEFAULT_ROOTS:
        raise KeyError(f"Unknown data root {name!r}.")
    if cli_value:
        value, source = cli_value, "cli"
    elif config.get(name):
        value, source = config[name], "config"
    else:
        variables = os.environ if environ is None else environ
        env_value = variables.get(ENV_NAMES[name])
        if env_value:
            value, source = env_value, "environment"
        else:
            value, source = str(DEFAULT_ROOTS[name]), "default"
    return resolve_project_path(value), source


def ensure_path_outside_raw(path: Path, raw_data_root: Path, *, name: str) -> None:
    """Guard against accidentally placing outputs below the strictly read-only raw root."""
    candidate = path.resolve()
    raw_root = raw_data_root.resolve()
    try:
        candidate.relative_to(raw_root)
    except ValueError:
        return
    raise ValueError(f"{name} must be outside RAW_DATA_ROOT ({raw_root}); received {candidate}.")
