"""Portable root precedence and raw-data write guards."""


import pytest

from piwpct.data_paths import PROJECT_ROOT, DEFAULT_ROOTS, ensure_path_outside_raw, resolve_data_root


def test_path_precedence_is_cli_then_config_then_environment_then_default(tmp_path):
    cli_path = tmp_path / "cli"
    config_path = tmp_path / "config"
    env_path = tmp_path / "env"
    config = {"prepared_data_root": str(config_path)}
    environ = {"PREPARED_DATA_ROOT": str(env_path)}

    assert resolve_data_root("prepared_data_root", str(cli_path), config, environ=environ) == (cli_path.resolve(), "cli")
    assert resolve_data_root("prepared_data_root", None, config, environ=environ) == (config_path.resolve(), "config")
    assert resolve_data_root("prepared_data_root", None, {}, environ=environ) == (env_path.resolve(), "environment")
    assert resolve_data_root("prepared_data_root", None, {}, environ={}) == ((PROJECT_ROOT / DEFAULT_ROOTS["prepared_data_root"]).resolve(), "default")


def test_output_root_cannot_be_nested_under_raw_dataset(tmp_path):
    raw_root = tmp_path / "Dataset"
    raw_root.mkdir()
    with pytest.raises(ValueError, match="outside RAW_DATA_ROOT"):
        ensure_path_outside_raw(raw_root / "prepared", raw_root, name="PREPARED_DATA_ROOT")
    ensure_path_outside_raw(tmp_path / "prepared", raw_root, name="PREPARED_DATA_ROOT")
