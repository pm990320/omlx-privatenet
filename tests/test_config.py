from __future__ import annotations

from pathlib import Path

import pytest

from router.config import DEFAULT_CONFIG_PATH, load_config, resolve_config_path


def test_default_config_values(tmp_path: Path):
    missing = tmp_path / "missing-router.json"

    config = load_config(missing)

    assert config.host == "0.0.0.0"
    assert config.port == 8741
    assert config.failure_threshold == 3
    assert config.local_models == []
    assert config.source_path == missing.resolve()


def test_loading_from_router_json(write_config):
    path = write_config({"host": "127.0.0.1", "port": 9999, "local_models": ["model-a"], "local_max_concurrent": 4})

    config = load_config(path)

    assert config.host == "127.0.0.1"
    assert config.port == 9999
    assert config.local_models == ["model-a"]
    assert config.local_max_concurrent == 4
    assert config.source_path == path.resolve()


def test_environment_variables_override_config(monkeypatch, write_config):
    path = write_config({"host": "0.0.0.0", "port": 8741, "local_models": ["model-a"]})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_HOST", "127.0.0.1")
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_PORT", "8841")
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_LOCAL_MODELS", "model-b,model-c")

    config = load_config(path)

    assert config.host == "127.0.0.1"
    assert config.port == 8841
    assert config.local_models == ["model-b", "model-c"]


def test_resolve_config_path_uses_environment_override(monkeypatch, tmp_path: Path):
    path = tmp_path / "router.json"
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_CONFIG", str(path))

    resolved = resolve_config_path()

    assert resolved == path.resolve()
    assert resolved != DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# _to_bool string parsing (lines 91-93)
# ---------------------------------------------------------------------------


def test_to_bool_string_values():
    from router.config import _to_bool

    assert _to_bool("true") is True
    assert _to_bool("True") is True
    assert _to_bool("1") is True
    assert _to_bool("yes") is True
    assert _to_bool("Yes") is True
    assert _to_bool("false") is False
    assert _to_bool("0") is False
    assert _to_bool("no") is False


def test_to_bool_non_string_non_bool():
    """Test _to_bool with non-string, non-bool values."""
    from router.config import _to_bool

    assert _to_bool(1) is True
    assert _to_bool(0) is False


# ---------------------------------------------------------------------------
# _to_optional_int (lines 99-104)
# ---------------------------------------------------------------------------


def test_to_optional_int_string_values():
    from router.config import _to_optional_int

    assert _to_optional_int(None) is None
    assert _to_optional_int("42") == 42
    assert _to_optional_int("  42  ") == 42
    assert _to_optional_int("") is None
    assert _to_optional_int("none") is None
    assert _to_optional_int("None") is None
    assert _to_optional_int("null") is None
    assert _to_optional_int(10) == 10


# ---------------------------------------------------------------------------
# local_models must be a JSON array (line 55)
# ---------------------------------------------------------------------------


def test_local_models_not_a_list_raises(write_config):
    from router.config import RouterConfig

    with pytest.raises(ValueError, match="local_models.*must be a JSON array"):
        RouterConfig.from_dict({"local_models": "not-a-list"})


# ---------------------------------------------------------------------------
# Env var overrides for auto_download, trusted_orgs, auto_update (lines 148-149, 151)
# ---------------------------------------------------------------------------


def test_env_override_auto_download(monkeypatch, write_config):
    path = write_config({"auto_download": False})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_AUTO_DOWNLOAD", "true")

    config = load_config(path)
    assert config.auto_download is True


def test_env_override_auto_download_false(monkeypatch, write_config):
    path = write_config({"auto_download": True})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_AUTO_DOWNLOAD", "false")

    config = load_config(path)
    assert config.auto_download is False


def test_env_override_auto_update(monkeypatch, write_config):
    path = write_config({"auto_update": False})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_AUTO_UPDATE", "yes")

    config = load_config(path)
    assert config.auto_update is True


def test_env_override_trusted_orgs(monkeypatch, write_config):
    path = write_config({"trusted_orgs": ["mlx-community"]})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_TRUSTED_ORGS", "org-a, org-b")

    config = load_config(path)
    assert config.trusted_orgs == ["org-a", "org-b"]


def test_env_override_auto_download_max_gb_none(monkeypatch, write_config):
    path = write_config({"auto_download_max_gb": 50})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_AUTO_DOWNLOAD_MAX_GB", "none")

    config = load_config(path)
    assert config.auto_download_max_gb is None


def test_env_override_auto_download_max_gb_value(monkeypatch, write_config):
    path = write_config({})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_AUTO_DOWNLOAD_MAX_GB", "100")

    config = load_config(path)
    assert config.auto_download_max_gb == 100


def test_env_override_overload_threshold_none(monkeypatch, write_config):
    path = write_config({"overload_threshold": 10})
    monkeypatch.setenv("OMLX_PRIVATENET_ROUTER_OVERLOAD_THRESHOLD", "null")

    config = load_config(path)
    assert config.overload_threshold is None


# ---------------------------------------------------------------------------
# Non-dict config file (line 172)
# ---------------------------------------------------------------------------


def test_config_file_not_a_dict_raises(tmp_path):
    """If the config file is a JSON array instead of object, raise ValueError."""
    path = tmp_path / "router.json"
    path.write_text('[1, 2, 3]', encoding="utf-8")

    with pytest.raises(ValueError, match="Router config must be a JSON object"):
        load_config(path)
