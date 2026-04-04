from __future__ import annotations

from pathlib import Path

from router.config import DEFAULT_CONFIG_PATH, load_config, resolve_config_path


def test_default_config_values(tmp_path: Path):
    missing = tmp_path / "missing-router.json"

    config = load_config(missing)

    assert config.host == "0.0.0.0"
    assert config.port == 8741
    assert config.failure_threshold == 3
    assert config.local_models
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
