from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from router.cli import main as cli_main


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the CLI at a temporary state directory."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def with_router_config(state_dir):
    """Write a minimal router config so the CLI can read node_id."""
    config = state_dir / "router.json"
    config.write_text('{"local_node_id": "test-node"}', encoding="utf-8")
    return state_dir


def test_status_shows_enabled_by_default(with_router_config, capsys):
    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "enabled" in captured.out
    assert "test-node" in captured.out


def test_disable_creates_file(with_router_config, capsys):
    result = cli_main(["disable"])
    assert result == 0
    assert (with_router_config / "disabled").exists()
    captured = capsys.readouterr()
    assert "disabled" in captured.out


def test_disable_idempotent(with_router_config, capsys):
    cli_main(["disable"])
    result = cli_main(["disable"])
    assert result == 0
    captured = capsys.readouterr()
    assert "already disabled" in captured.out


def test_enable_removes_file(with_router_config, capsys):
    cli_main(["disable"])
    result = cli_main(["enable"])
    assert result == 0
    assert not (with_router_config / "disabled").exists()
    captured = capsys.readouterr()
    assert "enabled" in captured.out


def test_enable_idempotent(with_router_config, capsys):
    result = cli_main(["enable"])
    assert result == 0
    captured = capsys.readouterr()
    assert "already enabled" in captured.out


def test_status_shows_disabled_after_disable(with_router_config, capsys):
    cli_main(["disable"])
    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "disabled" in captured.out


def test_models_shows_all_by_default(with_router_config, capsys):
    result = cli_main(["models"])
    assert result == 0
    captured = capsys.readouterr()
    assert "all models" in captured.out


def test_models_set_writes_allowlist(with_router_config, capsys):
    result = cli_main(["models", "set", "model-a", "model-b"])
    assert result == 0
    import json
    with (with_router_config / "router.json").open() as f:
        config = json.load(f)
    assert config["advertise_models"] == ["model-a", "model-b"]


def test_models_reset_removes_allowlist(with_router_config, capsys):
    cli_main(["models", "set", "model-a"])
    result = cli_main(["models", "reset"])
    assert result == 0
    import json
    with (with_router_config / "router.json").open() as f:
        config = json.load(f)
    assert "advertise_models" not in config


def test_models_set_no_args_returns_error(with_router_config, capsys):
    result = cli_main(["models", "set"])
    assert result == 1
    captured = capsys.readouterr()
    assert "No models specified" in captured.out


def test_models_shows_allowlist_after_set(with_router_config, capsys):
    cli_main(["models", "set", "model-x"])
    result = cli_main(["models"])
    assert result == 0
    captured = capsys.readouterr()
    assert "allowlist" in captured.out
    assert "model-x" in captured.out


def test_no_command_prints_help(capsys):
    result = cli_main([])
    assert result == 1


def test_status_without_config_falls_back_to_hostname(state_dir, capsys):
    """When router.json doesn't exist, falls back to hostname."""
    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Node ID:" in captured.out


# ---------------------------------------------------------------------------
# Registry CLI tests
# ---------------------------------------------------------------------------


def test_registry_list_empty(with_router_config, capsys):
    result = cli_main(["registry", "list"])
    assert result == 0
    captured = capsys.readouterr()
    assert "No models" in captured.out


def test_registry_add_and_list(with_router_config, capsys):
    result = cli_main(["registry", "add", "mlx-community/Llama-3.2-3B-Instruct-4bit"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Added" in captured.out
    assert "Llama-3.2-3B-Instruct-4bit" in captured.out

    result = cli_main(["registry", "list"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Llama-3.2-3B-Instruct-4bit" in captured.out
    assert "mlx-community" in captured.out
    assert "test-node" in captured.out


def test_registry_add_with_priority(with_router_config, capsys):
    result = cli_main(["registry", "add", "mlx-community/some-model", "--priority", "9"])
    assert result == 0

    # Verify priority in the registry file
    reg_path = with_router_config / "registry.json"
    data = json.loads(reg_path.read_text())
    models = data if isinstance(data, list) else data.get("models", data)
    # Find our model
    found = [m for m in models if m["id"] == "some-model"]
    assert len(found) == 1
    assert found[0]["priority"] == 9


def test_registry_add_invalid_repo(with_router_config, capsys):
    result = cli_main(["registry", "add", "invalid-no-slash"])
    assert result == 1
    captured = capsys.readouterr()
    assert "Invalid repo format" in captured.out


def test_registry_remove(with_router_config, capsys):
    cli_main(["registry", "add", "mlx-community/model-to-remove"])
    result = cli_main(["registry", "remove", "model-to-remove"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Removed" in captured.out

    # Verify it's gone
    result = cli_main(["registry", "list"])
    assert result == 0
    captured = capsys.readouterr()
    assert "No models" in captured.out


def test_registry_remove_nonexistent(with_router_config, capsys):
    result = cli_main(["registry", "remove", "no-such-model"])
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_registry_add_sets_node_id(with_router_config, capsys):
    cli_main(["registry", "add", "mlx-community/test-model"])
    reg_path = with_router_config / "registry.json"
    data = json.loads(reg_path.read_text())
    models = data if isinstance(data, list) else data.get("models", data)
    found = [m for m in models if m["id"] == "test-model"]
    assert found[0]["added_by"] == "test-node"


def test_registry_no_action_shows_usage(with_router_config, capsys):
    result = cli_main(["registry"])
    assert result == 1
    captured = capsys.readouterr()
    assert "Usage" in captured.out


# ---------------------------------------------------------------------------
# Config CLI tests
# ---------------------------------------------------------------------------


def test_config_get(with_router_config, capsys):
    result = cli_main(["config", "get", "local_node_id"])
    assert result == 0
    captured = capsys.readouterr()
    assert "test-node" in captured.out


def test_config_get_missing_key(with_router_config, capsys):
    result = cli_main(["config", "get", "nonexistent_key"])
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_config_set_string(with_router_config, capsys):
    result = cli_main(["config", "set", "local_node_id", "new-node"])
    assert result == 0
    # Verify it was written
    data = json.loads((with_router_config / "router.json").read_text())
    assert data["local_node_id"] == "new-node"


def test_config_set_bool(with_router_config, capsys):
    result = cli_main(["config", "set", "auto_download", "true"])
    assert result == 0
    data = json.loads((with_router_config / "router.json").read_text())
    assert data["auto_download"] is True


def test_config_set_int(with_router_config, capsys):
    result = cli_main(["config", "set", "auto_download_max_gb", "50"])
    assert result == 0
    data = json.loads((with_router_config / "router.json").read_text())
    assert data["auto_download_max_gb"] == 50


def test_config_set_list(with_router_config, capsys):
    result = cli_main(["config", "set", "trusted_orgs", '["mlx-community", "my-org"]'])
    assert result == 0
    data = json.loads((with_router_config / "router.json").read_text())
    assert data["trusted_orgs"] == ["mlx-community", "my-org"]


def test_config_no_action_shows_usage(with_router_config, capsys):
    result = cli_main(["config"])
    assert result == 1
    captured = capsys.readouterr()
    assert "Usage" in captured.out
