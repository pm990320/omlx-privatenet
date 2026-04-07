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


# ---------------------------------------------------------------------------
# _check_router_health tests
# ---------------------------------------------------------------------------


def test_check_router_health_returns_none_on_failure(state_dir, monkeypatch):
    """When the health endpoint is unreachable, _check_router_health returns None."""
    from router.cli import _check_router_health

    # Mock urlopen to raise an exception (lines 56-57)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("refused")))
    result = _check_router_health()
    assert result is None


def test_status_with_filtered_models(with_router_config, capsys):
    """When advertise_models is set, status shows filtered count (line 78)."""
    config_path = with_router_config / "router.json"
    import json
    data = json.loads(config_path.read_text())
    data["advertise_models"] = ["model-a", "model-b"]
    config_path.write_text(json.dumps(data))

    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "filtered" in captured.out
    assert "model-a" in captured.out


def test_status_health_not_responding(with_router_config, capsys, monkeypatch):
    """When health check fails, status shows 'not responding' (line 84)."""
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("refused")))
    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "not responding" in captured.out


def test_status_health_with_rollback(with_router_config, capsys, monkeypatch):
    """When rollback info is present in health response (lines 95-103)."""
    import time
    health_data = {
        "status": "ok",
        "cluster": [],
        "models": [],
        "rollback": {
            "rolled_back_from": "0.3.0",
            "rolled_back_to": "0.2.0",
            "timestamp": time.time(),
        },
    }

    def fake_urlopen(url, timeout=None):
        import io

        class FakeResp:
            def read(self):
                return json.dumps(health_data).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Rollback" in captured.out
    assert "0.3.0" in captured.out
    assert "0.2.0" in captured.out


def test_status_health_with_rollback_no_timestamp(with_router_config, capsys, monkeypatch):
    """When rollback info has no timestamp (lines 101-102)."""
    health_data = {
        "status": "ok",
        "cluster": [],
        "models": [],
        "rollback": {
            "rolled_back_from": "0.3.0",
            "rolled_back_to": "0.2.0",
            "timestamp": None,
        },
    }

    def fake_urlopen(url, timeout=None):
        class FakeResp:
            def read(self):
                return json.dumps(health_data).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "unknown" in captured.out


# ---------------------------------------------------------------------------
# Models evict / pin / unpin tests (lines 186, 190, 194, 228-266, 271-287, 292-308)
# ---------------------------------------------------------------------------


def test_models_evict_nothing_to_evict(with_router_config, capsys, monkeypatch):
    """Test models evict when nothing to evict (lines 186, 228-248)."""
    from router.eviction import EvictionPlan

    fake_config = type("C", (), {
        "auto_download_max_gb": 100,
        "advertise_models": None,
        "local_omlx_url": "http://127.0.0.1:5741",
        "local_omlx_api_key": None,
    })()
    monkeypatch.setattr("router.config.load_config", lambda *a, **kw: fake_config)
    monkeypatch.setattr(
        "router.eviction.plan_eviction",
        lambda **kw: EvictionPlan(models_to_evict=[], reason="Under cap"),
    )

    result = cli_main(["models", "evict"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Nothing to evict" in captured.out


def test_models_evict_dry_run(with_router_config, capsys, monkeypatch):
    """Test models evict with --dry-run (lines 250-262)."""
    from router.eviction import EvictionPlan

    fake_config = type("C", (), {
        "auto_download_max_gb": 10,
        "advertise_models": None,
        "local_omlx_url": "http://127.0.0.1:5741",
        "local_omlx_api_key": None,
    })()
    monkeypatch.setattr("router.config.load_config", lambda *a, **kw: fake_config)
    monkeypatch.setattr(
        "router.eviction.plan_eviction",
        lambda **kw: EvictionPlan(models_to_evict=["old-model"], reason="Over cap", bytes_to_free=1000),
    )
    monkeypatch.setattr("router.eviction.execute_eviction", lambda plan, model_dir, dry_run=False: [])

    result = cli_main(["models", "evict", "--dry-run"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Would evict" in captured.out
    assert "Dry run" in captured.out


def test_models_evict_real_run(with_router_config, capsys, monkeypatch):
    """Test models evict for real (lines 259, 263-266)."""
    from router.eviction import EvictionPlan

    fake_config = type("C", (), {
        "auto_download_max_gb": 10,
        "advertise_models": None,
        "local_omlx_url": "http://127.0.0.1:5741",
        "local_omlx_api_key": None,
    })()
    monkeypatch.setattr("router.config.load_config", lambda *a, **kw: fake_config)
    monkeypatch.setattr(
        "router.eviction.plan_eviction",
        lambda **kw: EvictionPlan(models_to_evict=["old-model"], reason="Over cap", bytes_to_free=1000),
    )
    monkeypatch.setattr("router.eviction.execute_eviction", lambda plan, model_dir, dry_run=False: ["old-model"])

    result = cli_main(["models", "evict"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Evicting" in captured.out
    assert "Evicted 1 model(s)" in captured.out


def test_models_pin(with_router_config, capsys, monkeypatch):
    """Test models pin (lines 190, 271-287)."""
    monkeypatch.setattr("router.eviction.load_pinned", lambda: [])
    saved = []
    monkeypatch.setattr("router.eviction.save_pinned", lambda p: saved.append(p))

    result = cli_main(["models", "pin", "my-model"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Pinned my-model" in captured.out
    assert saved[0] == ["my-model"]


def test_models_pin_already_pinned(with_router_config, capsys, monkeypatch):
    """Test models pin when already pinned (line 281)."""
    monkeypatch.setattr("router.eviction.load_pinned", lambda: ["my-model"])

    result = cli_main(["models", "pin", "my-model"])
    assert result == 0
    captured = capsys.readouterr()
    assert "already pinned" in captured.out


def test_models_pin_no_model_name(with_router_config, capsys, monkeypatch):
    """Test models pin with no model specified (lines 274-277)."""
    monkeypatch.setattr("router.eviction.load_pinned", lambda: [])

    result = cli_main(["models", "pin"])
    assert result == 1
    captured = capsys.readouterr()
    assert "No model specified" in captured.out


def test_models_unpin(with_router_config, capsys, monkeypatch):
    """Test models unpin (lines 194, 292-308)."""
    monkeypatch.setattr("router.eviction.load_pinned", lambda: ["my-model"])
    saved = []
    monkeypatch.setattr("router.eviction.save_pinned", lambda p: saved.append(p))

    result = cli_main(["models", "unpin", "my-model"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Unpinned my-model" in captured.out
    assert saved[0] == []


def test_models_unpin_not_pinned(with_router_config, capsys, monkeypatch):
    """Test models unpin when not pinned (line 302)."""
    monkeypatch.setattr("router.eviction.load_pinned", lambda: [])

    result = cli_main(["models", "unpin", "my-model"])
    assert result == 0
    captured = capsys.readouterr()
    assert "not pinned" in captured.out


def test_models_unpin_no_model_name(with_router_config, capsys, monkeypatch):
    """Test models unpin with no model specified (lines 294-298)."""
    monkeypatch.setattr("router.eviction.load_pinned", lambda: [])

    result = cli_main(["models", "unpin"])
    assert result == 1
    captured = capsys.readouterr()
    assert "No model specified" in captured.out


# ---------------------------------------------------------------------------
# Registry add duplicate (lines 355-357)
# ---------------------------------------------------------------------------


def test_registry_add_raises_value_error(with_router_config, capsys, monkeypatch):
    """When registry.add raises ValueError, the error is shown (lines 355-357)."""
    from router.registry import Registry

    original_add = Registry.add
    call_count = [0]

    def failing_add(self, model):
        call_count[0] += 1
        raise ValueError("Registry is at capacity (50 models).")

    monkeypatch.setattr(Registry, "add", failing_add)
    result = cli_main(["registry", "add", "mlx-community/fail-model"])
    assert result == 1
    captured = capsys.readouterr()
    assert "at capacity" in captured.out


# ---------------------------------------------------------------------------
# Models show with health data (lines 210->222, 213->218, 219->222)
# ---------------------------------------------------------------------------


def test_models_show_with_health_local_and_cluster(with_router_config, capsys, monkeypatch):
    """Test models show when health endpoint returns local node and cluster data."""
    health_data = {
        "status": "ok",
        "cluster": [
            {"local": True, "models": ["model-a", "model-b"]},
            {"local": False, "models": ["model-c"]},
        ],
        "models": ["model-a", "model-b", "model-c"],
    }

    def fake_urlopen(url, timeout=None):
        class FakeResp:
            def read(self):
                return json.dumps(health_data).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = cli_main(["models"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Currently advertised" in captured.out
    assert "Cluster total" in captured.out


def test_models_show_health_no_local_node(with_router_config, capsys, monkeypatch):
    """When health returns but no local node found (line 213->218 branch)."""
    health_data = {
        "status": "ok",
        "cluster": [{"local": False, "models": ["model-c"]}],
        "models": [],
    }

    def fake_urlopen(url, timeout=None):
        class FakeResp:
            def read(self):
                return json.dumps(health_data).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = cli_main(["models"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Currently advertised" not in captured.out
    assert "Cluster total" not in captured.out


def test_models_show_health_no_response(with_router_config, capsys, monkeypatch):
    """When health endpoint is unreachable, models show still works (line 210->222)."""
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("refused")))
    result = cli_main(["models"])
    assert result == 0
    captured = capsys.readouterr()
    assert "all models" in captured.out


# ---------------------------------------------------------------------------
# Update command tests (lines 416-445)
# ---------------------------------------------------------------------------


def test_update_check(with_router_config, capsys, monkeypatch):
    """Test update --check (lines 416-430)."""
    from router.updater import UpdateInfo

    monkeypatch.setattr(
        "router.updater.check_for_update",
        lambda: UpdateInfo(
            available=True,
            local_version="0.2.0",
            remote_version="0.3.0",
            local_sha="abc1234",
            remote_sha="def5678",
        ),
    )

    result = cli_main(["update", "--check"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Update Check" in captured.out
    assert "0.2.0" in captured.out
    assert "0.3.0" in captured.out
    assert "update available" in captured.out


def test_update_check_up_to_date(with_router_config, capsys, monkeypatch):
    """Test update --check when up to date."""
    from router.updater import UpdateInfo

    monkeypatch.setattr(
        "router.updater.check_for_update",
        lambda: UpdateInfo(
            available=False,
            local_version="0.2.0",
            remote_version="0.2.0",
            local_sha="abc1234",
            remote_sha="abc1234",
        ),
    )

    result = cli_main(["update", "--check"])
    assert result == 0
    captured = capsys.readouterr()
    assert "up to date" in captured.out


def test_update_already_up_to_date(with_router_config, capsys, monkeypatch):
    """Test update when already up to date (lines 432-434)."""
    from router.updater import UpdateInfo

    monkeypatch.setattr(
        "router.updater.check_for_update",
        lambda: UpdateInfo(
            available=False,
            local_version="0.2.0",
            remote_version="0.2.0",
            local_sha="abc1234",
            remote_sha="abc1234",
        ),
    )

    result = cli_main(["update"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Already up to date" in captured.out


def test_update_success(with_router_config, capsys, monkeypatch):
    """Test successful update (lines 436-442)."""
    from router.updater import UpdateInfo, UpdateResult

    monkeypatch.setattr(
        "router.updater.check_for_update",
        lambda: UpdateInfo(
            available=True,
            local_version="0.2.0",
            remote_version="0.3.0",
            local_sha="abc1234",
            remote_sha="def5678",
        ),
    )
    monkeypatch.setattr(
        "router.updater.drain_and_run",
        lambda cb: UpdateResult(success=True, previous_sha="abc1234", new_sha="def5678", error=None),
    )

    result = cli_main(["update"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Updated successfully" in captured.out
    assert "abc1234" in captured.out
    assert "def5678" in captured.out


def test_update_failure(with_router_config, capsys, monkeypatch):
    """Test failed update (lines 443-445)."""
    from router.updater import UpdateInfo, UpdateResult

    monkeypatch.setattr(
        "router.updater.check_for_update",
        lambda: UpdateInfo(
            available=True,
            local_version="0.2.0",
            remote_version="0.3.0",
            local_sha="abc1234",
            remote_sha="def5678",
        ),
    )
    monkeypatch.setattr(
        "router.updater.drain_and_run",
        lambda cb: UpdateResult(success=False, previous_sha="abc1234", new_sha="abc1234", error="git conflict"),
    )

    result = cli_main(["update"])
    assert result == 1
    captured = capsys.readouterr()
    assert "Update failed" in captured.out
    assert "git conflict" in captured.out


# ---------------------------------------------------------------------------
# __main__ guard (line 506)
# ---------------------------------------------------------------------------


def test_main_guard_line(with_router_config, monkeypatch):
    """Cover the if __name__ == '__main__' block (line 505-506)."""
    import runpy
    # runpy re-executes the module, but main() will get sys.argv which includes
    # pytest arguments, so it will fail with argparse error -> SystemExit(2).
    with pytest.raises(SystemExit):
        runpy.run_module("router.cli", run_name="__main__", alter_sys=True)
