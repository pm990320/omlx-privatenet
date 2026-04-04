from __future__ import annotations

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


def test_no_command_prints_help(capsys):
    result = cli_main([])
    assert result == 1


def test_status_without_config_falls_back_to_hostname(state_dir, capsys):
    """When router.json doesn't exist, falls back to hostname."""
    result = cli_main(["status"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Node ID:" in captured.out
