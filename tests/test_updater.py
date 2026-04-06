from __future__ import annotations

"""Tests for router.updater — version tracking, update execution, drain, and rollback."""

import json
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

from router.updater import (
    UpdateInfo,
    UpdateResult,
    check_for_update,
    drain_and_run,
    get_local_version,
    get_rollback_info,
    rollback,
    run_update,
    update_with_rollback,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo-like directory with a VERSION file."""
    version_file = tmp_path / "VERSION"
    version_file.write_text("0.3.0\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def tmp_state(tmp_path: Path) -> Path:
    """Create a temporary state directory."""
    state = tmp_path / "state"
    state.mkdir()
    return state


# ---------------------------------------------------------------------------
# 1. get_local_version
# ---------------------------------------------------------------------------

class TestGetLocalVersion:
    def test_reads_version_file(self, tmp_repo: Path) -> None:
        with patch("router.updater._repo_root", return_value=tmp_repo):
            assert get_local_version() == "0.3.0"

    def test_falls_back_to_git_sha(self, tmp_path: Path) -> None:
        # No VERSION file exists
        with patch("router.updater._repo_root", return_value=tmp_path):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="abc1234\n", returncode=0)
                result = get_local_version()
                assert result == "abc1234"

    def test_returns_unknown_on_failure(self, tmp_path: Path) -> None:
        with patch("router.updater._repo_root", return_value=tmp_path):
            with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
                assert get_local_version() == "unknown"


# ---------------------------------------------------------------------------
# 2. check_for_update
# ---------------------------------------------------------------------------

class TestCheckForUpdate:
    def test_update_available(self, tmp_repo: Path) -> None:
        commit_response = json.dumps({"sha": "deadbeef1234567890"}).encode()
        version_response = b"0.4.0\n"

        def mock_urlopen(req_or_url, *, timeout=None):
            url = req_or_url.full_url if hasattr(req_or_url, "full_url") else req_or_url
            mock_resp = MagicMock()
            if "api.github.com" in url:
                mock_resp.read.return_value = commit_response
            else:
                mock_resp.read.return_value = version_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("router.updater._repo_root", return_value=tmp_repo):
            with patch("router.updater._get_local_sha", return_value="abc1234"):
                with patch("urllib.request.urlopen", side_effect=mock_urlopen):
                    info = check_for_update()

        assert info.available is True
        assert info.local_version == "0.3.0"
        assert info.remote_version == "0.4.0"
        assert info.remote_sha == "deadbee"

    def test_up_to_date(self, tmp_repo: Path) -> None:
        commit_response = json.dumps({"sha": "abc1234xxxxxxxxxxxx"}).encode()
        version_response = b"0.3.0\n"

        def mock_urlopen(req_or_url, *, timeout=None):
            url = req_or_url.full_url if hasattr(req_or_url, "full_url") else req_or_url
            mock_resp = MagicMock()
            if "api.github.com" in url:
                mock_resp.read.return_value = commit_response
            else:
                mock_resp.read.return_value = version_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("router.updater._repo_root", return_value=tmp_repo):
            with patch("router.updater._get_local_sha", return_value="abc1234"):
                with patch("urllib.request.urlopen", side_effect=mock_urlopen):
                    info = check_for_update()

        assert info.available is False
        assert info.local_version == "0.3.0"
        assert info.remote_version == "0.3.0"

    def test_network_error_returns_not_available(self, tmp_repo: Path) -> None:
        with patch("router.updater._repo_root", return_value=tmp_repo):
            with patch("router.updater._get_local_sha", return_value="abc1234"):
                with patch("urllib.request.urlopen", side_effect=OSError("no network")):
                    info = check_for_update()

        assert info.available is False
        assert info.local_version == "0.3.0"


# ---------------------------------------------------------------------------
# 3. run_update
# ---------------------------------------------------------------------------

class TestRunUpdate:
    def test_success(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"

        # Mock subprocess.run calls
        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.returncode = 0
            result.stdout = "abc1234\n"
            result.stderr = ""
            return result

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = run_update()

        assert result.success is True
        assert result.previous_sha == "abc1234"
        assert result.error is None
        # State file should be written
        assert (state / "update-state.json").exists()

    def test_git_failure(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"

        call_index = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                # rev-parse for previous SHA
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc1234\n"
                result.stderr = ""
                return result
            # git fetch fails
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="fetch failed")

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = run_update()

        assert result.success is False
        assert "git update failed" in (result.error or "")

    def test_omlx_update_when_present(self, tmp_path: Path) -> None:
        """When oMLX source dir exists, run_update should update it too."""
        src = tmp_path / "src"
        src.mkdir()
        omlx_src = tmp_path / "omlx"
        omlx_src.mkdir()
        state = tmp_path / "state"

        commands_run: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            commands_run.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            # For git describe --tags, return a tag name
            if "describe" in cmd:
                result.stdout = "v0.5.0\n"
            else:
                result.stdout = "abc1234\n"
            return result

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("router.updater._omlx_src", return_value=omlx_src),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = run_update()

        assert result.success is True

        # Verify oMLX-related commands were issued
        flat = [" ".join(c) for c in commands_run]
        assert any("fetch --tags" in c and str(omlx_src) in c for c in flat), \
            f"Expected oMLX git fetch --tags, got: {flat}"
        assert any("checkout" in c and "v0.5.0" in c for c in flat), \
            f"Expected git checkout v0.5.0, got: {flat}"
        assert any("install" in c and "-e" in c and str(omlx_src) in c for c in flat), \
            f"Expected pip install -e omlx, got: {flat}"
        assert any("mlx-lm" in c for c in flat), \
            f"Expected mlx-lm fork install, got: {flat}"

    def test_omlx_update_skipped_when_absent(self, tmp_path: Path) -> None:
        """When oMLX source dir doesn't exist, run_update should skip it."""
        src = tmp_path / "src"
        src.mkdir()
        omlx_src = tmp_path / "omlx_does_not_exist"  # not created
        state = tmp_path / "state"

        commands_run: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            commands_run.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = "abc1234\n"
            result.stderr = ""
            return result

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("router.updater._omlx_src", return_value=omlx_src),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = run_update()

        assert result.success is True
        # No oMLX-related commands should appear
        flat = [" ".join(c) for c in commands_run]
        assert not any("fetch --tags" in c for c in flat), \
            f"Should not have oMLX fetch, got: {flat}"

    def test_omlx_update_failure(self, tmp_path: Path) -> None:
        """If oMLX update fails, run_update should return an error."""
        src = tmp_path / "src"
        src.mkdir()
        omlx_src = tmp_path / "omlx"
        omlx_src.mkdir()
        state = tmp_path / "state"

        call_index = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_index
            call_index += 1
            # First 3 calls: rev-parse, git fetch origin, git reset (privatenet)
            if call_index <= 3:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc1234\n"
                result.stderr = ""
                return result
            # 4th call: oMLX git fetch --tags fails
            raise subprocess.CalledProcessError(
                1, cmd, output="", stderr="omlx fetch failed"
            )

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("router.updater._omlx_src", return_value=omlx_src),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = run_update()

        assert result.success is False
        assert "oMLX update failed" in (result.error or "")


# ---------------------------------------------------------------------------
# 4. drain_and_run
# ---------------------------------------------------------------------------

class TestDrainAndRun:
    def test_drains_and_runs_callback(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        disabled_file = state / "disabled"

        # Set up a simple HTTP server that returns in_flight=0
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = json.dumps({
                    "status": "disabled",
                    "cluster": [{"local": True, "in_flight": 0}],
                })
                self.wfile.write(body.encode())

            def log_message(self, format, *args):
                pass  # Suppress output

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        callback_called = False

        def my_callback():
            nonlocal callback_called
            callback_called = True
            return 42

        try:
            with (
                patch("router.updater._state_dir", return_value=state),
                patch("router.updater._disabled_file", return_value=disabled_file),
            ):
                result = drain_and_run(my_callback, health_url=f"http://127.0.0.1:{port}", timeout=5)
        finally:
            server.shutdown()

        assert callback_called is True
        assert result == 42
        # Disabled file should be removed after callback
        assert not disabled_file.exists()

    def test_waits_for_inflight(self, tmp_path: Path) -> None:
        """Drain should poll until in_flight drops to 0 before running callback."""
        state = tmp_path / "state"
        disabled_file = state / "disabled"

        request_count = 0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal request_count
                request_count += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                # First two requests report in_flight > 0, then 0
                in_flight = 2 if request_count <= 2 else 0
                body = json.dumps({
                    "status": "disabled",
                    "cluster": [{"local": True, "in_flight": in_flight}],
                })
                self.wfile.write(body.encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        callback_called = False

        def my_callback():
            nonlocal callback_called
            callback_called = True
            return "done"

        try:
            with (
                patch("router.updater._state_dir", return_value=state),
                patch("router.updater._disabled_file", return_value=disabled_file),
                patch("router.updater.time.sleep"),  # speed up polling
            ):
                result = drain_and_run(my_callback, health_url=f"http://127.0.0.1:{port}", timeout=30)
        finally:
            server.shutdown()

        assert callback_called is True
        assert result == "done"
        # Health endpoint was polled multiple times before callback ran
        assert request_count >= 3
        assert not disabled_file.exists()

    def test_reenables_on_callback_error(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        disabled_file = state / "disabled"

        def failing_callback():
            raise RuntimeError("boom")

        with (
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._disabled_file", return_value=disabled_file),
            patch("urllib.request.urlopen", side_effect=OSError("no server")),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                drain_and_run(failing_callback, timeout=1)

        # Disabled file should still be removed (re-enabled) even on error
        assert not disabled_file.exists()


# ---------------------------------------------------------------------------
# 5. rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_success(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        # Write update-state.json
        update_state = state / "update-state.json"
        update_state.write_text(json.dumps({
            "previous_sha": "abc1234",
            "new_sha": "def5678",
        }), encoding="utf-8")

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "def5678\n"
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            success = rollback(privatenet_src=src, state_dir=state)

        assert success is True
        # rollback.json should exist
        rollback_file = state / "rollback.json"
        assert rollback_file.exists()
        info = json.loads(rollback_file.read_text())
        assert info["rolled_back_to"] == "abc1234"
        assert info["rolled_back_from"] == "def5678"

    def test_rollback_no_state(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        success = rollback(privatenet_src=src, state_dir=state)
        assert success is False

    def test_rollback_git_failure(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        update_state = state / "update-state.json"
        update_state.write_text(json.dumps({"previous_sha": "abc1234"}), encoding="utf-8")

        call_index = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                # rev-parse current SHA
                result = MagicMock()
                result.returncode = 0
                result.stdout = "def5678\n"
                result.stderr = ""
                return result
            # git checkout fails
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", side_effect=mock_run):
            success = rollback(privatenet_src=src, state_dir=state)

        assert success is False

    def test_get_rollback_info(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        state.mkdir()

        # No rollback.json yet
        assert get_rollback_info(state_dir=state) is None

        # Write one
        rollback_file = state / "rollback.json"
        data = {"timestamp": 1234567890, "reason": "test", "rolled_back_from": "a", "rolled_back_to": "b"}
        rollback_file.write_text(json.dumps(data), encoding="utf-8")

        info = get_rollback_info(state_dir=state)
        assert info is not None
        assert info["reason"] == "test"


# ---------------------------------------------------------------------------
# 6. update_with_rollback
# ---------------------------------------------------------------------------

class TestUpdateWithRollback:
    def test_update_with_rollback_healthy(self, tmp_path: Path) -> None:
        """When health returns 'ok' after update, no rollback should occur."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "abc1234\n"
            result.stderr = ""
            return result

        def mock_urlopen(req, *, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("subprocess.run", side_effect=mock_run),
            patch("router.updater.time.sleep"),  # skip waits
            patch("urllib.request.urlopen", side_effect=mock_urlopen),
        ):
            result = update_with_rollback(
                privatenet_src=src,
                state_dir=state,
                venv_dir=tmp_path / "venv",
            )

        assert result.success is True
        assert result.error is None
        # No rollback.json should exist
        assert not (state / "rollback.json").exists()

    def test_update_with_rollback_unhealthy(self, tmp_path: Path) -> None:
        """When health check fails after update, rollback should be triggered."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        # Pre-write update-state so rollback() can find the previous SHA
        update_state_file = state / "update-state.json"
        update_state_file.write_text(json.dumps({
            "previous_sha": "old1234",
            "new_sha": "new5678",
        }), encoding="utf-8")

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "abc1234\n"
            result.stderr = ""
            return result

        def mock_urlopen(req, *, timeout=None):
            """Always return an error status to simulate unhealthy router."""
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"status": "error"}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("subprocess.run", side_effect=mock_run),
            patch("router.updater.time.sleep"),  # skip waits
            patch("router.updater.time.monotonic", side_effect=[0, 0, 100]),  # first call for deadline, second for while check, third to exceed deadline
            patch("urllib.request.urlopen", side_effect=mock_urlopen),
        ):
            result = update_with_rollback(
                privatenet_src=src,
                state_dir=state,
                venv_dir=tmp_path / "venv",
            )

        assert result.success is False
        assert "rolled back" in (result.error or "").lower()
        # rollback.json should exist
        rollback_file = state / "rollback.json"
        assert rollback_file.exists()
        info = json.loads(rollback_file.read_text())
        assert info["reason"] == "automatic rollback after failed health check"
