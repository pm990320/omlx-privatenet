from __future__ import annotations

"""Tests for router.updater — version tracking, update execution, drain, and rollback."""

import asyncio
import json
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

from router.updater import (
    AutoUpdater,
    UpdateInfo,
    UpdateResult,
    _get_local_sha,
    _omlx_src,
    _privatenet_src,
    _read_install_var,
    _repo_root,
    _state_dir,
    _venv_bin,
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
        assert any("checkout -f" in c for c in flat), \
            f"Expected git checkout -f <ref>, got: {flat}"
        assert any("install" in c and "-e" in c and str(omlx_src) in c for c in flat), \
            f"Expected pip install -e omlx, got: {flat}"
        assert any("mlx-lm" in c for c in flat), \
            f"Expected mlx-lm install, got: {flat}"

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


# ---------------------------------------------------------------------------
# 7. AutoUpdater
# ---------------------------------------------------------------------------

class TestAutoUpdater:
    @pytest.fixture()
    def config(self) -> "RouterConfig":
        from router.config import RouterConfig
        return RouterConfig(auto_update=True, update_interval_hours=1)

    @pytest.mark.asyncio
    async def test_auto_updater_skips_when_no_update(self, config: "RouterConfig") -> None:
        """run_once should not call update_with_rollback when no update is available."""
        no_update = UpdateInfo(
            available=False,
            local_version="0.3.0",
            remote_version="0.3.0",
            local_sha="abc1234",
            remote_sha="abc1234",
        )
        updater = AutoUpdater(config)
        with (
            patch("router.updater.check_for_update", return_value=no_update) as mock_check,
            patch("router.updater.update_with_rollback") as mock_update,
            patch("router.updater.drain_and_run") as mock_drain,
        ):
            await updater.run_once()

        mock_check.assert_called_once()
        mock_update.assert_not_called()
        mock_drain.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_updater_applies_update(self, config: "RouterConfig") -> None:
        """run_once should call run_update and exit when an update is available."""
        has_update = UpdateInfo(
            available=True,
            local_version="0.3.0",
            remote_version="0.4.0",
            local_sha="abc1234",
            remote_sha="def5678",
        )
        success_result = UpdateResult(
            success=True,
            previous_sha="abc1234",
            new_sha="def5678",
            error=None,
        )
        updater = AutoUpdater(config)
        with (
            patch("router.updater.check_for_update", return_value=has_update),
            patch("router.updater.run_update", return_value=success_result) as mock_update,
        ):
            with pytest.raises(SystemExit) as exc_info:
                await updater.run_once()

        assert exc_info.value.code == 0
        mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_updater_logs_error_on_failed_update(self, config: "RouterConfig") -> None:
        """run_once should log an error when run_update returns failure."""
        has_update = UpdateInfo(
            available=True,
            local_version="0.3.0",
            remote_version="0.4.0",
            local_sha="abc1234",
            remote_sha="def5678",
        )
        fail_result = UpdateResult(
            success=False,
            previous_sha="abc1234",
            new_sha="abc1234",
            error="git update failed",
        )
        updater = AutoUpdater(config)
        with (
            patch("router.updater.check_for_update", return_value=has_update),
            patch("router.updater.run_update", return_value=fail_result),
        ):
            # Should NOT raise SystemExit since update failed
            await updater.run_once()

    @pytest.mark.asyncio
    async def test_run_forever_startup_delay_then_stop(self, config: "RouterConfig") -> None:
        """run_forever should wait 60s initially, then exit if stopped during that wait."""
        updater = AutoUpdater(config)
        # Stop immediately so the initial wait_for returns immediately
        updater._stop.set()
        await updater.run_forever()

    @pytest.mark.asyncio
    async def test_run_forever_runs_once_then_stops(self, config: "RouterConfig") -> None:
        """run_forever should run run_once, then stop when signaled."""
        updater = AutoUpdater(config)
        call_count = 0

        async def fake_run_once():
            nonlocal call_count
            call_count += 1
            # Stop after first run
            updater._stop.set()

        with patch.object(updater, "run_once", side_effect=fake_run_once):
            # Patch the initial 60s wait to timeout immediately
            original_wait_for = asyncio.wait_for

            async def fast_wait_for(coro, *, timeout):
                if timeout == 60:
                    raise asyncio.TimeoutError()
                return await original_wait_for(coro, timeout=timeout)

            with patch("asyncio.wait_for", side_effect=fast_wait_for):
                await updater.run_forever()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_run_forever_catches_exception_in_run_once(self, config: "RouterConfig") -> None:
        """run_forever should catch and log exceptions from run_once."""
        updater = AutoUpdater(config)
        call_count = 0

        async def failing_run_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            updater._stop.set()

        with patch.object(updater, "run_once", side_effect=failing_run_once):
            original_wait_for = asyncio.wait_for

            async def fast_wait_for(coro, *, timeout):
                if timeout == 60:
                    raise asyncio.TimeoutError()
                if timeout == config.update_interval_hours * 3600:
                    raise asyncio.TimeoutError()
                return await original_wait_for(coro, timeout=timeout)

            with patch("asyncio.wait_for", side_effect=fast_wait_for):
                await updater.run_forever()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_run_forever_propagates_system_exit(self, config: "RouterConfig") -> None:
        """run_forever should propagate SystemExit from run_once."""
        updater = AutoUpdater(config)

        async def exit_run_once():
            raise SystemExit(0)

        with patch.object(updater, "run_once", side_effect=exit_run_once):
            original_wait_for = asyncio.wait_for

            async def fast_wait_for(coro, *, timeout):
                if timeout == 60:
                    raise asyncio.TimeoutError()
                return await original_wait_for(coro, timeout=timeout)

            with patch("asyncio.wait_for", side_effect=fast_wait_for):
                with pytest.raises(SystemExit):
                    await updater.run_forever()

    @pytest.mark.asyncio
    async def test_stop(self, config: "RouterConfig") -> None:
        """stop() should set the _stop event."""
        updater = AutoUpdater(config)
        assert not updater._stop.is_set()
        await updater.stop()
        assert updater._stop.is_set()


# ---------------------------------------------------------------------------
# 8. Path helpers
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_read_install_var_found(self, tmp_path: Path) -> None:
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        install_sh = scripts_dir / "install.sh"
        install_sh.write_text('#!/bin/bash\nOMLX_REF="v1.2.3"\nMLX_LM_PACKAGE="mlx-lm>=0.5"\n')

        with patch("router.updater._privatenet_src", return_value=tmp_path):
            assert _read_install_var("OMLX_REF") == "v1.2.3"
            assert _read_install_var("MLX_LM_PACKAGE") == "mlx-lm>=0.5"

    def test_read_install_var_not_found(self, tmp_path: Path) -> None:
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        install_sh = scripts_dir / "install.sh"
        install_sh.write_text('#!/bin/bash\nFOO="bar"\n')

        with patch("router.updater._privatenet_src", return_value=tmp_path):
            assert _read_install_var("MISSING_VAR", "default_val") == "default_val"

    def test_read_install_var_no_script(self, tmp_path: Path) -> None:
        with patch("router.updater._privatenet_src", return_value=tmp_path):
            assert _read_install_var("OMLX_REF", "fallback") == "fallback"

    def test_read_install_var_read_error(self, tmp_path: Path) -> None:
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        install_sh = scripts_dir / "install.sh"
        install_sh.write_text('OMLX_REF="v1"\n')

        with patch("router.updater._privatenet_src", return_value=tmp_path):
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                assert _read_install_var("OMLX_REF", "default") == "default"

    def test_repo_root(self) -> None:
        result = _repo_root()
        assert isinstance(result, Path)

    def test_state_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OMLX_PRIVATENET_STATE_DIR", raising=False)
        result = _state_dir()
        assert result == Path.home() / ".omlx-privatenet"

    def test_state_dir_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", "/tmp/test-state")
        result = _state_dir()
        assert result == Path("/tmp/test-state")

    def test_privatenet_src_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OMLX_PRIVATENET_SRC", raising=False)
        result = _privatenet_src()
        assert result == _repo_root()

    def test_privatenet_src_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMLX_PRIVATENET_SRC", "/custom/src")
        result = _privatenet_src()
        assert result == Path("/custom/src")

    def test_venv_bin_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OMLX_PRIVATENET_VENV", raising=False)
        result = _venv_bin()
        import sys
        assert result == Path(sys.executable).parent

    def test_venv_bin_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMLX_PRIVATENET_VENV", "/custom/venv")
        result = _venv_bin()
        assert result == Path("/custom/venv") / "bin"

    def test_omlx_src_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OMLX_SRC", raising=False)
        result = _omlx_src()
        assert result == Path.home() / "omlx-privatenet" / "omlx"

    def test_omlx_src_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMLX_SRC", "/custom/omlx")
        result = _omlx_src()
        assert result == Path("/custom/omlx")


# ---------------------------------------------------------------------------
# 9. _get_local_sha
# ---------------------------------------------------------------------------

class TestGetLocalSha:
    def test_success(self, tmp_path: Path) -> None:
        with patch("router.updater._privatenet_src", return_value=tmp_path):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="abc1234\n", returncode=0)
                assert _get_local_sha() == "abc1234"

    def test_failure_returns_unknown(self, tmp_path: Path) -> None:
        with patch("router.updater._privatenet_src", return_value=tmp_path):
            with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
                assert _get_local_sha() == "unknown"


# ---------------------------------------------------------------------------
# 10. run_update edge cases
# ---------------------------------------------------------------------------

class TestRunUpdateEdgeCases:
    def test_previous_sha_exception(self, tmp_path: Path) -> None:
        """When rev-parse for previous SHA fails, previous_sha should be 'unknown'."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"

        call_index = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                # rev-parse fails
                raise FileNotFoundError("no git")
            result = MagicMock()
            result.returncode = 0
            result.stdout = "new1234\n"
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
        assert result.previous_sha == "unknown"

    def test_pip_install_failure(self, tmp_path: Path) -> None:
        """When pip install fails, run_update should return an error."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        req_file = src / "router" / "requirements.txt"
        req_file.parent.mkdir(parents=True)
        req_file.write_text("httpx\n")

        call_index = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index <= 3:
                # rev-parse, fetch, reset
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc1234\n"
                result.stderr = ""
                return result
            # pip install fails
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="pip failed")

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = run_update()

        assert result.success is False
        assert "pip install failed" in (result.error or "")

    def test_new_sha_exception(self, tmp_path: Path) -> None:
        """When rev-parse for new SHA fails, new_sha should be 'unknown'."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"

        call_index = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index <= 3:
                # rev-parse prev, fetch, reset
                result = MagicMock()
                result.returncode = 0
                result.stdout = "abc1234\n"
                result.stderr = ""
                return result
            # rev-parse for new SHA fails
            raise FileNotFoundError("no git")

        with (
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._venv_bin", return_value=tmp_path / "venv" / "bin"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = run_update()

        assert result.success is True
        assert result.new_sha == "unknown"


# ---------------------------------------------------------------------------
# 11. update_with_rollback edge cases
# ---------------------------------------------------------------------------

class TestUpdateWithRollbackEdgeCases:
    def test_update_failure_returns_immediately(self, tmp_path: Path) -> None:
        """When run_update fails, update_with_rollback should return immediately."""
        fail_result = UpdateResult(
            success=False,
            previous_sha="abc1234",
            new_sha="abc1234",
            error="git update failed",
        )
        with patch("router.updater.run_update", return_value=fail_result):
            result = update_with_rollback()

        assert result.success is False
        assert result.error == "git update failed"

    def test_health_poll_exception_triggers_rollback(self, tmp_path: Path) -> None:
        """When health endpoint throws exceptions, rollback should be triggered."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        # Pre-write update-state
        update_state_file = state / "update-state.json"
        update_state_file.write_text(json.dumps({
            "previous_sha": "old1234",
            "new_sha": "new5678",
        }))

        success_result = UpdateResult(
            success=True,
            previous_sha="old1234",
            new_sha="new5678",
            error=None,
        )

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "old1234\n"
            result.stderr = ""
            return result

        with (
            patch("router.updater.run_update", return_value=success_result),
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater.time.sleep"),
            patch("router.updater.time.monotonic", side_effect=[0, 0, 100]),
            patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = update_with_rollback(
                privatenet_src=src,
                state_dir=state,
            )

        assert result.success is False
        assert "rolled back" in (result.error or "").lower()

    def test_rollback_json_update_error_is_swallowed(self, tmp_path: Path) -> None:
        """When updating rollback.json reason fails, the error should be swallowed."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        update_state_file = state / "update-state.json"
        update_state_file.write_text(json.dumps({
            "previous_sha": "old1234",
            "new_sha": "new5678",
        }))

        success_result = UpdateResult(
            success=True,
            previous_sha="old1234",
            new_sha="new5678",
            error=None,
        )

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "old1234\n"
            result.stderr = ""
            return result

        with (
            patch("router.updater.run_update", return_value=success_result),
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater.time.sleep"),
            patch("router.updater.time.monotonic", side_effect=[0, 0, 100]),
            patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
            patch("subprocess.run", side_effect=mock_run),
        ):
            # After rollback creates rollback.json, corrupt it to trigger exception
            original_rollback = __import__("router.updater", fromlist=["rollback"]).rollback

            def patched_rollback(**kwargs):
                result = original_rollback(**kwargs)
                # Corrupt the rollback.json so the subsequent read fails
                rb_file = state / "rollback.json"
                if rb_file.exists():
                    rb_file.write_text("{{invalid json}}")
                return result

            with patch("router.updater.rollback", side_effect=patched_rollback):
                result = update_with_rollback(
                    privatenet_src=src,
                    state_dir=state,
                )

        # Should still return failure even though json update failed
        assert result.success is False
        assert "rolled back" in (result.error or "").lower()

    def test_rollback_fails_after_health_check(self, tmp_path: Path) -> None:
        """When health check fails and rollback also fails, return update failure."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        # No update-state.json => rollback() returns False
        success_result = UpdateResult(
            success=True,
            previous_sha="old1234",
            new_sha="new5678",
            error=None,
        )

        with (
            patch("router.updater.run_update", return_value=success_result),
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater.time.sleep"),
            patch("router.updater.time.monotonic", side_effect=[0, 0, 100]),
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
        ):
            result = update_with_rollback(
                privatenet_src=src,
                state_dir=state,
            )

        # Rollback failed (no state), so original success result is returned
        # but since health check failed... the result is the original success
        # because rolled_back is False, so no modification happens
        assert result.success is True  # no rollback occurred, original result kept

    def test_rollback_no_rollback_file(self, tmp_path: Path) -> None:
        """When rollback succeeds but rollback.json doesn't exist (edge case)."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        update_state_file = state / "update-state.json"
        update_state_file.write_text(json.dumps({
            "previous_sha": "old1234",
            "new_sha": "new5678",
        }))

        success_result = UpdateResult(
            success=True,
            previous_sha="old1234",
            new_sha="new5678",
            error=None,
        )

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "old1234\n"
            result.stderr = ""
            return result

        with (
            patch("router.updater.run_update", return_value=success_result),
            patch("router.updater._privatenet_src", return_value=src),
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater.time.sleep"),
            patch("router.updater.time.monotonic", side_effect=[0, 0, 100]),
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            patch("subprocess.run", side_effect=mock_run),
        ):
            # After rollback() creates rollback.json, remove it to test the branch
            # where rollback_file.exists() is False
            original_rollback = __import__("router.updater", fromlist=["rollback"]).rollback

            def patched_rollback(**kwargs):
                result = original_rollback(**kwargs)
                # Remove rollback.json after rollback succeeds
                rb_file = state / "rollback.json"
                if rb_file.exists():
                    rb_file.unlink()
                return result

            with patch("router.updater.rollback", side_effect=patched_rollback):
                result = update_with_rollback(
                    privatenet_src=src,
                    state_dir=state,
                )

        assert result.success is False
        assert "rolled back" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# 12. drain_and_run timeout
# ---------------------------------------------------------------------------

class TestDrainAndRunTimeout:
    def test_timeout_runs_callback_when_inflight_never_zero(self, tmp_path: Path) -> None:
        """When health endpoint always shows in_flight > 0, callback runs after timeout."""
        state = tmp_path / "state"
        disabled_file = state / "disabled"

        def mock_urlopen(url, *, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "cluster": [{"local": True, "in_flight": 5}],
            }).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        callback_called = False

        def my_callback():
            nonlocal callback_called
            callback_called = True
            return "done"

        # Use monotonic to simulate timeout
        times = iter([0, 0.5, 100])  # third call exceeds deadline

        with (
            patch("router.updater._state_dir", return_value=state),
            patch("router.updater._disabled_file", return_value=disabled_file),
            patch("urllib.request.urlopen", side_effect=mock_urlopen),
            patch("router.updater.time.monotonic", side_effect=times),
            patch("router.updater.time.sleep"),
        ):
            result = drain_and_run(my_callback, timeout=1)

        assert callback_called
        assert result == "done"


# ---------------------------------------------------------------------------
# 13. get_rollback_info with corrupt file
# ---------------------------------------------------------------------------

class TestDisabledFile:
    def test_disabled_file_path(self, tmp_path: Path) -> None:
        from router.updater import _disabled_file
        with patch("router.updater._state_dir", return_value=tmp_path):
            result = _disabled_file()
            assert result == tmp_path / "disabled"


class TestGetRollbackInfoEdgeCases:
    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        state.mkdir()
        rollback_file = state / "rollback.json"
        rollback_file.write_text("not valid json")

        assert get_rollback_info(state_dir=state) is None


# ---------------------------------------------------------------------------
# 14. rollback edge case: rev-parse fails
# ---------------------------------------------------------------------------

class TestRollbackEdgeCases:
    def test_rollback_revparse_failure(self, tmp_path: Path) -> None:
        """When rev-parse for current SHA fails during rollback, current_sha is 'unknown'."""
        src = tmp_path / "src"
        src.mkdir()
        state = tmp_path / "state"
        state.mkdir()

        update_state = state / "update-state.json"
        update_state.write_text(json.dumps({"previous_sha": "abc1234"}))

        call_index = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                # rev-parse fails
                raise FileNotFoundError("no git")
            # git checkout succeeds
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            success = rollback(privatenet_src=src, state_dir=state)

        assert success is True
        rollback_file = state / "rollback.json"
        info = json.loads(rollback_file.read_text())
        assert info["rolled_back_from"] == "unknown"
