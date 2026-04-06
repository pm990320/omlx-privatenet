from __future__ import annotations

"""Auto-updater for oMLX PrivateNet: version tracking, update execution, drain, and rollback."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import RouterConfig

T = TypeVar("T")

logger = logging.getLogger(__name__)

MLX_LM_FORK = "git+https://github.com/pm990320/mlx-lm@feat/gemma4-tool-calling"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (parent of the router/ package)."""
    return Path(__file__).resolve().parent.parent


def _state_dir() -> Path:
    return Path(os.environ.get("OMLX_PRIVATENET_STATE_DIR", Path.home() / ".omlx-privatenet"))


def _privatenet_src() -> Path:
    """Return the source checkout path used for git operations."""
    env = os.environ.get("OMLX_PRIVATENET_SRC")
    if env:
        return Path(env)
    return _repo_root()


def _venv_bin() -> Path:
    """Return the venv bin directory for pip installs."""
    env = os.environ.get("OMLX_PRIVATENET_VENV")
    if env:
        return Path(env) / "bin"
    # Walk up from the running interpreter to find the venv bin
    return Path(sys.executable).parent


def _omlx_src() -> Path:
    """Return the oMLX source checkout path (~/omlx-privatenet/omlx by default)."""
    env = os.environ.get("OMLX_SRC")
    if env:
        return Path(env)
    return Path.home() / "omlx-privatenet" / "omlx"


def _disabled_file() -> Path:
    return _state_dir() / "disabled"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UpdateInfo:
    """Result of checking for an available update."""

    available: bool
    local_version: str
    remote_version: str
    local_sha: str
    remote_sha: str


@dataclass(slots=True)
class UpdateResult:
    """Result of running an update."""

    success: bool
    previous_sha: str
    new_sha: str
    error: str | None


# ---------------------------------------------------------------------------
# 1. Version tracking
# ---------------------------------------------------------------------------

def get_local_version() -> str:
    """Read the VERSION file from the repo root, falling back to git short SHA."""
    version_file = _repo_root() / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 2. GitHub version check
# ---------------------------------------------------------------------------

def _get_local_sha() -> str:
    """Return the local git short SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_privatenet_src()),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def check_for_update(
    repo: str = "pm990320/omlx-privatenet",
    branch: str = "main",
) -> UpdateInfo:
    """Check GitHub for a newer version of oMLX PrivateNet."""
    local_version = get_local_version()
    local_sha = _get_local_sha()

    try:
        # Fetch latest commit SHA
        commit_url = f"https://api.github.com/repos/{repo}/commits/{branch}"
        req = urllib.request.Request(commit_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            commit_data = json.loads(resp.read())
        remote_sha = commit_data["sha"][:7]

        # Fetch remote VERSION file
        version_url = f"https://raw.githubusercontent.com/{repo}/{branch}/VERSION"
        with urllib.request.urlopen(version_url, timeout=10) as resp:
            remote_version = resp.read().decode("utf-8").strip()

    except Exception:
        return UpdateInfo(
            available=False,
            local_version=local_version,
            remote_version=local_version,
            local_sha=local_sha,
            remote_sha=local_sha,
        )

    available = (remote_version != local_version) or (remote_sha != local_sha)
    return UpdateInfo(
        available=available,
        local_version=local_version,
        remote_version=remote_version,
        local_sha=local_sha,
        remote_sha=remote_sha,
    )


# ---------------------------------------------------------------------------
# 3. Update execution
# ---------------------------------------------------------------------------

def _read_update_state(state_dir: Path) -> dict[str, Any]:
    state_file = state_dir / "update-state.json"
    if state_file.exists():
        with state_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_update_state(state_dir: Path, data: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "update-state.json"
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def run_update(
    repo: str = "pm990320/omlx-privatenet",
    branch: str = "main",
) -> UpdateResult:
    """Perform a git-based update of the oMLX PrivateNet source."""
    src = _privatenet_src()
    state = _state_dir()
    venv_bin = _venv_bin()

    # Capture current SHA before update
    try:
        result = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        previous_sha = result.stdout.strip()
    except Exception:
        previous_sha = "unknown"

    # Fetch and reset
    try:
        subprocess.run(
            ["git", "-C", str(src), "fetch", "origin"],
            capture_output=True, text=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(src), "reset", "--hard", f"origin/{branch}"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        return UpdateResult(
            success=False,
            previous_sha=previous_sha,
            new_sha=previous_sha,
            error=f"git update failed: {exc.stderr or exc.stdout or str(exc)}",
        )

    # Install requirements
    requirements_file = src / "router" / "requirements.txt"
    if requirements_file.exists():
        try:
            pip = str(venv_bin / "pip")
            subprocess.run(
                [pip, "install", "-r", str(requirements_file)],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            return UpdateResult(
                success=False,
                previous_sha=previous_sha,
                new_sha=previous_sha,
                error=f"pip install failed: {exc.stderr or exc.stdout or str(exc)}",
            )

    # Update oMLX source if present (server mode only)
    omlx_src = _omlx_src()
    if omlx_src.is_dir():
        try:
            # Fetch latest tags
            subprocess.run(
                ["git", "-C", str(omlx_src), "fetch", "--tags", "origin"],
                capture_output=True, text=True, check=True,
            )
            # Get the latest tag
            tag_result = subprocess.run(
                ["git", "-C", str(omlx_src), "describe", "--tags", "--abbrev=0",
                 "origin/HEAD"],
                capture_output=True, text=True, check=True,
            )
            latest_tag = tag_result.stdout.strip()
            if latest_tag:
                subprocess.run(
                    ["git", "-C", str(omlx_src), "checkout", latest_tag],
                    capture_output=True, text=True, check=True,
                )
            # Reinstall oMLX in editable mode
            pip = str(venv_bin / "pip")
            subprocess.run(
                [pip, "install", "-e", str(omlx_src)],
                capture_output=True, text=True, check=True,
            )
            # Reinstall the mlx-lm fork
            subprocess.run(
                [pip, "install", "--upgrade", "--force-reinstall", MLX_LM_FORK],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            return UpdateResult(
                success=False,
                previous_sha=previous_sha,
                new_sha=previous_sha,
                error=f"oMLX update failed: {exc.stderr or exc.stdout or str(exc)}",
            )

    # Capture new SHA
    try:
        result = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        new_sha = result.stdout.strip()
    except Exception:
        new_sha = "unknown"

    # Store previous SHA for rollback
    _write_update_state(state, {
        "previous_sha": previous_sha,
        "new_sha": new_sha,
        "timestamp": time.time(),
    })

    return UpdateResult(
        success=True,
        previous_sha=previous_sha,
        new_sha=new_sha,
        error=None,
    )


# ---------------------------------------------------------------------------
# 4. Drain helper
# ---------------------------------------------------------------------------

def drain_and_run(
    callback: Callable[[], T],
    health_url: str = "http://127.0.0.1:8741",
    timeout: float = 60,
) -> T:
    """Disable the node, wait for in-flight requests to drain, run callback, re-enable."""
    state = _state_dir()
    state.mkdir(parents=True, exist_ok=True)
    disabled_file = _disabled_file()

    # Create the disabled file
    disabled_file.write_text("Disabled for update\n", encoding="utf-8")

    try:
        # Poll health endpoint until in_flight == 0 or timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{health_url}/health", timeout=5) as resp:
                    data = json.loads(resp.read())
                cluster = data.get("cluster", [])
                # Find local node
                local_node = next((n for n in cluster if n.get("local")), None)
                if local_node is None or local_node.get("in_flight", 0) == 0:
                    break
            except Exception:
                # Health endpoint not reachable; proceed anyway
                break
            time.sleep(1)

        return callback()
    finally:
        # Re-enable
        if disabled_file.exists():
            disabled_file.unlink()


# ---------------------------------------------------------------------------
# 5. Rollback
# ---------------------------------------------------------------------------

def rollback(
    privatenet_src: Path | None = None,
    state_dir: Path | None = None,
) -> bool:
    """Roll back to the previous SHA recorded in update-state.json."""
    src = privatenet_src or _privatenet_src()
    state = state_dir or _state_dir()

    update_state = _read_update_state(state)
    previous_sha = update_state.get("previous_sha")
    if not previous_sha:
        return False

    # Capture current SHA (the one we're rolling back from)
    try:
        result = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        current_sha = result.stdout.strip()
    except Exception:
        current_sha = "unknown"

    try:
        subprocess.run(
            ["git", "-C", str(src), "checkout", previous_sha],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError:
        return False

    # Write rollback.json
    rollback_info = {
        "timestamp": time.time(),
        "reason": "manual rollback",
        "rolled_back_from": current_sha,
        "rolled_back_to": previous_sha,
    }
    state.mkdir(parents=True, exist_ok=True)
    rollback_file = state / "rollback.json"
    with rollback_file.open("w", encoding="utf-8") as f:
        json.dump(rollback_info, f, indent=2)
        f.write("\n")

    return True


def update_with_rollback(
    privatenet_src: Path | None = None,
    state_dir: Path | None = None,
    venv_dir: Path | None = None,
    health_url: str = "http://127.0.0.1:8741",
) -> UpdateResult:
    """Run an update and verify health afterward, rolling back on failure.

    1. Calls run_update() to perform the update.
    2. If the update succeeded, waits 10 seconds for the router to restart.
    3. Polls GET {health_url}/health for up to 30 seconds (every 5s).
       If status != "ok", calls rollback().
    4. Returns the update result with rollback info if applicable.
    """
    result = run_update()
    if not result.success:
        return result

    # Wait for the router to restart after code update
    time.sleep(10)

    # Poll health endpoint
    src = privatenet_src or _privatenet_src()
    state = state_dir or _state_dir()
    healthy = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{health_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            if data.get("status") == "ok":
                healthy = True
                break
        except Exception:
            pass
        time.sleep(5)

    if not healthy:
        rolled_back = rollback(privatenet_src=src, state_dir=state)
        if rolled_back:
            # Update the rollback.json reason to reflect automatic rollback
            rollback_file = state / "rollback.json"
            if rollback_file.exists():
                try:
                    with rollback_file.open("r", encoding="utf-8") as f:
                        info = json.load(f)
                    info["reason"] = "automatic rollback after failed health check"
                    with rollback_file.open("w", encoding="utf-8") as f:
                        json.dump(info, f, indent=2)
                        f.write("\n")
                except Exception:
                    pass
            result = UpdateResult(
                success=False,
                previous_sha=result.previous_sha,
                new_sha=result.new_sha,
                error="Health check failed after update; rolled back",
            )

    return result


def get_rollback_info(state_dir: Path | None = None) -> dict[str, Any] | None:
    """Read rollback.json if it exists, for health endpoint reporting."""
    state = state_dir or _state_dir()
    rollback_file = state / "rollback.json"
    if not rollback_file.exists():
        return None
    try:
        with rollback_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 7. Background auto-update loop
# ---------------------------------------------------------------------------


class AutoUpdater:
    """Background auto-update loop.

    Periodically checks for a newer version and applies the update using
    drain_and_run + update_with_rollback when one is available.
    """

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        """Check for updates every ``update_interval_hours``."""
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001
                logger.exception("Auto-update cycle failed")
            try:
                interval = self.config.update_interval_hours * 3600
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> None:
        """Check for an update and apply it if available."""
        info = await asyncio.to_thread(check_for_update)
        if not info.available:
            logger.debug(
                "Auto-update: up to date (local=%s, remote=%s)",
                info.local_version,
                info.remote_version,
            )
            return

        logger.info(
            "Auto-update: update available %s -> %s, applying...",
            info.local_version,
            info.remote_version,
        )

        health_url = f"http://127.0.0.1:{self.config.port}"

        result: UpdateResult = await asyncio.to_thread(
            lambda: drain_and_run(
                lambda: update_with_rollback(
                    privatenet_src=_privatenet_src(),
                    state_dir=_state_dir(),
                    venv_dir=_venv_bin().parent,
                    health_url=health_url,
                ),
                health_url=health_url,
            )
        )

        if result.success:
            logger.info(
                "Auto-update: successfully updated %s -> %s",
                result.previous_sha,
                result.new_sha,
            )
        else:
            logger.error("Auto-update: update failed: %s", result.error)

    async def stop(self) -> None:
        """Signal the loop to exit."""
        self._stop.set()
