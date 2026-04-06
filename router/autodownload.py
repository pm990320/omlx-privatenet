from __future__ import annotations

"""Auto-download loop for registry models missing locally.

Periodically checks the model registry and downloads missing models that
fit within node capacity and the configured GB cap.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import RouterConfig
from .fitcheck import FitResult, check_model_fit
from .registry import Registry, RegistryModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path.home() / ".omlx" / "models"


def _model_dir() -> Path:
    env = os.getenv("OMLX_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_MODEL_DIR


def _state_dir() -> Path:
    env = os.getenv("OMLX_PRIVATENET_STATE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".omlx-privatenet"


def _get_local_models(omlx_url: str, omlx_api_key: Optional[str] = None) -> list[str]:
    """Fetch model IDs from the local oMLX /v1/models endpoint."""
    import urllib.request
    import urllib.error

    url = f"{omlx_url.rstrip('/')}/v1/models"
    req = urllib.request.Request(url)
    if omlx_api_key:
        req.add_header("Authorization", f"Bearer {omlx_api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("Could not fetch local models from oMLX: %s", exc)
        return []

    # Handle both {data: [...]} and plain list formats
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        items = data
    else:
        return []

    results: list[str] = []
    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id")
            if model_id:
                results.append(str(model_id))
        elif isinstance(item, str) and item:
            results.append(item)
    return results


def _existing_download_size_gb(model_dir: Path) -> float:
    """Sum the size of all existing model directories in GB."""
    if not model_dir.exists():
        return 0.0
    total = 0
    for entry in model_dir.iterdir():
        if entry.is_dir():
            for f in entry.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
    return total / (1024 ** 3)


def _run_download(repo: str, target_dir: Path) -> subprocess.CompletedProcess[str]:
    """Download a model from HuggingFace Hub.

    Tries huggingface-cli first, falls back to python -m huggingface_hub.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("huggingface-cli"):
        cmd = ["huggingface-cli", "download", repo, "--local-dir", str(target_dir)]
    else:
        cmd = ["python", "-m", "huggingface_hub", "download", repo, "--local-dir", str(target_dir)]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600)


class AutoDownloader:
    """Background service that downloads registry models missing locally."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._stop = asyncio.Event()
        self._interval = config.discovery_interval_seconds * 2

    async def run_forever(self) -> None:
        """Continuously check for and download missing models."""
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001
                logger.exception("Auto-download cycle failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> None:
        """Run a single auto-download cycle."""
        if not self.config.auto_download:
            return

        # 1. Load registry
        registry = Registry(path=_state_dir() / "registry.json")
        registry.load()
        registry_models = registry.models
        if not registry_models:
            logger.debug("Auto-download: registry is empty, nothing to do")
            return

        # 2. Get locally available models
        local_model_ids = await asyncio.to_thread(
            _get_local_models,
            self.config.local_omlx_url,
            self.config.local_omlx_api_key,
        )
        local_set = set(local_model_ids)

        # 3. Find missing models, sorted by priority (ascending = most important first)
        missing = [m for m in registry_models if m.id not in local_set]
        missing.sort(key=lambda m: m.priority)

        if not missing:
            logger.debug("Auto-download: all registry models are available locally")
            return

        model_dir = _model_dir()

        # Check GB cap
        existing_gb = await asyncio.to_thread(_existing_download_size_gb, model_dir)
        max_gb = self.config.auto_download_max_gb

        for model in missing:
            if self._stop.is_set():
                break

            # Check safetensors requirement
            fit: FitResult = await asyncio.to_thread(
                check_model_fit,
                model.repo,
                self.config.local_omlx_url,
                self.config.local_omlx_api_key,
            )

            if model.safetensors_only and not fit.has_safetensors:
                logger.info(
                    "Auto-download: skipping %s — safetensors required but not available",
                    model.id,
                )
                continue

            if not fit.fits:
                logger.info(
                    "Auto-download: skipping %s — %s",
                    model.id,
                    fit.reason,
                )
                continue

            # Check GB cap (estimate: use fit model_size_bytes)
            model_gb = fit.model_size_bytes / (1024 ** 3)
            if max_gb is not None and (existing_gb + model_gb) > max_gb:
                logger.info(
                    "Auto-download: skipping %s — would exceed cap (%.1f + %.1f > %d GB)",
                    model.id,
                    existing_gb,
                    model_gb,
                    max_gb,
                )
                continue

            target = model_dir / model.id
            logger.info("Auto-download: downloading %s from %s", model.id, model.repo)

            result = await asyncio.to_thread(_run_download, model.repo, target)

            if result.returncode == 0:
                logger.info("Auto-download: successfully downloaded %s", model.id)
                existing_gb += model_gb
                # Run eviction if over GB cap
                if max_gb is not None:
                    await self._maybe_evict(registry, model_dir, max_gb)
            else:
                logger.error(
                    "Auto-download: failed to download %s — exit code %d: %s",
                    model.id,
                    result.returncode,
                    result.stderr[:500] if result.stderr else "(no output)",
                )

    async def _maybe_evict(
        self, registry: Registry, model_dir: Path, max_gb: float
    ) -> None:
        """Run eviction if disk usage exceeds the GB cap."""
        from .eviction import execute_eviction, plan_eviction

        try:
            plan = await asyncio.to_thread(
                plan_eviction,
                model_dir,
                registry,
                max_gb,
                self.config.advertise_models,
                self.config.local_omlx_url,
                self.config.local_omlx_api_key,
            )
            if plan.models_to_evict:
                logger.info(
                    "Auto-download eviction: %s (%s)",
                    plan.reason,
                    ", ".join(plan.models_to_evict),
                )
                deleted = await asyncio.to_thread(
                    execute_eviction, plan, model_dir
                )
                if deleted:
                    logger.info("Auto-download eviction: removed %s", ", ".join(deleted))
        except Exception:  # noqa: BLE001
            logger.exception("Eviction after download failed")

    async def stop(self) -> None:
        """Signal the loop to exit."""
        self._stop.set()
