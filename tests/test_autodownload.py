from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from router.autodownload import AutoDownloader, _existing_download_size_gb
from router.config import RouterConfig
from router.fitcheck import FitResult
from router.registry import RegistryModel


def _make_config(tmp_path: Path, **overrides) -> RouterConfig:
    """Build a RouterConfig pointing at isolated temp dirs."""
    defaults = {
        "auto_download": True,
        "auto_download_max_gb": 100,
        "local_omlx_url": "http://127.0.0.1:5741",
        "local_omlx_api_key": None,
        "discovery_interval_seconds": 10,
        "trusted_orgs": ["mlx-community"],
    }
    defaults.update(overrides)
    return RouterConfig(**defaults)


def _make_registry_file(state_dir: Path, models: list[dict]) -> None:
    registry_path = state_dir / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(models, indent=2), encoding="utf-8")


SAMPLE_MODEL = {
    "repo": "mlx-community/gemma-2b",
    "id": "gemma-2b",
    "priority": 1,
    "added_by": "test",
    "safetensors_only": True,
}

FIT_OK = FitResult(
    fits=True,
    model_size_bytes=2_000_000_000,
    max_memory_bytes=16_000_000_000,
    reason="Model fits: 2.0 GB model within 16.0 GB capacity",
    has_safetensors=True,
)

FIT_TOO_LARGE = FitResult(
    fits=False,
    model_size_bytes=20_000_000_000,
    max_memory_bytes=16_000_000_000,
    reason="Model too large: 20.0 GB model exceeds 16.0 GB capacity",
    has_safetensors=True,
)

FIT_NO_SAFETENSORS = FitResult(
    fits=True,
    model_size_bytes=2_000_000_000,
    max_memory_bytes=16_000_000_000,
    reason="Model fits",
    has_safetensors=False,
)


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    """Point state and model dirs at tmp_path to avoid touching real dirs."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OMLX_MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "state").mkdir()
    (tmp_path / "models").mkdir()


@pytest.mark.asyncio
async def test_skips_when_auto_download_false(tmp_path):
    """run_once should do nothing when auto_download is disabled."""
    config = _make_config(tmp_path, auto_download=False)
    downloader = AutoDownloader(config)

    # If it tried to load the registry or fetch models, it would fail because
    # there's no registry file. No error means it returned early.
    await downloader.run_once()


@pytest.mark.asyncio
async def test_downloads_missing_model(tmp_path, monkeypatch):
    """Should issue a download command for a missing model that fits."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])

    config = _make_config(tmp_path)

    # Mock _get_local_models to return empty (no models locally)
    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )

    # Mock check_model_fit to return fits=True
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_OK,
    )

    # Mock _run_download to track the call
    download_calls: list[tuple[str, Path]] = []

    def fake_download(repo: str, target_dir: Path):
        download_calls.append((repo, target_dir))
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr("router.autodownload._run_download", fake_download)

    downloader = AutoDownloader(config)
    await downloader.run_once()

    assert len(download_calls) == 1
    assert download_calls[0][0] == "mlx-community/gemma-2b"
    assert download_calls[0][1].name == "gemma-2b"


@pytest.mark.asyncio
async def test_skips_model_that_doesnt_fit(tmp_path, monkeypatch):
    """Should not download a model when fitcheck says it doesn't fit."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])

    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_TOO_LARGE,
    )

    download_calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "router.autodownload._run_download",
        lambda repo, target: download_calls.append((repo, target)) or MagicMock(returncode=0),
    )

    downloader = AutoDownloader(config)
    await downloader.run_once()

    assert len(download_calls) == 0


@pytest.mark.asyncio
async def test_respects_max_gb_cap(tmp_path, monkeypatch):
    """Should skip download when existing models exceed the GB cap."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])

    # Set a very low cap
    config = _make_config(tmp_path, auto_download_max_gb=1)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )

    # FIT_OK has model_size_bytes=2GB, which exceeds cap of 1GB
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_OK,
    )

    download_calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "router.autodownload._run_download",
        lambda repo, target: download_calls.append((repo, target)) or MagicMock(returncode=0),
    )

    downloader = AutoDownloader(config)
    await downloader.run_once()

    assert len(download_calls) == 0


@pytest.mark.asyncio
async def test_skips_model_without_safetensors(tmp_path, monkeypatch):
    """Should skip when safetensors_only=True but has_safetensors=False."""
    state_dir = tmp_path / "state"
    model = dict(SAMPLE_MODEL, safetensors_only=True)
    _make_registry_file(state_dir, [model])

    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_NO_SAFETENSORS,
    )

    download_calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "router.autodownload._run_download",
        lambda repo, target: download_calls.append((repo, target)) or MagicMock(returncode=0),
    )

    downloader = AutoDownloader(config)
    await downloader.run_once()

    assert len(download_calls) == 0
