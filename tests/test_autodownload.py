from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from router.autodownload import (
    AutoDownloader,
    _existing_download_size_gb,
    _get_local_models,
    _model_dir,
    _run_download,
    _state_dir,
)
from router.config import RouterConfig
from router.fitcheck import FitResult
from router.registry import Registry, RegistryModel


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


# ---------------------------------------------------------------------------
# _model_dir / _state_dir defaults (lines 31, 38)
# ---------------------------------------------------------------------------


def test_model_dir_default(monkeypatch):
    """_model_dir returns default when env var is unset."""
    monkeypatch.delenv("OMLX_MODELS_DIR", raising=False)
    result = _model_dir()
    assert result == Path.home() / ".omlx" / "models"


def test_state_dir_default(monkeypatch):
    """_state_dir returns default when env var is unset."""
    monkeypatch.delenv("OMLX_PRIVATENET_STATE_DIR", raising=False)
    result = _state_dir()
    assert result == Path.home() / ".omlx-privatenet"


# ---------------------------------------------------------------------------
# _get_local_models (lines 43-73)
# ---------------------------------------------------------------------------


def test_get_local_models_dict_data(monkeypatch):
    """_get_local_models parses {data: [{id: ...}]} response."""
    body = json.dumps({"data": [{"id": "model-a"}, {"id": "model-b"}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: FakeResp()
    )
    result = _get_local_models("http://fake:5741")
    assert result == ["model-a", "model-b"]


def test_get_local_models_with_api_key(monkeypatch):
    """_get_local_models adds auth header when api key is given."""
    body = json.dumps({"data": [{"id": "m1"}]}).encode()
    captured_reqs = []

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=None):
        captured_reqs.append(req)
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = _get_local_models("http://fake:5741", omlx_api_key="secret")
    assert result == ["m1"]
    assert captured_reqs[0].get_header("Authorization") == "Bearer secret"


def test_get_local_models_plain_list(monkeypatch):
    """_get_local_models handles plain list response."""
    body = json.dumps(["model-x", "model-y"]).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: FakeResp()
    )
    result = _get_local_models("http://fake:5741")
    assert result == ["model-x", "model-y"]


def test_get_local_models_unexpected_type(monkeypatch):
    """_get_local_models returns [] for non-dict/non-list response."""
    body = json.dumps("just-a-string").encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: FakeResp()
    )
    result = _get_local_models("http://fake:5741")
    assert result == []


def test_get_local_models_url_error(monkeypatch):
    """_get_local_models returns [] on connection error."""
    import urllib.error

    def fail(*a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    result = _get_local_models("http://fake:5741")
    assert result == []


def test_get_local_models_models_key(monkeypatch):
    """_get_local_models handles {models: [...]} response."""
    body = json.dumps({"models": [{"id": "m1"}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: FakeResp()
    )
    result = _get_local_models("http://fake:5741")
    assert result == ["m1"]


def test_get_local_models_string_items_in_list(monkeypatch):
    """_get_local_models handles list with string items and skips empty."""
    body = json.dumps({"data": ["alpha", "beta", ""]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: FakeResp()
    )
    result = _get_local_models("http://fake:5741")
    assert result == ["alpha", "beta"]


def test_get_local_models_dict_item_missing_id(monkeypatch):
    """_get_local_models skips dict items without id."""
    body = json.dumps({"data": [{"name": "no-id"}, {"id": "has-id"}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: FakeResp()
    )
    result = _get_local_models("http://fake:5741")
    assert result == ["has-id"]


# ---------------------------------------------------------------------------
# _existing_download_size_gb (lines 79, 82-85)
# ---------------------------------------------------------------------------


def test_existing_download_size_gb_nonexistent(tmp_path):
    """Returns 0 when model dir doesn't exist."""
    assert _existing_download_size_gb(tmp_path / "nope") == 0.0


def test_existing_download_size_gb_empty_dir(tmp_path):
    """Returns 0 when model dir exists but is empty (no subdirectories)."""
    model_dir = tmp_path / "models_empty"
    model_dir.mkdir()
    assert _existing_download_size_gb(model_dir) == 0.0


def test_existing_download_size_gb_empty_subdir(tmp_path):
    """Returns 0 when model subdir exists but has no files."""
    model_dir = tmp_path / "models_nosub"
    sub = model_dir / "empty-model"
    sub.mkdir(parents=True)
    assert _existing_download_size_gb(model_dir) == 0.0


def test_existing_download_size_gb_with_files(tmp_path):
    """Sums sizes of files in subdirectories."""
    model_dir = tmp_path / "models"
    sub = model_dir / "some-model"
    sub.mkdir(parents=True)
    (sub / "weights.bin").write_bytes(b"\0" * 1024)
    result = _existing_download_size_gb(model_dir)
    assert result == pytest.approx(1024 / (1024 ** 3), rel=1e-3)


def test_existing_download_size_gb_skips_non_dir(tmp_path):
    """Skips plain files at the top level of model_dir."""
    model_dir = tmp_path / "models_flat"
    model_dir.mkdir()
    (model_dir / "README.txt").write_text("hello")
    assert _existing_download_size_gb(model_dir) == 0.0


def test_existing_download_size_gb_nested_subdir(tmp_path):
    """Handles nested subdirectories (rglob yields dirs that aren't files)."""
    model_dir = tmp_path / "models_nested"
    sub = model_dir / "some-model" / "nested"
    sub.mkdir(parents=True)
    (sub / "weights.bin").write_bytes(b"\0" * 512)
    result = _existing_download_size_gb(model_dir)
    assert result == pytest.approx(512 / (1024 ** 3), rel=1e-3)


# ---------------------------------------------------------------------------
# _run_download (lines 94-101)
# ---------------------------------------------------------------------------


def test_run_download_uses_huggingface_cli(tmp_path, monkeypatch):
    """_run_download uses huggingface-cli when available."""
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/huggingface-cli")
    captured = []

    def fake_run(cmd, capture_output, text, timeout):
        captured.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    target = tmp_path / "models" / "test-model"
    _run_download("mlx-community/test-model", target)
    assert captured[0][0] == "huggingface-cli"
    assert "--local-dir" in captured[0]


def test_run_download_falls_back_to_python(tmp_path, monkeypatch):
    """_run_download falls back to python -m when huggingface-cli not found."""
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    captured = []

    def fake_run(cmd, capture_output, text, timeout):
        captured.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    target = tmp_path / "models" / "test-model"
    _run_download("mlx-community/test-model", target)
    assert captured[0][0] == "python"
    assert captured[0][1] == "-m"


# ---------------------------------------------------------------------------
# run_forever loop (lines 114-122)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_forever_runs_then_stops(tmp_path, monkeypatch):
    """run_forever calls run_once and stops when stop() is called."""
    config = _make_config(tmp_path)
    downloader = AutoDownloader(config)
    downloader._interval = 0.01

    call_count = 0

    async def counting_run_once():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            await downloader.stop()

    monkeypatch.setattr(downloader, "run_once", counting_run_once)
    await downloader.run_forever()
    assert call_count >= 2


@pytest.mark.asyncio
async def test_run_forever_handles_exception(tmp_path, monkeypatch):
    """run_forever catches exceptions in run_once and continues."""
    config = _make_config(tmp_path)
    downloader = AutoDownloader(config)
    downloader._interval = 0.01

    call_count = 0

    async def failing_run_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        await downloader.stop()

    monkeypatch.setattr(downloader, "run_once", failing_run_once)
    await downloader.run_forever()
    assert call_count >= 2


# ---------------------------------------------------------------------------
# run_once: empty registry (lines 134-135)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_empty_registry(tmp_path, monkeypatch):
    """run_once returns early if registry has no models."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [])
    config = _make_config(tmp_path)
    downloader = AutoDownloader(config)
    await downloader.run_once()


# ---------------------------------------------------------------------------
# run_once: all models present (lines 150-151)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_all_models_present(tmp_path, monkeypatch):
    """run_once returns early when all registry models are already local."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: ["gemma-2b"],
    )

    downloader = AutoDownloader(config)
    await downloader.run_once()


# ---------------------------------------------------------------------------
# run_once: stop mid-loop (line 161)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_stop_mid_loop(tmp_path, monkeypatch):
    """run_once breaks when stop is set before iteration begins."""
    state_dir = tmp_path / "state"
    model_a = dict(SAMPLE_MODEL, id="model-a", repo="mlx-community/model-a", priority=1)
    model_b = dict(SAMPLE_MODEL, id="model-b", repo="mlx-community/model-b", priority=2)
    _make_registry_file(state_dir, [model_a, model_b])

    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_OK,
    )

    downloader = AutoDownloader(config)
    downloader._stop.set()

    download_calls = []
    monkeypatch.setattr(
        "router.autodownload._run_download",
        lambda repo, target: download_calls.append(repo) or MagicMock(returncode=0),
    )

    await downloader.run_once()
    assert len(download_calls) == 0


# ---------------------------------------------------------------------------
# run_once: download failure (line 210)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_download_failure(tmp_path, monkeypatch):
    """run_once logs error when download returns non-zero."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_OK,
    )

    def failing_download(repo, target):
        m = MagicMock()
        m.returncode = 1
        m.stderr = "fatal error"
        return m

    monkeypatch.setattr("router.autodownload._run_download", failing_download)

    downloader = AutoDownloader(config)
    await downloader.run_once()


@pytest.mark.asyncio
async def test_run_once_download_failure_no_stderr(tmp_path, monkeypatch):
    """run_once handles download failure with empty stderr."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_OK,
    )

    def failing_download(repo, target):
        m = MagicMock()
        m.returncode = 1
        m.stderr = ""
        return m

    monkeypatch.setattr("router.autodownload._run_download", failing_download)

    downloader = AutoDownloader(config)
    await downloader.run_once()


# ---------------------------------------------------------------------------
# _maybe_evict integration (lines 207->159, 234-245)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_evict_called_on_success(tmp_path, monkeypatch):
    """After successful download with max_gb set, _maybe_evict runs."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    config = _make_config(tmp_path, auto_download_max_gb=100)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_OK,
    )
    monkeypatch.setattr(
        "router.autodownload._run_download",
        lambda repo, target: MagicMock(returncode=0),
    )

    evict_calls = []

    async def fake_maybe_evict(registry, model_dir, max_gb):
        evict_calls.append((model_dir, max_gb))

    downloader = AutoDownloader(config)
    monkeypatch.setattr(downloader, "_maybe_evict", fake_maybe_evict)
    await downloader.run_once()

    assert len(evict_calls) == 1
    assert evict_calls[0][1] == 100


@pytest.mark.asyncio
async def test_maybe_evict_with_eviction_plan(tmp_path, monkeypatch):
    """_maybe_evict calls plan_eviction and execute_eviction."""
    from router.eviction import EvictionPlan

    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    registry = Registry(path=state_dir / "registry.json")
    registry.load()

    config = _make_config(tmp_path, auto_download_max_gb=100)
    downloader = AutoDownloader(config)

    plan = EvictionPlan(
        models_to_evict=["old-model"],
        bytes_to_free=1_000_000,
        reason="over cap",
    )

    execute_calls = []
    monkeypatch.setattr(
        "router.eviction.plan_eviction",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        "router.eviction.execute_eviction",
        lambda p, md: execute_calls.append(p) or ["old-model"],
    )

    model_dir = tmp_path / "models"
    await downloader._maybe_evict(registry, model_dir, 100)
    assert len(execute_calls) == 1


@pytest.mark.asyncio
async def test_maybe_evict_empty_plan(tmp_path, monkeypatch):
    """_maybe_evict does nothing when plan has no models to evict."""
    from router.eviction import EvictionPlan

    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    registry = Registry(path=state_dir / "registry.json")
    registry.load()

    config = _make_config(tmp_path, auto_download_max_gb=100)
    downloader = AutoDownloader(config)

    plan = EvictionPlan(models_to_evict=[], bytes_to_free=0, reason="under cap")
    execute_calls = []
    monkeypatch.setattr(
        "router.eviction.plan_eviction",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        "router.eviction.execute_eviction",
        lambda p, md: execute_calls.append(p) or [],
    )

    model_dir = tmp_path / "models"
    await downloader._maybe_evict(registry, model_dir, 100)
    assert len(execute_calls) == 0


@pytest.mark.asyncio
async def test_maybe_evict_handles_exception(tmp_path, monkeypatch):
    """_maybe_evict catches exceptions from eviction logic."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    registry = Registry(path=state_dir / "registry.json")
    registry.load()

    config = _make_config(tmp_path, auto_download_max_gb=100)
    downloader = AutoDownloader(config)

    def exploding_plan(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "router.eviction.plan_eviction",
        exploding_plan,
    )

    model_dir = tmp_path / "models"
    await downloader._maybe_evict(registry, model_dir, 100)


@pytest.mark.asyncio
async def test_maybe_evict_execute_returns_empty(tmp_path, monkeypatch):
    """_maybe_evict handles execute_eviction returning empty list."""
    from router.eviction import EvictionPlan

    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    registry = Registry(path=state_dir / "registry.json")
    registry.load()

    config = _make_config(tmp_path, auto_download_max_gb=100)
    downloader = AutoDownloader(config)

    plan = EvictionPlan(
        models_to_evict=["gone-model"],
        bytes_to_free=1_000_000,
        reason="over cap",
    )
    monkeypatch.setattr(
        "router.eviction.plan_eviction",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        "router.eviction.execute_eviction",
        lambda p, md: [],
    )

    model_dir = tmp_path / "models"
    await downloader._maybe_evict(registry, model_dir, 100)


# ---------------------------------------------------------------------------
# stop() (line 249)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop(tmp_path):
    """stop() sets the internal stop event."""
    config = _make_config(tmp_path)
    downloader = AutoDownloader(config)
    assert not downloader._stop.is_set()
    await downloader.stop()
    assert downloader._stop.is_set()


# ---------------------------------------------------------------------------
# run_once: no max_gb set, skips eviction (line 207->159 branch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_no_max_gb_skips_evict(tmp_path, monkeypatch):
    """Successful download with max_gb=None does NOT call _maybe_evict."""
    state_dir = tmp_path / "state"
    _make_registry_file(state_dir, [SAMPLE_MODEL])
    config = _make_config(tmp_path, auto_download_max_gb=None)

    monkeypatch.setattr(
        "router.autodownload._get_local_models",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "router.autodownload.check_model_fit",
        lambda *args, **kwargs: FIT_OK,
    )
    monkeypatch.setattr(
        "router.autodownload._run_download",
        lambda repo, target: MagicMock(returncode=0),
    )

    evict_calls = []

    async def fake_maybe_evict(registry, model_dir, max_gb):
        evict_calls.append(True)

    downloader = AutoDownloader(config)
    monkeypatch.setattr(downloader, "_maybe_evict", fake_maybe_evict)
    await downloader.run_once()

    assert len(evict_calls) == 0
