from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from router.eviction import (
    EvictionPlan,
    _dir_size_bytes,
    _get_loaded_models,
    _model_dir,
    _state_dir,
    _total_models_size_bytes,
    execute_eviction,
    load_pinned,
    plan_eviction,
    save_pinned,
)
from router.registry import Registry, RegistryModel


def _make_model_dir(model_dir: Path, model_id: str, size_bytes: int = 1_000_000) -> Path:
    """Create a fake model directory with a file of the given size."""
    d = model_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    # Write a file of the specified size
    filler = d / "model.safetensors"
    filler.write_bytes(b"\0" * size_bytes)
    return d


def _make_registry(tmp_path: Path, models: list[dict]) -> Registry:
    """Build a registry with the given model entries."""
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(models, indent=2), encoding="utf-8")
    registry = Registry(path=reg_path)
    registry.load()
    return registry


def _patch_loaded(loaded_ids: set[str]):
    """Patch _get_loaded_models to return the given set."""
    return patch("router.eviction._get_loaded_models", return_value=loaded_ids)


def _patch_pinned(pinned_ids: list[str]):
    """Patch load_pinned to return the given list."""
    return patch("router.eviction.load_pinned", return_value=pinned_ids)


# ---- Tests ----------------------------------------------------------------


def test_eviction_targets_unregistered_first(tmp_path: Path):
    """3 models on disk, 1 in registry -- unregistered evicted first."""
    model_dir = tmp_path / "models"
    # Each model is 500 MB
    size = 500_000_000
    _make_model_dir(model_dir, "registered-model", size)
    _make_model_dir(model_dir, "unregistered-a", size)
    _make_model_dir(model_dir, "unregistered-b", size)

    registry = _make_registry(tmp_path, [
        {"repo": "mlx-community/registered-model", "id": "registered-model", "priority": 5},
    ])

    # Cap at 1 GB -- total is 1.5 GB, need to free 0.5 GB
    with _patch_loaded(set()), _patch_pinned([]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=1.0,
            omlx_url="http://127.0.0.1:5741",
        )

    # Should evict one of the unregistered models (not the registered one)
    assert len(plan.models_to_evict) >= 1
    assert plan.models_to_evict[0] in ("unregistered-a", "unregistered-b")
    assert "registered-model" not in plan.models_to_evict


def test_eviction_respects_priority(tmp_path: Path):
    """2 registry models, lower priority (higher number) evicted first."""
    model_dir = tmp_path / "models"
    size = 500_000_000
    _make_model_dir(model_dir, "important-model", size)
    _make_model_dir(model_dir, "less-important-model", size)

    registry = _make_registry(tmp_path, [
        {"repo": "mlx-community/important-model", "id": "important-model", "priority": 1},
        {"repo": "mlx-community/less-important-model", "id": "less-important-model", "priority": 9},
    ])

    # Cap at 0.5 GB -- total is 1 GB, need to free 0.5 GB
    with _patch_loaded(set()), _patch_pinned([]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=0.5,
            omlx_url="http://127.0.0.1:5741",
        )

    # less-important-model (priority 9) should be evicted first
    assert plan.models_to_evict[0] == "less-important-model"


def test_eviction_protects_loaded_models(tmp_path: Path):
    """Mock oMLX showing model as loaded -- verify not evicted."""
    model_dir = tmp_path / "models"
    size = 500_000_000
    _make_model_dir(model_dir, "loaded-model", size)
    _make_model_dir(model_dir, "idle-model", size)

    registry = _make_registry(tmp_path, [])

    # Cap at 0.5 GB -- total is 1 GB, need to free 0.5 GB
    with _patch_loaded({"loaded-model"}), _patch_pinned([]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=0.5,
            omlx_url="http://127.0.0.1:5741",
        )

    assert "loaded-model" not in plan.models_to_evict
    assert "idle-model" in plan.models_to_evict


def test_eviction_protects_pinned(tmp_path: Path):
    """Add model to pinned.json -- verify not evicted."""
    model_dir = tmp_path / "models"
    size = 500_000_000
    _make_model_dir(model_dir, "pinned-model", size)
    _make_model_dir(model_dir, "unpinned-model", size)

    registry = _make_registry(tmp_path, [])

    # Cap at 0.5 GB -- total is 1 GB
    with _patch_loaded(set()), _patch_pinned(["pinned-model"]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=0.5,
            omlx_url="http://127.0.0.1:5741",
        )

    assert "pinned-model" not in plan.models_to_evict
    assert "unpinned-model" in plan.models_to_evict


def test_eviction_protects_advertised(tmp_path: Path):
    """Model in advertise_models list -- verify not evicted."""
    model_dir = tmp_path / "models"
    size = 500_000_000
    _make_model_dir(model_dir, "advertised-model", size)
    _make_model_dir(model_dir, "other-model", size)

    registry = _make_registry(tmp_path, [])

    with _patch_loaded(set()), _patch_pinned([]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=0.5,
            advertise_models=["advertised-model"],
            omlx_url="http://127.0.0.1:5741",
        )

    assert "advertised-model" not in plan.models_to_evict
    assert "other-model" in plan.models_to_evict


def test_dry_run_doesnt_delete(tmp_path: Path):
    """Verify files still exist after dry_run."""
    model_dir = tmp_path / "models"
    size = 500_000_000
    _make_model_dir(model_dir, "model-a", size)
    _make_model_dir(model_dir, "model-b", size)

    plan = EvictionPlan(
        models_to_evict=["model-a", "model-b"],
        bytes_to_free=size * 2,
        reason="test",
    )

    deleted = execute_eviction(plan, model_dir, dry_run=True)

    # Models should be reported as "deleted" but still exist on disk
    assert "model-a" in deleted
    assert "model-b" in deleted
    assert (model_dir / "model-a").exists()
    assert (model_dir / "model-b").exists()


def test_execute_eviction_deletes(tmp_path: Path):
    """Verify files are actually deleted when dry_run=False."""
    model_dir = tmp_path / "models"
    size = 100_000
    _make_model_dir(model_dir, "model-a", size)

    plan = EvictionPlan(
        models_to_evict=["model-a"],
        bytes_to_free=size,
        reason="test",
    )

    deleted = execute_eviction(plan, model_dir, dry_run=False)

    assert "model-a" in deleted
    assert not (model_dir / "model-a").exists()


def test_pinned_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test save_pinned / load_pinned roundtrip."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    save_pinned(["model-b", "model-a"])
    result = load_pinned()
    assert result == ["model-a", "model-b"]  # sorted + deduplicated


def test_no_eviction_when_under_cap(tmp_path: Path):
    """No eviction needed when under the cap."""
    model_dir = tmp_path / "models"
    _make_model_dir(model_dir, "small-model", 1000)

    registry = _make_registry(tmp_path, [])

    with _patch_loaded(set()), _patch_pinned([]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=1.0,
            omlx_url="http://127.0.0.1:5741",
        )

    assert plan.models_to_evict == []
    assert "Under cap" in plan.reason


def test_no_eviction_when_no_cap(tmp_path: Path):
    """No eviction when max_gb is None."""
    model_dir = tmp_path / "models"
    _make_model_dir(model_dir, "model-a", 500_000_000)

    registry = _make_registry(tmp_path, [])

    plan = plan_eviction(
        model_dir=model_dir,
        registry=registry,
        max_gb=None,
        omlx_url="http://127.0.0.1:5741",
    )

    assert plan.models_to_evict == []
    assert "No GB cap" in plan.reason


# ---------------------------------------------------------------------------
# _model_dir / _state_dir defaults (lines 26-29, 36)
# ---------------------------------------------------------------------------


def test_model_dir_with_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """_model_dir returns path from env when set."""
    monkeypatch.setenv("OMLX_MODELS_DIR", str(tmp_path / "custom"))
    result = _model_dir()
    assert result == (tmp_path / "custom").resolve()


def test_model_dir_default(monkeypatch: pytest.MonkeyPatch):
    """_model_dir returns default when env var is unset."""
    monkeypatch.delenv("OMLX_MODELS_DIR", raising=False)
    result = _model_dir()
    assert result == Path.home() / ".omlx" / "models"


def test_state_dir_default(monkeypatch: pytest.MonkeyPatch):
    """_state_dir returns default when env var is unset."""
    monkeypatch.delenv("OMLX_PRIVATENET_STATE_DIR", raising=False)
    result = _state_dir()
    assert result == Path.home() / ".omlx-privatenet"


# ---------------------------------------------------------------------------
# load_pinned edge cases (lines 47, 53-55)
# ---------------------------------------------------------------------------


def test_load_pinned_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_pinned returns [] when pinned.json doesn't exist."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path / "empty"))
    result = load_pinned()
    assert result == []


def test_load_pinned_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_pinned returns [] on malformed JSON."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    pinned_path = tmp_path / "pinned.json"
    pinned_path.write_text("{bad json", encoding="utf-8")
    result = load_pinned()
    assert result == []


def test_load_pinned_non_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_pinned returns [] when JSON is not a list."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    pinned_path = tmp_path / "pinned.json"
    pinned_path.write_text('{"not": "a list"}', encoding="utf-8")
    result = load_pinned()
    assert result == []


# ---------------------------------------------------------------------------
# _get_loaded_models (lines 72-99)
# ---------------------------------------------------------------------------


def test_get_loaded_models_success(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models parses loaded models from {data: [...]}."""
    body = json.dumps({
        "data": [
            {"id": "loaded-a", "loaded": True},
            {"id": "idle-b", "loaded": False},
            {"model": "loaded-c", "loaded": True},
        ]
    }).encode()

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
    result = _get_loaded_models("http://fake:5741")
    assert "loaded-a" in result
    assert "idle-b" not in result
    assert "loaded-c" in result


def test_get_loaded_models_with_api_key(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models sends auth header."""
    body = json.dumps({"data": []}).encode()
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
    _get_loaded_models("http://fake:5741", omlx_api_key="key123")
    assert captured_reqs[0].get_header("Authorization") == "Bearer key123"


def test_get_loaded_models_url_error(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models returns empty set on connection error."""
    import urllib.error

    def fail(*a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    result = _get_loaded_models("http://fake:5741")
    assert result == set()


def test_get_loaded_models_list_response(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models handles plain list response."""
    body = json.dumps([
        {"id": "m1", "loaded": True},
        {"id": "m2", "loaded": False},
    ]).encode()

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
    result = _get_loaded_models("http://fake:5741")
    assert "m1" in result
    assert "m2" not in result


def test_get_loaded_models_models_key(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models handles {models: [...]} key."""
    body = json.dumps({
        "models": [{"id": "mx", "loaded": True}]
    }).encode()

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
    result = _get_loaded_models("http://fake:5741")
    assert "mx" in result


def test_get_loaded_models_unexpected_type(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models returns empty set for non-dict/non-list response."""
    body = json.dumps("just a string").encode()

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
    result = _get_loaded_models("http://fake:5741")
    assert result == set()


def test_get_loaded_models_empty_items(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models returns empty set when data list is empty."""
    body = json.dumps({"data": []}).encode()

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
    result = _get_loaded_models("http://fake:5741")
    assert result == set()


def test_get_loaded_models_non_dict_items(monkeypatch: pytest.MonkeyPatch):
    """_get_loaded_models skips non-dict items in list."""
    body = json.dumps({"data": ["not-a-dict", 42, {"id": "real", "loaded": True}]}).encode()

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
    result = _get_loaded_models("http://fake:5741")
    assert result == {"real"}


# ---------------------------------------------------------------------------
# _dir_size_bytes / _total_models_size_bytes (lines 105->109, 107->106, 115, 118->117)
# ---------------------------------------------------------------------------


def test_dir_size_bytes_empty_dir(tmp_path: Path):
    """_dir_size_bytes returns 0 for empty directory."""
    d = tmp_path / "empty"
    d.mkdir()
    assert _dir_size_bytes(d) == 0


def test_dir_size_bytes_with_files(tmp_path: Path):
    """_dir_size_bytes sums file sizes recursively."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "a.bin").write_bytes(b"\0" * 100)
    (d / "b.bin").write_bytes(b"\0" * 200)
    assert _dir_size_bytes(d) == 300


def test_dir_size_bytes_not_a_dir(tmp_path: Path):
    """_dir_size_bytes returns 0 for a non-directory path."""
    f = tmp_path / "file.txt"
    f.write_text("hello")
    assert _dir_size_bytes(f) == 0


def test_dir_size_bytes_nested_subdir(tmp_path: Path):
    """_dir_size_bytes handles nested subdirs (rglob yields non-file entries)."""
    d = tmp_path / "model"
    nested = d / "subdir"
    nested.mkdir(parents=True)
    (nested / "data.bin").write_bytes(b"\0" * 150)
    assert _dir_size_bytes(d) == 150


def test_total_models_size_bytes_nonexistent(tmp_path: Path):
    """_total_models_size_bytes returns 0 when dir doesn't exist."""
    assert _total_models_size_bytes(tmp_path / "nope") == 0


def test_total_models_size_bytes_with_subdirs(tmp_path: Path):
    """_total_models_size_bytes sums all model subdirectories."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    m1 = model_dir / "model-1"
    m1.mkdir()
    (m1 / "w.bin").write_bytes(b"\0" * 500)
    m2 = model_dir / "model-2"
    m2.mkdir()
    (m2 / "w.bin").write_bytes(b"\0" * 300)
    # Also put a plain file (not a dir) -- should be skipped
    (model_dir / "README.txt").write_text("hi")
    assert _total_models_size_bytes(model_dir) == 800


def test_total_models_size_bytes_no_dirs(tmp_path: Path):
    """_total_models_size_bytes returns 0 when dir has only files (no subdirs)."""
    model_dir = tmp_path / "models_flat"
    model_dir.mkdir()
    (model_dir / "stray.txt").write_text("data")
    assert _total_models_size_bytes(model_dir) == 0


def test_total_models_size_bytes_empty(tmp_path: Path):
    """_total_models_size_bytes returns 0 for empty directory."""
    model_dir = tmp_path / "models_empty"
    model_dir.mkdir()
    assert _total_models_size_bytes(model_dir) == 0


# ---------------------------------------------------------------------------
# plan_eviction: skips non-dir entries, file-only in model_dir (line 178)
# ---------------------------------------------------------------------------


def test_plan_eviction_skips_files_in_model_dir(tmp_path: Path):
    """plan_eviction ignores plain files in model_dir."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # Create a non-directory entry
    (model_dir / "README.txt").write_bytes(b"\0" * 500_000_000)
    # Create a real model dir to trigger overage
    _make_model_dir(model_dir, "evictable", 500_000_000)

    registry = _make_registry(tmp_path, [])

    with _patch_loaded(set()), _patch_pinned([]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=0.0001,  # very small cap
            omlx_url="http://127.0.0.1:5741",
        )

    # Only the directory should be considered for eviction
    assert "evictable" in plan.models_to_evict
    assert "README.txt" not in plan.models_to_evict


# ---------------------------------------------------------------------------
# execute_eviction: target not found (lines 218-219)
# ---------------------------------------------------------------------------


def test_execute_eviction_target_not_found(tmp_path: Path):
    """execute_eviction skips models whose dirs don't exist."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    plan = EvictionPlan(
        models_to_evict=["nonexistent-model"],
        bytes_to_free=100,
        reason="test",
    )

    deleted = execute_eviction(plan, model_dir)
    assert deleted == []


# ---------------------------------------------------------------------------
# execute_eviction: OSError on rmtree (lines 228-229)
# ---------------------------------------------------------------------------


def test_execute_eviction_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """execute_eviction logs error when rmtree fails."""
    model_dir = tmp_path / "models"
    _make_model_dir(model_dir, "stubborn-model", 1000)

    plan = EvictionPlan(
        models_to_evict=["stubborn-model"],
        bytes_to_free=1000,
        reason="test",
    )

    def failing_rmtree(path, **kw):
        raise OSError("permission denied")

    monkeypatch.setattr("shutil.rmtree", failing_rmtree)
    deleted = execute_eviction(plan, model_dir)
    assert deleted == []


# ---------------------------------------------------------------------------
# plan_eviction: model_dir doesn't exist (line 175->190)
# ---------------------------------------------------------------------------


def test_plan_eviction_model_dir_nonexistent(tmp_path: Path):
    """plan_eviction handles non-existent model_dir gracefully."""
    model_dir = tmp_path / "nonexistent"
    registry = _make_registry(tmp_path, [])

    # Since dir doesn't exist, _total_models_size_bytes returns 0.
    with _patch_loaded(set()), _patch_pinned([]):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=1.0,
            omlx_url="http://127.0.0.1:5741",
        )
    assert plan.models_to_evict == []


def test_plan_eviction_model_dir_nonexistent_over_cap(tmp_path: Path):
    """plan_eviction with nonexistent model_dir but forced overage via mock."""
    model_dir = tmp_path / "ghost"
    registry = _make_registry(tmp_path, [])

    # Force _total_models_size_bytes to return a high value even though dir
    # doesn't exist, so we exercise the model_dir.exists() False branch
    # inside the eviction candidate scanning code.
    with (
        _patch_loaded(set()),
        _patch_pinned([]),
        patch("router.eviction._total_models_size_bytes", return_value=10 * (1024 ** 3)),
    ):
        plan = plan_eviction(
            model_dir=model_dir,
            registry=registry,
            max_gb=1.0,
            omlx_url="http://127.0.0.1:5741",
        )
    # No candidates because dir doesn't exist
    assert plan.models_to_evict == []
