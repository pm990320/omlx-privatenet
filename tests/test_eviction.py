from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from router.eviction import (
    EvictionPlan,
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
