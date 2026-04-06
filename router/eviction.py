from __future__ import annotations

"""Priority-based model eviction for oMLX PrivateNet.

When disk usage exceeds the configured GB cap, this module plans and executes
eviction of models based on priority ordering.  Protected models (loaded,
pinned, or in the advertise allowlist) are never evicted.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import Registry

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


def _pinned_path() -> Path:
    return _state_dir() / "pinned.json"


def load_pinned() -> list[str]:
    """Load the list of pinned model IDs from pinned.json."""
    path = _pinned_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [str(item) for item in data]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read pinned.json: %s", exc)
    return []


def save_pinned(model_ids: list[str]) -> None:
    """Write the list of pinned model IDs to pinned.json."""
    path = _pinned_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(sorted(set(model_ids)), fh, indent=2)
        fh.write("\n")


def _get_loaded_models(
    omlx_url: str = "http://127.0.0.1:5741",
    omlx_api_key: Optional[str] = None,
) -> set[str]:
    """Query oMLX /v1/models/status to find currently loaded models."""
    import urllib.request
    import urllib.error

    url = f"{omlx_url.rstrip('/')}/v1/models/status"
    req = urllib.request.Request(url)
    if omlx_api_key:
        req.add_header("Authorization", f"Bearer {omlx_api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("Could not query oMLX model status: %s", exc)
        return set()

    loaded: set[str] = set()
    items = []
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        items = data

    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("model")
            is_loaded = item.get("loaded", False)
            if model_id and is_loaded:
                loaded.add(str(model_id))
    return loaded


def _dir_size_bytes(path: Path) -> int:
    """Calculate the total size of a directory in bytes."""
    total = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def _total_models_size_bytes(model_dir: Path) -> int:
    """Sum size of all model directories."""
    if not model_dir.exists():
        return 0
    total = 0
    for entry in model_dir.iterdir():
        if entry.is_dir():
            total += _dir_size_bytes(entry)
    return total


@dataclass
class EvictionPlan:
    """Describes which models should be evicted and why."""

    models_to_evict: list[str] = field(default_factory=list)  # model IDs to delete
    bytes_to_free: int = 0
    reason: str = ""


def plan_eviction(
    model_dir: Path,
    registry: Registry,
    max_gb: float | None,
    advertise_models: list[str] | None = None,
    omlx_url: str = "http://127.0.0.1:5741",
    omlx_api_key: str | None = None,
) -> EvictionPlan:
    """Plan which models to evict to stay under the GB cap.

    Eviction order (first to go):
    1. Models NOT in the registry (unregistered)
    2. Registry models with highest priority number (least important)

    Protected (never evicted):
    - Models currently loaded in oMLX
    - Models in the advertise_models allowlist
    - Models marked as pinned in pinned.json
    """
    if max_gb is None:
        return EvictionPlan(reason="No GB cap configured")

    max_bytes = int(max_gb * (1024 ** 3))
    current_bytes = _total_models_size_bytes(model_dir)

    if current_bytes <= max_bytes:
        return EvictionPlan(
            reason=f"Under cap: {current_bytes / (1024**3):.1f} GB / {max_gb} GB"
        )

    overage = current_bytes - max_bytes

    # Build protected set
    loaded = _get_loaded_models(omlx_url, omlx_api_key)
    pinned = set(load_pinned())
    advertised = set(advertise_models) if advertise_models else set()
    protected = loaded | pinned | advertised

    # Build registry lookup
    registry_map = {m.id: m for m in registry.models}

    # Enumerate on-disk models with their sizes
    candidates: list[tuple[str, int, bool, int]] = []  # (model_id, size, in_registry, priority)
    if model_dir.exists():
        for entry in model_dir.iterdir():
            if not entry.is_dir():
                continue
            model_id = entry.name
            if model_id in protected:
                continue
            size = _dir_size_bytes(entry)
            in_registry = model_id in registry_map
            priority = registry_map[model_id].priority if in_registry else 0
            candidates.append((model_id, size, in_registry, priority))

    # Sort: unregistered first (in_registry=False sorts before True when negated),
    # then by priority DESC (highest number = least important = evict first),
    # then alphabetically by model_id for stability
    candidates.sort(key=lambda c: (c[2], -c[3], c[0]))

    # Select models to evict until we free enough space
    to_evict: list[str] = []
    freed = 0
    for model_id, size, _in_reg, _pri in candidates:
        if freed >= overage:
            break
        to_evict.append(model_id)
        freed += size

    return EvictionPlan(
        models_to_evict=to_evict,
        bytes_to_free=freed,
        reason=f"Over cap by {overage / (1024**3):.1f} GB, plan frees {freed / (1024**3):.1f} GB",
    )


def execute_eviction(
    plan: EvictionPlan,
    model_dir: Path,
    dry_run: bool = False,
) -> list[str]:
    """Delete model directories. Returns list of deleted model IDs."""
    deleted: list[str] = []
    for model_id in plan.models_to_evict:
        target = model_dir / model_id
        if not target.exists():
            logger.warning("Eviction target not found: %s", target)
            continue
        if dry_run:
            logger.info("Dry run: would delete %s", target)
            deleted.append(model_id)
            continue
        try:
            shutil.rmtree(target)
            logger.info("Evicted model: %s", model_id)
            deleted.append(model_id)
        except OSError as exc:
            logger.error("Failed to evict %s: %s", model_id, exc)
    return deleted
