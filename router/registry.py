from __future__ import annotations

"""Model registry for oMLX PrivateNet.

Tracks which models the network wants available and which nodes volunteered
to host them.  The registry file is a simple JSON list that lives alongside
the router config.
"""

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

TRUSTED_ORGS: frozenset[str] = frozenset({"mlx-community"})
MAX_REGISTRY_MODELS: int = 50

_STATE_DIR_ENV = "OMLX_PRIVATENET_STATE_DIR"
_DEFAULT_STATE_DIR = Path.home() / ".omlx-privatenet"


def _state_dir() -> Path:
    env = os.getenv(_STATE_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_STATE_DIR


DEFAULT_REGISTRY_PATH = _state_dir() / "registry.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegistryModel:
    """A single model entry in the registry."""

    repo: str
    id: str
    priority: int = 5
    added_by: str = ""
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    safetensors_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryModel:
        return cls(
            repo=str(data["repo"]),
            id=str(data["id"]),
            priority=int(data.get("priority", 5)),
            added_by=str(data.get("added_by", "")),
            added_at=str(data.get("added_at", datetime.now(timezone.utc).isoformat())),
            safetensors_only=bool(data.get("safetensors_only", True)),
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_model_id(model_id: str) -> None:
    """Reject IDs with path-traversal characters or other unsafe content."""
    if not model_id:
        raise ValueError("Model ID must not be empty.")
    if "\x00" in model_id:
        raise ValueError("Model ID must not contain null bytes.")
    if ".." in model_id:
        raise ValueError("Model ID must not contain '..'.")
    if "/" in model_id or "\\" in model_id:
        raise ValueError("Model ID must not contain '/' or '\\'.")
    if not _ID_PATTERN.match(model_id):
        raise ValueError(
            f"Model ID contains invalid characters: {model_id!r}. "
            "Only alphanumeric, hyphens, underscores, and dots are allowed."
        )


def _validate_repo_org(repo: str, trusted_orgs: frozenset[str]) -> None:
    """Ensure the repo belongs to a trusted organisation."""
    parts = repo.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Repo must be in 'org/name' format, got: {repo!r}")
    org = parts[0]
    if org not in trusted_orgs:
        raise ValueError(
            f"Repo org {org!r} is not in the trusted allowlist. "
            f"Allowed: {sorted(trusted_orgs)}"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    """In-memory model registry backed by a JSON file."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        trusted_orgs: frozenset[str] | None = None,
        max_models: int = MAX_REGISTRY_MODELS,
    ) -> None:
        self.path: Path = (path or DEFAULT_REGISTRY_PATH).expanduser().resolve()
        self.trusted_orgs: frozenset[str] = trusted_orgs if trusted_orgs is not None else TRUSTED_ORGS
        self.max_models: int = max_models
        self._models: dict[str, RegistryModel] = {}

    # -- accessors ----------------------------------------------------------

    @property
    def models(self) -> list[RegistryModel]:
        return list(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    def get(self, model_id: str) -> RegistryModel | None:
        return self._models.get(model_id)

    # -- mutators -----------------------------------------------------------

    def add(self, model: RegistryModel) -> None:
        """Add a model after validation.  Raises ``ValueError`` on bad input."""
        _validate_model_id(model.id)
        _validate_repo_org(model.repo, self.trusted_orgs)
        if model.id not in self._models and len(self._models) >= self.max_models:
            raise ValueError(
                f"Registry is at capacity ({self.max_models} models). "
                "Remove a model before adding a new one."
            )
        self._models[model.id] = model

    def remove(self, model_id: str) -> bool:
        """Remove a model by ID.  Returns True if it existed."""
        return self._models.pop(model_id, None) is not None

    def merge(self, other: Registry) -> None:
        """Merge *other* into this registry (union, latest-wins by added_at)."""
        for model in other.models:
            existing = self._models.get(model.id)
            if existing is None or model.added_at > existing.added_at:
                self._models[model.id] = model

    # -- persistence --------------------------------------------------------

    def load(self) -> None:
        """Load models from the JSON file.  No-op if file is missing."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Registry file must contain a JSON array.")
        self._models = {}
        for entry in data:
            model = RegistryModel.from_dict(entry)
            self._models[model.id] = model

    def save(self) -> None:
        """Atomically write the registry to disk (write-tmp + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            [m.to_dict() for m in self._models.values()],
            indent=2,
            sort_keys=True,
        ) + "\n"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".registry-",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_path, str(self.path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
