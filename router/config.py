from __future__ import annotations

"""Configuration loading for the oMLX PrivateNet router."""

import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".omlx-privatenet" / "router.json"
DEFAULT_LOCAL_MODELS: tuple[str, ...] = ()
ENV_PREFIX = "OMLX_PRIVATENET_ROUTER_"


@dataclass(slots=True)
class RouterConfig:
    """Runtime configuration for a router instance."""

    host: str = "0.0.0.0"
    port: int = 8741
    api_key: str | None = None
    connect_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 600.0
    discovery_interval_seconds: int = 30
    health_check_timeout_seconds: float = 5.0
    failure_threshold: int = 3
    prefix_message_count: int = 3
    overload_threshold: int | None = None
    consistent_hash_replicas: int = 128
    prefer_local: bool = True
    tailscale_tag: str = "tag:omlx-node"
    tailscale_bin: str = "tailscale"
    local_node_id: str = field(default_factory=lambda: socket.gethostname())
    local_tailscale_ip: str | None = None
    local_omlx_url: str = "http://127.0.0.1:5741"
    local_omlx_api_key: str | None = None
    local_models: list[str] = field(default_factory=lambda: list(DEFAULT_LOCAL_MODELS))
    local_max_concurrent: int = 8
    advertise_models: list[str] | None = None  # None = all models, list = only these
    auto_download: bool = False
    auto_download_max_gb: int | None = None
    trusted_orgs: list[str] = field(default_factory=lambda: ["mlx-community"])
    auto_update: bool = False
    update_interval_hours: int = 6
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: Path | None = None) -> "RouterConfig":
        """Build a config object from decoded JSON-like data."""
        local_models = data.get("local_models")
        if local_models is None:
            local_models = list(DEFAULT_LOCAL_MODELS)
        if not isinstance(local_models, list):
            raise ValueError("`local_models` must be a JSON array.")

        overload_threshold = data.get("overload_threshold")
        return cls(
            host=str(data.get("host", "0.0.0.0")),
            port=int(data.get("port", 8741)),
            api_key=(str(data["api_key"]) if data.get("api_key") else None),
            connect_timeout_seconds=float(data.get("connect_timeout_seconds", 10.0)),
            request_timeout_seconds=float(data.get("request_timeout_seconds", 600.0)),
            discovery_interval_seconds=int(data.get("discovery_interval_seconds", 30)),
            health_check_timeout_seconds=float(data.get("health_check_timeout_seconds", 5.0)),
            failure_threshold=max(1, int(data.get("failure_threshold", 3))),
            prefix_message_count=max(1, int(data.get("prefix_message_count", 3))),
            overload_threshold=(int(overload_threshold) if overload_threshold is not None else None),
            consistent_hash_replicas=max(1, int(data.get("consistent_hash_replicas", 128))),
            prefer_local=_to_bool(data.get("prefer_local", True)),
            tailscale_tag=str(data.get("tailscale_tag", "tag:omlx-node")),
            tailscale_bin=str(data.get("tailscale_bin", "tailscale")),
            local_node_id=str(data.get("local_node_id") or socket.gethostname()),
            local_tailscale_ip=(str(data["local_tailscale_ip"]) if data.get("local_tailscale_ip") else None),
            local_omlx_url=str(data.get("local_omlx_url", "http://127.0.0.1:5741")).rstrip("/"),
            local_omlx_api_key=(str(data["local_omlx_api_key"]) if data.get("local_omlx_api_key") else None),
            local_models=[str(model) for model in local_models],
            local_max_concurrent=max(1, int(data.get("local_max_concurrent", 8))),
            advertise_models=([str(m) for m in data["advertise_models"]] if isinstance(data.get("advertise_models"), list) else None),
            auto_download=_to_bool(data.get("auto_download", False)),
            auto_download_max_gb=(_to_optional_int(data.get("auto_download_max_gb"))),
            trusted_orgs=([str(o) for o in data["trusted_orgs"]] if isinstance(data.get("trusted_orgs"), list) else ["mlx-community"]),
            auto_update=_to_bool(data.get("auto_update", False)),
            update_interval_hours=max(1, int(data.get("update_interval_hours", 6))),
            source_path=source_path,
        )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"", "none", "null"}:
            return None
        return int(value)
    return int(value)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(data)
    env_to_field = {
        f"{ENV_PREFIX}HOST": "host",
        f"{ENV_PREFIX}PORT": "port",
        f"{ENV_PREFIX}API_KEY": "api_key",
        f"{ENV_PREFIX}CONNECT_TIMEOUT_SECONDS": "connect_timeout_seconds",
        f"{ENV_PREFIX}REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
        f"{ENV_PREFIX}DISCOVERY_INTERVAL_SECONDS": "discovery_interval_seconds",
        f"{ENV_PREFIX}HEALTH_CHECK_TIMEOUT_SECONDS": "health_check_timeout_seconds",
        f"{ENV_PREFIX}FAILURE_THRESHOLD": "failure_threshold",
        f"{ENV_PREFIX}PREFIX_MESSAGE_COUNT": "prefix_message_count",
        f"{ENV_PREFIX}OVERLOAD_THRESHOLD": "overload_threshold",
        f"{ENV_PREFIX}CONSISTENT_HASH_REPLICAS": "consistent_hash_replicas",
        f"{ENV_PREFIX}TAILSCALE_TAG": "tailscale_tag",
        f"{ENV_PREFIX}TAILSCALE_BIN": "tailscale_bin",
        f"{ENV_PREFIX}LOCAL_NODE_ID": "local_node_id",
        f"{ENV_PREFIX}LOCAL_TAILSCALE_IP": "local_tailscale_ip",
        f"{ENV_PREFIX}LOCAL_OMLX_URL": "local_omlx_url",
        f"{ENV_PREFIX}LOCAL_OMLX_API_KEY": "local_omlx_api_key",
        f"{ENV_PREFIX}LOCAL_MODELS": "local_models",
        f"{ENV_PREFIX}LOCAL_MAX_CONCURRENT": "local_max_concurrent",
        f"{ENV_PREFIX}AUTO_DOWNLOAD": "auto_download",
        f"{ENV_PREFIX}AUTO_DOWNLOAD_MAX_GB": "auto_download_max_gb",
        f"{ENV_PREFIX}TRUSTED_ORGS": "trusted_orgs",
        f"{ENV_PREFIX}AUTO_UPDATE": "auto_update",
        f"{ENV_PREFIX}UPDATE_INTERVAL_HOURS": "update_interval_hours",
    }

    for env_name, field_name in env_to_field.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        if field_name in ("local_models", "trusted_orgs"):
            overrides[field_name] = [item.strip() for item in raw_value.split(",") if item.strip()]
        elif field_name in ("overload_threshold", "auto_download_max_gb"):
            value = raw_value.strip()
            overrides[field_name] = None if value.lower() in {"", "none", "null"} else int(value)
        elif field_name in ("auto_download", "auto_update"):
            overrides[field_name] = raw_value.lower() in {"true", "1", "yes"}
        else:
            overrides[field_name] = raw_value

    return overrides


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the effective config path, honoring legacy env overrides."""
    env_path = os.getenv("OMLX_PRIVATENET_ROUTER_CONFIG") or os.getenv("OMLX_PRIVATENET_CONFIG")
    raw = config_path or env_path or DEFAULT_CONFIG_PATH
    return Path(raw).expanduser().resolve()


def load_config(config_path: str | Path | None = None) -> RouterConfig:
    """Load config from disk plus supported environment overrides."""
    path = resolve_config_path(config_path)
    raw: dict[str, Any] = {}
    if path.exists():
        decoded = _read_json(path)
        if not isinstance(decoded, dict):
            raise ValueError("Router config must be a JSON object.")
        raw = decoded

    raw = _apply_env_overrides(raw)
    return RouterConfig.from_dict(raw, source_path=path)
