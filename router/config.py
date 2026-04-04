from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".omlx-privatenet" / "router.json"
DEFAULT_LOCAL_MODELS = [
    "gemma-4-26b-a4b-it-4bit",
    "gemma-4-31b-it-4bit",
]


@dataclass(slots=True)
class RouterConfig:
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
    tailscale_tag: str = "tag:omlx-node"
    tailscale_bin: str = "tailscale"
    local_node_id: str = field(default_factory=lambda: socket.gethostname())
    local_tailscale_ip: str | None = None
    local_omlx_url: str = "http://127.0.0.1:5741"
    local_omlx_api_key: str | None = None
    local_models: list[str] = field(default_factory=lambda: list(DEFAULT_LOCAL_MODELS))
    local_max_concurrent: int = 8
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: Path | None = None) -> "RouterConfig":
        local_models = data.get("local_models") or list(DEFAULT_LOCAL_MODELS)
        if not isinstance(local_models, list) or not local_models:
            raise ValueError("`local_models` must be a non-empty JSON array.")

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
            tailscale_tag=str(data.get("tailscale_tag", "tag:omlx-node")),
            tailscale_bin=str(data.get("tailscale_bin", "tailscale")),
            local_node_id=str(data.get("local_node_id") or socket.gethostname()),
            local_tailscale_ip=(str(data["local_tailscale_ip"]) if data.get("local_tailscale_ip") else None),
            local_omlx_url=str(data.get("local_omlx_url", "http://127.0.0.1:5741")).rstrip("/"),
            local_omlx_api_key=(str(data["local_omlx_api_key"]) if data.get("local_omlx_api_key") else None),
            local_models=[str(model) for model in local_models],
            local_max_concurrent=max(1, int(data.get("local_max_concurrent", 8))),
            source_path=source_path,
        )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    env_path = os.getenv("OMLX_PRIVATENET_ROUTER_CONFIG") or os.getenv("OMLX_PRIVATENET_CONFIG")
    raw = config_path or env_path or DEFAULT_CONFIG_PATH
    return Path(raw).expanduser().resolve()


def load_config(config_path: str | Path | None = None) -> RouterConfig:
    path = resolve_config_path(config_path)
    if not path.exists():
        return RouterConfig(source_path=path)

    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Router config must be a JSON object.")
    return RouterConfig.from_dict(raw, source_path=path)
