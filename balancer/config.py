from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CLUSTER_PATH = Path(__file__).resolve().parent.parent / "cluster.json"


@dataclass(slots=True)
class NodeConfig:
    tailscale_ip: str
    api_key: str
    models: list[str]
    port: int = 5741
    name: str | None = None
    max_inflight: int | None = None

    @property
    def node_id(self) -> str:
        return self.name or f"{self.tailscale_ip}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.tailscale_ip}:{self.port}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeConfig":
        models = data.get("models") or []
        if not isinstance(models, list) or not models:
            raise ValueError("Each node must declare at least one model in `models`.")
        return cls(
            tailscale_ip=str(data["tailscale_ip"]),
            port=int(data.get("port", 5741)),
            api_key=str(data["api_key"]),
            models=[str(model) for model in models],
            name=(str(data["name"]) if data.get("name") else None),
            max_inflight=(int(data["max_inflight"]) if data.get("max_inflight") is not None else None),
        )


@dataclass(slots=True)
class BalancerConfig:
    host: str = "0.0.0.0"
    port: int = 8741
    api_key: str | None = None
    health_interval_seconds: int = 30
    connect_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 600.0
    prefix_message_count: int = 3
    sticky_ttl_seconds: int = 12 * 60 * 60
    default_max_inflight: int = 2
    cluster_file: Path = field(default_factory=lambda: DEFAULT_CLUSTER_PATH)
    nodes: list[NodeConfig] = field(default_factory=list)
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_path: Path | None = None,
        cluster_nodes: list[NodeConfig] | None = None,
    ) -> "BalancerConfig":
        base_dir = source_path.parent if source_path else Path.cwd()
        cluster_file = data.get("cluster_file")
        if cluster_file:
            cluster_path = (base_dir / str(cluster_file)).expanduser().resolve()
        else:
            cluster_path = DEFAULT_CLUSTER_PATH.expanduser().resolve()

        inline_nodes = [NodeConfig.from_dict(item) for item in data.get("nodes", [])]
        nodes = cluster_nodes if cluster_nodes is not None else inline_nodes
        if not nodes:
            nodes = load_cluster_nodes(cluster_path)

        return cls(
            host=str(data.get("host", "0.0.0.0")),
            port=int(data.get("port", 8741)),
            api_key=(str(data["api_key"]) if data.get("api_key") else None),
            health_interval_seconds=int(data.get("health_interval_seconds", 30)),
            connect_timeout_seconds=float(data.get("connect_timeout_seconds", 10)),
            request_timeout_seconds=float(data.get("request_timeout_seconds", 600)),
            prefix_message_count=int(data.get("prefix_message_count", 3)),
            sticky_ttl_seconds=int(data.get("sticky_ttl_seconds", 12 * 60 * 60)),
            default_max_inflight=int(data.get("default_max_inflight", 2)),
            cluster_file=cluster_path,
            nodes=nodes,
            source_path=source_path,
        )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cluster_nodes(path: str | Path) -> list[NodeConfig]:
    cluster_path = Path(path).expanduser().resolve()
    if not cluster_path.exists():
        raise FileNotFoundError(
            f"Cluster file not found: {cluster_path}. Create cluster.json with your node list."
        )

    raw = _read_json(cluster_path)
    if isinstance(raw, dict):
        items = raw.get("nodes", [])
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("cluster.json must be either a JSON list or an object with a `nodes` array.")

    nodes = [NodeConfig.from_dict(item) for item in items]
    if not nodes:
        raise ValueError(f"Cluster file {cluster_path} contains no nodes.")
    return nodes


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    env_path = os.getenv("OMLX_PRIVATENET_CONFIG") or os.getenv("PRIVATENET_CONFIG")
    raw = config_path or env_path or DEFAULT_CONFIG_PATH
    return Path(raw).expanduser().resolve()


def load_config(config_path: str | Path | None = None) -> BalancerConfig:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Balancer config not found: {path}. Copy balancer/config.example.json to balancer/config.json first."
        )

    raw = _read_json(path)
    if isinstance(raw, list):
        return BalancerConfig(nodes=[NodeConfig.from_dict(item) for item in raw], source_path=path)
    if not isinstance(raw, dict):
        raise ValueError("Balancer config must be a JSON object or a JSON list of nodes.")

    cluster_nodes = None
    if raw.get("cluster_file"):
        cluster_nodes = load_cluster_nodes((path.parent / str(raw["cluster_file"])).expanduser().resolve())

    return BalancerConfig.from_dict(raw, source_path=path, cluster_nodes=cluster_nodes)
