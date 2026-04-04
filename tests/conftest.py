from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx
import pytest

from router.discovery import DiscoveredPeer
from router.health import NodeHealthMonitor
from router.router import NodeInfo
from router.server import create_app


class ChunkedAsyncStream(httpx.AsyncByteStream):
    """Small helper for mocking streamed SSE responses."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


@pytest.fixture
def make_node() -> Callable[..., NodeInfo]:
    def _make_node(
        node_id: str,
        *,
        tailscale_ip: str,
        models: list[str] | None = None,
        in_flight: int = 0,
        max_concurrent: int = 8,
        healthy: bool = True,
        uptime_seconds: int = 60,
        local: bool = False,
        online: bool = True,
        consecutive_failures: int = 0,
        last_error: str | None = None,
    ) -> NodeInfo:
        return NodeInfo(
            node_id=node_id,
            tailscale_ip=tailscale_ip,
            router_url=f"http://{tailscale_ip}:8741",
            models=models or ["gemma-4-26b-a4b-it-4bit"],
            in_flight=in_flight,
            max_concurrent=max_concurrent,
            healthy=healthy,
            uptime_seconds=uptime_seconds,
            local=local,
            online=online,
            consecutive_failures=consecutive_failures,
            last_error=last_error,
        )

    return _make_node


@pytest.fixture
def make_peer() -> Callable[..., DiscoveredPeer]:
    def _make_peer(
        node_id: str,
        *,
        tailscale_ip: str,
        local: bool = False,
        online: bool = True,
        host_name: str | None = None,
    ) -> DiscoveredPeer:
        return DiscoveredPeer(
            node_id=node_id,
            tailscale_ip=tailscale_ip,
            router_url=f"http://{tailscale_ip}:8741",
            host_name=host_name or node_id,
            online=online,
            local=local,
        )

    return _make_peer


@pytest.fixture
def tailscale_status_payload() -> dict[str, Any]:
    return {
        "Self": {
            "HostName": "local-node",
            "TailscaleIPs": ["100.64.0.1", "fd7a:115c:a1e0::1"],
        },
        "Peer": {
            "peer-1": {
                "HostName": "peer-a.tailnet.ts.net",
                "TailscaleIPs": ["100.64.0.2", "fd7a:115c:a1e0::2"],
                "Tags": ["tag:omlx-node"],
                "Online": True,
            },
            "peer-2": {
                "HostName": "peer-b.tailnet.ts.net",
                "TailscaleIPs": ["100.64.0.3"],
                "Tags": ["tag:other"],
                "Online": False,
            },
        },
    }


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict[str, Any] | None], Path]:
    def _write_config(overrides: dict[str, Any] | None = None) -> Path:
        payload = {
            "host": "127.0.0.1",
            "port": 8741,
            "api_key": None,
            "connect_timeout_seconds": 5,
            "request_timeout_seconds": 30,
            "discovery_interval_seconds": 30,
            "health_check_timeout_seconds": 5,
            "failure_threshold": 3,
            "prefix_message_count": 3,
            "overload_threshold": None,
            "consistent_hash_replicas": 32,
            "tailscale_tag": "tag:omlx-node",
            "tailscale_bin": "tailscale",
            "local_node_id": "local-node",
            "local_tailscale_ip": "100.64.0.1",
            "local_omlx_url": "http://127.0.0.1:5741",
            "local_omlx_api_key": "local-key",
            "local_models": ["gemma-4-26b-a4b-it-4bit", "text-embedding-3-small"],
            "local_max_concurrent": 8,
        }
        payload.update(overrides or {})
        path = tmp_path / "router.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    return _write_config


@pytest.fixture
def app_factory(
    monkeypatch: pytest.MonkeyPatch,
    write_config: Callable[[dict[str, Any] | None], Path],
    make_peer: Callable[..., DiscoveredPeer],
) -> Callable[..., Any]:
    async def _run_forever(self: NodeHealthMonitor) -> None:
        await self._stop.wait()

    monkeypatch.setattr(NodeHealthMonitor, "run_forever", _run_forever)

    def _factory(
        *,
        nodes: list[NodeInfo],
        local_node: NodeInfo | None = None,
        config_overrides: dict[str, Any] | None = None,
        outgoing_handler: Callable[[httpx.Request], httpx.Response] | None = None,
    ):
        path = write_config(config_overrides)
        effective_local_node = local_node or next((node for node in nodes if node.local), None)
        if effective_local_node is None:
            effective_local_node = NodeInfo(
                node_id="local-node",
                tailscale_ip="100.64.0.1",
                router_url="http://100.64.0.1:8741",
                models=["gemma-4-26b-a4b-it-4bit"],
                in_flight=0,
                max_concurrent=8,
                healthy=True,
                uptime_seconds=120,
                local=True,
            )

        async def _run_once(self: NodeHealthMonitor) -> None:
            self._known_peers = {
                node.node_id: make_peer(
                    node.node_id,
                    tailscale_ip=node.tailscale_ip,
                    local=node.local,
                    online=node.online,
                    host_name=node.node_id,
                )
                for node in nodes
            }
            self._last_nodes = {node.node_id: node for node in nodes}
            self.router.update_nodes(nodes)

        async def _current_local_node_info(self: NodeHealthMonitor) -> NodeInfo:
            return effective_local_node

        monkeypatch.setattr(NodeHealthMonitor, "run_once", _run_once)
        monkeypatch.setattr(NodeHealthMonitor, "current_local_node_info", _current_local_node_info)

        app = create_app(path)

        @asynccontextmanager
        async def _manager() -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
            async with app.router.lifespan_context(app):
                if outgoing_handler is not None:
                    await app.state.client.aclose()
                    upstream_client = httpx.AsyncClient(
                        transport=httpx.MockTransport(outgoing_handler),
                        follow_redirects=False,
                    )
                    app.state.client = upstream_client
                    app.state.monitor.client = upstream_client

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client:
                    yield app, client

        return _manager()

    return _factory
