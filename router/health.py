from __future__ import annotations

"""Cluster discovery and peer health monitoring."""

import asyncio
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from .config import RouterConfig
from .discovery import DiscoveredPeer, TailscaleDiscovery
from .router import ConsistentHashRouter, NodeInfo


class NodeHealthMonitor:
    """Maintain a live view of healthy PrivateNet peers."""

    def __init__(self, config: RouterConfig, router: ConsistentHashRouter, client: httpx.AsyncClient) -> None:
        self.config = config
        self.router = router
        self.client = client
        self.discovery = TailscaleDiscovery(config)
        self.started_at = time.time()
        self._stop = asyncio.Event()
        self._known_peers: dict[str, DiscoveredPeer] = {}
        self._last_nodes: dict[str, NodeInfo] = {}
        self._failures: dict[str, int] = {}

    async def run_forever(self) -> None:
        """Continuously refresh discovery and health state."""
        while not self._stop.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.discovery_interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        """Request the background health loop to exit."""
        self._stop.set()

    async def run_once(self) -> None:
        """Run one discovery and probe cycle."""
        discovered = await asyncio.to_thread(self.discovery.discover)
        self._known_peers = {peer.node_id: peer for peer in discovered}
        self._failures = {node_id: count for node_id, count in self._failures.items() if node_id in self._known_peers}
        self._last_nodes = {node_id: node for node_id, node in self._last_nodes.items() if node_id in self._known_peers}

        tasks = [self._probe_peer(peer) for peer in discovered]
        nodes = [node for node in await asyncio.gather(*tasks) if node is not None]
        self._last_nodes = {node.node_id: node for node in nodes}
        self.router.update_nodes(nodes)

    async def current_local_node_info(self) -> NodeInfo:
        """Return a freshly probed view of the local node."""
        peer = self._known_peers.get(self.config.local_node_id)
        if peer is None:
            discovered = await asyncio.to_thread(self.discovery.discover)
            for item in discovered:
                if item.local:
                    peer = item
                    self._known_peers[item.node_id] = item
                    break
        if peer is None:
            raise RuntimeError("Could not determine local Tailscale identity.")
        node = await self._probe_local_peer(peer)
        self._last_nodes[node.node_id] = node
        return node

    async def _probe_peer(self, peer: DiscoveredPeer) -> NodeInfo | None:
        try:
            node = await (self._probe_local_peer(peer) if peer.local else self._probe_remote_peer(peer))
        except Exception as exc:  # noqa: BLE001
            failures = self._failures.get(peer.node_id, 0) + 1
            self._failures[peer.node_id] = failures
            last = self._last_nodes.get(peer.node_id)
            if failures >= self.config.failure_threshold:
                return None
            if last is None:
                models = list(self.config.local_models) if peer.local else []
                max_concurrent = self.config.local_max_concurrent if peer.local else (self.config.overload_threshold or 1)
                return NodeInfo(
                    node_id=peer.node_id,
                    tailscale_ip=peer.tailscale_ip,
                    router_url=peer.router_url,
                    models=models,
                    in_flight=0,
                    max_concurrent=max_concurrent,
                    healthy=False,
                    uptime_seconds=int(time.time() - self.started_at),
                    local=peer.local,
                    consecutive_failures=failures,
                    last_error=str(exc),
                    online=peer.online,
                )
            return replace(
                last,
                healthy=False,
                consecutive_failures=failures,
                last_error=str(exc),
                online=peer.online,
            )

        self._failures[peer.node_id] = 0
        return replace(node, consecutive_failures=0, last_error=node.last_error, online=peer.online)

    async def _probe_local_peer(self, peer: DiscoveredPeer) -> NodeInfo:
        healthy = False
        available_models: list[str] = []
        try:
            health_task = self.client.get(
                f"{self.config.local_omlx_url}/health",
                timeout=self.config.health_check_timeout_seconds,
            )
            models_task = self.client.get(
                f"{self.config.local_omlx_url}/v1/models",
                headers=self._local_omlx_headers(),
                timeout=self.config.health_check_timeout_seconds,
            )
            health_response, models_response = await asyncio.gather(health_task, models_task)
            health_response.raise_for_status()
            models_response.raise_for_status()
            available_models = self._extract_available_models(models_response.json())
            healthy = True
        except Exception:  # noqa: BLE001
            healthy = False

        # Administratively disabled — report unhealthy so peers stop routing here
        disabled = self._is_node_disabled()
        if disabled:
            healthy = False

        node = self.router.get_node(peer.node_id)
        inflight = node.in_flight if node else 0
        last_error: str | None = None
        if disabled:
            last_error = "node administratively disabled"
        elif not healthy:
            last_error = "local oMLX did not respond in time"
        # Filter through advertise_models allowlist if configured
        models = available_models or list(self.config.local_models)
        if self.config.advertise_models is not None:
            allowed = set(self.config.advertise_models)
            models = [m for m in models if m in allowed]

        return NodeInfo(
            node_id=peer.node_id,
            tailscale_ip=peer.tailscale_ip,
            router_url=peer.router_url,
            models=models,
            in_flight=inflight,
            max_concurrent=self.config.local_max_concurrent,
            healthy=healthy,
            uptime_seconds=int(time.time() - self.started_at),
            local=True,
            online=True,
            last_error=last_error,
        )

    async def _probe_remote_peer(self, peer: DiscoveredPeer) -> NodeInfo:
        response = await self.client.get(
            f"{peer.router_url}/v1/node-info",
            timeout=self.config.health_check_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected node-info payload from {peer.node_id}")

        models = payload.get("models") or []
        if not isinstance(models, list):
            raise RuntimeError(f"Invalid models list from {peer.node_id}")

        return NodeInfo(
            node_id=str(payload.get("node_id") or peer.node_id),
            tailscale_ip=str(payload.get("tailscale_ip") or peer.tailscale_ip),
            router_url=peer.router_url,
            models=[str(model) for model in models],
            in_flight=max(0, int(payload.get("in_flight", 0))),
            max_concurrent=max(1, int(payload.get("max_concurrent", self.config.local_max_concurrent))),
            healthy=bool(payload.get("healthy", False)),
            uptime_seconds=max(0, int(payload.get("uptime_seconds", 0))),
            local=False,
            online=peer.online,
        )

    def _local_omlx_headers(self) -> dict[str, str]:
        if not self.config.local_omlx_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.local_omlx_api_key}"}

    @staticmethod
    def _is_node_disabled() -> bool:
        """Check whether this node has been administratively disabled."""
        state_dir = os.environ.get("OMLX_PRIVATENET_STATE_DIR", str(Path.home() / ".omlx-privatenet"))
        return Path(state_dir, "disabled").exists()

    @staticmethod
    def _extract_available_models(models_data: Any) -> list[str]:
        """Extract all available model IDs from oMLX /v1/models endpoint.

        oMLX loads models on demand, so all models reported by /v1/models are
        servable — not just ones currently resident in memory.
        """
        if isinstance(models_data, dict):
            items = models_data.get("data") or models_data.get("models") or []
        elif isinstance(models_data, list):
            items = models_data
        else:
            items = []

        results: list[str] = []
        for item in items:
            if isinstance(item, dict):
                model_id = item.get("id")
                if model_id:
                    results.append(str(model_id))
            elif isinstance(item, str) and item:
                results.append(item)
        return results
