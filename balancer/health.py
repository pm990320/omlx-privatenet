from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .config import BalancerConfig, NodeConfig
from .router import CacheAwareRouter


class NodeHealthMonitor:
    def __init__(self, config: BalancerConfig, router: CacheAwareRouter, client: httpx.AsyncClient) -> None:
        self.config = config
        self.router = router
        self.client = client
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.health_interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> None:
        tasks = [self._probe(node) for node in self.config.nodes]
        await asyncio.gather(*tasks)

    async def _probe(self, node: NodeConfig) -> None:
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {node.api_key}"}
        try:
            health_task = self.client.get(f"{node.base_url}/health")
            status_task = self.client.get(f"{node.base_url}/v1/models/status", headers=headers)
            health_response, status_response = await asyncio.gather(health_task, status_task)
            health_response.raise_for_status()
            status_response.raise_for_status()

            health_data: dict[str, Any] = health_response.json()
            status_data = status_response.json()
            loaded_models = self._extract_loaded_models(health_data, status_data)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self.router.update_node_health(
                node.node_id,
                healthy=True,
                active_requests=int(health_data.get("active_requests", 0)),
                waiting_requests=int(health_data.get("waiting_requests", 0)),
                loaded_models=loaded_models,
                latency_ms=latency_ms,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self.router.update_node_health(
                node.node_id,
                healthy=False,
                active_requests=0,
                waiting_requests=0,
                loaded_models=[],
                latency_ms=latency_ms,
                error=str(exc),
            )

    @staticmethod
    def _extract_loaded_models(health_data: dict[str, Any], status_data: Any) -> list[str]:
        loaded = health_data.get("loaded_models")
        if isinstance(loaded, list):
            return [str(model) for model in loaded]

        if isinstance(status_data, dict):
            items = status_data.get("data") or status_data.get("models") or []
        elif isinstance(status_data, list):
            items = status_data
        else:
            items = []

        results: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("loaded"):
                model_id = item.get("id")
                if model_id:
                    results.append(str(model_id))
        return results
