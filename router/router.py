from __future__ import annotations

"""Deterministic request routing for PrivateNet peers."""

import bisect
import hashlib
import json
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Iterable

SESSION_KEYS = (
    "session_id",
    "conversation_id",
    "thread_id",
    "chat_id",
)
METADATA_SESSION_KEYS = SESSION_KEYS + ("session", "conversation")


@dataclass(slots=True)
class NodeInfo:
    """Live metadata describing one router peer."""

    node_id: str
    tailscale_ip: str
    router_url: str
    models: list[str]
    in_flight: int
    max_concurrent: int
    healthy: bool
    uptime_seconds: int
    local: bool = False
    consecutive_failures: int = 0
    last_error: str | None = None
    online: bool = True

    def supports_model(self, model: str) -> bool:
        """Return whether this node can serve ``model``."""
        return model in self.models


@dataclass(slots=True)
class RouteDecision:
    """The outcome of selecting an upstream node for a request."""

    selected: NodeInfo
    primary: NodeInfo
    ordered_candidates: list[NodeInfo]
    routing_key: str
    affinity_kind: str
    session_id: str | None
    prefix_hashes: list[str]
    reason: str


class ConsistentHashRouter:
    """Pick stable nodes for chat and embedding requests."""

    def __init__(
        self,
        *,
        local_node_id: str,
        prefix_message_count: int = 3,
        overload_threshold: int | None = None,
        consistent_hash_replicas: int = 128,
        prefer_local: bool = True,
    ) -> None:
        self.local_node_id = local_node_id
        self.prefix_message_count = prefix_message_count
        self.overload_threshold = overload_threshold
        self.consistent_hash_replicas = consistent_hash_replicas
        self.prefer_local = prefer_local
        self._lock = Lock()
        self._nodes: dict[str, NodeInfo] = {}
        self._inflight_adjustments: dict[str, int] = {}
        self._ring_cache: dict[tuple[str, tuple[str, ...]], list[tuple[int, str]]] = {}

    def update_nodes(self, nodes: Iterable[NodeInfo]) -> None:
        """Replace the router's current cluster view."""
        with self._lock:
            refreshed = {node.node_id: node for node in nodes}
            self._nodes = refreshed
            self._inflight_adjustments = {
                node_id: delta for node_id, delta in self._inflight_adjustments.items() if node_id in refreshed and delta > 0
            }
            self._ring_cache.clear()

    def mark_node_unhealthy(self, node_id: str, error: str | None = None) -> None:
        """Mark a node unhealthy without removing it from the cluster view."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return
            self._nodes[node_id] = replace(node, healthy=False, last_error=error)

    def bump_inflight(self, node_id: str, delta: int) -> None:
        """Apply an optimistic in-flight adjustment for a node."""
        if delta == 0:
            return
        with self._lock:
            if node_id not in self._nodes:
                return
            next_value = self._inflight_adjustments.get(node_id, 0) + delta
            if next_value <= 0:
                self._inflight_adjustments.pop(node_id, None)
            else:
                self._inflight_adjustments[node_id] = next_value

    def release(self, node_id: str) -> None:
        """Release one optimistic in-flight slot from ``node_id``."""
        self.bump_inflight(node_id, -1)

    def get_node(self, node_id: str) -> NodeInfo | None:
        """Return the effective state for one node, including in-flight deltas."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return None
            return self._effective_node(node)

    def aggregate_models(self, *, healthy_only: bool = False) -> list[str]:
        """Return the sorted union of models across the cluster."""
        with self._lock:
            nodes = [node for node in self._nodes.values() if node.healthy] if healthy_only else list(self._nodes.values())
            models = {model for node in nodes for model in node.models}
        return sorted(models)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current cluster."""
        with self._lock:
            nodes = [self._node_to_dict(self._effective_node(node)) for node in sorted(self._nodes.values(), key=lambda item: item.node_id)]
        return {"nodes": nodes}

    def route_chat(self, payload: dict[str, Any]) -> RouteDecision:
        """Route a chat completion payload using sticky affinity then failover."""
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValueError("`model` is required.")

        session_id = self._extract_session_id(payload)
        prefix_hashes = self._build_prefix_hashes(payload)
        if session_id:
            routing_key = self._hash_key(f"session:{model}:{session_id}")
            affinity_kind = "session-hash"
            base_reason = "consistent hash from session/conversation id"
        elif prefix_hashes:
            routing_key = prefix_hashes[-1]
            affinity_kind = "prefix-hash"
            base_reason = "consistent hash from the first messages"
        else:
            routing_key = self._hash_key(f"anonymous:{model}")
            affinity_kind = "anonymous-hash"
            base_reason = "deterministic anonymous fallback"

        ordered = self._ordered_candidates(model, routing_key)
        primary = ordered[0]
        healthy_candidates = [node for node in ordered if node.healthy]
        available = [node for node in healthy_candidates if not self._is_overloaded(node)]

        # Prefer local node if enabled, healthy, has the model, and not overloaded
        if self.prefer_local:
            local = next((n for n in available if n.node_id == self.local_node_id), None)
            if local is not None:
                selected = local
                reason = f"prefer_local; local node {local.node_id} has model and is available"
                failover_candidates = [node for node in ordered if node.healthy]
                return RouteDecision(
                    selected=selected,
                    primary=primary,
                    ordered_candidates=failover_candidates or ordered,
                    routing_key=routing_key,
                    affinity_kind="local-preferred",
                    session_id=session_id,
                    prefix_hashes=prefix_hashes,
                    reason=reason,
                )

        if available:
            selected = available[0]
            if selected.node_id == primary.node_id:
                reason = f"{base_reason}; primary node {primary.node_id} is available"
            else:
                reason = (
                    f"{base_reason}; failed over from {primary.node_id} to {selected.node_id} "
                    "because the primary was unhealthy or overloaded"
                )
        elif healthy_candidates:
            selected = min(healthy_candidates, key=lambda node: (self._load(node), node.node_id))
            if selected.node_id == primary.node_id:
                reason = f"{base_reason}; all healthy nodes were overloaded, using least-loaded primary"
            else:
                reason = (
                    f"{base_reason}; all healthy nodes were overloaded, "
                    f"using least-loaded fallback {selected.node_id} instead of {primary.node_id}"
                )
        else:
            raise LookupError(f"No healthy nodes are currently available for model `{model}`.")

        # Only include healthy candidates in the failover list so the proxy
        # never attempts to send requests to disabled/unhealthy nodes.
        failover_candidates = [node for node in ordered if node.healthy]
        return RouteDecision(
            selected=selected,
            primary=primary,
            ordered_candidates=failover_candidates or ordered,
            routing_key=routing_key,
            affinity_kind=affinity_kind,
            session_id=session_id,
            prefix_hashes=prefix_hashes,
            reason=reason,
        )

    def route_embeddings(self, payload: dict[str, Any]) -> RouteDecision:
        """Route embeddings requests by least load across healthy nodes."""
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValueError("`model` is required.")

        with self._lock:
            candidates = [self._effective_node(node) for node in self._nodes.values() if node.supports_model(model)]
        if not candidates:
            raise LookupError(f"No nodes are configured for model `{model}`.")

        healthy = [node for node in candidates if node.healthy]
        if not healthy:
            raise LookupError(f"No healthy nodes are currently available for model `{model}`.")

        available = [node for node in healthy if not self._is_overloaded(node)] or healthy
        selected = min(available, key=lambda node: (self._load(node), node.node_id))
        return RouteDecision(
            selected=selected,
            primary=selected,
            ordered_candidates=sorted(available, key=lambda node: (self._load(node), node.node_id)),
            routing_key=self._hash_key(f"embeddings:{model}"),
            affinity_kind="least-load",
            session_id=None,
            prefix_hashes=[],
            reason="healthy least-load routing for embeddings",
        )

    def _ordered_candidates(self, model: str, routing_key: str) -> list[NodeInfo]:
        with self._lock:
            supported = [self._effective_node(node) for node in self._nodes.values() if node.supports_model(model)]
            if not supported:
                raise LookupError(f"No nodes are configured for model `{model}`.")

            ring = self._ring_for_model(model, supported)
            point = int(routing_key, 16)
            idx = bisect.bisect_left(ring, (point, ""))
            ordered_ids: list[str] = []
            seen: set[str] = set()
            for offset in range(len(ring)):
                _, node_id = ring[(idx + offset) % len(ring)]
                if node_id in seen:
                    continue
                seen.add(node_id)
                ordered_ids.append(node_id)
                if len(ordered_ids) == len(supported):
                    break

            return [next(node for node in supported if node.node_id == node_id) for node_id in ordered_ids]

    def _ring_for_model(self, model: str, supported: list[NodeInfo]) -> list[tuple[int, str]]:
        cache_key = (model, tuple(sorted(node.node_id for node in supported)))
        cached = self._ring_cache.get(cache_key)
        if cached is not None:
            return cached

        ring: list[tuple[int, str]] = []
        for node in supported:
            for replica in range(self.consistent_hash_replicas):
                value = int(self._hash_key(f"ring:{model}:{node.node_id}:{replica}"), 16)
                ring.append((value, node.node_id))
        ring.sort()
        self._ring_cache[cache_key] = ring
        return ring

    def _effective_node(self, node: NodeInfo) -> NodeInfo:
        delta = self._inflight_adjustments.get(node.node_id, 0)
        if delta == 0:
            return node
        return replace(node, in_flight=max(0, node.in_flight + delta))

    def _is_overloaded(self, node: NodeInfo) -> bool:
        return node.in_flight >= self._overload_limit(node)

    def _overload_limit(self, node: NodeInfo) -> int:
        return max(1, self.overload_threshold or node.max_concurrent)

    @staticmethod
    def _load(node: NodeInfo) -> int:
        return node.in_flight

    def _extract_session_id(self, payload: dict[str, Any]) -> str | None:
        for key in SESSION_KEYS:
            value = payload.get(key)
            if value:
                return str(value)

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in METADATA_SESSION_KEYS:
                value = metadata.get(key)
                if value:
                    return str(value)

        user = payload.get("user")
        if isinstance(user, str) and user.strip():
            return user.strip()
        return None

    def _build_prefix_hashes(self, payload: dict[str, Any]) -> list[str]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return []

        prefix_messages = messages[: self.prefix_message_count]
        chunks = [self._message_chunk(message) for message in prefix_messages]
        chunks = [chunk for chunk in chunks if chunk]
        if not chunks:
            return []

        prefix_hashes: list[str] = []
        previous = "root"
        for chunk in chunks:
            previous = self._hash_key(f"{chunk}|{previous}")
            prefix_hashes.append(previous)
        return prefix_hashes

    def _message_chunk(self, message: Any) -> str:
        if not isinstance(message, dict):
            return self._hash_key(json.dumps(message, sort_keys=True, ensure_ascii=False))

        normalized: dict[str, Any] = {
            "role": message.get("role"),
            "name": message.get("name"),
            "content": self._normalize_content(message.get("content")),
        }
        if "tool_calls" in message:
            normalized["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            normalized["tool_call_id"] = message["tool_call_id"]
        if "refusal" in message:
            normalized["refusal"] = message["refusal"]
        raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return self._hash_key(raw)

    def _normalize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            normalized = []
            for item in content:
                if isinstance(item, dict):
                    entry = {"type": item.get("type")}
                    if item.get("type") in {"text", "input_text"}:
                        entry["text"] = item.get("text")
                    elif item.get("type") in {"image_url", "input_image"}:
                        image_url = item.get("image_url") or item.get("image") or {}
                        entry["url"] = image_url.get("url") if isinstance(image_url, dict) else image_url
                    else:
                        entry.update(item)
                    normalized.append(entry)
                else:
                    normalized.append(item)
            return normalized
        return content

    @staticmethod
    def _hash_key(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _node_to_dict(self, node: NodeInfo) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "tailscale_ip": node.tailscale_ip,
            "router_url": node.router_url,
            "models": list(node.models),
            "in_flight": node.in_flight,
            "max_concurrent": node.max_concurrent,
            "healthy": node.healthy,
            "uptime_seconds": node.uptime_seconds,
            "local": node.local,
            "online": node.online,
            "consecutive_failures": node.consecutive_failures,
            "last_error": node.last_error,
            "overloaded": self._is_overloaded(node),
        }
