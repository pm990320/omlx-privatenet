from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable

from .config import NodeConfig

SESSION_KEYS = (
    "session_id",
    "conversation_id",
    "thread_id",
    "chat_id",
)
METADATA_SESSION_KEYS = SESSION_KEYS + ("session", "conversation")


@dataclass(slots=True)
class NodeRuntime:
    healthy: bool = False
    inflight: int = 0
    active_requests: int = 0
    waiting_requests: int = 0
    loaded_models: set[str] = field(default_factory=set)
    last_checked_at: float = 0.0
    last_error: str | None = None
    last_latency_ms: float | None = None


@dataclass(slots=True)
class AffinityRecord:
    node_id: str
    last_used_at: float
    expires_at: float


@dataclass(slots=True)
class RouteDecision:
    selected: NodeConfig
    ordered_candidates: list[NodeConfig]
    routing_key: str
    affinity_kind: str
    session_id: str | None
    prefix_hashes: list[str]
    reason: str


class CacheAwareRouter:
    """Session-first, prefix-aware, load-bounded router for oMLX PrivateNet."""

    def __init__(
        self,
        nodes: Iterable[NodeConfig],
        *,
        prefix_message_count: int = 3,
        sticky_ttl_seconds: int = 12 * 60 * 60,
        default_max_inflight: int = 2,
    ) -> None:
        self.prefix_message_count = prefix_message_count
        self.sticky_ttl_seconds = sticky_ttl_seconds
        self.default_max_inflight = default_max_inflight
        self._lock = Lock()
        self._rr_counter = 0
        self._nodes: dict[str, NodeConfig] = {}
        self._runtime: dict[str, NodeRuntime] = {}
        self._session_affinity: dict[str, AffinityRecord] = {}
        self._prefix_affinity: dict[str, AffinityRecord] = {}
        self.refresh_nodes(nodes)

    def refresh_nodes(self, nodes: Iterable[NodeConfig]) -> None:
        with self._lock:
            refreshed: dict[str, NodeConfig] = {}
            runtime: dict[str, NodeRuntime] = {}
            for node in nodes:
                refreshed[node.node_id] = node
                runtime[node.node_id] = self._runtime.get(node.node_id, NodeRuntime())
            self._nodes = refreshed
            self._runtime = runtime
            self._purge_expired_locked()
            self._drop_missing_affinity_locked()

    def update_node_health(
        self,
        node_id: str,
        *,
        healthy: bool,
        active_requests: int = 0,
        waiting_requests: int = 0,
        loaded_models: Iterable[str] | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            runtime = self._runtime.setdefault(node_id, NodeRuntime())
            runtime.healthy = healthy
            runtime.active_requests = active_requests
            runtime.waiting_requests = waiting_requests
            runtime.loaded_models = set(loaded_models or [])
            runtime.last_checked_at = time.time()
            runtime.last_latency_ms = latency_ms
            runtime.last_error = error

    def mark_inflight(self, node_id: str, delta: int) -> None:
        with self._lock:
            runtime = self._runtime.setdefault(node_id, NodeRuntime())
            runtime.inflight = max(0, runtime.inflight + delta)

    def release(self, node_id: str) -> None:
        self.mark_inflight(node_id, -1)

    def route_chat(self, payload: dict[str, Any]) -> RouteDecision:
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValueError("`model` is required.")

        session_id = self._extract_session_id(payload)
        prefix_hashes = [self._hash_key(f"model:{model}:{item}") for item in self._build_prefix_hashes(payload)]
        routing_key, affinity_kind, preferred_id, reason = self._resolve_affinity(
            model=model,
            session_id=session_id,
            prefix_hashes=prefix_hashes,
        )
        decision = self._select_node(
            model=model,
            routing_key=routing_key,
            preferred_id=preferred_id,
            affinity_kind=affinity_kind,
            reason=reason,
        )
        self._record_affinity(
            model=model,
            session_id=session_id,
            prefix_hashes=prefix_hashes,
            node_id=decision.selected.node_id,
        )
        return RouteDecision(
            selected=decision.selected,
            ordered_candidates=decision.ordered_candidates,
            routing_key=routing_key,
            affinity_kind=affinity_kind,
            session_id=session_id,
            prefix_hashes=prefix_hashes,
            reason=decision.reason,
        )

    def route_embeddings(self, payload: dict[str, Any]) -> RouteDecision:
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValueError("`model` is required.")
        routing_key = f"embedding:{model}"
        decision = self._select_node(
            model=model,
            routing_key=routing_key,
            preferred_id=None,
            affinity_kind="least-load",
            reason="embedding requests use least-load healthy routing",
        )
        return RouteDecision(
            selected=decision.selected,
            ordered_candidates=decision.ordered_candidates,
            routing_key=routing_key,
            affinity_kind="least-load",
            session_id=None,
            prefix_hashes=[],
            reason=decision.reason,
        )

    def aggregate_models(self) -> list[str]:
        with self._lock:
            models = {model for node in self._nodes.values() for model in node.models}
        return sorted(models)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._purge_expired_locked()
            nodes = []
            for node_id, node in sorted(self._nodes.items()):
                runtime = self._runtime.get(node_id, NodeRuntime())
                nodes.append(
                    {
                        "node_id": node_id,
                        "tailscale_ip": node.tailscale_ip,
                        "port": node.port,
                        "models": list(node.models),
                        "healthy": runtime.healthy,
                        "inflight": runtime.inflight,
                        "active_requests": runtime.active_requests,
                        "waiting_requests": runtime.waiting_requests,
                        "loaded_models": sorted(runtime.loaded_models),
                        "last_checked_at": runtime.last_checked_at,
                        "last_latency_ms": runtime.last_latency_ms,
                        "last_error": runtime.last_error,
                        "max_inflight": self._node_max_inflight(node),
                    }
                )
            return {
                "nodes": nodes,
                "session_affinity_entries": len(self._session_affinity),
                "prefix_affinity_entries": len(self._prefix_affinity),
            }

    def _resolve_affinity(
        self,
        *,
        model: str,
        session_id: str | None,
        prefix_hashes: list[str],
    ) -> tuple[str, str, str | None, str]:
        with self._lock:
            self._purge_expired_locked()

            if session_id:
                session_lookup = f"{model}:{session_id}"
                session_key = self._hash_key(f"session:{session_lookup}")
                record = self._session_affinity.get(session_lookup)
                if record and record.node_id in self._nodes:
                    return session_key, "session-affinity", record.node_id, "existing session affinity"
                return session_key, "session-affinity", None, "session hash"

            for prefix_hash in reversed(prefix_hashes):
                record = self._prefix_affinity.get(prefix_hash)
                if record and record.node_id in self._nodes:
                    return prefix_hash, "prefix-affinity", record.node_id, "longest known prefix match"

            if prefix_hashes:
                return prefix_hashes[-1], "prefix-hash", None, "chunked prefix hash"

            fallback = self._hash_key(f"model:{model}:anonymous")
            return fallback, "model-hash", None, "anonymous fallback hash"

    def _record_affinity(
        self,
        *,
        model: str,
        session_id: str | None,
        prefix_hashes: list[str],
        node_id: str,
    ) -> None:
        now = time.time()
        expiry = now + self.sticky_ttl_seconds
        record = AffinityRecord(node_id=node_id, last_used_at=now, expires_at=expiry)
        with self._lock:
            if session_id:
                self._session_affinity[f"{model}:{session_id}"] = record
            for prefix_hash in prefix_hashes:
                self._prefix_affinity[prefix_hash] = record
            self._purge_expired_locked()

    def _select_node(
        self,
        *,
        model: str,
        routing_key: str,
        preferred_id: str | None,
        affinity_kind: str,
        reason: str,
    ) -> RouteDecision:
        with self._lock:
            self._purge_expired_locked()
            healthy_candidates = [
                node for node in self._nodes.values() if model in node.models and self._runtime.get(node.node_id, NodeRuntime()).healthy
            ]
            if not healthy_candidates:
                configured = [node for node in self._nodes.values() if model in node.models]
                if not configured:
                    raise LookupError(f"No nodes are configured for model `{model}`.")
                raise LookupError(f"No healthy nodes are currently available for model `{model}`.")

            loaded_candidates = [
                node
                for node in healthy_candidates
                if model in self._runtime.get(node.node_id, NodeRuntime()).loaded_models
            ]

            ordered = self._rank_candidates(routing_key, healthy_candidates)

            if preferred_id:
                preferred = next((node for node in healthy_candidates if node.node_id == preferred_id), None)
                if preferred and not self._is_overloaded(preferred):
                    order = [preferred] + [node for node in ordered if node.node_id != preferred.node_id]
                    return RouteDecision(preferred, order, routing_key, affinity_kind, None, [], f"{reason}; preferred node is healthy")

            if not preferred_id and len(loaded_candidates) == 1 and not self._is_overloaded(loaded_candidates[0]):
                loaded = loaded_candidates[0]
                order = [loaded] + [node for node in ordered if node.node_id != loaded.node_id]
                return RouteDecision(loaded, order, routing_key, affinity_kind, None, [], "single loaded-model owner")

            for node in ordered:
                if not self._is_overloaded(node):
                    return RouteDecision(node, ordered, routing_key, affinity_kind, None, [], f"{reason}; rendezvous hash with bounded load")

            fallback = self._least_loaded(ordered)
            ordered_fallback = [fallback] + [node for node in ordered if node.node_id != fallback.node_id]
            return RouteDecision(
                fallback,
                ordered_fallback,
                routing_key,
                affinity_kind,
                None,
                [],
                f"{reason}; all candidates overloaded, using least-loaded fallback",
            )

    def _rank_candidates(self, routing_key: str, candidates: list[NodeConfig]) -> list[NodeConfig]:
        scored = []
        for node in candidates:
            digest = hashlib.sha256(f"{routing_key}:{node.node_id}".encode("utf-8")).hexdigest()
            scored.append((digest, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored]

    def _rotate_for_fairness(self, ordered: list[NodeConfig]) -> list[NodeConfig]:
        if len(ordered) < 2:
            return ordered
        self._rr_counter += 1
        shift = self._rr_counter % len(ordered)
        return ordered[shift:] + ordered[:shift]

    def _least_loaded(self, candidates: list[NodeConfig]) -> NodeConfig:
        best = min(candidates, key=lambda node: (self._load(node), node.node_id))
        tied = [node for node in candidates if self._load(node) == self._load(best)]
        if len(tied) == 1:
            return best
        ordered = self._rotate_for_fairness(sorted(tied, key=lambda node: node.node_id))
        return ordered[0]

    def _is_overloaded(self, node: NodeConfig) -> bool:
        runtime = self._runtime.get(node.node_id, NodeRuntime())
        return runtime.inflight >= self._node_max_inflight(node)

    def _load(self, node: NodeConfig) -> int:
        runtime = self._runtime.get(node.node_id, NodeRuntime())
        return runtime.inflight + runtime.active_requests + runtime.waiting_requests

    def _node_max_inflight(self, node: NodeConfig) -> int:
        return node.max_inflight or self.default_max_inflight

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
        }
        content = message.get("content")
        normalized["content"] = self._normalize_content(content)
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
                    if item.get("type") == "text":
                        entry["text"] = item.get("text")
                    elif item.get("type") == "input_text":
                        entry["text"] = item.get("text")
                    elif item.get("type") in {"image_url", "input_image"}:
                        image_url = item.get("image_url") or item.get("image") or {}
                        if isinstance(image_url, dict):
                            entry["url"] = image_url.get("url")
                        else:
                            entry["url"] = image_url
                    else:
                        entry.update(item)
                    normalized.append(entry)
                else:
                    normalized.append(item)
            return normalized
        return content

    def _purge_expired_locked(self) -> None:
        now = time.time()
        self._session_affinity = {
            key: record for key, record in self._session_affinity.items() if record.expires_at > now
        }
        self._prefix_affinity = {
            key: record for key, record in self._prefix_affinity.items() if record.expires_at > now
        }

    def _drop_missing_affinity_locked(self) -> None:
        valid = set(self._nodes)
        self._session_affinity = {
            key: record for key, record in self._session_affinity.items() if record.node_id in valid
        }
        self._prefix_affinity = {
            key: record for key, record in self._prefix_affinity.items() if record.node_id in valid
        }

    @staticmethod
    def _hash_key(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
