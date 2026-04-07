from __future__ import annotations

import pytest

from router.router import ConsistentHashRouter, NodeInfo


MODEL = "gemma-4-26b-a4b-it-4bit"


def build_router(nodes: list[NodeInfo], *, overload_threshold: int | None = None, prefer_local: bool = False) -> ConsistentHashRouter:
    router = ConsistentHashRouter(
        local_node_id="node-a",
        prefix_message_count=3,
        overload_threshold=overload_threshold,
        consistent_hash_replicas=64,
        prefer_local=prefer_local,
    )
    router.update_nodes(nodes)
    return router


def payload_for_primary(router: ConsistentHashRouter, target_node_id: str, *, model: str = MODEL) -> dict[str, object]:
    for index in range(10_000):
        payload = {"model": model, "session_id": f"session-{index}", "messages": [{"role": "user", "content": "hi"}]}
        if router.route_chat(payload).primary.node_id == target_node_id:
            return payload
    raise AssertionError(f"Could not find payload mapping to {target_node_id}")


def test_consistent_hash_ring_is_deterministic(make_node):
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True),
        make_node("node-b", tailscale_ip="100.64.0.2"),
        make_node("node-c", tailscale_ip="100.64.0.3"),
    ]
    router_a = build_router(nodes)
    router_b = build_router(nodes)
    payload = {"model": MODEL, "session_id": "abc123", "messages": [{"role": "user", "content": "hello"}]}

    decision_a = router_a.route_chat(payload)
    decision_b = router_b.route_chat(payload)

    assert decision_a.primary.node_id == decision_b.primary.node_id
    assert decision_a.selected.node_id == decision_b.selected.node_id
    assert decision_a.routing_key == decision_b.routing_key


def test_same_session_id_routes_to_same_node_across_router_instances(make_node):
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    payload = {"model": MODEL, "session_id": "sticky-user", "messages": [{"role": "user", "content": "hello"}]}

    selected = {build_router(nodes).route_chat(payload).selected.node_id for _ in range(5)}

    assert selected == {next(iter(selected))}


def test_failover_uses_next_ring_node_when_primary_is_unhealthy(make_node):
    healthy_b = make_node("node-b", tailscale_ip="100.64.0.2")
    healthy_c = make_node("node-c", tailscale_ip="100.64.0.3")
    unhealthy_a = make_node("node-a", tailscale_ip="100.64.0.1", local=True, healthy=False)
    router = build_router([unhealthy_a, healthy_b, healthy_c])

    payload = payload_for_primary(router, "node-a")
    decision = router.route_chat(payload)

    assert decision.primary.node_id == "node-a"
    assert decision.selected.node_id != "node-a"
    assert decision.selected.healthy is True


def test_failover_uses_next_ring_node_when_primary_is_overloaded(make_node):
    overloaded_a = make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=4, max_concurrent=4)
    healthy_b = make_node("node-b", tailscale_ip="100.64.0.2", in_flight=1, max_concurrent=4)
    healthy_c = make_node("node-c", tailscale_ip="100.64.0.3", in_flight=0, max_concurrent=4)
    router = build_router([overloaded_a, healthy_b, healthy_c])

    payload = payload_for_primary(router, "node-a")
    decision = router.route_chat(payload)

    assert decision.primary.node_id == "node-a"
    assert decision.selected.node_id in {"node-b", "node-c"}
    assert decision.selected.node_id != "node-a"


def test_all_overloaded_nodes_fall_back_to_least_loaded(make_node):
    node_a = make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=5, max_concurrent=4)
    node_b = make_node("node-b", tailscale_ip="100.64.0.2", in_flight=2, max_concurrent=2)
    node_c = make_node("node-c", tailscale_ip="100.64.0.3", in_flight=3, max_concurrent=3)
    router = build_router([node_a, node_b, node_c])

    payload = payload_for_primary(router, "node-a")
    decision = router.route_chat(payload)

    assert decision.primary.node_id == "node-a"
    assert decision.selected.node_id == "node-b"
    assert "least-loaded fallback" in decision.reason


def test_model_filtering_only_uses_nodes_with_requested_model(make_node):
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, models=[MODEL]),
        make_node("node-b", tailscale_ip="100.64.0.2", models=["text-embedding-3-small"]),
        make_node("node-c", tailscale_ip="100.64.0.3", models=[MODEL, "text-embedding-3-small"]),
    ]
    router = build_router(nodes)

    decision = router.route_chat({"model": MODEL, "session_id": "sticky", "messages": [{"role": "user", "content": "hi"}]})

    assert all(node.supports_model(MODEL) for node in decision.ordered_candidates)
    assert decision.selected.node_id in {"node-a", "node-c"}


def test_prefix_hashing_is_used_when_session_id_is_missing(make_node):
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router_a = build_router(nodes)
    router_b = build_router(nodes)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Summarize this."},
            {"role": "assistant", "content": "Sure."},
        ],
    }

    decision_a = router_a.route_chat(payload)
    decision_b = router_b.route_chat(payload)

    assert decision_a.affinity_kind == "prefix-hash"
    assert decision_a.prefix_hashes
    assert decision_a.routing_key == decision_b.routing_key
    assert decision_a.selected.node_id == decision_b.selected.node_id


def test_empty_cluster_raises_lookup_error(make_node):
    router = build_router([])

    with pytest.raises(LookupError, match="No nodes are configured"):
        router.route_chat({"model": MODEL, "session_id": "empty", "messages": [{"role": "user", "content": "hi"}]})


def test_single_node_cluster_always_routes_to_same_node(make_node):
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])

    for index in range(10):
        decision = router.route_chat(
            {"model": MODEL, "session_id": f"session-{index}", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert decision.selected.node_id == "node-a"
        assert decision.primary.node_id == "node-a"


def test_recovered_node_reenters_the_ring(make_node):
    healthy_a = make_node("node-a", tailscale_ip="100.64.0.1", local=True)
    healthy_b = make_node("node-b", tailscale_ip="100.64.0.2")
    router = build_router([healthy_a, healthy_b])
    payload = payload_for_primary(router, "node-a")

    router.mark_node_unhealthy("node-a", "timed out")
    failed_over = router.route_chat(payload)
    assert failed_over.selected.node_id == "node-b"

    recovered_a = make_node("node-a", tailscale_ip="100.64.0.1", local=True, healthy=True)
    router.update_nodes([recovered_a, healthy_b])
    recovered = router.route_chat(payload)

    assert recovered.primary.node_id == "node-a"
    assert recovered.selected.node_id == "node-a"


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_route_chat_missing_model_raises(make_node):
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    with pytest.raises(ValueError, match="`model` is required"):
        router.route_chat({"model": "", "messages": [{"role": "user", "content": "hi"}]})


def test_route_chat_anonymous_hash_when_no_session_no_prefix(make_node):
    """When there's no session_id and no messages, should use anonymous-hash."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router = build_router(nodes)
    payload = {"model": MODEL, "messages": []}

    decision = router.route_chat(payload)
    assert decision.affinity_kind == "anonymous-hash"
    assert "deterministic anonymous fallback" in decision.reason


def test_route_chat_no_healthy_nodes_raises(make_node):
    """When all nodes supporting the model are unhealthy, raise LookupError."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, healthy=False),
        make_node("node-b", tailscale_ip="100.64.0.2", healthy=False),
    ]
    router = build_router(nodes)
    payload = {"model": MODEL, "session_id": "sess", "messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(LookupError, match="No healthy nodes"):
        router.route_chat(payload)


def test_route_chat_all_overloaded_primary_is_least_loaded(make_node):
    """When all nodes are overloaded but primary is least-loaded, use it."""
    # Ensure primary happens to be least loaded
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=2, max_concurrent=2),
        make_node("node-b", tailscale_ip="100.64.0.2", in_flight=5, max_concurrent=2),
    ]
    router = build_router(nodes)
    payload = payload_for_primary(router, "node-a")
    decision = router.route_chat(payload)

    # node-a has lower load, so even when all overloaded it should be selected
    assert decision.selected.node_id == "node-a"
    assert "least-loaded primary" in decision.reason


def test_mark_node_unhealthy_nonexistent(make_node):
    """mark_node_unhealthy on a non-existent node should do nothing."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    router.mark_node_unhealthy("nonexistent", "error")
    # Should not raise


def test_bump_inflight_zero_delta(make_node):
    """bump_inflight with delta=0 should be a no-op."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    router.bump_inflight("node-a", 0)
    node = router.get_node("node-a")
    assert node.in_flight == 0


def test_bump_inflight_nonexistent_node(make_node):
    """bump_inflight on a non-existent node should do nothing."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    router.bump_inflight("nonexistent", 1)


def test_get_node_nonexistent(make_node):
    """get_node for a non-existent node should return None."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    assert router.get_node("nonexistent") is None


def test_get_node_with_inflight_delta(make_node):
    """get_node should include inflight adjustments."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=2)])
    router.bump_inflight("node-a", 3)
    node = router.get_node("node-a")
    assert node.in_flight == 5


def test_snapshot(make_node):
    """snapshot should return all nodes with overloaded field."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=0),
        make_node("node-b", tailscale_ip="100.64.0.2", in_flight=9, max_concurrent=8),
    ]
    router = build_router(nodes)
    snap = router.snapshot()

    assert "nodes" in snap
    assert len(snap["nodes"]) == 2
    node_dict = {n["node_id"]: n for n in snap["nodes"]}
    assert node_dict["node-a"]["overloaded"] is False
    assert node_dict["node-b"]["overloaded"] is True
    assert "online" in node_dict["node-a"]
    assert "consecutive_failures" in node_dict["node-a"]
    assert "last_error" in node_dict["node-a"]


def test_route_embeddings_missing_model_raises(make_node):
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    with pytest.raises(ValueError, match="`model` is required"):
        router.route_embeddings({"model": ""})


def test_route_embeddings_no_nodes_for_model(make_node):
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True, models=["other-model"])])
    with pytest.raises(LookupError, match="No nodes are configured"):
        router.route_embeddings({"model": MODEL})


def test_route_embeddings_no_healthy_nodes(make_node):
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True, healthy=False)]
    router = build_router(nodes)
    with pytest.raises(LookupError, match="No healthy nodes"):
        router.route_embeddings({"model": MODEL})


def test_route_embeddings_least_load(make_node):
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=5, max_concurrent=8),
        make_node("node-b", tailscale_ip="100.64.0.2", in_flight=1, max_concurrent=8),
    ]
    router = build_router(nodes)
    decision = router.route_embeddings({"model": MODEL})

    assert decision.selected.node_id == "node-b"
    assert decision.affinity_kind == "least-load"
    assert decision.session_id is None
    assert decision.prefix_hashes == []


def test_route_embeddings_all_overloaded_falls_back_to_healthy(make_node):
    """When all healthy nodes are overloaded, embeddings should still use least-loaded."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=10, max_concurrent=8),
        make_node("node-b", tailscale_ip="100.64.0.2", in_flight=9, max_concurrent=8),
    ]
    router = build_router(nodes)
    decision = router.route_embeddings({"model": MODEL})

    assert decision.selected.node_id == "node-b"


def test_effective_node_with_inflight_delta(make_node):
    """Inflight adjustments should be reflected in routing decisions."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=0),
        make_node("node-b", tailscale_ip="100.64.0.2", in_flight=0),
    ]
    router = build_router(nodes)
    router.bump_inflight("node-a", 5)
    node = router.get_node("node-a")
    assert node.in_flight == 5


def test_message_chunk_non_dict_message(make_node):
    """Non-dict messages should still produce stable prefix hashes."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": ["just a string", "another string"],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "prefix-hash"
    assert len(decision.prefix_hashes) == 2


def test_message_chunk_with_tool_calls(make_node):
    """Messages with tool_calls should produce stable hashes."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "test"}}]},
        ],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "prefix-hash"


def test_message_chunk_with_tool_call_id(make_node):
    """Messages with tool_call_id should produce stable hashes."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "prefix-hash"


def test_message_chunk_with_refusal(make_node):
    """Messages with refusal field should produce stable hashes."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "assistant", "content": None, "refusal": "I cannot help with that"},
        ],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "prefix-hash"


def test_normalize_content_list_with_types(make_node):
    """Content list with various types should normalize correctly."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                    {"type": "input_text", "text": "input"},
                    {"type": "input_image", "image": {"url": "http://example.com/img2.png"}},
                    {"type": "unknown_type", "data": "something"},
                    "just a string in list",
                ],
            },
        ],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "prefix-hash"


def test_normalize_content_non_string_non_list(make_node):
    """Content that is neither string nor list should pass through unchanged."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": 12345},
        ],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "prefix-hash"


def test_session_id_from_metadata(make_node):
    """Session ID should be extracted from metadata dict."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "metadata": {"session": "meta-session-123"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "session-hash"
    assert decision.session_id == "meta-session-123"


def test_session_id_from_user_field(make_node):
    """Session ID should be extracted from user field as fallback."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "user": "user-123",
        "messages": [{"role": "user", "content": "hi"}],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "session-hash"
    assert decision.session_id == "user-123"


def test_release_decrements_inflight(make_node):
    """release() should decrement the inflight adjustment."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    router.bump_inflight("node-a", 3)
    assert router.get_node("node-a").in_flight == 3
    router.release("node-a")
    assert router.get_node("node-a").in_flight == 2


def test_aggregate_models_healthy_only(make_node):
    """aggregate_models with healthy_only should filter unhealthy nodes."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, models=["model-a"]),
        make_node("node-b", tailscale_ip="100.64.0.2", healthy=False, models=["model-b"]),
    ]
    router = build_router(nodes)

    all_models = router.aggregate_models()
    assert "model-a" in all_models
    assert "model-b" in all_models

    healthy_models = router.aggregate_models(healthy_only=True)
    assert "model-a" in healthy_models
    assert "model-b" not in healthy_models


def test_route_chat_failover_reason(make_node):
    """When primary is unhealthy, reason should mention failover."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, healthy=False),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router = build_router(nodes)
    payload = payload_for_primary(router, "node-a")
    decision = router.route_chat(payload)

    assert "failed over" in decision.reason
    assert decision.selected.node_id == "node-b"


def test_bump_inflight_decrement_to_zero(make_node):
    """bump_inflight should remove tracking when delta goes to zero."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    router.bump_inflight("node-a", 2)
    assert router.get_node("node-a").in_flight == 2
    router.bump_inflight("node-a", -2)
    assert router.get_node("node-a").in_flight == 0


def test_bump_inflight_decrement_below_zero(make_node):
    """bump_inflight with large negative delta should clamp to zero."""
    router = build_router([make_node("node-a", tailscale_ip="100.64.0.1", local=True)])
    router.bump_inflight("node-a", 1)
    router.bump_inflight("node-a", -5)
    assert router.get_node("node-a").in_flight == 0


def test_session_id_from_metadata_empty_falls_to_user(make_node):
    """When metadata dict has no session keys, should fall through to user field."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "metadata": {"some_other_key": "value"},
        "user": "fallback-user",
        "messages": [{"role": "user", "content": "hi"}],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "session-hash"
    assert decision.session_id == "fallback-user"


def test_normalize_content_image_with_direct_url(make_node):
    """image_url with non-dict image_url should handle gracefully."""
    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "http://example.com/direct.png"},
                ],
            },
        ],
    }
    decision = router.route_chat(payload)
    assert decision.affinity_kind == "prefix-hash"


def test_build_prefix_hashes_empty_chunks_fallback(make_node):
    """When all message chunks are falsy, should return empty prefix hashes."""
    from unittest.mock import patch

    nodes = [make_node("node-a", tailscale_ip="100.64.0.1", local=True)]
    router = build_router(nodes)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    }

    with patch.object(router, "_message_chunk", return_value=""):
        decision = router.route_chat(payload)

    # With empty chunks, falls back to anonymous-hash
    assert decision.affinity_kind == "anonymous-hash"


def test_ordered_candidates_ring_exhaustion(make_node):
    """When the ring walk completes without break, return partial results."""
    from unittest.mock import patch

    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router = build_router(nodes)

    # Inject a ring cache that only contains entries for node-a
    # but supported has both node-a and node-b
    # This forces the for loop to exhaust without finding node-b
    def fake_ring_for_model(model, supported):
        # Only put node-a entries, so node-b can never be found
        return [(100, "node-a")]

    with patch.object(router, "_ring_for_model", side_effect=fake_ring_for_model):
        payload = {"model": MODEL, "session_id": "test", "messages": [{"role": "user", "content": "hi"}]}
        decision = router.route_chat(payload)

    assert decision.selected.node_id == "node-a"


def test_prefer_local_routes_to_local_node(make_node):
    """When prefer_local is on, local node is selected if healthy and has model."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router = build_router(nodes, prefer_local=True)
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

    decision = router.route_chat(payload)

    assert decision.selected.node_id == "node-a"
    assert decision.affinity_kind == "local-preferred"
    assert "prefer_local" in decision.reason


def test_prefer_local_falls_back_when_local_overloaded(make_node):
    """When local node is overloaded, fall back to hash routing."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, in_flight=10, max_concurrent=8),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router = build_router(nodes, prefer_local=True)
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

    decision = router.route_chat(payload)

    assert decision.selected.node_id == "node-b"
    assert decision.affinity_kind != "local-preferred"


def test_prefer_local_falls_back_when_local_unhealthy(make_node):
    """When local node is unhealthy, fall back to remote."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True, healthy=False),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router = build_router(nodes, prefer_local=True)
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

    decision = router.route_chat(payload)

    assert decision.selected.node_id == "node-b"


def test_prefer_local_off_uses_hash_routing(make_node):
    """When prefer_local is off, consistent hash determines the node."""
    nodes = [
        make_node("node-a", tailscale_ip="100.64.0.1", local=True),
        make_node("node-b", tailscale_ip="100.64.0.2"),
    ]
    router = build_router(nodes, prefer_local=False)
    payload = {"model": MODEL, "session_id": "test", "messages": [{"role": "user", "content": "hi"}]}

    decision = router.route_chat(payload)

    assert decision.affinity_kind == "session-hash"
