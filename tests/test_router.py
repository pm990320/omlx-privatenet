from __future__ import annotations

import pytest

from router.router import ConsistentHashRouter, NodeInfo


MODEL = "gemma-4-26b-a4b-it-4bit"


def build_router(nodes: list[NodeInfo], *, overload_threshold: int | None = None) -> ConsistentHashRouter:
    router = ConsistentHashRouter(
        local_node_id="node-a",
        prefix_message_count=3,
        overload_threshold=overload_threshold,
        consistent_hash_replicas=64,
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
