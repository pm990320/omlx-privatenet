from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from router.config import load_config
from router.health import NodeHealthMonitor
from router.registry import Registry, RegistryModel
from router.router import ConsistentHashRouter


LOCAL_MODEL = "gemma-4-26b-a4b-it-4bit"
REMOTE_MODEL = "text-embedding-3-small"


@pytest.fixture(autouse=True)
def _isolate_state_dir(tmp_path, monkeypatch):
    """Prevent tests from reading the real ~/.omlx-privatenet/disabled file."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))


def make_transport_handler(remote_behavior):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://127.0.0.1:5741/health":
            return httpx.Response(200, json={"status": "healthy"})
        if url == "http://127.0.0.1:5741/v1/models":
            return httpx.Response(200, json={
                "object": "list",
                "data": [{"id": LOCAL_MODEL, "object": "model"}],
            })
        if url == "http://100.64.0.2:8741/v1/node-info":
            return remote_behavior(request)
        raise AssertionError(f"Unexpected request: {url}")

    return handler


@pytest.mark.asyncio
async def test_healthy_node_stays_in_routing(write_config, make_peer):
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(lambda request: httpx.Response(
        200,
        json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": [REMOTE_MODEL],
            "in_flight": 1,
            "max_concurrent": 4,
            "healthy": True,
            "uptime_seconds": 90,
        },
    ))))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
        make_peer("remote-node", tailscale_ip="100.64.0.2"),
    ]

    await monitor.run_once()

    assert router.get_node(config.local_node_id).healthy is True
    assert router.get_node("remote-node").healthy is True
    await client.aclose()


@pytest.mark.asyncio
async def test_unreachable_node_marked_unhealthy_after_timeout(write_config, make_peer):
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def remote_behavior(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(remote_behavior)))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
        make_peer("remote-node", tailscale_ip="100.64.0.2"),
    ]

    await monitor.run_once()

    remote = router.get_node("remote-node")
    assert remote is not None
    assert remote.healthy is False
    assert remote.consecutive_failures == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_three_consecutive_failures_remove_node_from_routing(write_config, make_peer):
    config = load_config(write_config({"failure_threshold": 3}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def remote_behavior(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(remote_behavior)))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
        make_peer("remote-node", tailscale_ip="100.64.0.2"),
    ]

    await monitor.run_once()
    assert router.get_node("remote-node") is not None
    await monitor.run_once()
    assert router.get_node("remote-node") is not None
    await monitor.run_once()

    assert router.get_node("remote-node") is None
    await client.aclose()


@pytest.mark.asyncio
async def test_recovered_node_is_added_back(write_config, make_peer):
    config = load_config(write_config({"failure_threshold": 2}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    state = {"healthy": False}

    def remote_behavior(request: httpx.Request) -> httpx.Response:
        if not state["healthy"]:
            raise httpx.ConnectTimeout("still down", request=request)
        return httpx.Response(
            200,
            json={
                "node_id": "remote-node",
                "tailscale_ip": "100.64.0.2",
                "models": [REMOTE_MODEL],
                "in_flight": 0,
                "max_concurrent": 6,
                "healthy": True,
                "uptime_seconds": 100,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(remote_behavior)))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
        make_peer("remote-node", tailscale_ip="100.64.0.2"),
    ]

    await monitor.run_once()
    await monitor.run_once()
    assert router.get_node("remote-node") is None

    state["healthy"] = True
    await monitor.run_once()

    recovered = router.get_node("remote-node")
    assert recovered is not None
    assert recovered.healthy is True
    assert recovered.models == [REMOTE_MODEL]
    await client.aclose()


@pytest.mark.asyncio
async def test_current_local_node_info_discovers_self_when_needed(write_config, make_peer):
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(lambda request: httpx.Response(
        200,
        json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": [REMOTE_MODEL],
            "in_flight": 0,
            "max_concurrent": 4,
            "healthy": True,
            "uptime_seconds": 90,
        },
    ))))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True)]

    node = await monitor.current_local_node_info()

    assert node.node_id == config.local_node_id
    assert node.local is True
    assert node.models == [LOCAL_MODEL]
    await client.aclose()


@pytest.mark.asyncio
async def test_remote_node_info_payload_is_parsed(write_config, make_peer):
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(lambda request: httpx.Response(
        200,
        json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": [REMOTE_MODEL],
            "in_flight": 3,
            "max_concurrent": 9,
            "healthy": True,
            "uptime_seconds": 777,
        },
    ))))
    monitor = NodeHealthMonitor(config, router, client)

    node = await monitor._probe_remote_peer(make_peer("remote-node", tailscale_ip="100.64.0.2"))

    assert node.node_id == "remote-node"
    assert node.models == [REMOTE_MODEL]
    assert node.in_flight == 3
    assert node.max_concurrent == 9
    assert node.uptime_seconds == 777
    await client.aclose()


@pytest.mark.asyncio
async def test_disabled_node_reports_unhealthy(write_config, make_peer, tmp_path, monkeypatch):
    """When the disabled file exists, local node should report unhealthy."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    (tmp_path / "disabled").write_text("disabled by test\n")

    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(lambda request: httpx.Response(
        200,
        json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": [REMOTE_MODEL],
            "in_flight": 0,
            "max_concurrent": 4,
            "healthy": True,
            "uptime_seconds": 90,
        },
    ))))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
    ]

    await monitor.run_once()

    local = router.get_node(config.local_node_id)
    assert local is not None
    assert local.healthy is False
    assert local.last_error == "node administratively disabled"
    await client.aclose()


@pytest.mark.asyncio
async def test_enabled_node_reports_healthy(write_config, make_peer, tmp_path, monkeypatch):
    """When the disabled file does not exist, local node should report healthy."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    # No disabled file — node is enabled

    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(lambda request: httpx.Response(
        200,
        json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": [REMOTE_MODEL],
            "in_flight": 0,
            "max_concurrent": 4,
            "healthy": True,
            "uptime_seconds": 90,
        },
    ))))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
    ]

    await monitor.run_once()

    local = router.get_node(config.local_node_id)
    assert local is not None
    assert local.healthy is True
    await client.aclose()


@pytest.mark.asyncio
async def test_registry_merge_from_remote_peer(write_config, make_peer, tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))

    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    # Seed local registry with one model
    local_registry = Registry(path=tmp_path / "registry.json")
    local_registry.add(RegistryModel(repo="mlx-community/local-model", id="local-model"))
    local_registry.save()

    remote_registry_payload = {
        "models": [
            {
                "repo": "mlx-community/remote-model",
                "id": "remote-model",
                "priority": 5,
                "added_by": "peer",
                "added_at": "2026-01-01T00:00:00+00:00",
                "safetensors_only": True,
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://127.0.0.1:5741/health":
            return httpx.Response(200, json={"status": "healthy"})
        if url == "http://127.0.0.1:5741/v1/models":
            return httpx.Response(200, json={
                "object": "list",
                "data": [{"id": LOCAL_MODEL, "object": "model"}],
            })
        if url == "http://100.64.0.2:8741/v1/node-info":
            return httpx.Response(200, json={
                "node_id": "remote-node",
                "tailscale_ip": "100.64.0.2",
                "models": [REMOTE_MODEL],
                "in_flight": 0,
                "max_concurrent": 4,
                "healthy": True,
                "uptime_seconds": 90,
            })
        if url == "http://100.64.0.2:8741/v1/registry":
            return httpx.Response(200, json=remote_registry_payload)
        raise AssertionError(f"Unexpected request: {url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
        make_peer("remote-node", tailscale_ip="100.64.0.2"),
    ]

    await monitor.run_once()

    # Reload registry from disk and verify merge happened
    result = Registry(path=tmp_path / "registry.json")
    result.load()
    model_ids = {m.id for m in result.models}
    assert "local-model" in model_ids
    assert "remote-model" in model_ids
    await client.aclose()


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_forever_loop_and_stop(write_config, make_peer):
    """run_forever should loop and exit when stopped."""
    config = load_config(write_config({"discovery_interval_seconds": 1}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(lambda request: httpx.Response(
        200,
        json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": [REMOTE_MODEL],
            "in_flight": 0,
            "max_concurrent": 4,
            "healthy": True,
            "uptime_seconds": 90,
        },
    ))))
    monitor = NodeHealthMonitor(config, router, client)
    monitor.discovery.discover = lambda: [
        make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True),
    ]

    run_count = 0
    original_run_once = monitor.run_once

    async def counting_run_once():
        nonlocal run_count
        run_count += 1
        await original_run_once()
        if run_count >= 2:
            await monitor.stop()

    monitor.run_once = counting_run_once
    await monitor.run_forever()

    assert run_count >= 2
    await client.aclose()


@pytest.mark.asyncio
async def test_probe_peer_failure_returns_none_after_threshold(write_config, make_peer):
    """When failures reach threshold, _probe_peer should return None."""
    config = load_config(write_config({"failure_threshold": 2}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def remote_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(remote_error)))
    monitor = NodeHealthMonitor(config, router, client)

    peer = make_peer("remote-node", tailscale_ip="100.64.0.2")

    # First failure: should return a node with healthy=False
    node = await monitor._probe_peer(peer)
    assert node is not None
    assert node.healthy is False
    assert node.consecutive_failures == 1

    # Second failure: should return None (threshold reached)
    node = await monitor._probe_peer(peer)
    assert node is None
    await client.aclose()


@pytest.mark.asyncio
async def test_probe_peer_failure_with_existing_last_node(write_config, make_peer):
    """When a peer fails but has a cached last_node, return updated copy."""
    config = load_config(write_config({"failure_threshold": 5}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def remote_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(remote_error)))
    monitor = NodeHealthMonitor(config, router, client)

    peer = make_peer("remote-node", tailscale_ip="100.64.0.2")

    # Manually populate _last_nodes to simulate a previously seen node
    from router.router import NodeInfo
    last_node = NodeInfo(
        node_id="remote-node",
        tailscale_ip="100.64.0.2",
        router_url="http://100.64.0.2:8741",
        models=[REMOTE_MODEL],
        in_flight=0,
        max_concurrent=4,
        healthy=True,
        uptime_seconds=90,
        local=False,
    )
    monitor._last_nodes["remote-node"] = last_node

    # Probe fails, but should return a modified copy of last_node
    node = await monitor._probe_peer(peer)
    assert node is not None
    assert node.healthy is False
    assert node.consecutive_failures == 1
    assert node.models == [REMOTE_MODEL]
    await client.aclose()


@pytest.mark.asyncio
async def test_probe_peer_failure_no_last_node_local_peer(write_config, make_peer):
    """When a local peer fails with no last_node, use config defaults."""
    config = load_config(write_config({"failure_threshold": 5}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(error_handler))
    monitor = NodeHealthMonitor(config, router, client)

    peer = make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True)

    node = await monitor._probe_peer(peer)
    assert node is not None
    assert node.healthy is False
    assert node.local is True
    assert node.models == list(config.local_models)
    assert node.max_concurrent == config.local_max_concurrent
    await client.aclose()


@pytest.mark.asyncio
async def test_remote_peer_non_dict_payload_raises(write_config, make_peer):
    """When remote node-info returns non-dict, should raise RuntimeError."""
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def bad_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(bad_response)))
    monitor = NodeHealthMonitor(config, router, client)

    peer = make_peer("remote-node", tailscale_ip="100.64.0.2")
    with pytest.raises(RuntimeError, match="Unexpected node-info payload"):
        await monitor._probe_remote_peer(peer)
    await client.aclose()


@pytest.mark.asyncio
async def test_remote_peer_non_list_models_raises(write_config, make_peer):
    """When remote node-info returns non-list models, should raise RuntimeError."""
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def bad_models(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": "not-a-list",
            "in_flight": 0,
            "max_concurrent": 4,
            "healthy": True,
            "uptime_seconds": 90,
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(bad_models)))
    monitor = NodeHealthMonitor(config, router, client)

    peer = make_peer("remote-node", tailscale_ip="100.64.0.2")
    with pytest.raises(RuntimeError, match="Invalid models list"):
        await monitor._probe_remote_peer(peer)
    await client.aclose()


@pytest.mark.asyncio
async def test_current_local_node_info_runtime_error_when_no_peer(write_config, make_peer):
    """When discovery can't find local peer, should raise RuntimeError."""
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    monitor = NodeHealthMonitor(config, router, client)
    # Return only remote peers, no local
    monitor.discovery.discover = lambda: [
        make_peer("remote-node", tailscale_ip="100.64.0.2"),
    ]

    with pytest.raises(RuntimeError, match="Could not determine local Tailscale identity"):
        await monitor.current_local_node_info()
    await client.aclose()


@pytest.mark.asyncio
async def test_local_probe_unhealthy_without_disabled(write_config, make_peer, tmp_path, monkeypatch):
    """When local oMLX is down but not disabled, last_error should mention oMLX."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(error_handler))
    monitor = NodeHealthMonitor(config, router, client)

    peer = make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True)
    node = await monitor._probe_local_peer(peer)

    assert node.healthy is False
    assert node.last_error == "local oMLX did not respond in time"
    await client.aclose()


@pytest.mark.asyncio
async def test_advertise_models_filter(write_config, make_peer, tmp_path, monkeypatch):
    """When advertise_models is set, only those models should be reported."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    config = load_config(write_config({"advertise_models": [LOCAL_MODEL]}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://127.0.0.1:5741/health":
            return httpx.Response(200, json={"status": "healthy"})
        if url == "http://127.0.0.1:5741/v1/models":
            return httpx.Response(200, json={
                "object": "list",
                "data": [
                    {"id": LOCAL_MODEL, "object": "model"},
                    {"id": "hidden-model", "object": "model"},
                ],
            })
        raise AssertionError(f"Unexpected: {url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monitor = NodeHealthMonitor(config, router, client)

    peer = make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True)
    node = await monitor._probe_local_peer(peer)

    assert LOCAL_MODEL in node.models
    assert "hidden-model" not in node.models
    await client.aclose()


@pytest.mark.asyncio
async def test_extract_available_models_various_formats(write_config, make_peer):
    """_extract_available_models should handle dict with data, dict with models, list, and other."""
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    monitor = NodeHealthMonitor(config, router, client)

    # dict with "data"
    assert monitor._extract_available_models({"data": [{"id": "m1"}, {"id": "m2"}]}) == ["m1", "m2"]

    # dict with "models"
    assert monitor._extract_available_models({"models": [{"id": "m3"}]}) == ["m3"]

    # plain list of strings
    assert monitor._extract_available_models(["model-a", "model-b"]) == ["model-a", "model-b"]

    # plain list of dicts
    assert monitor._extract_available_models([{"id": "m4"}, {"id": ""}]) == ["m4"]

    # non-standard type
    assert monitor._extract_available_models("not-a-list-or-dict") == []

    # empty string in list
    assert monitor._extract_available_models(["good", ""]) == ["good"]

    # list with non-dict non-string items
    assert monitor._extract_available_models([123, None]) == []

    await client.aclose()


@pytest.mark.asyncio
async def test_local_omlx_headers_no_api_key(write_config, make_peer):
    """When no API key is configured, headers should be empty."""
    config = load_config(write_config({"local_omlx_api_key": None}))
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    monitor = NodeHealthMonitor(config, router, client)

    assert monitor._local_omlx_headers() == {}
    await client.aclose()


@pytest.mark.asyncio
async def test_current_local_node_info_known_peer(write_config, make_peer):
    """When local peer is already in _known_peers, skip re-discovery."""
    config = load_config(write_config())
    router = ConsistentHashRouter(local_node_id=config.local_node_id)
    client = httpx.AsyncClient(transport=httpx.MockTransport(make_transport_handler(lambda request: httpx.Response(
        200,
        json={
            "node_id": "remote-node",
            "tailscale_ip": "100.64.0.2",
            "models": [REMOTE_MODEL],
            "in_flight": 0,
            "max_concurrent": 4,
            "healthy": True,
            "uptime_seconds": 90,
        },
    ))))
    monitor = NodeHealthMonitor(config, router, client)

    # Pre-populate _known_peers so discovery is skipped
    local_peer = make_peer(config.local_node_id, tailscale_ip="100.64.0.1", local=True)
    monitor._known_peers[config.local_node_id] = local_peer

    node = await monitor.current_local_node_info()
    assert node.node_id == config.local_node_id
    assert node.local is True
    await client.aclose()
