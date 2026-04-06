from __future__ import annotations

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
