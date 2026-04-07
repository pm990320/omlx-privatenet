from __future__ import annotations

import json

import httpx
import pytest

from router.registry import Registry, RegistryModel
from router.router import ConsistentHashRouter


CHAT_MODEL = "gemma-4-26b-a4b-it-4bit"
EMBED_MODEL = "text-embedding-3-small"


@pytest.fixture(autouse=True)
def _isolate_state_dir(tmp_path, monkeypatch):
    """Prevent tests from reading the real ~/.omlx-privatenet/disabled file."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))


class MockAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_get_models_returns_omlx_compatible_format(app_factory, make_node):
    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL]),
        make_node("remote-node", tailscale_ip="100.64.0.2", models=[EMBED_MODEL], healthy=False),
    ]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"] == [
        {
            "id": CHAT_MODEL,
            "object": "model",
            "created": payload["data"][0]["created"],
            "owned_by": "omlx-privatenet",
        }
    ]


@pytest.mark.asyncio
async def test_get_node_info_returns_local_metadata(app_factory, make_node):
    local_node = make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL], in_flight=2)

    async with app_factory(nodes=[local_node], local_node=local_node) as (_, client):
        response = await client.get("/v1/node-info")

    assert response.status_code == 200
    assert response.json() == {
        "node_id": "local-node",
        "tailscale_ip": "100.64.0.1",
        "models": [CHAT_MODEL],
        "in_flight": 2,
        "max_concurrent": 8,
        "healthy": True,
        "uptime_seconds": 60,
    }


@pytest.mark.asyncio
async def test_chat_completions_proxy_to_selected_node(app_factory, make_node):
    captured: dict[str, object] = {}

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"id": "chatcmpl-1", "object": "chat.completion", "choices": []})

    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL]),
        make_node("remote-node", tailscale_ip="100.64.0.2", models=[CHAT_MODEL]),
    ]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer shared-key"},
            json={"model": CHAT_MODEL, "session_id": "sticky", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.headers["x-omlx-privatenet-node"] == "remote-node"
    assert captured["url"] == "http://100.64.0.2:8741/v1/chat/completions"
    assert captured["headers"]["x-omlx-router-local-only"] == "1"
    assert captured["headers"]["x-omlx-routed-by"] == "local-node"
    # Client auth must NOT be forwarded to peers (security: prevents credential leakage)
    assert "authorization" not in captured["headers"]


@pytest.mark.asyncio
async def test_embeddings_proxy_to_selected_node(app_factory, make_node):
    captured: dict[str, object] = {}

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"object": "list", "data": [{"embedding": [0.1, 0.2], "index": 0}]})

    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL]),
        make_node("remote-node", tailscale_ip="100.64.0.2", models=[EMBED_MODEL], in_flight=0),
    ]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post(
            "/v1/embeddings",
            json={"model": EMBED_MODEL, "input": "hello"},
        )

    assert response.status_code == 200
    assert response.headers["x-omlx-privatenet-node"] == "remote-node"
    assert captured["url"] == "http://100.64.0.2:8741/v1/embeddings"


@pytest.mark.asyncio
async def test_streaming_sse_passthrough_works(app_factory, make_node):
    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncStream([b"data: {\"delta\":\"hello\"}\n\ndata: [DONE]\n\n"]),
        )

    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL]),
        make_node("remote-node", tailscale_ip="100.64.0.2", models=[CHAT_MODEL]),
    ]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": CHAT_MODEL, "session_id": "stream", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body == b"data: {\"delta\":\"hello\"}\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
async def test_no_healthy_nodes_returns_503(app_factory, make_node):
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL], healthy=False)]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": CHAT_MODEL, "session_id": "none", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 503
    assert "No healthy nodes" in response.json()["detail"]


@pytest.mark.asyncio
async def test_local_only_requests_proxy_to_local_omlx(app_factory, make_node):
    captured: dict[str, object] = {}

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"id": "local-chat", "choices": []})

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            headers={"x-omlx-router-local-only": "1"},
            json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.headers["x-omlx-privatenet-node"] == "local-node"
    assert captured["url"] == "http://127.0.0.1:5741/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer local-key"


@pytest.mark.asyncio
async def test_router_api_key_is_enforced(app_factory, make_node):
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, config_overrides={"api_key": "router-key"}) as (_, client):
        unauthenticated = await client.get("/v1/models")
        authenticated = await client.get("/v1/models", headers={"Authorization": "Bearer router-key"})
        health = await client.get("/health")

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert health.status_code == 200


@pytest.mark.asyncio
async def test_invalid_json_body_returns_400(app_factory, make_node):
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            content="not-json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert "Invalid JSON body" in response.json()["detail"]


@pytest.mark.asyncio
async def test_remote_500_fails_over_to_next_candidate(app_factory, make_node):
    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL]),
        make_node("remote-a", tailscale_ip="100.64.0.2", models=[CHAT_MODEL]),
        make_node("remote-b", tailscale_ip="100.64.0.3", models=[CHAT_MODEL]),
    ]
    router = ConsistentHashRouter(local_node_id="local-node", consistent_hash_replicas=32)
    router.update_nodes(nodes)
    payload = None
    for index in range(10_000):
        candidate = {"model": CHAT_MODEL, "session_id": f"retry-{index}", "messages": [{"role": "user", "content": "hi"}]}
        if router.route_chat(candidate).primary.node_id == "remote-a":
            payload = candidate
            break
    assert payload is not None

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://100.64.0.2:8741/v1/chat/completions":
            return httpx.Response(500, json={"error": "boom"})
        if str(request.url) == "http://100.64.0.3:8741/v1/chat/completions":
            return httpx.Response(200, json={"id": "fallback", "choices": []})
        raise AssertionError(f"Unexpected request: {request.url}")

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["x-omlx-privatenet-node"] == "remote-b"


@pytest.mark.asyncio
async def test_health_endpoint_reports_cluster(app_factory, make_node):
    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL]),
        make_node("remote-node", tailscale_ip="100.64.0.2", models=[EMBED_MODEL], healthy=False),
    ]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.get("/health")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["router"]["node_id"] == "local-node"
    assert {node["node_id"] for node in payload["cluster"]} == {"local-node", "remote-node"}


@pytest.mark.asyncio
async def test_health_reports_disabled_when_node_disabled(app_factory, make_node, tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    (tmp_path / "disabled").write_text("disabled by test\n")

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.get("/health")

    assert response.json()["status"] == "disabled"


@pytest.mark.asyncio
async def test_node_info_reports_disabled_when_node_disabled(app_factory, make_node, tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    (tmp_path / "disabled").write_text("disabled by test\n")

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.get("/v1/node-info")

    payload = response.json()
    assert payload["healthy"] is False
    assert payload["disabled"] is True


@pytest.mark.asyncio
async def test_get_registry_returns_local_models(app_factory, make_node, tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    registry = Registry(path=tmp_path / "registry.json")
    registry.add(RegistryModel(repo="mlx-community/gemma-4", id="gemma-4-26b"))
    registry.save()

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.get("/v1/registry")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 1
    assert payload["models"][0]["id"] == "gemma-4-26b"
    assert payload["models"][0]["repo"] == "mlx-community/gemma-4"


@pytest.mark.asyncio
async def test_registry_endpoint_is_unauthenticated(app_factory, make_node, tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, config_overrides={"api_key": "router-key"}) as (_, client):
        # No Authorization header — should still succeed
        response = await client.get("/v1/registry")

    assert response.status_code == 200
    assert "models" in response.json()


# ---------------------------------------------------------------------------
# Auto-download lifespan (lines 76-81, 97, 105, 107-111)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_with_auto_download(app_factory, make_node, monkeypatch):
    """Auto-download tasks are created and cleaned up during lifespan."""
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, config_overrides={"auto_download": True}) as (app, client):
        assert hasattr(app.state, "auto_downloader")
        assert hasattr(app.state, "auto_download_task")
        assert app.state.auto_downloader is not None
        assert app.state.auto_download_task is not None
    # After context exit, cleanup happened (no error)


# ---------------------------------------------------------------------------
# Auto-update lifespan (lines 86-91, 99-103)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_with_auto_update(app_factory, make_node, monkeypatch):
    """Auto-update tasks are created and cleaned up during lifespan."""
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, config_overrides={"auto_update": True}) as (app, client):
        assert hasattr(app.state, "auto_updater")
        assert hasattr(app.state, "auto_update_task")
        assert app.state.auto_updater is not None
        assert app.state.auto_update_task is not None
    # After context exit, cleanup happened (no error)


# ---------------------------------------------------------------------------
# Health with rollback info (line 148)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint_with_rollback_info(app_factory, make_node, tmp_path, monkeypatch):
    """When rollback info file exists, health endpoint includes it."""
    monkeypatch.setenv("OMLX_PRIVATENET_STATE_DIR", str(tmp_path))
    import json
    rollback_data = {
        "rolled_back_from": "0.3.0",
        "rolled_back_to": "0.2.0",
        "timestamp": 1700000000,
    }
    (tmp_path / "rollback.json").write_text(json.dumps(rollback_data))

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.get("/health")

    payload = response.json()
    assert response.status_code == 200
    if "rollback" in payload:
        assert payload["rollback"]["rolled_back_from"] == "0.3.0"


# ---------------------------------------------------------------------------
# Embeddings local-only proxy (line 209)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_local_only(app_factory, make_node):
    """Embeddings with local-only header proxies to local oMLX."""
    captured: dict[str, object] = {}

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"object": "list", "data": [{"embedding": [0.1], "index": 0}]})

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL])]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post(
            "/v1/embeddings",
            headers={"x-omlx-router-local-only": "1"},
            json={"model": EMBED_MODEL, "input": "hello"},
        )

    assert response.status_code == 200
    assert captured["url"] == "http://127.0.0.1:5741/v1/embeddings"


# ---------------------------------------------------------------------------
# Embeddings route error (lines 213-214)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_no_healthy_nodes_returns_503(app_factory, make_node):
    """When no healthy nodes for embeddings, return 503."""
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL], healthy=False)]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.post(
            "/v1/embeddings",
            json={"model": EMBED_MODEL, "input": "hello"},
        )

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Streaming local proxy (lines 245-256, 327-336)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_local_proxy(app_factory, make_node):
    """Streaming chat completions via local proxy."""
    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncStream([b"data: {\"delta\":\"hi\"}\n\ndata: [DONE]\n\n"]),
        )

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"x-omlx-router-local-only": "1"},
            json={"model": CHAT_MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert response.status_code == 200
    assert b"data: {\"delta\":\"hi\"}" in body


# ---------------------------------------------------------------------------
# Streaming remote 500 failover (lines 267-271)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_remote_500_failover(app_factory, make_node):
    """Streaming request to remote node that returns 500 fails over to next candidate."""
    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL]),
        make_node("remote-a", tailscale_ip="100.64.0.2", models=[CHAT_MODEL]),
        make_node("remote-b", tailscale_ip="100.64.0.3", models=[CHAT_MODEL]),
    ]
    # Find a session that routes primarily to remote-a
    router_obj = ConsistentHashRouter(local_node_id="local-node", consistent_hash_replicas=32)
    router_obj.update_nodes(nodes)
    payload = None
    for index in range(10_000):
        candidate = {"model": CHAT_MODEL, "session_id": f"stream-retry-{index}", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        if router_obj.route_chat(candidate).primary.node_id == "remote-a":
            payload = candidate
            break
    assert payload is not None

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        if "100.64.0.2" in str(request.url):
            return httpx.Response(500, stream=MockAsyncStream([b"error"]))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncStream([b"data: ok\n\n"]),
        )

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        async with client.stream(
            "POST", "/v1/chat/completions", json=payload,
        ) as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert response.status_code == 200
    assert response.headers["x-omlx-privatenet-node"] == "remote-b"


# ---------------------------------------------------------------------------
# HTTP error failover (lines 285-287)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error_failover(app_factory, make_node):
    """When httpx raises an HTTPError, fail over to next candidate."""
    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL]),
        make_node("remote-a", tailscale_ip="100.64.0.2", models=[CHAT_MODEL]),
        make_node("remote-b", tailscale_ip="100.64.0.3", models=[CHAT_MODEL]),
    ]
    router_obj = ConsistentHashRouter(local_node_id="local-node", consistent_hash_replicas=32)
    router_obj.update_nodes(nodes)
    payload = None
    for index in range(10_000):
        candidate = {"model": CHAT_MODEL, "session_id": f"err-{index}", "messages": [{"role": "user", "content": "hi"}]}
        if router_obj.route_chat(candidate).primary.node_id == "remote-a":
            payload = candidate
            break
    assert payload is not None

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        if "100.64.0.2" in str(request.url):
            raise httpx.ConnectError("Connection refused")
        return httpx.Response(200, json={"id": "fallback", "choices": []})

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["x-omlx-privatenet-node"] == "remote-b"


# ---------------------------------------------------------------------------
# All candidates fail -> 503 (line 292)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_candidates_fail_returns_503(app_factory, make_node):
    """When all candidates fail, 503 with last error."""
    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL]),
        make_node("remote-a", tailscale_ip="100.64.0.2", models=[CHAT_MODEL]),
    ]

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": CHAT_MODEL, "session_id": "fail-all", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Streaming response with decision=None (line 391->395)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_local_only_no_decision(app_factory, make_node):
    """Streaming local-only does not include routing headers (decision is None)."""
    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncStream([b"data: hi\n\n"]),
        )

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"x-omlx-router-local-only": "1"},
            json={"model": CHAT_MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert response.status_code == 200
    assert "x-omlx-routing-key" not in response.headers


# ---------------------------------------------------------------------------
# JSON body not an object (line 417)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_body_not_object_returns_400(app_factory, make_node):
    """When JSON body is an array, return 400."""
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            content='[1, 2, 3]',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert "must be an object" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Peer headers with api_key (line 445, 432->434)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_peer_headers_include_api_key(app_factory, make_node):
    """When the router has an api_key, peer headers include Authorization."""
    captured: dict[str, object] = {}

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"id": "chatcmpl-1", "choices": []})

    nodes = [
        make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[EMBED_MODEL]),
        make_node("remote-node", tailscale_ip="100.64.0.2", models=[CHAT_MODEL]),
    ]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler, config_overrides={"api_key": "router-key"}) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer router-key"},
            json={"model": CHAT_MODEL, "session_id": "peer-auth", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert captured["headers"].get("authorization") == "Bearer router-key"


# ---------------------------------------------------------------------------
# Local oMLX headers without api_key (line 432->434)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_omlx_headers_no_api_key(app_factory, make_node):
    """When no local_omlx_api_key, headers don't include Authorization."""
    captured: dict[str, object] = {}

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"id": "local-chat", "choices": []})

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler, config_overrides={"local_omlx_api_key": None}) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            headers={"x-omlx-router-local-only": "1"},
            json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert "authorization" not in captured["headers"]


# ---------------------------------------------------------------------------
# Streaming local route via _proxy_via_selected_node (lines 245-256)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_local_via_route(app_factory, make_node):
    """When router routes streaming to local node through _proxy_via_selected_node."""
    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncStream([b"data: local\n\n"]),
        )

    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": CHAT_MODEL, "session_id": "local-stream", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert response.status_code == 200
    assert b"data: local" in body


# ---------------------------------------------------------------------------
# Auth middleware with wrong key (line 321->324 partial)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_api_key_returns_401(app_factory, make_node):
    """Wrong API key returns 401 with WWW-Authenticate header."""
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, config_overrides={"api_key": "correct-key"}) as (_, client):
        response = await client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing router API key."


# ---------------------------------------------------------------------------
# Non-streaming local route through _proxy_via_selected_node (branch 254->256)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffered_local_route_via_proxy_selected_node(app_factory, make_node):
    """Non-streaming chat completions routed to local node via _proxy_via_selected_node."""
    captured: dict[str, object] = {}

    def outgoing_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "local-buf", "choices": []})

    # Only local node has the model, so routing must go through local
    nodes = [make_node("local-node", tailscale_ip="100.64.0.1", local=True, models=[CHAT_MODEL])]

    async with app_factory(nodes=nodes, outgoing_handler=outgoing_handler) as (_, client):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": CHAT_MODEL, "session_id": "buf-local", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.headers["x-omlx-privatenet-node"] == "local-node"
    assert captured["url"] == "http://127.0.0.1:5741/v1/chat/completions"
