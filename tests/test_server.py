from __future__ import annotations

import httpx
import pytest

from router.router import ConsistentHashRouter


CHAT_MODEL = "gemma-4-26b-a4b-it-4bit"
EMBED_MODEL = "text-embedding-3-small"


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
    assert captured["headers"]["authorization"] == "Bearer shared-key"


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
