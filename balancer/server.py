from __future__ import annotations

import argparse
import asyncio
import os
import plistlib
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .config import BalancerConfig, NodeConfig, load_config, resolve_config_path
from .health import NodeHealthMonitor
from .router import CacheAwareRouter, RouteDecision

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
LAUNCH_AGENT_LABEL = "com.omlx-privatenet.balancer"


def create_app(config_path: str | Path | None = None) -> FastAPI:
    config = load_config(config_path)
    timeout = httpx.Timeout(
        connect=config.connect_timeout_seconds,
        read=None,
        write=config.request_timeout_seconds,
        pool=config.request_timeout_seconds,
    )
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    router = CacheAwareRouter(
        config.nodes,
        prefix_message_count=config.prefix_message_count,
        sticky_ttl_seconds=config.sticky_ttl_seconds,
        default_max_inflight=config.default_max_inflight,
    )
    monitor = NodeHealthMonitor(config, router, client)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.client = client
        app.state.router = router
        app.state.monitor = monitor
        await monitor.run_once()
        task = asyncio.create_task(monitor.run_forever(), name="omlx-privatenet-health")
        app.state.health_task = task
        try:
            yield
        finally:
            await monitor.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await client.aclose()

    app = FastAPI(title="oMLX PrivateNet Balancer", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def enforce_auth(request: Request, call_next):  # type: ignore[override]
        if request.url.path == "/health":
            return await call_next(request)
        _require_balancer_api_key(request, config)
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "host": config.host,
            "port": config.port,
            "nodes": router.snapshot()["nodes"],
            "models": router.aggregate_models(),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        data = [
            {
                "id": model,
                "object": "model",
                "owned_by": "omlx-privatenet",
            }
            for model in router.aggregate_models()
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        payload = await _read_json_body(request)
        try:
            decision = router.route_chat(payload)
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return await _proxy_request(
            request=request,
            payload=payload,
            endpoint="/v1/chat/completions",
            decision=decision,
            stream=bool(payload.get("stream")),
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        payload = await _read_json_body(request)
        try:
            decision = router.route_embeddings(payload)
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return await _proxy_request(
            request=request,
            payload=payload,
            endpoint="/v1/embeddings",
            decision=decision,
            stream=False,
        )

    return app


async def _proxy_request(
    *,
    request: Request,
    payload: dict[str, Any],
    endpoint: str,
    decision: RouteDecision,
    stream: bool,
) -> Response:
    app = request.app
    client: httpx.AsyncClient = app.state.client
    router: CacheAwareRouter = app.state.router

    last_error: str | None = None
    for node in decision.ordered_candidates:
        acquired = False
        streaming_response: httpx.Response | None = None
        router.mark_inflight(node.node_id, 1)
        acquired = True
        try:
            if stream:
                streaming_response = await _send_streaming(client=client, node=node, endpoint=endpoint, payload=payload)
                if streaming_response.status_code >= 500 and node is not decision.ordered_candidates[-1]:
                    status_code = streaming_response.status_code
                    await streaming_response.aclose()
                    streaming_response = None
                    last_error = f"{node.node_id} returned {status_code}"
                    router.update_node_health(node.node_id, healthy=False, error=last_error)
                    router.release(node.node_id)
                    acquired = False
                    continue
                return _streaming_response(router=router, node=node, upstream=streaming_response, decision=decision)

            upstream = await client.post(
                f"{node.base_url}{endpoint}",
                json=payload,
                headers=_upstream_headers(node),
            )
            if upstream.status_code >= 500 and node is not decision.ordered_candidates[-1]:
                last_error = f"{node.node_id} returned {upstream.status_code}"
                router.update_node_health(node.node_id, healthy=False, error=last_error)
                continue
            return _buffered_response(node=node, upstream=upstream, decision=decision)
        except httpx.HTTPError as exc:
            last_error = f"{node.node_id}: {exc}"
            router.update_node_health(node.node_id, healthy=False, error=last_error)
        finally:
            if acquired and (not stream or streaming_response is None):
                router.release(node.node_id)

    raise HTTPException(status_code=503, detail=last_error or "No healthy upstream node accepted the request.")


async def _send_streaming(
    *,
    client: httpx.AsyncClient,
    node: NodeConfig,
    endpoint: str,
    payload: dict[str, Any],
) -> httpx.Response:
    request = client.build_request(
        "POST",
        f"{node.base_url}{endpoint}",
        json=payload,
        headers=_upstream_headers(node),
    )
    return await client.send(request, stream=True)


def _buffered_response(
    *,
    node: NodeConfig,
    upstream: httpx.Response,
    decision: RouteDecision,
) -> Response:
    headers = _filtered_headers(upstream.headers)
    headers["x-omlx-privatenet-node"] = node.node_id
    headers["x-omlx-routing-key"] = decision.routing_key
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers=headers,
    )


def _streaming_response(
    *,
    router: CacheAwareRouter,
    node: NodeConfig,
    upstream: httpx.Response,
    decision: RouteDecision,
) -> StreamingResponse:
    headers = _filtered_headers(upstream.headers)
    headers["x-omlx-privatenet-node"] = node.node_id
    headers["x-omlx-routing-key"] = decision.routing_key

    async def iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            router.release(node.node_id)

    return StreamingResponse(
        iterator(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=headers,
    )


async def _read_json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")
    return payload


def _filtered_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in dict(headers).items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        result[key] = value
    return result


def _upstream_headers(node: NodeConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {node.api_key}"}


def _require_balancer_api_key(request: Request, config: BalancerConfig) -> None:
    if not config.api_key:
        return
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {config.api_key}"
    if auth != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing balancer API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def install_launchagent(config_path: str | Path, *, load: bool = True) -> Path:
    config = load_config(config_path)
    resolved_config = resolve_config_path(config_path)
    repo_root = Path(__file__).resolve().parent.parent
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
    logs_dir = Path.home() / "Library" / "Logs" / "omlx-privatenet"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    program = [
        sys.executable,
        "-m",
        "balancer.server",
        "--config",
        str(resolved_config),
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    payload = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": program,
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logs_dir / "balancer.stdout.log"),
        "StandardErrorPath": str(logs_dir / "balancer.stderr.log"),
    }
    with plist_path.open("wb") as handle:
        plistlib.dump(payload, handle)

    if load:
        uid = str(os.getuid())
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(plist_path)], check=False)
        result = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)], check=False)
        if result.returncode != 0:
            subprocess.run(["launchctl", "load", "-w", str(plist_path)], check=True)
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/{LAUNCH_AGENT_LABEL}"], check=False)
    return plist_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the oMLX PrivateNet load balancer.")
    parser.add_argument("--config", default=str(resolve_config_path()), help="Path to balancer config JSON.")
    parser.add_argument("--host", default=None, help="Override bind host.")
    parser.add_argument("--port", default=None, type=int, help="Override bind port.")
    parser.add_argument("--install-launchagent", action="store_true", help="Write ~/Library/LaunchAgents/com.omlx-privatenet.balancer.plist and optionally load it.")
    parser.add_argument("--no-load", action="store_true", help="When used with --install-launchagent, only write the plist.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.install_launchagent:
        plist = install_launchagent(args.config, load=not args.no_load)
        print(f"LaunchAgent written to {plist}")
        return 0

    config = load_config(args.config)
    host = args.host or config.host
    port = args.port or config.port
    app = create_app(args.config)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
