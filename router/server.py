from __future__ import annotations

"""FastAPI application exposing the PrivateNet router API."""

import argparse
import asyncio
import hmac
import os
import plistlib
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import RouterConfig, load_config, resolve_config_path
from .health import NodeHealthMonitor
from .registry import Registry
from .router import ConsistentHashRouter, NodeInfo, RouteDecision

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
LOCAL_ONLY_HEADER = "x-omlx-router-local-only"
ROUTED_BY_HEADER = "x-omlx-routed-by"
LAUNCH_AGENT_LABEL = "com.omlx-privatenet.router"


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create the FastAPI application for the router service."""
    config = load_config(config_path)
    timeout = httpx.Timeout(
        connect=config.connect_timeout_seconds,
        read=None,
        write=config.request_timeout_seconds,
        pool=config.request_timeout_seconds,
    )
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    router = ConsistentHashRouter(
        local_node_id=config.local_node_id,
        prefix_message_count=config.prefix_message_count,
        overload_threshold=config.overload_threshold,
        consistent_hash_replicas=config.consistent_hash_replicas,
    )
    monitor = NodeHealthMonitor(config, router, client)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.client = client
        app.state.router = router
        app.state.monitor = monitor
        await monitor.run_once()
        task = asyncio.create_task(monitor.run_forever(), name="omlx-privatenet-discovery")
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

    app = FastAPI(title="oMLX PrivateNet Router", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def enforce_auth(request: Request, call_next):  # type: ignore[override]
        if request.url.path in {"/health", "/v1/node-info", "/v1/registry"}:
            return await call_next(request)
        try:
            _require_router_api_key(request, config)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Return local router status plus the currently known cluster view."""
        disabled = _is_node_disabled()
        return {
            "status": "disabled" if disabled else "ok",
            "router": {
                "host": config.host,
                "port": config.port,
                "node_id": config.local_node_id,
            },
            "cluster": router.snapshot()["nodes"],
            "models": router.aggregate_models(),
        }

    @app.get("/v1/node-info")
    async def node_info() -> dict[str, Any]:
        """Return metadata peers use for health and load balancing."""
        node = await monitor.current_local_node_info()
        payload = _node_info_payload(node)
        if _is_node_disabled():
            payload["healthy"] = False
            payload["disabled"] = True
        return payload

    @app.get("/v1/registry")
    async def get_registry() -> dict[str, Any]:
        """Return the local model registry as JSON."""
        state_dir = Path(os.environ.get("OMLX_PRIVATENET_STATE_DIR", Path.home() / ".omlx-privatenet"))
        registry = Registry(path=state_dir / "registry.json")
        registry.load()
        return {"models": [m.to_dict() for m in registry.models]}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """Aggregate model IDs across healthy nodes in oMLX-compatible format."""
        now = int(time.time())
        all_models = router.aggregate_models(healthy_only=True)
        data = [
            {
                "id": model,
                "object": "model",
                "created": now,
                "owned_by": "omlx-privatenet",
            }
            for model in all_models
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        """Route or locally proxy chat completion requests."""
        payload = await _read_json_body(request)
        if _is_local_only(request):
            return await _proxy_to_local_omlx(request=request, payload=payload, endpoint="/v1/chat/completions", stream=bool(payload.get("stream")))

        try:
            decision = router.route_chat(payload)
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return await _proxy_via_selected_node(
            request=request,
            payload=payload,
            endpoint="/v1/chat/completions",
            decision=decision,
            stream=bool(payload.get("stream")),
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        """Route or locally proxy embeddings requests."""
        payload = await _read_json_body(request)
        if _is_local_only(request):
            return await _proxy_to_local_omlx(request=request, payload=payload, endpoint="/v1/embeddings", stream=False)

        try:
            decision = router.route_embeddings(payload)
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return await _proxy_via_selected_node(
            request=request,
            payload=payload,
            endpoint="/v1/embeddings",
            decision=decision,
            stream=False,
        )

    return app


async def _proxy_via_selected_node(
    *,
    request: Request,
    payload: dict[str, Any],
    endpoint: str,
    decision: RouteDecision,
    stream: bool,
) -> Response:
    app = request.app
    client: httpx.AsyncClient = app.state.client
    router: ConsistentHashRouter = app.state.router
    config: RouterConfig = app.state.config

    last_error: str | None = None
    for node in decision.ordered_candidates:
        router.bump_inflight(node.node_id, 1)
        release_on_exit = True
        try:
            if node.local or node.node_id == config.local_node_id:
                response = await _proxy_to_local_omlx(
                    request=request,
                    payload=payload,
                    endpoint=endpoint,
                    stream=stream,
                    selected=node,
                    decision=decision,
                    manage_inflight=False,
                )
                if stream:
                    release_on_exit = False
                return response

            if stream:
                upstream = await _send_streaming(
                    client=client,
                    node=node,
                    endpoint=endpoint,
                    payload=payload,
                    headers=_peer_headers(request, config),
                )
                if upstream.status_code >= 500 and node.node_id != decision.ordered_candidates[-1].node_id:
                    status_code = upstream.status_code
                    await upstream.aclose()
                    router.mark_node_unhealthy(node.node_id, f"{node.node_id} returned {status_code}")
                    last_error = f"{node.node_id} returned {status_code}"
                    continue
                release_on_exit = False
                return _streaming_response(router=router, node=node, upstream=upstream, decision=decision)

            upstream = await client.post(
                f"{node.router_url}{endpoint}",
                json=payload,
                headers=_peer_headers(request, config),
            )
            if upstream.status_code >= 500 and node.node_id != decision.ordered_candidates[-1].node_id:
                last_error = f"{node.node_id} returned {upstream.status_code}"
                router.mark_node_unhealthy(node.node_id, last_error)
                continue
            return _buffered_response(node=node, upstream=upstream, decision=decision)
        except httpx.HTTPError as exc:
            last_error = f"{node.node_id}: {exc}"
            router.mark_node_unhealthy(node.node_id, last_error)
        finally:
            if release_on_exit:
                router.release(node.node_id)

    raise HTTPException(status_code=503, detail=last_error or "No healthy upstream node accepted the request.")


async def _proxy_to_local_omlx(
    *,
    request: Request,
    payload: dict[str, Any],
    endpoint: str,
    stream: bool,
    selected: NodeInfo | None = None,
    decision: RouteDecision | None = None,
    manage_inflight: bool = True,
) -> Response:
    app = request.app
    client: httpx.AsyncClient = app.state.client
    router: ConsistentHashRouter = app.state.router
    config: RouterConfig = app.state.config
    node = selected or router.get_node(config.local_node_id) or NodeInfo(
        node_id=config.local_node_id,
        tailscale_ip=config.local_tailscale_ip or "127.0.0.1",
        router_url=f"http://{config.local_tailscale_ip or '127.0.0.1'}:{config.port}",
        models=list(config.local_models),
        in_flight=0,
        max_concurrent=config.local_max_concurrent,
        healthy=False,
        uptime_seconds=0,
        local=True,
    )

    if manage_inflight:
        router.bump_inflight(node.node_id, 1)

    release_on_exit = manage_inflight
    try:
        if stream:
            upstream = await _send_streaming(
                client=client,
                node=node,
                endpoint=endpoint,
                payload=payload,
                headers=_local_omlx_headers(config),
                base_url=config.local_omlx_url,
            )
            release_on_exit = False
            return _streaming_response(router=router, node=node, upstream=upstream, decision=decision)

        upstream = await client.post(
            f"{config.local_omlx_url}{endpoint}",
            json=payload,
            headers=_local_omlx_headers(config),
        )
        return _buffered_response(node=node, upstream=upstream, decision=decision)
    finally:
        if release_on_exit:
            router.release(node.node_id)


async def _send_streaming(
    *,
    client: httpx.AsyncClient,
    node: NodeInfo,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    base_url: str | None = None,
) -> httpx.Response:
    target_base = base_url or node.router_url
    request = client.build_request(
        "POST",
        f"{target_base}{endpoint}",
        json=payload,
        headers=headers,
    )
    return await client.send(request, stream=True)


def _buffered_response(*, node: NodeInfo, upstream: httpx.Response, decision: RouteDecision | None) -> Response:
    headers = _filtered_headers(upstream.headers)
    headers["x-omlx-privatenet-node"] = node.node_id
    if decision is not None:
        headers["x-omlx-routing-key"] = decision.routing_key
        headers["x-omlx-routing-reason"] = decision.reason
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers=headers,
    )


def _streaming_response(
    *,
    router: ConsistentHashRouter,
    node: NodeInfo,
    upstream: httpx.Response,
    decision: RouteDecision | None,
) -> StreamingResponse:
    headers = _filtered_headers(upstream.headers)
    headers["x-omlx-privatenet-node"] = node.node_id
    if decision is not None:
        headers["x-omlx-routing-key"] = decision.routing_key
        headers["x-omlx-routing-reason"] = decision.reason

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


def _local_omlx_headers(config: RouterConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    if config.local_omlx_api_key:
        headers["Authorization"] = f"Bearer {config.local_omlx_api_key}"
    return headers


def _peer_headers(request: Request, config: RouterConfig) -> dict[str, str]:
    """Build headers for peer-to-peer forwarding.

    Uses the router's own API key (if configured) rather than forwarding
    the client's auth header to untrusted peers.
    """
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    headers[LOCAL_ONLY_HEADER] = "1"
    headers[ROUTED_BY_HEADER] = config.local_node_id
    return headers


def _require_router_api_key(request: Request, config: RouterConfig) -> None:
    if not config.api_key:
        return
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {config.api_key}"
    if not hmac.compare_digest(auth.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing router API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _is_local_only(request: Request) -> bool:
    return request.headers.get(LOCAL_ONLY_HEADER, "").strip() == "1"


def _is_node_disabled() -> bool:
    """Check whether this node has been administratively disabled."""
    return Path(os.environ.get("OMLX_PRIVATENET_STATE_DIR", Path.home() / ".omlx-privatenet"), "disabled").exists()


def _node_info_payload(node: NodeInfo) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "tailscale_ip": node.tailscale_ip,
        "models": list(node.models),
        "in_flight": node.in_flight,
        "max_concurrent": node.max_concurrent,
        "healthy": node.healthy,
        "uptime_seconds": node.uptime_seconds,
    }


def install_launchagent(config_path: str | Path | None = None, *, load: bool = True) -> Path:  # pragma: no cover - macOS launchctl integration
    """Write a LaunchAgent plist for the router, optionally loading it."""
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
        "router.server",
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
        "StandardOutPath": str(logs_dir / "router.stdout.log"),
        "StandardErrorPath": str(logs_dir / "router.stderr.log"),
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


def build_parser() -> argparse.ArgumentParser:  # pragma: no cover - thin CLI wrapper
    """Build the CLI parser for the router entrypoint."""
    parser = argparse.ArgumentParser(description="Run the oMLX PrivateNet router.")
    parser.add_argument("--config", default=str(resolve_config_path()), help="Path to router config JSON.")
    parser.add_argument("--host", default=None, help="Override bind host.")
    parser.add_argument("--port", default=None, type=int, help="Override bind port.")
    parser.add_argument(
        "--install-launchagent",
        action="store_true",
        help="Write ~/Library/LaunchAgents/com.omlx-privatenet.router.plist and optionally load it.",
    )
    parser.add_argument("--no-load", action="store_true", help="When used with --install-launchagent, only write the plist.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    """CLI entrypoint for launching the router service."""
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
