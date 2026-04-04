# Architecture

## Overview

`omlx-privatenet` is a Tailscale-only distributed inference layer for oMLX.

```text
                   ┌────────────────────┐
                   │ OpenClaw / Clients │
                   └─────────┬──────────┘
                             │ OpenAI-compatible API
                             v
                  ┌──────────────────────────┐
                  │  PrivateNet Balancer     │
                  │  FastAPI + httpx proxy   │
                  │  :8741                   │
                  └───────┬────────┬─────────┘
                          │        │
                          v        v
                 ┌────────────┐ ┌────────────┐
                 │ Edge Node  │ │ Edge Node  │
                 │ oMLX :5741 │ │ oMLX :5741 │
                 └────────────┘ └────────────┘
```

## Edge nodes

Each edge node is an Apple Silicon Mac that runs:

- Tailscale
- Python 3.13
- `pm990320/omlx` at `v0.3.2`
- the `pm990320/mlx-lm@feat/gemma4-tool-calling` fork
- oMLX bound to `0.0.0.0:5741`
- API-key auth via `OMLX_API_KEY`

The installer also downloads these two models by default:

- `gemma-4-26b-a4b-it-4bit`
- `gemma-4-31b-it-4bit`

## Cluster registry

There is no service discovery layer.

The admin maintains `cluster.json` manually. Each entry contains:

- `tailscale_ip`
- `port`
- `api_key`
- `models`
- optional `name`
- optional `max_inflight`

This keeps the system simple and private.

## Balancer internals

### Config loading

`balancer/config.py` loads:

- the balancer config file (`balancer/config.json`)
- the cluster file (`cluster.json`)

Relative paths are resolved from the config file location, so `cluster_file: "../cluster.json"` works cleanly.

### Health checks

`balancer/health.py` probes each node every 30 seconds by default:

- `GET /health`
- `GET /v1/models/status`

From those responses the balancer tracks:

- node health
- in-flight activity signals (`active_requests`, `waiting_requests`)
- which models appear loaded right now

### Router design

`balancer/router.py` uses a simple inferred-cache strategy inspired by SGLang router / Preble-style ideas:

1. **Session affinity**
   - if a request carries a session-like identifier, always try to send it back to the same node.
2. **Prefix affinity**
   - otherwise, chunk-hash the first few messages.
   - each hash is chained: `hash(chunk_i + previous_hash)`.
   - the balancer records which node most recently served those prefixes.
3. **Rendezvous hashing with bounded load**
   - when there is no direct affinity hit, rank candidate nodes with rendezvous hashing.
   - pick the first healthy node whose in-flight count is below its configured threshold.
4. **Least-loaded fallback**
   - if every healthy candidate is above threshold, route to the least-loaded one anyway.

This gives good locality without needing explicit KV telemetry from oMLX.

## Request flow

### Chat completions

1. Client sends `POST /v1/chat/completions` to the balancer.
2. Balancer filters nodes by requested `model`.
3. Router tries, in order:
   - existing session affinity
   - longest known prefix match
   - single healthy node already running that model in memory
   - rendezvous hash candidate under load threshold
   - least-loaded healthy fallback
4. Balancer proxies the request with the node's own API key.
5. For streaming, SSE bytes are passed through directly.
6. Router records the selected node as the likely cache owner for that session/prefix.

### Embeddings

Embeddings do not benefit from the same prefix locality rules, so the balancer uses healthy least-load routing for the requested embedding model.

## Failure model

- **Node down**: health check marks it unhealthy; router skips it.
- **Node overloaded**: router prefers the next ranked candidate.
- **All healthy nodes overloaded**: least-loaded node still receives the request.
- **Upstream 5xx before stream starts**: balancer can try the next candidate.
- **Manual node changes**: update `cluster.json` and restart the balancer.

## Security model

- Tailscale provides encrypted transport and private addressing.
- Every edge node has its own oMLX API key.
- The balancer can also require its own API key for clients.
- No public internet exposure is required.
