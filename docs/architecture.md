# Architecture

## Overview

`omlx-privatenet` is now a peer-to-peer Tailscale router for oMLX.

Every Mac runs the same two local services:

```text
┌──────────────────────────────────────────────┐
│ Peer Node                                    │
│                                              │
│  Router :8741                                │
│    - OpenAI-compatible API                   │
│    - Tailscale discovery                     │
│    - consistent-hash routing                 │
│                                              │
│  oMLX :5741                                  │
│    - local inference only                    │
│    - not exposed directly to other clients   │
└──────────────────────────────────────────────┘
```

A client can connect to any router in the tailnet. That router decides which peer should serve the request, then forwards the request to that peer's local oMLX server through the peer router.

## Discovery

There is no central registry and no manual node enrollment file.

Each router periodically runs:

```bash
tailscale status --json
```

The discovery loop:

1. reads the local tailnet peer list
2. filters peers by Tailscale tag `tag:omlx-node`
3. includes itself as a peer too
4. probes each peer's router at `http://<peer_ip>:8741/v1/node-info`
5. builds the routing table from those live node-info responses

### Why this is enough

Tailscale already gives us:

- authenticated peer identity
- encrypted transport
- routable private IPs
- a local, no-API-key-needed peer listing command

That means we do not need:

- `cluster.json`
- an admin-maintained node manifest
- a service registry
- gossip or leader election

## `/v1/node-info`

Every router exposes the state of its **local** oMLX instance:

```json
{
  "node_id": "my-mac-studio",
  "tailscale_ip": "100.x.y.z",
  "models": ["gemma-4-26b-a4b-it-4bit", "gemma-4-31b-it-4bit"],
  "in_flight": 2,
  "max_concurrent": 8,
  "healthy": true,
  "uptime_seconds": 3600
}
```

Routers use this to make the same routing decision independently.

## Routing algorithm

### Primary: session affinity via consistent hashing

If a request includes a stable session-like identifier, the router hashes that value onto a consistent-hash ring built from the discovered peer list.

Inputs checked:

- `session_id`
- `conversation_id`
- `thread_id`
- `chat_id`
- `metadata.session_id`
- `metadata.conversation_id`
- `user`

All routers use the same ring and the same hash, so they all pick the same primary node.

### Prefix hashing when there is no session ID

If no session key exists, the router hashes the first few messages (default: 3) and uses the resulting prefix hash as the ring key.

This keeps the original cache-locality idea:

- identical or similar conversation openings land on the same peer
- different routers still agree on the chosen primary

### Failover: availability first

If the primary node is:

- unhealthy, or
- overloaded

the router immediately walks clockwise to the next node on the ring.

This is deterministic too, so all routers agree on the same fallback order.

**Important principle:** availability is more important than cache locality.

When the original node recovers, new requests for that session may hash back to it naturally. There is no forced migration of in-flight work.

### Load awareness

Each node advertises:

- `in_flight`
- `max_concurrent`

Routing treats a node as overloaded when:

```text
in_flight >= overload_threshold
```

where `overload_threshold` is:

- the configured override, if set, otherwise
- the node's own advertised `max_concurrent`

## Health model

Health checks are based on `/v1/node-info`.

Defaults:

- discovery interval: `30s`
- health timeout: `5s`
- failure threshold: `3`

Behavior:

- a non-responsive node is immediately treated as unhealthy for routing
- after 3 consecutive failures it remains fully out of the routing path
- probing continues in the background
- once `/v1/node-info` succeeds again, the node is restored automatically

## Request flow

### External request

1. Client sends `POST /v1/chat/completions` to any router.
2. That router chooses a primary via session hash or prefix hash.
3. If needed, it walks the ring to the next healthy node.
4. If the selected node is local, it proxies directly to local oMLX on `127.0.0.1:5741`.
5. If the selected node is remote, it forwards the request to the remote router with a local-only header.
6. The remote router proxies the request to its own local oMLX.

### Why forward through the peer router?

Discovery only needs peer IPs from Tailscale. It does **not** need other nodes' private oMLX credentials.

By forwarding through the peer router, each node can keep its own local oMLX API key private while still participating in cluster routing.

## Security model

- Tailscale provides private addressing and encrypted transport.
- oMLX binds only to `127.0.0.1:5741`.
- The router is the only network-facing component.
- Router auth is optional; if enabled, use the same shared router API key on every node.
- Discovery requires no extra secrets and no external service.

## Operational implications

- Adding a node is simple: install it, connect to Tailscale, tag it `tag:omlx-node`.
- Removing a node is simple: shut it down or remove the tag.
- No central load balancer exists anymore.
- Any node can act as the cluster entry point.
