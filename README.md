# omlx-privatenet

Peer-to-peer oMLX inference for trusted Macs on the same Tailscale tailnet.

```text
Every Mac runs:

┌──────────────────────────────────────────────┐
│ Mac A / Mac B / Mac C / ...                  │
│                                              │
│  oMLX                 127.0.0.1:5741         │
│  Router               0.0.0.0:8741           │
│  Tailscale discovery  tag:omlx-node          │
└──────────────────────────────────────────────┘

Any client can connect to ANY router on :8741.
That router discovers peers over Tailscale and forwards the request
to the best node's local oMLX server.
```

`omlx-privatenet` now gives you a single edge install that turns every Mac into a full peer:

1. **oMLX** bound locally on `127.0.0.1:5741`
2. **Router** bound on `0.0.0.0:8741` with an OpenAI-compatible API

There is no central load balancer, no `cluster.json`, and no admin collecting node manifests.

## Why this exists

oMLX already does a great job with prefix cache reuse on one machine. This repo extends that idea across many Macs on one private network:

- keep traffic on **Tailscale only**
- discover peers automatically with **`tailscale status --json`**
- route repeat conversations back to the **same node** with consistent hashing
- use **prefix-based hashing** for requests that do not include a session ID
- fail over immediately when a node is down or overloaded
- keep the system simple: no external coordinator, no gossip, no API keys for discovery

## Quickstart

### 1) Install a peer node on each Mac

Run this on every Apple Silicon Mac that should join the cluster:

```bash
curl -fsSL https://raw.githubusercontent.com/pm990320/omlx-privatenet/main/scripts/install.sh | bash
```

The installer will:

- verify macOS + Apple Silicon
- install Homebrew, Python 3.13, git, and Tailscale if missing
- connect Tailscale and try to advertise `tag:omlx-node`
- clone **`pm990320/omlx`** at **`v0.3.2`**
- clone this repo so the router code is installed locally too
- create `~/.omlx-privatenet/venv`
- install oMLX, router requirements, `huggingface-hub`, `xgrammar`, and the **Gemma 4 tool-calling mlx-lm fork**
- download both default models:
  - `gemma-4-26b-a4b-it-4bit`
  - `gemma-4-31b-it-4bit`
- configure oMLX to bind to `127.0.0.1:5741`
- write `~/.omlx-privatenet/router.json`
- create two LaunchAgents:
  - `~/Library/LaunchAgents/com.omlx-privatenet.omlx.plist`
  - `~/Library/LaunchAgents/com.omlx-privatenet.router.plist`
- start both services automatically

If tag advertisement fails, the installer prints the exact follow-up for the Tailscale admin:

> Your Tailscale admin needs to add this to the ACL policy: `tag:omlx-node`

### 2) Point clients at any node's router

Use any peer's Tailscale IP or MagicDNS name:

```text
http://100.x.y.z:8741/v1
```

See [docs/openclaw-config.md](docs/openclaw-config.md) for an OpenClaw example.

## Discovery model

Every router periodically runs:

```bash
tailscale status --json
```

It then:

1. finds peers tagged `tag:omlx-node`
2. gets their Tailscale IPs
3. probes `http://<peer_ip>:8741/v1/node-info`
4. builds a live routing table from those responses

There is no manual node registry.

## Routing strategy

### 1) Session affinity via consistent hashing

If a request includes `session_id`, `conversation_id`, `thread_id`, `chat_id`, `metadata.session_id`, or `user`, the router hashes that value onto a shared consistent-hash ring.

Because every router sees the same peer list and uses the same hash function, they all pick the same primary node.

### 2) Prefix hashing for requests without session IDs

If there is no session identifier, the router hashes the first few messages (default: 3) and uses that prefix hash as the ring key.

That preserves the original cache-locality idea without requiring a central coordinator.

### 3) Deterministic failover

If the primary node is:

- unhealthy, or
- overloaded (`in_flight >= overload_threshold`, or `max_concurrent` when no override is set)

then the router walks clockwise around the same consistent-hash ring until it finds the next healthy node.

**Availability wins over cache locality.** A cache miss is acceptable. A dead primary is not.

### 4) Health checks

Each router probes peers through `/v1/node-info`.

- timeout: `5s` by default
- interval: `30s` by default
- after **1 failure** the node is treated as unhealthy for routing
- after **3 consecutive failures** it stays out of the routing path until it recovers
- recovery is automatic as soon as `/v1/node-info` responds again

## Repo layout

```text
omlx-privatenet/
├── README.md
├── scripts/
│   └── install.sh
├── router/
│   ├── __init__.py
│   ├── config.py
│   ├── config.example.json
│   ├── discovery.py
│   ├── health.py
│   ├── Makefile
│   ├── requirements.txt
│   ├── router.py
│   └── server.py
├── docs/
│   ├── architecture.md
│   └── openclaw-config.md
└── LICENSE
```

## Router config

The router reads optional local settings from:

```text
~/.omlx-privatenet/router.json
```

Example:

```json
{
  "host": "0.0.0.0",
  "port": 8741,
  "api_key": null,
  "discovery_interval_seconds": 30,
  "health_check_timeout_seconds": 5,
  "failure_threshold": 3,
  "overload_threshold": null,
  "local_node_id": "macbook-patrick",
  "local_tailscale_ip": "100.64.0.10",
  "local_omlx_url": "http://127.0.0.1:5741",
  "local_omlx_api_key": "pn-local-omlx-key",
  "local_models": [
    "gemma-4-26b-a4b-it-4bit",
    "gemma-4-31b-it-4bit"
  ],
  "local_max_concurrent": 8
}
```

Notes:

- `api_key` is optional. If you enable it, use the **same shared router API key on every node** so peer-to-peer forwarding keeps working cleanly.
- `overload_threshold: null` means: use each node's advertised `max_concurrent` value.
- discovery uses only Tailscale; no external service or registry is required.

## LaunchAgent support

Each node installs two separate LaunchAgents:

- `com.omlx-privatenet.omlx`
- `com.omlx-privatenet.router`

Manual install from the repo is still available:

```bash
cd router
cp config.example.json ~/.omlx-privatenet/router.json
make install-launchagent
```

## API surface

The router exposes:

- `GET /health`
- `GET /v1/node-info`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

The `/v1/node-info` endpoint describes the local node's oMLX state and is what peers use for health + load information.

## Notes

- Traffic stays inside **Tailscale**.
- oMLX stays local-only on `127.0.0.1:5741`.
- The router is the only network-facing process.
- No `cluster.json` is used anywhere.
- No admin role is required for normal day-to-day operation.

## License

MIT
