# omlx-privatenet

Distributed oMLX inference for trusted Macs on the same Tailscale tailnet.

```text
[OpenClaw / any client]
        |
        v
  [Load Balancer :8741]
   /      |       \
  v       v        v
[Mac1]  [Mac2]   [Mac3]
  :5741   :5741    :5741
```

`omlx-privatenet` gives you two pieces:

1. **An edge-node installer** for Apple Silicon Macs running oMLX behind Tailscale.
2. **A FastAPI load balancer** that exposes an OpenAI-compatible API and routes requests to the best node while preserving KV-cache locality as much as possible.

## Why this exists

oMLX already does a great job with prefix cache reuse on one machine. This repo extends that idea across many Macs on one private network:

- keep traffic on **Tailscale only**
- use **oMLX's built-in API key auth**
- route repeat conversations back to the **same node**
- use **prefix-based affinity** for new sessions with the same system prompt or opening turns
- fall back safely when a preferred node is down or overloaded

## Quickstart

### 1) Install an edge node on a Mac

Run this on an Apple Silicon Mac that should join the cluster:

```bash
curl -fsSL https://raw.githubusercontent.com/pm990320/omlx-privatenet/main/scripts/install.sh | bash
```

The installer will:

- verify macOS + Apple Silicon
- install Homebrew, Python 3.13, git, and Tailscale if missing
- clone **`pm990320/omlx`** at **`v0.3.2`** into `~/omlx-privatenet/omlx`
- create `~/.omlx-privatenet/venv`
- install oMLX, `huggingface-hub`, `xgrammar`, and the **Gemma 4 tool-calling mlx-lm fork**
- download both default models:
  - `gemma-4-26b-a4b-it-4bit`
  - `gemma-4-31b-it-4bit`
- configure oMLX to bind to `0.0.0.0:5741`
- generate and persist a node API key in `~/.omlx-privatenet/node.env`
- create `~/Library/LaunchAgents/com.omlx-privatenet.edge.plist`
- write `~/.omlx-privatenet/node.json`

When it finishes, send `~/.omlx-privatenet/node.json` to the cluster admin.

### 2) Build the cluster file on the balancer host

Copy `cluster.example.json` to `cluster.json` and add the node JSON payload from each Mac.

```json
[
  {
    "name": "mac-mini-m4",
    "tailscale_ip": "100.64.0.11",
    "port": 5741,
    "api_key": "pn-node-key-1",
    "models": [
      "gemma-4-26b-a4b-it-4bit",
      "gemma-4-31b-it-4bit"
    ],
    "max_inflight": 2
  }
]
```

### 3) Start the balancer

```bash
cd balancer
cp config.example.json config.json
make run
```

Default balancer listen address: `0.0.0.0:8741`

### 4) Point OpenClaw at the balancer

See [docs/openclaw-config.md](docs/openclaw-config.md).

## Request routing strategy

This is the v1 router design:

1. **Session affinity first**
   - if a request includes `session_id`, `conversation_id`, `thread_id`, `chat_id`, `metadata.session_id`, or `user`, the balancer hashes that value and keeps the session on one node whenever possible.
2. **Prefix affinity second**
   - for new sessions, the balancer chunk-hashes the first few messages and records which node likely has that prefix cached.
3. **Bounded-load fallback**
   - if the preferred node is unhealthy or over the node's `max_inflight` threshold, the balancer walks the rendezvous-hash ordering and picks the first node under the load cap.
4. **Least-loaded safety net**
   - if every candidate is overloaded, the balancer still serves the request on the least-loaded healthy node.

This is intentionally simple: no oMLX changes, no central registry, and no deep cache introspection.

## Repo layout

```text
omlx-privatenet/
├── README.md
├── cluster.example.json
├── scripts/
│   └── install.sh
├── balancer/
│   ├── __init__.py
│   ├── config.py
│   ├── config.example.json
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

## Balancer config

`balancer/config.example.json` controls the balancer process itself:

```json
{
  "host": "0.0.0.0",
  "port": 8741,
  "api_key": "change-me-balancer-api-key",
  "health_interval_seconds": 30,
  "connect_timeout_seconds": 10,
  "request_timeout_seconds": 600,
  "prefix_message_count": 3,
  "sticky_ttl_seconds": 43200,
  "default_max_inflight": 2,
  "cluster_file": "../cluster.json"
}
```

`cluster.json` is the node registry. It is manually maintained by the admin and intentionally not committed.

## LaunchAgent support

### Edge nodes

The installer writes:

- `~/Library/LaunchAgents/com.omlx-privatenet.edge.plist`
- `~/.omlx-privatenet/start-edge.sh`

### Balancer

Install the balancer as a LaunchAgent from the repo root:

```bash
cd balancer
cp config.example.json config.json
make install-launchagent
```

That writes `~/Library/LaunchAgents/com.omlx-privatenet.balancer.plist` and loads it for the current macOS user.

## API surface

The balancer exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

All `/v1/*` endpoints use the balancer API key from `balancer/config.json` when set.

## Notes

- Traffic stays inside **Tailscale**.
- No TLS is required on top because Tailscale already encrypts the transport.
- Each edge node keeps its **own oMLX API key**.
- The balancer stores those node keys in `cluster.json`.
- The balancer does not require any changes to oMLX internals.

## License

MIT
