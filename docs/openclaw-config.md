# OpenClaw configuration

Point OpenClaw at any PrivateNet router as if it were a normal OpenAI-compatible provider.

## Example provider block

```json
{
  "models": {
    "providers": {
      "privatenet": {
        "baseUrl": "http://100.x.y.z:8741/v1",
        "apiKey": "optional-shared-router-api-key",
        "api": "openai-completions",
        "models": [
          {
            "id": "gemma-4-26b-a4b-it-4bit",
            "name": "Gemma 4 26B (PrivateNet)",
            "reasoning": true,
            "contextWindow": 128000,
            "maxOutput": 32768
          },
          {
            "id": "gemma-4-31b-it-4bit",
            "name": "Gemma 4 31B (PrivateNet)",
            "reasoning": true,
            "contextWindow": 128000,
            "maxOutput": 32768
          }
        ]
      }
    }
  }
}
```

Replace:

- `100.x.y.z` with the **Tailscale IP or MagicDNS name of any peer node**
- `optional-shared-router-api-key` with your shared router API key if you enabled one

## Recommended setup

1. Pick any stable peer as the default entry point.
2. Better yet, use **Tailscale MagicDNS** so the URL stays readable.
3. Reuse a stable OpenClaw session/conversation ID when possible so the router can preserve deterministic session affinity.
4. If you enable router auth, use the **same shared router API key on every node**.

## Example with MagicDNS

```json
{
  "baseUrl": "http://my-mac-studio.tailnet-name.ts.net:8741/v1",
  "apiKey": "optional-shared-router-api-key",
  "api": "openai-completions"
}
```

## Operational notes

- `/v1/models` is aggregated from the discovered peers.
- `/v1/chat/completions` uses session-hash routing first, then prefix-hash routing, then deterministic failover.
- `/v1/embeddings` uses healthy least-load routing.
- You do **not** need to maintain `cluster.json` anywhere.
- You can fail over manually by switching OpenClaw to another node's router URL; the routing logic stays the same.
