# OpenClaw configuration

Point OpenClaw at the PrivateNet balancer as if it were a normal OpenAI-compatible provider.

## Example provider block

```json
{
  "models": {
    "providers": {
      "privatenet": {
        "baseUrl": "http://100.x.y.z:8741/v1",
        "apiKey": "balancer-api-key",
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

- `100.x.y.z` with the **Tailscale IP of the balancer host**
- `balancer-api-key` with the `api_key` from `balancer/config.json`

## Recommended setup

1. Put the balancer on a stable machine in the same tailnet.
2. Keep `cluster.json` updated when nodes join or leave.
3. Reuse a stable OpenClaw session/conversation ID when possible so the balancer can preserve node affinity and improve cache reuse.

## Operational notes

- `/v1/models` is aggregated across every configured node.
- `/v1/chat/completions` uses session affinity first, then prefix affinity, then bounded-load fallback.
- `/v1/embeddings` uses healthy least-load routing for the requested model.
