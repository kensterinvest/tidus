# Pricing Sync

Tidus uses a two-layer pricing update system to keep model costs accurate — accurate prices are critical because the routing score weights cost at **70%**.

## How It Works

```
┌─────────────────────────────────┐     every Sunday 03:00 (host)
│  sync_pricing.py  (local host)  │ ──────────────────────────────────┐
│  - fetches OpenRouter API        │                                    │
│  - diffs against models.yaml     │                                    ▼
│  - updates models.yaml + sync.py │              ┌──────────────────────────┐
│  - git commit + push → GitHub    │              │   GitHub (main branch)    │
└─────────────────────────────────┘              │   config/models.yaml      │
                                                 └──────────────────────────┘
┌─────────────────────────────────┐                         │
│  Tidus server (APScheduler)     │  git pull / Docker pull │
│  - runs every Sunday 02:00 UTC  │◄────────────────────────┘
│  - diffs _KNOWN_PRICES vs live  │
│  - updates in-memory registry   │
│  - writes PriceChangeRecord DB  │
└─────────────────────────────────┘
```

### Layer 1 — Host script (external, runs locally)

`sync_pricing.py` is a standalone Python script that runs on the machine maintaining the Tidus deployment. It is **not part of the open-source repo** — it is host-specific.

What it does each Sunday:
1. Calls the [OpenRouter public API](https://openrouter.ai/api/v1/models) to get live prices for ~30 models across all major vendors
2. Compares against current values in `config/models.yaml`
3. Updates any prices that changed by ≥ 5% (configurable via `--threshold`)
4. Updates `_KNOWN_PRICES` in `tidus/sync/price_sync.py` to match
5. Git commits both files and pushes to `origin/main`
6. Optionally triggers `POST /api/v1/sync/prices` on the running server

### Layer 2 — Server sync (internal, runs inside Tidus)

`TidusScheduler` fires `run_price_sync()` every Sunday at 02:00 UTC (configured in `config/policies.yaml`).

What it does:
- Compares the live model registry against `_KNOWN_PRICES`
- Updates model prices in-memory when delta ≥ threshold
- Writes a `PriceChangeRecord` to the database for each change (audit trail)
- Changes are visible immediately in routing — no restart needed

## Configuring the Sync Schedule

Edit `config/policies.yaml`:

```yaml
pricing_sync:
  day_of_week: 6      # 0=Monday … 6=Sunday
  hour_utc: 2         # run at 02:00 UTC
  change_threshold: 0.05  # 5% delta triggers an update
```

## Manually Triggering a Sync

Any admin can trigger an immediate price sync via the API:

```bash
curl -X POST http://localhost:8000/api/v1/sync/prices \
  -H "Authorization: Bearer <admin-token>"
```

Response:

```json
{
  "changes_detected": 3,
  "changes": [
    {
      "model_id": "deepseek-v3",
      "field": "input_price",
      "old_value": 0.00028,
      "new_value": 0.00032,
      "delta_pct": 14.3
    }
  ]
}
```

## Viewing Price Change History

All detected changes are stored in the `price_change_logs` table and exposed via the audit API:

```bash
curl http://localhost:8000/api/v1/audit/events?action=price_change \
  -H "Authorization: Bearer <admin-token>"
```

## Adding New Models to the Sync

When a new model is added to `config/models.yaml`:

1. Add its price to `_KNOWN_PRICES` in `tidus/sync/price_sync.py`
2. If the vendor is on OpenRouter, add a mapping entry to `OPENROUTER_MAP` in `sync_pricing.py`

## Price Sources

| Vendor | Primary source | Covered by OpenRouter? |
|--------|---------------|----------------------|
| OpenAI | openai.com/api/pricing | ✅ Yes |
| Anthropic | platform.claude.com/docs/pricing | ✅ Yes |
| Google | ai.google.dev/gemini-api/docs/pricing | ✅ Yes |
| Mistral | mistral.ai/pricing | ✅ Yes |
| DeepSeek | api-docs.deepseek.com/quick_start/pricing | ✅ Yes |
| xAI | docs.x.ai/developers/models | ✅ Yes |
| Moonshot/Kimi | platform.moonshot.ai/docs/pricing | ✅ Yes |
| Cohere | cohere.com/pricing | ✅ Yes |
| Qwen | alibabacloud.com/help/en/model-studio/model-pricing | Partial |
| Perplexity | docs.perplexity.ai | ✅ Yes |
| Together AI | together.ai/pricing | ✅ Yes |
| Groq | groq.com/pricing | ✅ Yes |

Models not covered by OpenRouter fall back to the hardcoded values in `_KNOWN_PRICES` and must be updated manually when vendor prices change.
