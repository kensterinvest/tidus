# Pricing Sync

Accurate prices are critical — the routing score weights cost at **70%**. Since v1.1.0, Tidus
has used a multi-source consensus pipeline that creates versioned, audited DB revisions whenever
prices change significantly; v1.2.0 added a recency tie-breaker and a retired-model reconciliation
pass so models removed from `models.yaml` are dropped from new revisions.

## Architecture

```
┌──────────────────┐   ┌─────────────────────────────┐
│  HardcodedSource │   │  TidusPricingFeedSource      │
│  confidence=0.7  │   │  confidence=0.85 (optional)  │
│  41 models       │   │  TIDUS_PRICING_FEED_URL       │
└────────┬─────────┘   └──────────────┬──────────────┘
         │ PriceQuote[]               │ PriceQuote[]
         └──────────┬─────────────────┘
                    ▼
         ┌──────────────────────┐
         │   PriceConsensus     │  MAD-based outlier detection
         │   (per model_id)     │  confidence-weighted selection
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  RegistryPipeline    │  Three-tier validation
         │  run_price_sync_cycle│  Atomic two-phase DB write
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ model_catalog_       │  New ACTIVE revision
         │ revisions (DB)       │  Old revision → SUPERSEDED
         └──────────────────────┘
```

Every Sunday at 02:00 UTC, `TidusScheduler` fires `run_price_sync_cycle()`:

1. All available `PricingSource` instances are queried concurrently
2. `PriceConsensus` resolves quotes using MAD outlier detection (see below)
3. Models where the consensus price differs from the current active revision by ≥ `change_threshold` (5%) are collected
4. If no models changed: exit — no new revision
5. Tier 1 (Pydantic schema) + Tier 2 (cross-field invariants) validation run
6. If validation passes: Phase A write inserts a PENDING revision + all entries
7. Tier 3 canary probe samples up to 3 models with live health checks
8. If canary passes: Phase B atomic flip promotes revision to ACTIVE (old → SUPERSEDED)
9. `EffectiveRegistry` is refreshed, changes become live immediately

## Pricing Sources

### HardcodedSource (`confidence=0.7`)

Built-in price table in `tidus/sync/pricing/hardcoded_source.py`. Covers 41 models across
Anthropic, OpenAI, Google, Mistral, DeepSeek, xAI, Moonshot, Groq, Cohere, Qwen, Together AI.
Always available. Updated manually when vendor prices change.

### TidusPricingFeedSource (`confidence=0.85`)

Optional remote feed. Enabled by setting `TIDUS_PRICING_FEED_URL`. Sends only a single
`GET {url}/prices?schema_version=1` — **no customer data, no messages, no team IDs**.

**Response format:**
```json
{"prices": [{"model_id": "gpt-4o", "input_price": 0.0025, "output_price": 0.01, "updated_at": "2026-04-09", "confidence": 0.9}]}
```

**Signature verification:** If `TIDUS_PRICING_FEED_SIGNING_KEY` is set, the feed response must
include `X-Tidus-Signature: hmac-sha256=<hex>`. Unsigned responses are rejected. If the env var
is not set, unsigned responses are accepted with a `pricing_feed_unsigned` warning in logs.

**Circuit breaker:** After `PRICING_FEED_FAILURE_THRESHOLD` (default 5) consecutive failures,
the circuit opens. No requests are made until `PRICING_FEED_RESET_TIMEOUT_SECONDS` (default 300)
elapses, then one probe request is allowed (HALF-OPEN). On success → CLOSED. On failure → back
to OPEN. State resets on server restart.

### Adding a Custom PricingSource

Subclass `tidus.sync.pricing.base.PricingSource` and implement `fetch_quotes()`:

```python
from tidus.sync.pricing.base import PriceQuote, PricingSource
from datetime import UTC, date, datetime

class MySource(PricingSource):
    @property
    def source_name(self) -> str:
        return "my_source"

    @property
    def confidence(self) -> float:
        return 0.75

    async def fetch_quotes(self) -> list[PriceQuote]:
        # fetch prices from your data source
        return [
            PriceQuote(
                model_id="gpt-4o",
                input_price=0.0025,
                output_price=0.01,
                cache_read_price=0.0,
                cache_write_price=0.0,
                currency="USD",
                effective_date=date.today(),
                retrieved_at=datetime.now(UTC),
                source_name=self.source_name,
                source_confidence=self.confidence,
            )
        ]
```

Then register it in `tidus/sync/scheduler.py` alongside `HardcodedSource`.

## MAD Outlier Detection

When multiple sources provide quotes for the same model, `PriceConsensus` uses the
Modified Z-Score (Median Absolute Deviation) to detect outliers:

```
median_price = median(all input_price quotes)
MAD = median(|price_i − median_price| for each quote)
modified_z_score(i) = 0.6745 × |price_i − median_price| / MAD

Reject quote if modified_z_score > 3.5 (configurable: consensus.outlier_z_threshold)
```

Special cases:
- `MAD == 0` (all sources agree exactly): no rejection
- Single source: accepted but `single_source=True`, effective confidence lowered by 0.2
- All sources rejected: revision fails, alert logged

## Configuring the Sync Schedule

Edit `config/policies.yaml`:

```yaml
pricing_sync:
  day_of_week: 6          # 0=Monday … 6=Sunday
  hour_utc: 2             # 02:00 UTC
  change_threshold: 0.05  # 5% delta triggers a revision
  min_feed_interval_seconds: 3600  # rate guard for TidusPricingFeedSource
```

## Manually Triggering a Sync

```bash
curl -X POST http://localhost:8000/api/v1/sync/prices \
  -H "Authorization: Bearer <admin-token>"
```

**Dry run** (validate + consensus without writing a revision):

```bash
curl -X POST "http://localhost:8000/api/v1/sync/prices?dry_run=true" \
  -H "Authorization: Bearer <admin-token>"
```

Response:

```json
{
  "revision_id": "abc123",
  "changes_detected": 3,
  "sources_used": ["hardcoded", "tidus_feed"],
  "single_source_models": ["kimi-k2.5"],
  "ingestion_run_ids": ["run-1", "run-2"],
  "changes": [
    {
      "model_id": "deepseek-v3",
      "field": "input_price",
      "old_value": 0.00028,
      "new_value": 0.00027,
      "delta_pct": -3.57
    }
  ]
}
```

Note: `delta_pct` is signed — negative values indicate price drops.

## Viewing Price Change History

```bash
curl http://localhost:8000/api/v1/audit/events?action=price_change \
  -H "Authorization: Bearer <admin-token>"
```

The `pricing_ingestion_runs` table records one row per source per sync cycle, including
`raw_payload`, `quotes_valid`, `quotes_rejected`, and `rejection_reasons`.

## Revision Lifecycle

Each successful sync creates a new `model_catalog_revisions` entry:

| Status | Meaning |
|---|---|
| `pending` | Phase A written; validation in progress |
| `validating` | Canary probe running |
| `active` | Current live revision; used by all routing |
| `superseded` | Replaced by a newer revision; retained 90 days |
| `failed` | Validation or canary failed; never promoted |

To roll back to a previous revision:

```bash
curl -X POST http://localhost:8000/api/v1/registry/revisions/{superseded_id}/activate \
  -H "Authorization: Bearer <admin-token>"
```

## Price Sources Reference

| Vendor | Models tracked | Notes |
|--------|---------------|-------|
| Anthropic | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6 | |
| OpenAI | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-5-codex, o3, o4-mini, codex-mini-latest, gpt-oss-120b | |
| Google | gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro, gemini-3.1-flash, gemini-3.1-pro | |
| DeepSeek | deepseek-r1, deepseek-v3, deepseek-v4 | |
| xAI | grok-3, grok-3-fast | |
| Mistral | codestral, devstral, devstral-small, mistral-large-3, mistral-medium, mistral-nemo, mistral-small | |
| Moonshot | kimi-k2.5 | |
| Groq | groq-deepseek-r1, groq-llama4-maverick | |
| Cohere | command-r, command-r-plus | disabled in YAML, prices tracked |
| Qwen | qwen-max, qwen-plus, qwen-flash | disabled in YAML, prices tracked |
| Perplexity | sonar, sonar-pro | disabled in YAML, prices tracked |
| Together AI | together-llama4-maverick | |
