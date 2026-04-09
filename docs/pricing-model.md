# How Tidus Prices AI Requests — A Complete Guide

> **Audience:** Engineers, finance teams, and operators who want to understand exactly how Tidus
> obtains, validates, and applies AI model pricing — from raw vendor data to a per-request cost
> estimate in dollars.

---

## The Problem With Hardcoded Prices

AI vendor pricing changes frequently and without warning. In 2026 alone, Qwen cut prices by 90%,
Google halved Gemini Flash pricing twice, and OpenAI launched three new model tiers at different
price points than their predecessors. A system that ships prices in source code goes stale within
days and can systematically over-charge (routing to expensive models when cheap ones are
available) or under-charge (creating budget leakage against provider invoices).

Tidus solves this with a **multi-source, self-healing pricing registry** that:

1. Ingests prices from one or more sources
2. Validates them with statistical outlier detection (MAD)
3. Writes audited, versioned revisions to the database
4. Detects drift between declared prices and provider invoices
5. Exposes metrics when prices become stale

---

## Price Units

All prices in Tidus are stored in **dollars per 1,000 tokens ($/1K tokens)**:

```
input_price  = 0.003   →   $3.00 per 1M tokens
output_price = 0.015   →   $15.00 per 1M tokens
```

To convert: `$/1M = stored_value × 1000`.

Four price fields exist per model:

| Field | Meaning |
|---|---|
| `input_price` | Cost per 1K input (prompt) tokens |
| `output_price` | Cost per 1K output (completion) tokens |
| `cache_read_price` | Cost per 1K tokens read from prompt cache (0.0 if unsupported) |
| `cache_write_price` | Cost per 1K tokens written to prompt cache (0.0 if unsupported) |

Local (Ollama) models always have all four fields set to `0.0`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Pricing Sources                                   │
│                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │  HardcodedSource     │    │  TidusPricingFeedSource           │  │
│  │  confidence = 0.7    │    │  confidence = 0.85                │  │
│  │  (always available)  │    │  (optional, needs FEED_URL)       │  │
│  └──────────┬───────────┘    └─────────────────┬────────────────┘  │
│             └──────────────┬──────────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
                             │ list[PriceQuote]
                   ┌─────────▼──────────────┐
                   │   PriceConsensus       │
                   │   MAD outlier filter   │
                   │   confidence weighting │
                   └─────────┬──────────────┘
                             │ consensus quotes
                   ┌─────────▼──────────────┐
                   │   RegistryPipeline     │
                   │   Tier 1: schema valid │
                   │   Tier 2: invariants   │
                   │   Tier 3: canary probe │
                   │   Phase A + Phase B    │
                   └─────────┬──────────────┘
                             │ new active revision
                   ┌─────────▼──────────────┐
                   │  model_catalog_entries  │
                   │  (active revision)      │
                   └─────────┬──────────────┘
                             │ read at routing time
                   ┌─────────▼──────────────┐
                   │   CostEngine.estimate() │
                   │   per-request USD cost  │
                   └─────────────────────────┘
```

---

## Step 1 — Price Sources

### HardcodedSource (confidence = 0.7)

**File:** `tidus/sync/pricing/hardcoded_source.py`

The built-in source wraps a hand-maintained `_KNOWN_PRICES` dictionary. It is always available
(`is_available = True`) and serves as the fallback when no external feed is configured.

```python
_KNOWN_PRICES = {
    "claude-haiku-4-5":  {"input": 0.0008,  "output": 0.004},    # $0.80/$4 per 1M
    "claude-opus-4-6":   {"input": 0.015,   "output": 0.075},    # $15/$75 per 1M
    "gpt-4.1-mini":      {"input": 0.0004,  "output": 0.0016},   # $0.40/$1.60 per 1M
    # ... 22 models total
}
```

Confidence of **0.7** reflects that these prices are manually verified and may lag vendor changes
by days-to-weeks. The `_EFFECTIVE_DATE` field records when the table was last verified.

**The script `tidus/sync/pricing/hardcoded_source.py` is updated before each release** — the
comment at the top records the verification date and sources checked (official vendor pricing
pages, not OpenRouter or aggregators).

### TidusPricingFeedSource (confidence = 0.85)

**File:** `tidus/sync/pricing/feed_source.py`

An optional HTTP source that pulls from a hosted pricing feed endpoint. Enabled by setting
`TIDUS_PRICING_FEED_URL` in the environment. The feed returns only pricing data — no customer
data, no prompts, no team IDs are ever sent to the feed.

```
GET {TIDUS_PRICING_FEED_URL}/prices?schema_version=1
→ {"prices": [{model_id, input_price, output_price, updated_at, confidence}]}
```

Confidence of **0.85** reflects that a hosted feed is updated more frequently than a manual table.

**Security:** Feed responses must include `X-Tidus-Signature: hmac-sha256=<hex>`. If
`TIDUS_PRICING_FEED_SIGNING_KEY` is set, Tidus rejects unsigned or tampered responses. If the
key is not set, Tidus accepts unsigned responses but logs `pricing_feed_unsigned` as a warning.

**Circuit breaker:** After 5 consecutive failures (`PRICING_FEED_FAILURE_THRESHOLD`), the feed
source opens its circuit and returns empty quotes for 5 minutes before re-trying. This prevents
a flaky feed from blocking the sync pipeline indefinitely.

**Rate guard:** The feed is called at most once per hour regardless of how often the scheduler
fires, preventing accidental DDoS of the feed endpoint.

### Adding a Custom Source

To add a new pricing source (e.g., pulling from OpenRouter's API), implement the `PricingSource`
abstract base class:

```python
class MyCustomSource(PricingSource):
    @property
    def source_name(self) -> str:
        return "my-custom-source"

    @property
    def confidence(self) -> float:
        return 0.80   # how much to trust this source (0–1)

    @property
    def is_available(self) -> bool:
        return bool(os.getenv("MY_FEED_URL"))   # only activate if configured

    async def fetch_quotes(self) -> list[PriceQuote]:
        # fetch and return PriceQuote objects
        ...
```

Then register it in `tidus/sync/scheduler.py` → `_run_price_sync()`.

---

## Step 2 — Consensus (MAD Outlier Detection)

**File:** `tidus/sync/pricing/consensus.py`

When multiple sources provide quotes for the same model, Tidus cannot blindly average them — one
source could report a price that is significantly wrong due to a data pipeline error, stale cache,
or malicious tampering. The **Modified Z-Score** method detects statistical outliers.

### The Algorithm

For each model with N quotes from different sources:

```
1. median_price = median of all source prices for this model

2. MAD = median(|price_i − median_price| for each source i)
         (MAD = Median Absolute Deviation)

3. modified_z_score_i = 0.6745 × |price_i − median_price| / MAD

4. Reject source i if modified_z_score_i > 3.5
   (threshold configurable via consensus.outlier_z_threshold)
```

### Why MAD instead of standard deviation?

Standard deviation is sensitive to the outliers it is trying to detect — one extreme value inflates
the deviation, making it harder to flag that same value. MAD uses the median of deviations, which
is robust: a single extreme outlier does not change the MAD.

**Example:**

Three sources report input prices for `claude-opus-4-6`:
- HardcodedSource: $15.00/1M (= 0.015 stored)
- FeedSource A: $15.00/1M
- FeedSource B: $1.50/1M ← potential error (10× off)

```
median = $15.00
MAD = median(|15-15|, |15-15|, |1.5-15|) = median(0, 0, 13.5) = 0
```

When MAD = 0 (all but one source agree exactly), the outlier is flagged as infinite z-score →
rejected. The consensus price is $15.00 from the two agreeing sources.

**Legitimate large price changes** — when a vendor cuts prices by 50%, both sources will update
and the MAD will reflect the genuine change. A single source dropping while others remain is
what triggers outlier detection.

### Special Cases

| Situation | Behaviour |
|---|---|
| Only one source has data for a model | Accepted; confidence lowered by 0.2; `single_source=True` flag set |
| All sources agree exactly (MAD = 0) | All accepted — no rejections possible |
| All sources rejected as outliers | Revision rejected entirely; alert logged |
| Two sources disagree without a clear majority | Higher-confidence source wins |

### Source Confidence Weighting

When multiple non-outlier sources provide different prices (within acceptable range), the
source with higher confidence wins:

```
HardcodedSource: confidence = 0.7
FeedSource:      confidence = 0.85   ← wins if both are within threshold
```

---

## Step 3 — The Registry Pipeline

**File:** `tidus/registry/pipeline.py`

The pipeline is the transactional heart of the pricing system. It runs on a **weekly schedule**
(Sunday 02:00 UTC by default) and produces a new database revision if prices have changed.

### Pipeline Steps

```
1. Acquire distributed lock (PostgreSQL advisory lock)
   → Only one replica runs the sync in a k8s deployment

2. Clean up stale PENDING revisions (> 1 hour old)

3. Ingest from all available sources concurrently
   → Each source writes a pricing_ingestion_runs row (full audit trail)

4. Consensus: MAD outlier detection across all source quotes

5. Normalize: compare consensus prices to current active revision
   → Only models with ≥ 5% price change are included in the new revision
   → Models with < 5% change keep their existing prices

6. If no changes: return None (no revision created)

7. Tier 1 validation — Pydantic schema check
   (prices ≥ 0, context_window > 0, tier in 1–4, etc.)

8. Tier 2 validation — Cross-field invariants
   (local models must have price = 0, min_complexity ≤ max_complexity, etc.)

9. If dry_run=True: return DryRunResult without writing to DB

10. Phase A write: INSERT revision (status=PENDING) + all entries
    → Atomic transaction; other replicas still see old ACTIVE revision

11. Tier 3 canary probe: send a real request to 3 randomly sampled models
    → Up to 3 retries per model; revision passes if ≥ 67% of models pass
    → Stores results in canary_results JSON column for audit

12. If canary fails: set revision status=FAILED; old revision stays ACTIVE

13. Phase B flip (atomic): old ACTIVE → SUPERSEDED, new PENDING → ACTIVE
    → Two-row UPDATE in one transaction; zero downtime

14. Write PriceChangeRecord rows (backward compatibility)

15. Trigger EffectiveRegistry.refresh()

16. Update Prometheus metrics

17. Release advisory lock
```

### The Two-Phase Write Protocol

The reason Tidus uses two separate transactions (Phase A and Phase B) is to prevent routers from
reading a half-written revision:

```
Phase A:  INSERT revision (status=PENDING) + INSERT all 53 entries
           ↑ Routers never read PENDING revisions — they are invisible

Phase B:  BEGIN
          UPDATE status='superseded' WHERE status='active'
          UPDATE status='active' WHERE revision_id=$new
          COMMIT
           ↑ A single atomic 2-row UPDATE — either both succeed or neither does
```

If the server crashes between Phase A and Phase B, the PENDING revision is cleaned up by
`_cleanup_stale_pending()` on the next run.

### Revision States

```
PENDING → (validation passes) → VALIDATING
PENDING → (Tier 3 canary) → ACTIVE  (Phase B flip)
ACTIVE  → (new revision promoted) → SUPERSEDED
PENDING/VALIDATING → (any failure) → FAILED

SUPERSEDED revisions are retained for 90 days (REGISTRY_REVISION_RETENTION_DAYS).
FAILED revisions are never auto-retried.
Admin can force-activate a SUPERSEDED revision (skips Tier 3 only).
```

---

## Step 4 — Cost Estimation at Request Time

**File:** `tidus/cost/engine.py`

When a routing request arrives, `CostEngine.estimate()` computes the expected cost using prices
from the current ACTIVE revision:

```python
cost = (input_tokens × spec.input_price + output_tokens × spec.output_price) / 1000
     × (1 + estimate_buffer_pct)   # default 15% safety buffer
```

The buffer accounts for:
- **Tokenization variance**: different tokenizers count tokens differently; the caller's estimate
  may not match the model's actual tokenizer.
- **Output length uncertainty**: the model may produce more tokens than the caller estimated.

If prompt caching is used:
```python
cost += cache_read_tokens × spec.cache_read_price / 1000
cost += cache_write_tokens × spec.cache_write_price / 1000
```

The estimate is used in **Stage 4 (budget filtering)** and is stored in `routing_decisions` for
later comparison against actual cost.

---

## Step 5 — Price Multiplier Overrides

The three-layer registry merge allows price adjustments without touching the base catalog:

```python
# Create a price_multiplier override for team A (e.g., they negotiated a 20% discount)
POST /api/v1/registry/overrides
{
    "override_type": "price_multiplier",
    "scope": "team",
    "scope_id": "team-a",
    "payload": {"multiplier": 0.80},
    "justification": "Negotiated enterprise discount with Anthropic"
}
```

At routing time, `merge.py` applies the multiplier to all four price fields:

```python
def apply_price_multiplier(spec: ModelSpec, multiplier: float) -> ModelSpec:
    return spec.model_copy(update={
        "input_price":       spec.input_price * multiplier,
        "output_price":      spec.output_price * multiplier,
        "cache_read_price":  spec.cache_read_price * multiplier,
        "cache_write_price": spec.cache_write_price * multiplier,
    })
```

This affects **cost estimates** (Stage 4) and therefore affects which models survive budget
filtering and how they score in Stage 5.

---

## Step 6 — Billing Reconciliation

**File:** `tidus/billing/reconciler.py`

After the fact, enterprises can upload their actual provider invoices and compare them to what
Tidus estimated. This catches **systematic pricing errors** (e.g., a model price was wrong for
a week before the sync caught it) and **leakage** (calls made outside Tidus).

### Normalized CSV Format

```csv
model_id,date,provider_cost_usd
claude-opus-4-6,2026-04-01,127.45
gpt-4o,2026-04-01,89.20
```

### Variance Calculation

```
tidus_cost    = SUM(cost_records.cost_usd) for this model + date range
provider_cost = from CSV row
variance_pct  = (tidus_cost - provider_cost) / provider_cost

Status thresholds:
  matched:  |variance_pct| ≤ 5%
  warning:  5% < |variance_pct| ≤ 25%
  critical: |variance_pct| > 25%
```

A `critical` variance means Tidus and the provider disagree by more than 25% — likely indicating
a stale price or missed model in the catalog.

---

## Observability

Tidus exposes the following Prometheus metrics for pricing health:

| Metric | Type | Description |
|---|---|---|
| `tidus_registry_last_successful_sync_timestamp` | Gauge | Unix timestamp of last successful sync |
| `tidus_registry_model_last_price_update_timestamp` | Gauge | Per-model last price update time |
| `tidus_registry_model_confidence` | Gauge | Per-model confidence score |
| `tidus_registry_models_stale_count` | Gauge | Models with price data older than 8 days |
| `tidus_registry_drift_events_total` | Counter | Drift events by model and type |

### Staleness Alerts

```yaml
# Fires if sync hasn't run in 2 days:
alert: TidusRegistrySyncStale
expr: (time() - tidus_registry_last_successful_sync_timestamp) > 172800

# Fires if > 10 models have stale prices (>8 days old):
alert: TidusRegistryStaleModelCount
expr: tidus_registry_models_stale_count > 10
```

---

## Frequently Asked Questions

**Q: How often do prices update?**  
The weekly sync runs every Sunday at 02:00 UTC. Price changes smaller than 5% are ignored to
prevent noise. A new revision is only created if at least one model changed by ≥ 5%.

**Q: What if a vendor announces a price change mid-week?**  
The `POST /api/v1/sync/prices` endpoint triggers an immediate sync. The weekly job is a safety
net; any operator can trigger an ad-hoc sync at any time.

**Q: Models that are in the YAML but not in the hardcoded source — what price do they use?**  
Their price from the YAML seed (the initial `seed-v0` revision) is preserved. The pipeline only
updates prices for models that have a matching entry in the consensus output. Unknown models keep
their last-known price.

**Q: Why is the hardcoded source confidence 0.7 and not higher?**  
0.7 reflects that the table is manually verified and may be up to 14 days stale between releases.
A live feed with programmatic verification warrants 0.85. 0.7 is honest — it's good data, but
not real-time.

**Q: Can prices go negative or to zero through a price_multiplier override?**  
No. The merge layer clamps prices to `max(0, computed_price)`. A multiplier of 0.0 would make a
model appear free, which is allowed (useful for internal models where you absorb cost centrally).
A negative multiplier is invalid and rejected at override creation time.

**Q: What prevents a bad feed from poisoning prices?**  
Three layers of defense:
1. MAD outlier detection rejects quotes that deviate from the consensus
2. Tier 1 + Tier 2 validation rejects malformed or invariant-violating specs
3. HMAC-SHA256 signature verification rejects tampered feed responses

---

## Price Data Lineage

For any model at any point in time, you can trace the price to its origin:

```
GET /api/v1/registry/revisions/{revision_id}
→ {status, activated_at, source, canary_results, ...}

For each entry:
GET /api/v1/registry/revisions/{revision_id}/diff?base={prev_revision_id}
→ {model_id, changed_fields: {input_price: {from: 0.005, to: 0.003}}}

For source provenance:
SELECT * FROM pricing_ingestion_runs WHERE revision_id_created = '{revision_id}'
→ {source_name, status, quotes_valid, quotes_rejected, raw_payload}

For billing verification:
GET /api/v1/billing/reconciliations?date_from=2026-04-01&date_to=2026-04-30
→ [{model_id, tidus_cost_usd, provider_cost_usd, variance_pct, status}]
```

Every price in production has a full audit chain from source quote → consensus → revision → cost
estimate → actual spend → invoice reconciliation.
