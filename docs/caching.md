# Caching — Pillar 3: Cache Everything

Response caching is the third cost-control pillar. After Pillars 1+2 (tiered model selection + smart routing) reduce costs by 87–93%, caching eliminates duplicate compute entirely — the cheapest model call is the one you never make.

---

## Why Caching Matters

Enterprise AI workloads have high repetition:
- 10 users asking the same onboarding question pay 10× instead of 1×
- A recurring report generation workflow re-processes the same documents daily
- A classification pipeline runs the same prompt template on similar inputs thousands of times

**Expected reduction from caching: 30–50%** on top of Pillars 1+2.

---

## Cache Layers

### Layer 1 — Exact Match Cache

Caches responses by a hash of `(team_id, messages, model_id)`.

```
hash_key = SHA256(team_id + json(messages) + model_id)
if cache.get(hash_key):
    return cached_response  # zero cost
else:
    execute → store → return
```

- Zero false positives — exact same input always returns the exact same output
- Very high hit rate for repeated templates (classification, extraction, FAQ)
- TTL: 1 hour (configurable)
- Team-scoped: responses from `team-engineering` are never returned to `team-finance`

### Layer 2 — Semantic Cache

Caches responses by embedding similarity. Two prompts that mean the same thing return the same cached response.

```
embedding = embed(messages)
nearest, similarity = vector_store.nearest(embedding)
if similarity >= THRESHOLD (default: 0.95):
    return nearest.response
else:
    execute → store embedding + response → return
```

- Catches "same question, different phrasing" without exact-match overhead
- Uses `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no extra API cost)
- Threshold tunable: higher = stricter (fewer false hits), lower = more aggressive caching
- TTL: 15 minutes (shorter than exact — semantic similarity is approximate)
- Graceful no-op if `sentence-transformers` is not installed

### Layer 3 — Workflow Cache (Roadmap)

Caches sub-results within a multi-step agent workflow.

```
workflow_step_key = hash(workflow_id + step_name + input_hash)
if cache.get(workflow_step_key):
    skip step → use cached sub-result
```

- Prevents re-running completed steps when an agent retries after partial failure
- Essential for long document processing pipelines
- Planned for a future release

---

## Privacy & Cache Isolation

Cache keys always include `team_id`. A cached response from `team-engineering` is never returned to `team-finance`, even for identical prompts. Tasks with `privacy: "confidential"` are never cached — they are always executed fresh to prevent cross-contamination of sensitive data.

---

## Cache Backends

| Backend | Use Case | Configuration |
|---------|---------|--------------|
| In-memory dict | Development, single-process | Default — no configuration required |
| Redis | Production, multi-instance | `REDIS_URL=redis://localhost:6379/0` (roadmap) |

The in-memory backend is the default and works for single-server deployments. Redis support for multi-instance horizontal scaling is on the roadmap.

---

## Configuration

Cache behaviour is controlled via environment variables in `.env`:

```env
CACHE_EXACT_TTL_SECONDS=3600        # Exact cache TTL (default: 1 hour)
CACHE_SEMANTIC_TTL_SECONDS=900      # Semantic cache TTL (default: 15 minutes)
SEMANTIC_CACHE_THRESHOLD=0.95       # Cosine similarity threshold (default: 0.95)
```

And via `config/policies.yaml`:

```yaml
cache:
  enabled: true
  max_cached_response_tokens: 4000  # don't cache very long responses
  excluded_domains:                 # never cache these (real-time data, compliance)
    - creative
```

---

## Cache Hit Rate in the Dashboard

The dashboard at `/dashboard/` shows cache performance:
- Exact cache hits vs. misses (7-day trend)
- Semantic cache hits vs. misses
- Estimated cost avoided by caching (in USD)
- Current cache entry count

---

## Expected ROI from Caching

For a 500-user enterprise with Pillars 1+2 already saving 87%:

| Without Caching | With Caching |
|----------------|-------------|
| 3,000,000 API calls/month | ~1,500,000 API calls/month (50% hit rate) |
| $840/month AI cost | ~$420/month AI cost |
| — | Additional $420/month saving |

Caching is most effective for:
- FAQ and support bots (high query repetition)
- Document classification pipelines (same template, many documents)
- Recurring report generation (same prompt, same data)
- Agent workflows with retry logic (exact cache re-uses prior step outputs)
