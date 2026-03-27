# Architecture

Tidus is built around **five cost-control pillars**. Each pillar is a discrete system that reduces enterprise AI spend independently — together they achieve 70–96% cost reduction.

## The Five Pillars

```
Pillar 1 — Tiered Model Strategy      LIVE
  High-end (tier 1) : planning, reasoning, compliance, critical decisions
  Mid-tier  (tier 2) : summarisation, extraction, code, routine tasks
  Economy   (tier 3) : classification, filtering, pre-processing
  Local     (tier 4) : simple logic, on-prem, zero API cost

Pillar 2 — Router Agent Intelligence   LIVE
  5-stage selector decides: which model, local vs. cloud,
  multi-agent need, budget feasibility — before any compute runs

Pillar 3 — Cache Everything            LIVE
  Exact-match cache  : hash(messages+model) → stored response
  Semantic cache     : embed query → cosine similarity → threshold
  Expected reduction : 30–50% on top of Pillars 1+2

Pillar 4 — Agent Autonomy Limits       LIVE
  max_agent_depth        : 5        (live)
  max_tokens_per_step    : 8,000    (live)
  max_retries_per_task   : 3        (live)

Pillar 5 — Vendor-Agnostic Design      LIVE
  @register_adapter pattern : add vendor in one file
  8 built-in adapters       : OpenAI, Anthropic, Google, Mistral,
                              DeepSeek, xAI, Moonshot, Ollama
  MCP server (tidus-mcp)    : Claude Desktop, Cursor, any MCP client
  Open model support        : Llama 4, Mistral, Gemma, Phi-4 via Ollama
```

---

## Request Lifecycle

```
AI Client (app / agent / MCP tool)
          │
          │  POST /api/v1/route  or  POST /api/v1/complete
          ▼
┌──────────────────────────────────────────────────────────────┐
│                    Tidus Router Service                        │
│                                                              │
│  Cache Check ──────── (Phase 4) exact + semantic hit?        │
│       │ miss                                                  │
│       ▼                                                      │
│  Stage 1: Hard Constraints                                    │
│    enabled, context fits, domain supported,                   │
│    confidential → local only,                                 │
│    complexity within model's min/max range                    │
│       │                                                      │
│  Stage 2: Guardrails                                          │
│    agent_depth, tokens_per_step, retry_count                  │
│       │                                                      │
│  Stage 3: Complexity Tier Ceiling                             │
│    simple→any  moderate→≤3  complex→≤2  critical→1           │
│       │                                                      │
│  Stage 4: Budget Filter                                       │
│    can_spend(team_id, estimated_cost)?                        │
│       │                                                      │
│  Stage 5: Score & Select                                      │
│    score = cost×0.70 + tier×0.20 + latency×0.10              │
│    pick minimum                                               │
│       │                                                      │
│  /route returns RoutingDecision here ─────────────────────►  │
│       │                                                      │
│  /complete continues:                                         │
│    AdapterFactory → get_adapter(vendor)                       │
│    Adapter.complete(model, task) → vendor API call            │
│    CostLogger → write CostRecord to DB                        │
│    BudgetEnforcer.deduct(actual_cost)                         │
│    CacheStore.set(key, response)  ── (Phase 4)                │
└──────────────────────────────────────────────────────────────┘
          │
          │  TaskResult {model_id, cost_usd, tokens, response}
          ▼
      AI Client
```

---

## Components

### Model Selection Engine (`tidus/router/selector.py`)

Five sequential stages — the first failure produces a typed `RejectionReason` with full observability:

| Stage | Check | Rejection Reason |
|-------|-------|-----------------|
| 1a | `enabled=True`, `deprecated=False` | `model_disabled` |
| 1b | `estimated_input_tokens <= max_context` | `context_too_large` |
| 1c | required domain capability present | `domain_not_supported` |
| 1d | `confidential` task → `is_local=True` only | `privacy_violation` |
| 1e | task complexity within `[min_complexity, max_complexity]` | `complexity_mismatch` |
| 2a | `agent_depth <= max_agent_depth` | `agent_depth_exceeded` |
| 2b | `estimated_input_tokens <= max_tokens_per_step` | `token_limit_exceeded` |
| 3 | tier <= complexity ceiling | `complexity_ceiling` |
| 4 | estimated cost <= remaining budget | `budget_exceeded` |
| 5 | score = cost×0.70 + tier×0.20 + latency×0.10 | — (select minimum) |

The `complexity_mismatch` check (stage 1e) is critical: it ensures that models designed for
`complex`/`critical` work (e.g. `o3`, `deepseek-r1`, `grok-3`) cannot win `simple` tasks through
scoring alone, and that economy models cannot be selected for tasks that exceed their quality range.

### Vendor Abstraction Layer (`tidus/adapters/`) — Phase 4

Every vendor implements `AbstractModelAdapter`:
```python
async def complete(model, task) -> AdapterResponse
async def stream_complete(model, task) -> AsyncIterator[str]
async def health_check() -> bool
async def count_tokens(model_id, messages) -> int
```

Adding a new vendor = one file + `@register_adapter` decorator + YAML entry. Zero changes to router or API.

| Adapter | Vendor | Auth | Token Counting |
|---------|--------|------|---------------|
| `anthropic_adapter.py` | Claude family | `ANTHROPIC_API_KEY` | Anthropic count_tokens endpoint |
| `openai_adapter.py` | GPT / o3 / Codex | `OPENAI_API_KEY` | tiktoken (local, fast) |
| `google_adapter.py` | Gemini family | `GOOGLE_API_KEY` | google-generativeai SDK |
| `mistral_adapter.py` | Mistral / Codestral / Devstral | `MISTRAL_API_KEY` | sentencepiece |
| `deepseek_adapter.py` | DeepSeek R1/V3 | `DEEPSEEK_API_KEY` | tiktoken cl100k |
| `xai_adapter.py` | Grok 3 | `XAI_API_KEY` | tiktoken cl100k |
| `moonshot_adapter.py` | Kimi K2.5 | `MOONSHOT_API_KEY` | tiktoken cl100k |
| `ollama_adapter.py` | Local models | none | Ollama tokenize endpoint |

### Cost Engine (`tidus/cost/`)

- Provider-native token counting — avoids 10–20% estimation error vs. character-based counting
- 15% safety buffer on all estimates: `buffered = raw * 1.15`
- Actual token counts logged post-call; drift tracked in DB

### Budget Enforcer (`tidus/budget/`)

- `can_spend(team_id, workflow_id, usd)` — atomic check against in-process counters
- `deduct(team_id, workflow_id, usd)` — atomic decrement after actual cost is known
- Periods: `daily`, `weekly`, `monthly`, `rolling_30d`
- Hard stop: reject request if budget would be exceeded
- Warn-only: allow but emit alert at `warn_at_pct` threshold (default 80%)

### Response Cache (`tidus/cache/`) — Phase 4

```
Request arrives
    │
    ├─► exact_cache.get(hash(team+messages+model)) → HIT: return stored response
    │
    ├─► semantic_cache.get(embed(messages), threshold=0.95) → HIT: return nearest match
    │
    └─► MISS: proceed to selector → execute → store in both caches
```

Cache backends: in-memory dict (dev) → Redis (`REDIS_URL` env var) for production.
Cache TTL: configurable per team/domain (default 1 hour for exact, 15 min for semantic).

### Agent Guardrails (`tidus/guardrails/`) — Pillar 4

Current limits (enforced in Stage 2):

| Limit | Default | Purpose |
|-------|---------|---------|
| `max_agent_depth` | 5 | Prevents recursive agent explosion |
| `max_tokens_per_step` | 8,000 | Caps per-call token spend |
| `max_retries_per_task` | 3 | Limits retry-driven cost amplification |

Planned (Phase 4):

| Limit | Default | Purpose |
|-------|---------|---------|
| `max_concurrent_agents` | 10 | Prevents fan-out explosion in parallel workflows |
| `max_reflection_loops` | 3 | Stops self-critique spirals (10× token multiplier risk) |
| `max_total_tokens_session` | 50,000 | Session-level hard cap across all agents |

### Health & Sync (`tidus/sync/`) — Phase 4

- **Health probe** every 5 minutes: 10-token test prompt per model, records latency.
  Three consecutive failures → `enabled=False` + alert. Updates `latency_p50_ms` (rolling P50).
- **Price sync** weekly (Sunday 02:00 UTC): compares current vendor pricing against YAML.
  Delta >5% → update YAML + write `PriceChangeRecord` to DB + emit alert.

### MCP Server (`tidus/mcp/`) — Phase 6

Exposes Tidus as MCP tools for Claude Desktop, Cursor, and any A2A-compatible agent:
- `tidus_route_task` — select best model for a task
- `tidus_complete_task` — route + execute
- `tidus_get_budget_status` — current spend for a team
- `tidus_list_models` — available models with health status

Transport: stdio (local) or SSE (remote).

---

## Data Flow

```
config/models.yaml  ──► ModelRegistry  (in-memory, loaded at startup)
config/budgets.yaml ──► BudgetPolicies (in-memory + DB)
config/policies.yaml ──► GuardrailPolicy + RoutingWeights

TaskDescriptor
    │
    ├─► CacheStore ──────────────────────────────► HIT: CachedResponse
    │       │ miss
    ├─► ModelSelector
    │       ├─► CapabilityMatcher ──► eligible models
    │       ├─► BudgetEnforcer    ──► cost-feasible models
    │       └─► Scorer            ──► RoutingDecision
    │
    │   (/route endpoint returns here)
    │
    │   (/complete continues)
    ├─► AdapterFactory ──► VendorAdapter ──► Vendor API
    │                               │
    ├─► CostLogger ─────────────────┘──► CostRecord (DB)
    ├─► BudgetEnforcer.deduct(actual_cost)
    └─► CacheStore.set(key, response)
```

## Database Schema

| Table | Purpose |
|-------|---------|
| `cost_records` | Every executed task with actual token counts and cost |
| `routing_decisions` | Every routing decision (accepted or rejected) with rejection reason |
| `budget_policies` | Current budget policy configuration |
| `price_change_log` | Audit trail of all vendor price changes detected by sync job |
