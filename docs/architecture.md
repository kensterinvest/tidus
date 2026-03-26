# Architecture

## Request Lifecycle

```
AI Client (app / agent / MCP tool)
          │
          │  POST /api/v1/complete  {TaskDescriptor}
          ▼
┌─────────────────────────────────────────────────────┐
│                  Tidus Router Service                │
│                                                     │
│  1. GuardrailCheck  ─── agent depth / tokens / retries
│  2. ModelSelector   ─── 5-stage filtering + scoring │
│  3. BudgetEnforcer  ─── can_spend(team, workflow)   │
│  4. CostEngine      ─── estimate with 15% buffer    │
│  5. AdapterFactory  ─── get_adapter(vendor)         │
│  6. Adapter.complete ── call vendor API             │
│  7. CostLogger      ─── write CostRecord to DB      │
│  8. BudgetDeduct    ─── deduct actual cost          │
└─────────────────────────────────────────────────────┘
          │
          │  TaskResult  {model_id, cost_usd, response}
          ▼
      AI Client

```

## Components

### Model Selection Engine (`tidus/router/selector.py`)
Five sequential filtering stages — the first failure rejects with a typed reason:
1. **Hard constraints** — enabled, context fits, domain supported, privacy/local check
2. **Guardrails** — depth, tokens-per-step, retry count
3. **Complexity tier ceiling** — simple→any, moderate→tier≤3, complex→tier≤2, critical→tier 1
4. **Budget filter** — cost estimate vs. remaining budget
5. **Score & select** — `cost×0.70 + tier×0.20 + latency×0.10`

### Vendor Abstraction Layer (`tidus/adapters/`)
Every vendor implements `AbstractModelAdapter`. The factory maps `vendor` string → adapter class.
Adding a new vendor = one file + `@register_adapter` + YAML entry.

### Cost Engine (`tidus/cost/`)
- Provider-native token counting (tiktoken, Anthropic API, Google SDK, sentencepiece, Ollama)
- 15% safety buffer on all estimates
- Actual token counts logged post-call

### Budget Enforcer (`tidus/budget/`)
- `can_spend(team_id, workflow_id, estimated_usd)` → allowed/rejected
- `deduct(team_id, workflow_id, actual_usd)` → atomic counter update
- Periods: daily, weekly, monthly, rolling_30d

### Health & Sync (`tidus/sync/`)
- Health probe every 5 min — 3 failures → auto-disable
- Weekly price sync — detects >5% price change → updates YAML + DB log

### MCP Server (`tidus/mcp/`)
Exposes Tidus as MCP tools consumable by Claude Desktop, Cursor, and any A2A agent.

## Data Flow Diagram

```
config/models.yaml ──► ModelRegistry (in-memory)
                              │
config/budgets.yaml ──► BudgetPolicies (DB)
config/policies.yaml ──► GuardrailPolicy + RoutingWeights

TaskDescriptor ──► Selector ──► RoutingDecision ──► Adapter ──► Vendor API
                      │                                │
                  BudgetEnforcer               CostRecord (DB)
                      │
                  SessionStore (in-memory)
```

## Database Schema

| Table | Purpose |
|-------|---------|
| `cost_records` | Every executed task with actual token counts and cost |
| `routing_decisions` | Every routing decision (accepted or rejected) |
| `budget_policies` | Current budget policy configuration |
| `price_change_log` | Audit trail of all vendor price changes detected |
