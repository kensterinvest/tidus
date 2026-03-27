# Budgets & Guardrails

Tidus enforces two categories of spend control: **budget policies** (how much a team can spend) and **guardrails** (how far an agent can go per request). Together they implement Pillars 2 and 4 of the five cost-control pillars.

---

## Budget Policies

### How Budgets Work

Every routing request carries a `team_id`. Before a model is selected, the `BudgetEnforcer` checks whether the team's current spend plus the estimated request cost would exceed the configured limit.

```
request arrives (team_id="team-engineering", estimated_cost=$0.005)
    │
    ├─ BudgetEnforcer.can_spend("team-engineering", "$0.005")
    │       ├─ current_spend = $123.45
    │       ├─ limit         = $500.00
    │       ├─ remaining     = $376.55
    │       └─ $0.005 < $376.55 → ALLOW
    │
    └─ model selected, request executes
           │
           └─ BudgetEnforcer.deduct("team-engineering", actual_cost=$0.0048)
```

If `hard_stop=true` and the budget would be exceeded, the request is rejected with HTTP 422 and `rejection_reason: budget_exceeded`. If `hard_stop=false`, the request is allowed but a warning is emitted.

### Configuring Budgets (`config/budgets.yaml`)

```yaml
budgets:
  - policy_id: "team-eng-monthly"
    scope: team
    scope_id: "team-engineering"
    period: monthly
    limit_usd: 500.00
    warn_at_pct: 0.80       # warn when 80% spent
    hard_stop: true         # reject at 100%

  - policy_id: "workflow-onboarding-daily"
    scope: workflow
    scope_id: "onboarding-bot"
    period: daily
    limit_usd: 10.00
    warn_at_pct: 0.90
    hard_stop: false        # warn only, don't block
```

| Field | Type | Description |
|-------|------|-------------|
| `policy_id` | string | Unique identifier |
| `scope` | `team` \| `workflow` | Whether the limit applies to a team or a specific workflow |
| `scope_id` | string | The `team_id` or `workflow_id` this policy covers |
| `period` | `daily` \| `weekly` \| `monthly` \| `rolling_30d` | Budget reset cadence |
| `limit_usd` | float | Maximum USD spend for the period |
| `warn_at_pct` | float | Alert threshold (0.0–1.0, default: 0.80) |
| `hard_stop` | bool | `true` = reject requests; `false` = warn only |

### Checking Budget Status

Via the API:
```bash
curl http://localhost:8000/api/v1/budgets/status/team/team-engineering
```

```json
{
  "team_id": "team-engineering",
  "spent_usd": 123.45,
  "limit_usd": 500.00,
  "utilisation_pct": 24.69,
  "is_hard_stopped": false
}
```

### Per-Request Cost Ceiling

You can also set a per-request ceiling in the route request itself:

```json
{
  "team_id": "team-engineering",
  "max_cost_usd": 0.01,
  ...
}
```

Any candidate model whose estimated cost exceeds `max_cost_usd` is eliminated at stage 4 before scoring.

---

## Guardrails — Pillar 4: Limit Agent Autonomy

Without limits, multi-agent workflows can run away:

```
Agent reflects → delegates → re-plans → re-evaluates → re-summarises
Each hop costs tokens. 5 hops × 2,000 tokens = 10× the expected compute.
```

Tidus enforces hard stops at multiple points in the agent lifecycle.

### Current Limits (`config/policies.yaml`)

```yaml
guardrails:
  max_agent_depth: 5          # maximum recursive agent depth
  max_tokens_per_step: 8000   # maximum tokens in a single step
  max_retries_per_task: 3     # maximum retries before rejection
```

These are enforced at **Stage 2** of the 5-stage selector. Any request that violates these limits is rejected before any model selection or cost estimation occurs.

| Limit | Default | What It Prevents |
|-------|---------|-----------------|
| `max_agent_depth` | 5 | Sub-agents spawning sub-agents indefinitely |
| `max_tokens_per_step` | 8,000 | A single step consuming excessive context |
| `max_retries_per_task` | 3 | Retry loops that multiply cost without progress |

### Roadmap Limits

| Limit | Default | What It Prevents |
|-------|---------|-----------------|
| `max_concurrent_agents` | 10 | Parallel agent fan-out explosion |
| `max_reflection_loops` | 3 | Self-critique spirals (agent critiques its own output repeatedly) |
| `max_total_tokens_session` | 50,000 | Session-level compute cap across all agents in one workflow |

Reflection loops are particularly dangerous: an agent that critiques its own output 10 times burns 10× the tokens for marginal quality improvement. The `max_reflection_loops` limit will cap this at the session level when implemented.

### Agent Sessions

To track agent state across multiple steps, create a session:

```bash
# Start a session
curl -X POST http://localhost:8000/api/v1/guardrails/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-001", "team_id": "team-engineering"}'

# Advance depth (checks all guardrails)
curl -X POST http://localhost:8000/api/v1/guardrails/sessions/advance \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-001", "tokens_used": 1200}'

# Check session state
curl http://localhost:8000/api/v1/guardrails/sessions/sess-001

# Terminate session
curl -X DELETE http://localhost:8000/api/v1/guardrails/sessions/sess-001
```

When `agent_depth` in a route request exceeds `max_agent_depth`, the request is rejected with:
```json
{
  "detail": "No capable model found",
  "failure_stage": 2,
  "failure_reason": "agent_depth_exceeded"
}
```

---

## Why These Limits Matter

Real enterprise cost patterns:

| Without limits | With Tidus guardrails |
|---------------|----------------------|
| Agent depth 15+ common | Hard stop at depth 5 |
| 50K tokens per step | Hard stop at 8K |
| Infinite retries on failure | Max 3 retries |
| Reflection loops of 20+ | Max 3 loops (Phase 4) |
| Unbounded concurrent agents | Max 10 agents (Phase 4) |

A single runaway agent workflow can consume $100+ in a few minutes. Guardrails convert that from a surprise invoice into a predictable, bounded cost.

---

## Interaction Between Budgets and Guardrails

Budgets and guardrails operate at different levels:

| | Budgets | Guardrails |
|--|---------|-----------|
| **Scope** | Team / workflow spend over time | Per-request / per-session depth |
| **When enforced** | Stage 4 (after cost estimation) | Stage 2 (before cost estimation) |
| **What they prevent** | Overspending across many requests | Runaway compute in a single workflow |
| **Config location** | `config/budgets.yaml` | `config/policies.yaml` |

Both are enforced before any vendor API call is made. Tidus never spends money and then checks — it checks first, then spends.
