# API Reference

All endpoints are available at `http://localhost:8000` by default. Interactive Swagger UI is at `/docs`.

---

## Routing

### POST /api/v1/route

Select the best model for a task without executing it. Returns a routing decision.

**Request body:**

```json
{
  "team_id": "team-engineering",
  "complexity": "simple",
  "domain": "chat",
  "estimated_input_tokens": 500,
  "messages": [{"role": "user", "content": "Hello"}],

  "privacy": "public",
  "estimated_output_tokens": 256,
  "agent_depth": 0,
  "preferred_model_id": null,
  "max_cost_usd": null,
  "workflow_id": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `team_id` | string | Yes | Team identifier for budget tracking |
| `complexity` | enum | Yes | `simple` \| `moderate` \| `complex` \| `critical` |
| `domain` | enum | Yes | `chat` \| `code` \| `reasoning` \| `extraction` \| `classification` \| `summarization` \| `creative` |
| `estimated_input_tokens` | int | Yes | Estimated prompt token count |
| `messages` | array | Yes | OpenAI message format `[{"role": "user", "content": "..."}]` |
| `privacy` | enum | No | `public` (default) \| `internal` \| `confidential` |
| `estimated_output_tokens` | int | No | Expected output length (default: 256) |
| `agent_depth` | int | No | Current recursive agent depth (default: 0) |
| `preferred_model_id` | string | No | Preferred model; used if eligible |
| `max_cost_usd` | float | No | Per-request cost ceiling in USD |
| `workflow_id` | string | No | Workflow identifier for per-workflow budgets |

**Response 200 — accepted:**

```json
{
  "task_id": "3f8a2c1d-...",
  "accepted": true,
  "chosen_model_id": "deepseek-v3",
  "estimated_cost_usd": 0.000056,
  "score": 0.096
}
```

**Response 422 — no capable model:**

```json
{
  "detail": "No capable model found",
  "failure_stage": 3,
  "failure_reason": "complexity_ceiling",
  "rejections": [
    {"model_id": "phi-4-ollama", "reason": "complexity_mismatch"},
    {"model_id": "gemini-nano", "reason": "complexity_mismatch"}
  ]
}
```

**Rejection reasons:**

| Reason | Stage | Cause |
|--------|-------|-------|
| `model_disabled` | 1 | Model has `enabled=false` or `deprecated=true` |
| `context_too_large` | 1 | `estimated_input_tokens > max_context` |
| `domain_not_supported` | 1 | Model lacks the required domain capability |
| `privacy_violation` | 1 | Confidential task routed to a non-local model |
| `complexity_mismatch` | 1 | Task complexity outside model's `[min_complexity, max_complexity]` range |
| `agent_depth_exceeded` | 2 | `agent_depth > max_agent_depth` (default: 5) |
| `token_limit_exceeded` | 2 | `estimated_input_tokens > max_tokens_per_step` (default: 8,000) |
| `complexity_ceiling` | 3 | Model's tier exceeds the ceiling for this complexity level |
| `budget_exceeded` | 4 | Estimated cost would exceed team or workflow budget |
| `no_capable_model` | — | All candidates rejected; catch-all |

---

## Models

### GET /api/v1/models

List all models in the registry.

**Query parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled_only` | bool | Return only enabled, non-deprecated models |
| `tier` | int | Filter by tier (1–4) |

**Response:**
```json
[
  {
    "model_id": "deepseek-r1",
    "vendor": "deepseek",
    "tier": 1,
    "max_context": 128000,
    "input_price": 0.00055,
    "output_price": 0.00219,
    "latency_p50_ms": 2000,
    "capabilities": ["reasoning", "code", "chat"],
    "min_complexity": "complex",
    "max_complexity": "critical",
    "is_local": false,
    "enabled": true,
    "deprecated": false
  }
]
```

### GET /api/v1/models/{model_id}

Get a single model by ID. Returns `404` if not found.

### PATCH /api/v1/models/{model_id}

Update model settings in-memory (changes not persisted to `models.yaml`).

```json
{
  "enabled": false,
  "latency_p50_ms": 2500
}
```

---

## Budgets

### GET /api/v1/budgets

List all configured budget policies.

### POST /api/v1/budgets

Create a budget policy.

```json
{
  "policy_id": "team-eng-monthly",
  "scope": "team",
  "scope_id": "team-engineering",
  "period": "monthly",
  "limit_usd": 500.00,
  "warn_at_pct": 0.80,
  "hard_stop": true
}
```

| Field | Options |
|-------|---------|
| `scope` | `team` \| `workflow` |
| `period` | `daily` \| `weekly` \| `monthly` \| `rolling_30d` |
| `hard_stop` | `true` = reject requests over limit; `false` = warn only |

### GET /api/v1/budgets/status/team/{team_id}

Live spend vs. limit for a team. Returns `404` if no policy exists for the team.

```json
{
  "team_id": "team-engineering",
  "spent_usd": 123.45,
  "limit_usd": 500.00,
  "utilisation_pct": 24.69,
  "is_hard_stopped": false,
  "period": "monthly"
}
```

---

## Usage

### GET /api/v1/usage/summary

Cost utilisation for all teams with active budget policies.

```json
[
  {
    "team_id": "team-engineering",
    "current_spend_usd": 123.45,
    "limit_usd": 500.00,
    "utilisation_pct": 24.69,
    "is_hard_stopped": false
  }
]
```

---

## Guardrails — Agent Sessions

### POST /api/v1/guardrails/sessions

Create a new agent session.

```json
{
  "session_id": "session-abc123",
  "team_id": "team-engineering",
  "max_depth": 5
}
```

Returns `409` if a session with that ID already exists.

### GET /api/v1/guardrails/sessions/{session_id}

Get session state: current depth, retry count, tokens used.

### DELETE /api/v1/guardrails/sessions/{session_id}

Terminate a session. Returns `204 No Content`.

### POST /api/v1/guardrails/sessions/advance

Check guardrails and increment agent depth. Returns `422` if any limit is exceeded.

```json
{
  "session_id": "session-abc123",
  "tokens_used": 1200
}
```

---

## Health

### GET /health

Liveness probe. Returns `200` when the server process is running.

```json
{"status": "ok"}
```

### GET /ready

Readiness probe. Returns `200` when the registry and DB are initialised.

---

## Interactive Docs

- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- OpenAPI schema: `GET /openapi.json`
