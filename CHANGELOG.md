# Changelog

All notable changes to Tidus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] — 2026-03-27 — MVP Release

### Added

#### Phase 4 — Vendor Adapters, /complete endpoint, Caching, Sync
- 8 vendor adapters via `@register_adapter` pattern: Ollama, Anthropic, OpenAI, Google, Mistral, DeepSeek, xAI, Moonshot — all with `complete()`, `health_check()`, `count_tokens()`
- `POST /api/v1/complete` — route + execute + log cost in one call; falls back to first fallback model on adapter error
- `POST /api/v1/sync/health` and `POST /api/v1/sync/prices` — admin-triggered manual sync
- `ExactCache` — SHA-256 keyed, TTL eviction, team-scoped (**Pillar 3, Layer 1**)
- `SemanticCache` — sentence-transformers `all-MiniLM-L6-v2`, cosine similarity threshold 0.95 (**Pillar 3, Layer 2**); graceful no-op if package not installed
- `CostLogger` — writes `CostRecord` to DB after each `/complete` (non-fatal)
- `TidusScheduler` (APScheduler): health probes every 5 min, price sync weekly Sunday 02:00 UTC
- `HealthProbe` — rolling P50 latency over 20 probes; auto-disables models after 3 failures
- `PriceSync` — compares registry vs hardcoded known-prices; writes `PriceChangeRecord` on >5% delta
- 9 new integration tests for `/complete` endpoint, budget enforcement, privacy routing, fallback, sync admin

#### Phase 5 — Dashboard SPA
- `GET /api/v1/dashboard/summary` — all dashboard metrics in one API call
- Vanilla JS/HTML/CSS SPA at `/dashboard/` (no build step):
  - 6 KPI cards, cost-by-model bar chart (Chart.js, tier colour-coded), budget utilization bars, active sessions table, registry health badges
  - Auto-refreshes every 30 seconds
  - All DOM manipulation uses safe DOM API — no innerHTML with external values

#### Phase 6 — MCP Server + Docker
- MCP server (`tidus-mcp` entry point) with 4 tools: `tidus_route_task`, `tidus_complete_task`, `tidus_get_budget_status`, `tidus_list_models`
- stdio transport — compatible with Claude Desktop, Cursor, and any MCP client
- `Dockerfile` — python:3.12-slim + uv, non-root user, SQLite volume at `/app/data`
- `docker-compose.yml` — tidus + optional Ollama profile (`--profile ollama`) for local inference

#### Phase 7 — Documentation + MCP Tests
- Full content for: `docs/mcp-integration.md`, `docs/deployment.md`, `docs/dashboard.md`, `docs/adapters.md`
- `README.md` updated: all phases marked complete, pillar statuses live, feature table current
- v0.1.0 CHANGELOG finalized

### Fixed
- `tidus/mcp/server.py` `_handle_route`: `decision.vendor`/`decision.tier` don't exist on `RoutingDecision` — now resolved via `registry.get(decision.chosen_model_id)`
- `tidus/mcp/server.py` `_handle_budget_status`: `enforcer.status()` never returns `None` — now detects no-policy via `status.policy_id == "none"` sentinel

### Tests
- **115 tests passing** across unit, integration, model selection intelligence, and MCP protocol suites
- `tests/unit/test_mcp_server.py` — 15 new tests: MCP tool registration, `tidus_list_models`, `tidus_route_task` (simple/confidential/budget-rejection), `tidus_get_budget_status` (with/without policy), `tidus_complete_task`

---

## [v0.1-phase3] — 2026-03-26 — API Layer

### Added
- `POST /api/v1/route` — route-only endpoint; returns `RoutingDecision` with model, cost, and score
- `GET /api/v1/models` — list model registry with optional `?enabled_only=true&tier=N` filters
- `GET /api/v1/models/{id}` — single model detail
- `PATCH /api/v1/models/{id}` — enable/disable model, update latency estimate (in-memory)
- `GET /api/v1/budgets` — list budget policies
- `POST /api/v1/budgets` — create budget policy
- `GET /api/v1/budgets/status/team/{team_id}` — live spend vs. limit
- `GET /api/v1/usage/summary` — cost utilisation across all tracked teams
- `POST /api/v1/guardrails/sessions` — create agent session (409 on duplicate)
- `GET /api/v1/guardrails/sessions/{id}` — session detail
- `DELETE /api/v1/guardrails/sessions/{id}` — terminate session (204)
- `POST /api/v1/guardrails/sessions/advance` — check guardrails and increment agent depth
- FastAPI dependency injection pattern (`tidus/api/deps.py`) — singletons built once at lifespan startup
- 26 model-selection-intelligence integration tests covering complexity enforcement, domain
  specialisation, scoring formula, savings quantification, and regression guard (91 tests total, all passing)
- 12 route-endpoint integration tests using `TestClient` against the real registry

### Changed
- `complexity_mismatch` rejection reason added to `RejectionReason` — **stage 1 hard constraint**
- `CapabilityMatcher` now enforces `min_complexity`/`max_complexity` per model as a hard binary
  constraint. A model designed for `complex`/`critical` work is explicitly rejected (not just
  deprioritised) for `simple` tasks. This closes a silent routing error where premium models
  could win low-complexity tasks through scoring drift.
- `BudgetEnforcer` gains `list_policies()` and `add_policy()` to support API endpoints
- `tidus/main.py` registers all v1 routers and calls `build_singletons()` in lifespan

### Fixed
- Model registry updated from 26 → 28 models; GPT-5/GPT-5.2 (non-existent models) replaced
  with `o3` and `o4-mini` (real OpenAI reasoning models, verified 2026-03-26)
- All vendor prices corrected to verified 2026-03-26 rates:
  - `gpt-oss-120b`: $0.000039/$0.0001 per 1K
  - `claude-opus-4-6`: $0.005/$0.025 per 1K, context window expanded to 1M tokens
  - `claude-sonnet-4-6`: context window expanded to 1M tokens
  - `claude-haiku-4-5`: $0.001/$0.005 per 1K
  - `gemini-3.1-pro`: $0.002/$0.012 per 1K
  - `gemini-3.1-flash`: $0.00025/$0.0015 per 1K
  - `mistral-large-3`: $0.0005/$0.0015 per 1K
  - `mistral-small`: $0.00007/$0.0002 per 1K
  - `codestral`: $0.0002/$0.0006 per 1K
  - `devstral`: $0.0004/$0.002 per 1K (new entry)

### Added to Model Registry
- `gpt-5-codex` — tier 1, 400K context, $0.00125/$0.010 per 1K; for complex/critical code tasks
- `codex-mini-latest` — tier 3, 200K context, $0.00075/$0.003 per 1K; simple-to-complex code
- `devstral` — Mistral's code-specialised instruct model, tier 2, 128K context

---

## [v0.1-phase2] — 2026-03-25 — Core Logic

### Added
- `ModelRegistry` — loads and validates `config/models.yaml`; supports `get()`, `list()`, `enabled()`
- `CapabilityMatcher` — stages 1–2 of the 5-stage selector: hard constraints + guardrail checks
- `CostEngine` — token pricing math with configurable safety buffer and provider-aware tokenizer dispatch
- `BudgetEnforcer` — `can_spend()`, `deduct()`, `status()` with atomic in-process spend counters
- `AgentGuard` — depth, retry, and token-per-step guardrail enforcement
- `SessionStore` — in-memory agent session tracker (Redis-ready interface)
- `ModelSelector` — complete 5-stage selection algorithm
- `ModelSelectionError` — structured error with `stage`, per-model rejection list, and `failure_reason`
- Unit tests: selector (12), cost engine (5), budget enforcer (10), guardrails (10) — all passing
- Integration test: real-registry selector test against `config/models.yaml`

---

## [v0.1-phase1] — 2026-03-24 — Foundation

### Added
- `pyproject.toml` with `uv` toolchain and all dependencies declared
- Pydantic v2 data models: `TaskDescriptor`, `ModelSpec`, `CostEstimate`, `CostRecord`,
  `BudgetPolicy`, `BudgetStatus`, `RoutingDecision`, `AgentSession`, `GuardrailPolicy`,
  `PriceChangeRecord`
- `TokenizerType`, `Capability`, `ModelTier`, `Domain`, `Complexity`, `Privacy` enums
- Full model registry `config/models.yaml` — 26 models across 8 vendor families
- Budget policies `config/budgets.yaml` with team and workflow examples
- Routing + guardrail policies `config/policies.yaml`
- SQLAlchemy async ORM: `cost_records`, `budget_policies`, `price_change_log`, `routing_decisions`
- FastAPI app factory with lifespan, `/health`, `/ready` probes
- Structured JSON logging via `structlog`
- Safe YAML loader with Pydantic validation (`tidus/utils/yaml_loader.py`)
- Open-source readiness files: `README.md`, `CONTRIBUTING.md`, `LICENSE` (Apache 2.0),
  `CODE_OF_CONDUCT.md`, `SECURITY.md`
- Documentation skeleton in `docs/` (11 files + enterprise stubs)
- `.gitignore`, `.env.example`

---

<!-- Links -->
[Unreleased]: https://github.com/lapkei01/tidus/compare/v0.1-phase3...HEAD
[v0.1-phase3]: https://github.com/lapkei01/tidus/compare/v0.1-phase2...v0.1-phase3
[v0.1-phase2]: https://github.com/lapkei01/tidus/compare/v0.1-phase1...v0.1-phase2
[v0.1-phase1]: https://github.com/lapkei01/tidus/releases/tag/v0.1-phase1
