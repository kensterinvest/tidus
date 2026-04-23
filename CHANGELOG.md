# Changelog

All notable changes to Tidus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Pricing sync cadence**: weekly Sunday → Sundays and Wednesdays at 02:00 UTC (2026-04-23). `.github/workflows/weekly-sync.yml` now uses `cron: '0 2 * * 0,3'` (≈104 runs/year vs. 52). The AI market moves fast enough that a 7-day gap leaves routing cost estimates stale; a 3-4 day cadence keeps pricing, the DB, the landing page magazine, and subscriber emails current. Local Windows Task Scheduler script (`C:\Users\OWNER\scripts\tidus\setup_windows_schedule.ps1`) follows the same cadence (Sun + Wed 03:00 local). File name `weekly-sync.yml` kept for git-history continuity; workflow display name is now "Pricing Sync (Sun + Wed)". Docs updated in `docs/pricing-sync.md`, `docs/deployment.md`, `docs/pricing-model.md`, `docs/architecture.md`, `docs/configuration.md`, `docs/api-reference.md`, `README.md`. User-facing "Weekly Edition" branding on the magazine + subscribe page intentionally left unchanged — existing subscribers signed up for a weekly report and will now receive it at a higher cadence without a breaking promise.

### Known gaps (follow-up candidates)

- **OpenRouter vendor map is hardcoded** (`C:\Users\OWNER\scripts\tidus\sync_pricing.py:OPENROUTER_MAP`, 34 entries). The Layer 2 sync does not discover new vendors or models that appear on OpenRouter — they must be added manually to the dict. The canonical Layer 1 (GitHub Actions) uses `HardcodedSource` + optional `TidusPricingFeedSource` from `tidus/sync/pricing/`, which has the same manual-curation model via `config/models.yaml`. A true "market research" step (diff OpenRouter catalog vs. local map, propose additions) is not implemented.

## [v1.3.0] — 2026-04-21 — Auto-Classification Layer (Stages A + B)

### Highlights

- **Callers no longer need to supply `complexity` / `domain` / `privacy` / `estimated_input_tokens`.** Tidus classifies every request internally via a five-tier cascade (T0 caller override → T1 regex/keywords → T2 trained encoder ∥ T2b Presidio NER → T5 Ollama LLM). Backward compatible: callers who do supply fields still win via the `caller_override` merge.
- **New endpoint `POST /api/v1/classify`** — run the cascade without routing. Useful for offline classification previews, UI surfaces, and integration tests.
- **Asymmetric-safety privacy merge** — ANY tier voting `confidential` forces `confidential` on the final verdict. Confidence scales with agreement count. `public_floor` prevents weakly-supported `public` emissions from the encoder.
- **Two SKU architecture** — CPU-only (89.2% confidential recall baseline) and Enterprise (GPU-bound, T5 LLM escalation for topic-bearing confidentials). See `docs/hardware-requirements.md`.
- **Stage B PII-safe telemetry** — per-request structured log with PCA-reduced embedding (384→64, 57.7% variance retained), type-only entity/pattern lists, and `model_routed` populated from the router's decision. Never emits raw prompts or matched values. Foundation for Stage C active-learning feedback loop.
- **Per-tier async locks** preserve intra-request T2∥T2b parallelism while serializing across concurrent FastAPI requests (torch + spaCy are not thread-safe).
- **Measured baseline:** E1 rule (PERSON alone triggers confidential) = 89.2% [80.7, 94.2] confidential recall on cross-family IRR-adjudicated ground truth (n=83 confidentials in n=2,249 total). E2 (PERSON + encoder-non-public) = 83.1% [73.7, 89.7] at 3.5× better precision.

### Added

- **Classification package (`tidus/classification/`)**: `models.py` (Pydantic types — `EncoderResult.embedding`, `PresidioResult.entity_scores`, `ClassificationTier` literal), `classifier.py` (`TaskClassifier` T0→T5 orchestrator), `heuristics.py` (T1 regex — 13 pattern IDs, Luhn+BIN for credit cards), `keywords.py` (medical/legal/financial/hr/hardship/credential_request — keyword veto for complexity floor), `encoder.py` (T2 — frozen MiniLM + 3 sklearn LR heads; `EncoderProtocol` for DI), `presidio_wrapper.py` (T2b — spaCy `en_core_web_sm`, `HIGH_TRUST_ENTITIES` frozenset), `llm_classifier.py` (T5 — Ollama localhost, JSON-format, `_SlidingWindowLimiter`, `_TTLCache`).
- **Observability package additions** (`tidus/observability/`): `classification_metrics.py` (Prometheus counters `tidus_classify_t5_calls_total`, `tidus_classify_t5_flips_total`, latency histogram `tidus_classify_t5_latency_seconds`), `classification_telemetry.py` (Stage B emitter — PCA load/cache/reduce, structlog `classification` event).
- **Auth: `TokenPayload.tenant_id`** — JWT claim in OIDC mode; `X-Tenant-ID` header in dev mode only (header fallback dropped in OIDC to prevent per-tenant telemetry poisoning); falls back to `team_id`.
- **Settings**: `auto_classify_enabled`, `classify_encoder_dir`, `classify_encoder_max_chars`, `classify_tier5_enabled`, `classify_tier5_model`, `classify_tier5_rate_limit_per_minute`, `classify_privacy_threshold`, `classify_domain_threshold`, `classify_complexity_threshold`, `classify_privacy_public_floor`, `classify_presidio_enabled`, `classify_presidio_parallel`, `classify_presidio_max_chars`, `classify_presidio_rule` (E1/E2), `classify_cache_ttl_seconds`, `classify_cache_max_entries`, `classify_telemetry_enabled`, `classify_pca_path`, `oidc_tenant_claim`, `tenant_header_name`.
- **Scripts**: `fit_pca_64d.py` (one-time PCA fit on labeled corpus → `weights_b/pca_64d.joblib`), `uncertainty_sample.py` (Lever P1 uncertainty-sampled active learning — null result, see `tests/classification/p1_uncertain/RESULTS.md`).
- **Tests**: 756 passing — `tests/unit/classification/*` (7 files), `tests/unit/observability/*` (2 files), `tests/integration/test_{classify_endpoint,auto_classify_endpoints,stage_b_telemetry,encoder_integration,presidio_integration,tier5_integration}.py`.
- **Docs**: `docs/hardware-requirements.md` (two-SKU positioning + GPU FAQ), `findings.md` (research writeup — 3-rater IRR study, E1 vs E2 empirical trade-off, 50/50 entity/topic split).

### Changed

- **`POST /api/v1/complete`** and **`POST /api/v1/route`** — `complexity`, `domain`, `privacy`, `estimated_input_tokens` are now **Optional**. When omitted, Tidus auto-classifies. When all four supplied, T0 short-circuits the cascade (zero overhead). Partial override merged via `caller_override` rule; asymmetric safety still applies to `privacy`.
- **`/ready` endpoint** — now includes `classifier` health block (`encoder_loaded`, `presidio_loaded`, `llm_loaded`, `sku`).
- **`LLM_CONFIDENCE` renamed to `LLM_VOTER_STRENGTH`** — constant is voter weight in the merge arithmetic, NOT a calibrated probability. Rename + docstring makes the semantics self-documenting.
- **`EncoderResult` gains `embedding: list[float] | None`** — the 384-d MiniLM vector, surfaced so Stage B telemetry can dim-reduce to 64-d. Optional (None by default, populated by the concrete encoder).
- **`PresidioResult` gains `entity_scores: dict[str, float]`** — max score per detected entity type. Not plumbed into merge logic yet; surfaced in `debug` payload for observability.

### Fixed

- **`model_routed` was always `None` on `/complete` and `/route` telemetry** — observer fired before `selector.select()`, so the chosen model was unknown when the record was emitted. Fix: `_TelemetryCapture` buffers classifier intermediates; endpoint calls `capture.emit(model_routed=decision.chosen_model_id)` after routing completes.
- **Float precision drift** in `_merge_privacy` — `0.90 + 0.05 = 0.9500000000000001` broke downstream equality checks. Now rounded to 4 decimals.
- **T5 cache key missed model name** — stale cache hits from a prior model could leak into a new model's traffic. Key now `f"{model}|{text[:MAX_CHARS]}"`.
- **T5 over-eager triggering** — broad `bool(kw_hits)` gate fired T5 on every medical/legal/HR question, blowing the Enterprise GPU cost envelope. Now gated by encoder-uncertainty (`t2 is None OR t2.privacy == "public" OR t2.confidence["privacy"] < threshold`).
- **Tier label honesty** — `_apply_t5` now only sets `classification_tier="llm"` when T5 actually flipped the verdict (not when T5 agreed with the pre-T5 result).
- **Presidio `DATE_TIME` removed from `HIGH_TRUST_ENTITIES`** — every "tomorrow" or "next Tuesday" would have been a confidential vote. Now documented why.

### Security

- **Tenant-header poisoning gate** — in OIDC mode, `tenant_id` resolves from the JWT claim only. `X-Tenant-ID` header is *only* honored in dev mode. Prevents authenticated callers from writing to another tenant's Stage B / Stage C feedback stream.
- **`rationale` field stripped from Stage B records** — T5's free-text `rationale` could paraphrase the prompt verbatim. The telemetry emitter never includes it. Test `test_rationale_is_never_in_record` locks the invariant.

## [v1.2.0] — 2026-04-17 — Comprehensive Review Hardening

### Highlights

- **Cross-tenant data-leak plugs** — 4 endpoints (`/dashboard/summary`, `/guardrails/sessions`, `/budgets`) now enforce team-scoped access for non-admin/non-team_manager callers
- **Fallback path now safe** — `/complete` fallback re-runs the full 5-stage selector (privacy / budget / guardrails all re-applied) instead of short-circuiting to `spec.fallbacks[0]`
- **Budget reservation pattern** — `BudgetEnforcer.reserve()` atomically holds estimated cost across the adapter call; `deduct(..., reserved_usd=)` settles by delta. Eliminates the check-then-undo race
- **Redis SpendCounter backend** — `RedisSpendCounter` with Lua-backed `check_and_add` for multi-worker / multi-pod deployments. Picked automatically when `REDIS_URL` is set
- **Server-side agent-depth tracking** — `agent_depth > 0` now requires a real `agent_session_id`; server advances depth via `AgentGuard`. Client-supplied depths are no longer trusted (breaking)
- **Adapter hardening** — all 8 vendor adapters now use a shared `AdapterError` hierarchy, per-call timeout, and exponential backoff on transient failures. Auth/client errors fail fast
- **ExactCache wired into `/complete`** — response cache active for non-confidential tasks; new `cache_hit: bool` field on `CompleteResponse`
- **Audit on every error path** — `/complete` records an audit entry on each failure exit (SOC2/ISO friendly)

### Added

- `tidus/cost/counter.py::RedisSpendCounter` — Redis-backed counter with Lua `check_and_add`, `reset_workflow` via SCAN
- `tidus/budget/enforcer.py::reserve` and `refund` — reservation lifecycle; `deduct` accepts optional `reserved_usd` for delta-settle
- `tidus/adapters/base.py` — `AdapterError`, `AdapterAuthError`, `AdapterRateLimitError`, `AdapterTimeoutError`, `AdapterServerError`, `AdapterClientError`; `with_retry()` helper; `translate_vendor_exception()` for SDK-to-hierarchy mapping
- `tidus/api/deps.py::get_exact_cache` — FastAPI dependency exposing the shared `ExactCache` singleton
- `CompleteResponse.cache_hit: bool` — tells callers whether the response came from cache
- New settings: `cache_enabled`, `cache_ttl_seconds`, `cache_max_size`, `adapter_timeout_seconds`, `adapter_max_retries`, `adapter_base_delay_seconds`, `redis_spend_counter_prefix`
- `SpendCounter.reset_workflow(workflow_id)` — reset all `(team, workflow_id)` counters in one pass
- `ModelSelector.select(task, exclude_model_ids=...)` — optional parameter used by the `/complete` fallback re-selection
- Description entries for `claude-opus-4-7`, `grok-4`, `qwen-flash`
- `docs/enterprise/sso.md`, `docs/enterprise/rbac.md` now document shipped behavior (were "Roadmap")

### Changed

- `BudgetEnforcer.can_spend()` is now a **pure check** (no state mutation). The atomic reserve-and-hold moved to the new `reserve()` method
- `BudgetEnforcer.deduct(team, wf, amount_usd, *, reserved_usd=None)` — new kwarg; legacy callers that don't reserve still work
- `resolve_caller_id()` — JWT sub now overrides a mismatching `X-Titus-User-Id` header (was: header blindly trusted); impersonation attempts logged as `metering_header_impersonation_attempt`
- RBAC `_has_role()` — **admin is a super-role**: satisfies every required-role list even when not explicitly listed (defense in depth)
- `PriceConsensus` tie-breaker — ties on `source_confidence` now break by `effective_date` then `retrieved_at` (commit f5be789 promised this; now actually implemented)
- `RegistryPipeline.run_price_sync_cycle` — models missing from `models.yaml` are **retired** (removed from the new revision) instead of persisting forever
- `BudgetEnforcer.reset_period()` — workflow-scoped policies now reset all `(team, workflow_id)` counters via `counter.reset_workflow(scope_id)` instead of mis-keying `(workflow_id, None)`
- Anthropic adapter — joins **all** text blocks in a response (previously dropped every block except the first)
- Google adapter — reads the real `finish_reason` from the candidate instead of hardcoding `"stop"` (`safety`/`max_tokens` stops are now reported correctly)
- MCP server — singletons initialized exactly once per process via `_ensure_initialized`; `SpendCounter` and `SessionStore` identity preserved across tool calls
- README pricing table now mirrors `docs/pricing.md` 4-tier ladder (Community / Pro / Business / Enterprise)
- Magazine (`index.html`) — model count 55 (was 53), vendor count 13 (was 12), adapter list corrected (`Moonshot`, `Ollama` added; `Groq`, `Together AI` removed — they were listed but have no live adapter), per-vendor counts refreshed
- Model count corrections across docs (`README.md`, `docs/pricing.md`, `docs/selection-algorithm.md`) and descriptions (Opus 4.6 $5/$25, Haiku 4.5 $1/$5, DeepSeek-V3 $0.32/$0.89)

### Fixed

- `/complete` fallback path bypassed Stages 1–5 of the selector — confidential tasks could land on non-local fallback models. Fixed by re-running `selector.select(..., exclude_model_ids=...)`
- `BudgetEnforcer.can_spend → undo → deduct` race — concurrent requests could all pass the check and collectively overrun the limit. Fixed by the reservation pattern
- `agent_depth` was trusted from the request body — clients could claim depth=0 indefinitely and bypass recursion limits. Now validated against `AgentGuard`/`SessionStore` server-side
- `/dashboard/summary`: no team filter on cost records / budget rows / sessions. `read_only`/`service_account` callers could see every team's spend history
- `/guardrails/sessions/{id}` GET/DELETE: no team-ownership check; a caller who knew a session ID could read or terminate another team's agent session
- `/guardrails/sessions` POST and `/budgets` POST: accepted `team_id`/`scope_id` from the request body with no validation against the caller's JWT team
- `X-Titus-User-Id` header trusted without JWT verification — any caller could impersonate any user_id in the metering tables
- All 8 adapters collapsed every upstream failure to bare `Exception`. Zero timeouts on cloud calls (only Ollama had one). No retry/backoff. Auth errors and rate-limits were indistinguishable
- All `/complete` error paths raised `HTTPException` without an audit-log entry
- MCP server rebuilt `SpendCounter` + `SessionStore` on every tool call — budget state never accumulated across an MCP session
- `pricing_report` generator emitted `Tier ModelTier.mid` (enum repr) instead of `Tier 2` in the "New Models" block
- Duplicate `How It Works` nav entry in `index.html`
- Stale "Roadmap" labels on shipped SSO/RBAC features in 4 doc files
- Model count drift across README (53 → 55), `docs/pricing.md` (53 → 55), `docs/selection-algorithm.md` (53 → 55)

### Removed

- `www/` directory — stale first-iteration SaaS marketing page that contradicted the real product (`$99/mo` tier, `api.tidus.ai`, `hello@tidus.ai`, `github.com/tidusai/tidus`). The root `index.html` is the single source of truth for the landing page

### Breaking

- `POST /api/v1/complete`: `agent_depth > 0` now requires a valid `agent_session_id` from `POST /api/v1/guardrails/sessions`. Requests omitting the session id → HTTP 400
- `BudgetEnforcer.can_spend()` no longer mutates state. Any code relying on the old reserve-and-undo behavior needs to migrate to the new `reserve()` method
- Callers that passed `team_id`/`scope_id` for other teams on `/guardrails/sessions` POST, `/budgets` POST, or GET-ed other teams' session IDs on `/guardrails/sessions/{id}` now receive HTTP 403 / 404

### Tests

- +82 new regression tests (total 569, was 487) covering every fix above
- New test files: `test_adapter_base.py`, `test_adapter_integration.py`, `test_cross_team_api_isolation.py`, `test_rbac.py`, `test_redis_spend_counter.py`
- `fakeredis[lua]` added as a dev-group dep so Redis atomicity tests run without a Redis daemon

### Dependencies

- **Added**: `redis>=5.0.0` (runtime), `fakeredis[lua]>=2.20.0` (dev group)

---

## [v1.1.0] — 2026-04-06 — Multi-Source Self-Healing Registry

### Highlights

- **Versioned, audited registry** — model catalog changes create DB revisions with full source provenance; no more silent in-memory mutations
- **Layered merge architecture** — three-layer merge (base catalog → overrides → telemetry) with deterministic precedence rules
- **Scoped overrides** — RBAC-controlled overrides (price_multiplier, hard_disable, force_tier_ceiling, emergency_freeze) replace direct YAML edits
- **Multi-source consensus pricing** — MAD-based outlier detection across pluggable pricing sources
- **Drift detection + auto-remediation** — four detectors auto-disable models whose runtime behaviour diverges from catalog
- **Billing reconciliation** — upload provider invoice CSVs to flag cost leakage
- **Prometheus observability** — 9 custom metrics (6 Gauges, 3 Counters); alerting rules; runbooks

### Added

#### Phase 1 — DB Schema, Alembic Formalization, YAML Seeding

- `tidus/db/registry_orm.py` — 5 new ORM tables: `model_catalog_revisions`, `model_catalog_entries`, `model_overrides`, `model_telemetry`, `model_drift_events`
- `tidus/registry/seeder.py` — `RegistrySeeder.seed_from_yaml()` — idempotent YAML → DB import; creates revision 0 on first run
- Alembic catch-up migration for pre-existing tables; new migration for registry tables
- `ModelSpec` gains `cache_read_price`, `cache_write_price` (default 0.0), `retired_at`, `retirement_reason`
- `POST /api/v1/models/{id}/retire` — admin-only model retirement endpoint

#### Phase 2 — Layered Registry, Override Engine, API

- `tidus/registry/effective_registry.py` — `EffectiveRegistry`: drop-in `ModelRegistry` replacement with 3-layer merge; revision-aware cache key `(active_revision_id, override_generation)`
- `tidus/registry/merge.py` — pure merge functions: `merge_spec()`, `apply_price_multiplier()`
- `tidus/registry/override_manager.py` — `OverrideManager` with RBAC enforcement, conflict detection, and audit trail
- `tidus/registry/telemetry_reader.py` — `TelemetryReader` with 3-tier staleness: fresh (<24h), unknown (24–72h, base fallback), expired (>72h, excluded)
- `tidus/sync/override_expiry.py` — `OverrideExpiryJob` — deactivates expired overrides every 15 min; writes audit entries
- New registry API router (`/api/v1/registry`): revision CRUD, override CRUD (team_manager scoped), drift event listing, revision diff, revision preview

#### Phase 3 — Multi-Source Pricing Pipeline

- `tidus/sync/pricing/` — pluggable `PricingSource` abstraction: `HardcodedSource`, `TidusPricingFeedSource` (HMAC-SHA256 verified, circuit breaker), `PriceConsensus` (MAD outlier detection)
- `tidus/registry/pipeline.py` — `RegistryPipeline` with 3-tier validation (Schema → Invariant → Canary), atomic 2-phase write, PostgreSQL advisory lock for k8s multi-replica safety
- `tidus/registry/validators.py` — `SchemaValidator`, `InvariantValidator`, `CanaryProbe` with retry logic
- `pricing_ingestion_runs` table — full source provenance per sync cycle
- `POST /api/v1/sync/prices?dry_run=true` — validate consensus without writing a revision
- `TIDUS_PRICING_FEED_URL`, `TIDUS_PRICING_FEED_SIGNING_KEY`, `PRICING_FEED_FAILURE_THRESHOLD`, `PRICING_FEED_RESET_TIMEOUT_SECONDS` settings

#### Phase 4 — Drift Detection + Telemetry Persistence

- `tidus/sync/drift/detectors.py` — 4 drift detectors: `LatencyDriftDetector`, `ContextDriftDetector`, `TokenizationDriftDetector`, `PriceDriftDetector`
- `tidus/sync/drift/engine.py` — `DriftEngine`: concurrent detection, auto-disable on critical drift, auto-recovery after 3 consecutive healthy probes
- `tidus/sync/telemetry_writer.py` — `TelemetryWriter.write()` — persists health probe output to `model_telemetry` (survives restarts)
- 3-tier health probe sampling: Tier A (always live), Tier B (synthetic-first), Tier C (10% sample, synthetic-first)
- `probe_type: synthetic|live` tracked per telemetry row for cost attribution

#### Phase 5 — Billing Reconciliation

- `tidus/billing/reconciler.py` — `BillingReconciler` with matched/warning/critical classification
- `tidus/billing/csv_parser.py` — normalized billing CSV parser (UTF-8 BOM safe, duplicate-row deduplication)
- `billing_reconciliations` DB table
- Billing API (`/api/v1/billing`): `POST /reconcile`, `GET /reconciliations`, `GET /reconciliations/summary`

#### Phase 6 — Prometheus Metrics + Observability

- `tidus/observability/registry_metrics.py` — 9 custom metrics (see below)
- `tidus/observability/metrics_updater.py` — `MetricsUpdater`: refreshes 6 Gauges every 5 min and at startup
- `monitoring/alerting-rules.yaml` — 5 Prometheus alerting rules
- `monitoring/README.md` — metrics reference, alert setup, Grafana import instructions
- `docs/runbooks/emergency-freeze.md`, `drift-incident.md`, `override-abuse.md` — operational runbooks

### Metrics Added

| Metric | Type | Description |
|---|---|---|
| `tidus_registry_last_successful_sync_timestamp` | Gauge | Unix timestamp of last successful price sync |
| `tidus_registry_active_revision_activated_timestamp` | Gauge | Unix timestamp of active revision promotion |
| `tidus_registry_model_last_price_update_timestamp` | Gauge (per model) | Last price update time |
| `tidus_registry_model_confidence` | Gauge (per model) | 1.0 (fresh) or 0.5 (stale >8 days) |
| `tidus_registry_active_revision_id` | Gauge | Deterministic int hash of active revision UUID |
| `tidus_registry_models_stale_count` | Gauge | Models with stale price data |
| `tidus_probe_live_calls_total` | Counter | Live health probe calls by model and result |
| `tidus_probe_synthetic_calls_total` | Counter | Synthetic probe calls by model and result |
| `tidus_registry_drift_events_total` | Counter | Drift events by model, type, and severity |

### Changed

- `build_singletons()` is now `async` — `EffectiveRegistry.build()` requires a DB session
- `TidusScheduler` gains 5 new background jobs: registry_refresh (60s), override_expiry (15min), drift_engine (5min), metrics_updater (5min), monthly_budget_reset (1st of month)
- `HealthProbe` gains `session_factory` optional param; persists telemetry after every probe
- Price sync response enriched with `revision_id`, `sources_used`, `ingestion_run_ids` (backward-compatible)

### Settings Added

- `TIDUS_PRICING_FEED_URL` — optional remote pricing feed endpoint
- `TIDUS_PRICING_FEED_SIGNING_KEY` — HMAC-SHA256 signing key for feed verification
- `PRICING_FEED_FAILURE_THRESHOLD` — circuit breaker trip count (default 5)
- `PRICING_FEED_RESET_TIMEOUT_SECONDS` — circuit breaker reset window (default 300)
- `REGISTRY_REVISION_RETENTION_DAYS` — SUPERSEDED revision retention window (default 90)

---

## [v1.0.0-community] — 2026-04-02 — Community Release

This is the first public community release of Tidus. The codebase is production-ready and open-sourced under Apache 2.0.

### Highlights
- **Free community tier** — up to 10K requests/month, model routing with no budget enforcement required
- **One-command install** — `pip install tidus` (PyPI) or `docker run -p 8000:8000 tidus/tidus:latest`
- **28 models across 8 vendors** — OpenAI, Anthropic, Google, Mistral, DeepSeek, xAI, Kimi, Ollama (local)
- **87–96% AI cost reduction** via 5-pillar cost-control strategy (tiered routing, caching, guardrails)
- **MCP server** (`tidus-mcp`) compatible with Claude Desktop, Cursor, and any MCP client
- **Kubernetes-ready** — Helm chart, Prometheus/Grafana dashboards, HPA included (Phase 11)
- **Audit logs + PostgreSQL/Redis** production config (Phase 9–10)
- **SSO/OIDC + RBAC** for enterprise deployments (Phase 8)

### Added since v0.1.0
- Phase 8 — SSO/OIDC authentication + RBAC (role-based access control)
- Phase 9 — Structured audit logging with queryable log trail
- Phase 10 — PostgreSQL + Redis production configuration (SQLite remains default for dev)
- Phase 11 — Kubernetes manifests, Helm chart, Prometheus metrics, Grafana dashboards, HPA
- End-to-end load tests, cache correctness tests, and multi-tenant isolation tests
- Marketing website at `www/`
- Full how-to-use guide (`docs/how-to-use.md`)
- Community release preparation: repo URL cleanup, version bump to 1.0.0, PyPI + Docker Hub publish

### Changed
- Version bumped from `0.1.0` → `1.0.0` to signal production readiness
- GitHub repository canonical URL corrected to `github.com/kensterinvest/tidus`

---

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
[Unreleased]: https://github.com/kensterinvest/tidus/compare/v1.0.0-community...HEAD
[v1.0.0-community]: https://github.com/kensterinvest/tidus/compare/v0.1.0...v1.0.0-community
[v0.1.0]: https://github.com/kensterinvest/tidus/releases/tag/v0.1.0
[v0.1-phase3]: https://github.com/kensterinvest/tidus/compare/v0.1-phase2...v0.1-phase3
[v0.1-phase2]: https://github.com/kensterinvest/tidus/compare/v0.1-phase1...v0.1-phase2
[v0.1-phase1]: https://github.com/kensterinvest/tidus/releases/tag/v0.1-phase1
