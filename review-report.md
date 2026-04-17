# Tidus — Comprehensive Code, Docs, Tests, and UX Review

**Date:** 2026-04-17
**Reviewer:** Claude Opus 4.7 (1M context) with 8 parallel specialist agents
**Scope:** Full codebase (router, adapters, registry, auth/budget/cache, API/MCP/CLI), docs, tests (487 passing, 61% cov), magazine + public HTML
**Git state:** Branches clean of Claude/Paperclip co-authors (origin/main has only `Tidus Pricing Bot` by user preference). Tags `v1.0.0` and `v1.0.0-community` still have dirty ancestry — decision pending.

---

## Priority summary

| Severity | Count |
|---|---|
| Critical | 27 |
| High | 30 |
| Medium | 33 |
| Low | 31 |

**Top-5 must-fix before next release:**
1. Fallback path bypasses all 5 selector stages — confidential data can leak (`tidus/api/v1/complete.py:156-171`)
2. Budget enforcer race condition — concurrent requests can overrun the spend limit (`tidus/budget/enforcer.py:83-96`)
3. RBAC flat check instead of hierarchy; admin silently denied in several endpoints (`tidus/auth/rbac.py:30-54`)
4. Dashboard + session/budget endpoints leak cross-tenant data (`tidus/api/v1/dashboard.py:130-162`, `guardrails.py:75-100`, `budgets.py:62-82`)
5. MCP server rebuilds singletons on every handler call — budget state never accumulates (`tidus/mcp/server.py:40-41,85-86,147-148,168-169`)

---

## 1. End-to-end workflow + router core

### Critical
- **`tidus/api/v1/complete.py:156-190`** — Fallback path calls `adapter.complete(fallback_id, ...)` after a raw `registry.get()`, **fully bypassing Stages 1-5**: no privacy enforcement (confidential task can land on a non-local fallback), no `can_spend`, no guardrails, no tier ceiling, no `max_cost_usd` check.
- **`tidus/budget/enforcer.py:83-96`** — Hard-stop check is **not atomic**: `check_and_add()` reserves, then `add(-amount)` releases before the adapter runs. Under concurrency, N requests all pass `can_spend`, adapters fire, then `deduct()` commits all of them — overrunning the limit.
- **`tidus/api/v1/complete.py:93`** + **`tidus/router/capability_matcher.py:135-136`** — Agent depth comes from the request body (`req.agent_depth`), compared against policy max; `AgentGuard` and `SessionStore` are **never invoked** from `/complete`. Clients can claim `agent_depth=0` indefinitely and bypass recursion limits.
- **`tidus/api/v1/complete.py:118,133,139,176,183,187`** — All error paths raise `HTTPException` **without `audit.record(...)` and without `cost_logger.record(...)`**. Silent failures for SOC2/ISO compliance.

### High
- `tidus/api/v1/complete.py:40-54,146` — `CompleteRequest.stream: bool` accepted but handler unconditionally calls `adapter.complete(...)` instead of `stream_complete(...)`. `stream=True` silently returns non-streamed.
- `tidus/router/capability_matcher.py:92-126` — `_check_hard_constraints` never inspects `spec.retired_at`; retired models remain routable.
- `tidus/api/v1/complete.py:86` — `effective_team_id = _auth.team_id or req.team_id` — empty/missing JWT team claim falls through to body value, defeating cross-team abuse guarantee.
- `tidus/cost/counter.py` + `tidus/guardrails/session_store.py` — In-memory dicts only. `uvicorn --workers N` silently fragments budget counters per process; multi-pod deployments have no global enforcement.
- `tidus/api/deps.py:80` — `MeteringService` is built as a singleton but never called in the `/complete` hot path — billing metering is orphaned.

### Medium
- `tidus/api/v1/complete.py:157` — Only `spec.fallbacks[0]` is attempted, not the ordered chain. If the first fallback fails, request 502s even when later fallbacks exist.
- `tidus/cost/engine.py:54-55,86-87` — `int(raw * 1.15)` truncates: raw=1 → buffered=1 (zero buffer). Inconsistent at low volumes.
- `tidus/api/v1/complete.py:163-166` — `decision.fallback_from = decision.chosen_model_id` mutates pre-copy decision before `model_copy(update=...)` supersedes it. Dead code.
- `tidus/budget/enforcer.py:170` — Ternary precedence on `policy.period == BudgetPeriod(period)` is fragile.

### Low
- `tidus/models/routing.py:10` — Comment on `model_disabled` says "enabled=False or deprecated=True" but matcher only rejects on `enabled=False`. Stale comment.
- Deprecated `+0.15` penalty: scores min-max normalized to [0,1], so 0.15 is modest tiebreaker (working as designed per docstring at `selector.py:201-203`).
- `tidus/cost/logger.py:85-87` — Swallows all exceptions as non-fatal. Combined with Critical #4, a DB outage produces zero cost/audit records with no surfaced error.

---

## 2. Vendor adapters (all 10, including disabled)

### Critical
- **Missing adapters** — Qwen and Cohere listed as supported vendors (Qwen "partial", Cohere "disabled"), `landing_updater.py:34,38` lists them, but **no adapter implementations exist**. Any model with `vendor="qwen"/"cohere"` will `KeyError` in `get_adapter()` (`base.py:82`).
- **Tokenizer network calls with placeholder keys** — Anthropic/Google/Ollama `count_tokens()` make live API calls without checking placeholder keys. Health probe Tier B/C uses it as "free synthetic probe" (`health_probe.py:178`) — fails loudly on placeholder. Copy guard from `tidus/cost/tokenizers.py:112,138` into `anthropic_adapter.py:120-132`, `google_adapter.py:125-136`, `ollama_adapter.py:108-121`.

### High
- **Zero timeout on cloud calls** — `openai_adapter.py:47`, `anthropic_adapter.py:61`, `mistral_adapter.py:47`, `deepseek_adapter.py:51`, `xai_adapter.py:50`, `moonshot_adapter.py:50`, `google_adapter.py:73`. Only Ollama has timeout. A hung vendor blocks health probe + `/complete` indefinitely.
- **No retry/backoff anywhere** — grep for `retry|backoff` in `adapters/` returns nothing. Every 429/503 is immediate failure.
- **Error handling collapses to bare `Exception`** — callers cannot distinguish rate-limit vs auth vs server. Bad API key indistinguishable from rate-limit; both trigger auto-disable at `health_probe.py:134`.
- **OpenAI-compat param mismatch** — `openai_adapter.py:50` uses `max_completion_tokens`; `mistral_adapter.py:50`, `deepseek_adapter.py:54`, `xai_adapter.py:53`, `moonshot_adapter.py:53` use `max_tokens`. Modern OpenAI reasoning models (o3, o4-mini) require `max_completion_tokens` and reject `max_tokens`.

### Medium
- `google_adapter.py:98` — `finish_reason="stop"` hardcoded; never reads `response.candidates[0].finish_reason`. SAFETY/MAX_TOKENS misreported as normal completion.
- `google_adapter.py:100`, `mistral_adapter.py:73` — `raw={}` while other adapters return `response.model_dump()`. Inconsistent debugging payload.
- `anthropic_adapter.py:64` — Only reads `content[0].text`; silently drops tool-use blocks or multiple text blocks.
- `ollama_adapter.py:87` — `import json` inside streaming loop on every line.

### Low
- `adapter_factory.py:13-23` — Imports list doesn't mention Qwen/Cohere as deliberately excluded.
- `base.py:70` — `register_adapter` instantiates class at import time.
- `openai_adapter.py:106` — Tokenizer encoding substring match is fragile.
- `deepseek_adapter.py:108`, `xai_adapter.py:107`, `moonshot_adapter.py:107` — All three use `cl100k_base` as tokenizer stand-in. DeepSeek/Kimi use different BPE — billing will drift from vendor's meter.
- `ollama_adapter.py:42,81` — 120s timeout; 70B cold-load can exceed.

---

## 3. Registry, pricing, sync, billing

### Critical
- **`config/model_descriptions.yaml:17`** — Opus 4.6 described as "$15/$75 per 1M" but actual is **$5/$25** (3× overstated). Feeds weekly subscriber pricing report (`tidus/sync/scheduler.py:235-265`).
- **`config/model_descriptions.yaml`** — **No entry** for `claude-opus-4-7`, `grok-4`, `qwen-flash`. New models ship anonymous in reports.
- **`tidus/registry/pipeline.py:173,217-241`** — **Removed-model path is missing**. `new_specs = dict(current_by_id)` + add-only loop means a model deleted from `models.yaml` persists in DB forever. Seeder (`seeder.py:65-73`) skips on existing seed revision.

### High
- **`tidus/sync/pricing/consensus.py:134`** — Commit `f5be789`'s claimed "recency tie-breaker" is **not in code**. `max(non_outliers, key=source_confidence)` returns first on tie; `effective_date` stored but never consulted. Commit message ≠ code.
- `config/model_descriptions.yaml:32,174` — Haiku 4.5 says "$0.80/$4" but actual is **$1/$5**. DeepSeek-V3 says "$0.27/$0.89" but actual is **$0.32/$0.89**.
- `tidus/registry/pipeline.py:222-241` — "Detect new models from YAML" (e99aeba) only fires for models also in `consensus.quotes`. A new YAML entry missing from `_KNOWN_PRICES` is invisible.
- `tidus/sync/pricing/consensus.py:127` — `ConsensusError` for any one pathological model aborts **the entire** sync cycle instead of skipping just that model.

### Medium
- `tidus/registry/pipeline.py:166-171` — `ModelSpec.model_validate` failures silently skipped via `except Exception: continue`. Affected models freeze at previous prices with no alert.
- `tidus/models/model_registry.py:94-95` + `tidus/registry/effective_registry.py:156-164` — `retired_at`/`retirement_reason` defined but never read/written. `list_enabled()` docstring claims "enabled AND not retired" but filter is just `s.enabled`.
- `tidus/registry/effective_registry.py:120-142` — `refresh()` not guarded by lock. Two concurrent calls (scheduler + override_expiry) can both call `build()` and swap `_by_id` with mid-rebuild state.
- `tidus/sync/drift/detectors.py:197-206` + `policies.yaml:53-54` — Config comment says "avg requests use ≥5% of context" but code implements "fraction of requests exceeding 90%". Semantics disagree.
- `tidus/billing/reconciler.py:125-127` — No status distinguishes "untracked model" from "real variance".

### Low
- `tidus/sync/price_sync.py:7` — Docstring "verified 2026-04-05" but hardcoded_source says 2026-04-17.
- `tidus/registry/pipeline.py:644-662` — Advisory lock `return None, True` fail-open on exception. Two replicas + transient DB error = both sync.
- `tidus/registry/override_manager.py:64,155,189` — `hasattr(Role, actor.role)` checks class attrs, not enum members.
- `tidus/registry/seeder.py:39` — `_SEED_REVISION_ID = "seed-v0"` never promotable; if non-seed revisions all fail, DB stuck with seed forever.
- `tidus/sync/pricing/hardcoded_source.py:107-108` — `cache_read_price`, `cache_write_price` hardcoded 0.0 for all models. Anthropic/OpenAI prompt caching isn't free.
- `tidus/registry/pipeline.py:507` — Empty `canary_results` coerces to None, losing 0-sample fact.
- `config/models.yaml:977-981` — `sonar.max_context: 127072` looks like typo for `128000`.

---

## 4. Auth, budget, guardrails, cache, cost, metering, DB

### Critical
- **`tidus/auth/rbac.py:30-54`** — `_has_role` checks set membership, **NOT hierarchy**, despite `_ROLE_HIERARCHY` being defined and docstring promising "each role includes permissions of roles below". Callsites must list every allowed role explicitly; forgetting admin (as in `/budgets` POST line 65) silently denies admin users.
- **`tidus/api/v1/dashboard.py:130-162`** — `dashboard_summary` requires only `get_current_user` (any role incl. `read_only`/`service_account`), and `_get_cost_records` (line 97) has **NO team filter** — any user can dump cross-tenant cost records for up to 90 days.
- **`tidus/api/v1/guardrails.py:75-100`** — `get_session`/`terminate_session` do NOT verify the session's `team_id` matches the caller's JWT team. User in team-A can read/delete team-B's active agent sessions by knowing the session_id.
- **`tidus/api/v1/guardrails.py:52-67`**, **`tidus/api/v1/budgets.py:62-82`** — Both accept `team_id`/`scope_id` from body with **NO validation against `_auth.team_id`**. A `team_manager` in team-A can create a budget policy for team-B (e.g. `limit_usd=9999999`).
- **`tidus/cost/logger.py:85-87`** vs **`tidus/api/v1/complete.py:197-200`** — `enforcer.deduct()` (in-memory) runs before `cost_logger.record()` (DB). If DB write fails it's swallowed; counter has spend, DB doesn't. On process restart counter resets to zero while DB retained records — **drift in opposite direction**. No reconciliation path.
- **`tidus/cache/semantic_cache.py:93-121`** — SemanticCache filters only on `team_id`; does **NOT** filter by `privacy` level. A confidential response can be returned to a subsequent internal/public query from the same team above the 0.95 threshold. ExactCache docstring claims "Confidential tasks never cached (privacy guard in /complete)" — **the guard does not exist; neither cache is wired into `/complete` at all** (grep confirms zero callers outside tests).

### High
- `tidus/budget/enforcer.py:169-171` — `reset_period` for workflow-scoped policy resets wrong key: `self._counter.reset(policy.scope_id, None)` resets `(team_id=policy.scope_id, workflow_id=None)` when workflow counters are keyed `(team_id, workflow_id)`. **Workflow budgets effectively never reset.**
- `tidus/guardrails/session_store.py:21-55` — No TTL, no eviction, no cleanup task. Sessions accumulate until manually deleted. Unbounded memory leak.
- `tidus/budget/enforcer.py:65-115` — `can_spend` = check_and_add then add(-amount) — two separate lock operations. Another caller can observe tentative over-budget state.
- `tidus/cost/engine.py:72-102` — `estimate_from_counts` always applies 15% `_buffer_pct`, even though docstring says it's for post-execution actual counts.
- `tidus/metering/middleware.py:56,77-82` — `x-titus-user-id` header trusted verbatim, no JWT sub verification. Any caller can impersonate any user_id in metering. Also 4xx responses still metered (retry storm inflates counts).
- `tidus/auth/middleware.py:78-115` — Dev-mode grants admin when `OIDC_ISSUER_URL` empty; default bind is `0.0.0.0` — fresh install is admin-equivalent open service on any interface.
- `tidus/main.py:102-130` — `slowapi` Limiter constructed, attached to `app.state`, but **no endpoint uses `@_limiter.limit(...)`**. Rate limiting is dead code.

### Medium
- `tidus/api/v1/complete.py:146-197` — Cost deducted AFTER adapter returns. If response or audit write fails after that, user was charged with no idempotency key → client retry on 5xx double-charges.
- `tidus/auth/oidc.py:96-99` — `get_key` falls back to "first available key" when `kid` absent/unknown — masks key-id misconfigurations.
- `tidus/api/v1/dashboard.py:108-110` — `select(CostRecordORM).where(timestamp >= cutoff)` loads every row into memory; no limit, no team filter. OOM vector on 90-day window.
- `tidus/db/engine.py:141-160` — `pool_size=10, max_overflow=20`; no `pool_timeout` configured → waiting requests hang indefinitely. `echo=True` in dev spams logs.
- `tidus/observability/*` — No metrics for per-team spend, budget utilization, cache hit rate, guardrail rejections. Prom scrape gives no visibility into router core.
- `tidus/cost/counter.py:92-95` — `get_all()` returns full dict snapshot of all counters; unbounded as teams grow.

### Low
- `tidus/cache/exact_cache.py:70-82` — Evicts first 10% of keys by insertion order, not true LRU.
- `tidus/cache/semantic_cache.py:105-113` — O(N) linear scan per query.
- `tidus/auth/rbac.py:47-53` — Unknown role strings silently treated as unauthorized without logging.
- `tidus/cost/tokenizers.py:198-215` — Ollama tokenize response fully loaded into memory.
- `tidus/db/engine.py:109` — `metadata_ JSON nullable=True` has no size bound.
- `tidus/metering/service.py:180` — `get_trend_7d` fires 7 sequential COUNT DISTINCT queries (N+1).
- `tidus/metering/middleware.py:70` — Unbounded `asyncio.create_task` under burst.

---

## 5. API surface, MCP server, CLI wizard

### Critical
- **`tidus/mcp/server.py:40-41,85-86,147-148,168-169`** — Every handler calls `create_tables()` + `build_singletons()` per invocation. Rebuilds `EffectiveRegistry` from DB, re-instantiates `SpendCounter` on every tool call, wiping in-memory budget state. **Budgets never accumulate across an MCP session; `deduct()` is effectively a no-op.** This is the central bug of the MCP surface.
- **`tidus/cli/setup_wizard.py:1-267`** — Binary is `tidus-setup` not `tidus init` (per `pyproject.toml`). Wizard only creates `config/budgets.yaml`; does NOT create `.env`, seed OIDC admin, or initialize DB — those rely on app lifespan. First-run wizard failing to offer `.env` scaffolding is a completeness gap.
- **`tidus/main.py:182-184`** — `/ready` unconditionally returns `{"status":"ready"}` without checking DB, singleton completion, or scheduler state. K8s readinessProbe green-lights pods before lifespan finishes.

### High
- `tidus/api/v1/route.py:133-145` vs `tidus/api/v1/complete.py:118-129` — **Error envelope inconsistent** for same failure class. `route.py` emits `{error, message, failure_stage, rejections}`; `complete.py` emits `{detail, failure_stage, failure_reason, rejections}`. No RFC 7807. Clients must branch on endpoint.
- `tidus/auth/middleware.py:93-115` — Dev-mode grants admin on non-production environment; default bind is `0.0.0.0` (`main.py:232`). Fresh Tidus on any network interface is admin-equivalent.

### Medium
- `tidus/api/v1/dashboard.py:135` vs `tidus/api/v1/metering.py:49-51` — `/dashboard/summary` uses `get_current_user` (any role), exposes per-team budget rows + active session IDs to `read_only`/`service_account`.
- `tidus/cli/setup_wizard.py:232-235` — Printed curl example has malformed shell quoting (`'}}\''` produces literal `}}'`).
- `tidus/api/v1/subscribe.py:160-163` — Wraps `add_subscriber()` exceptions as HTTP 500 with raw `str(exc)` — file-write failures should be 503; could leak paths.
- `tidus/mcp/tools.py:12-117` — Schemas lack `description` fields for `team_id` (2 of 4 tools), `estimated_output_tokens`, `enabled_only`. `tidus_list_models` has no `vendor`/`tier`/`domain` filter for a 55-model catalog.
- `docs/mcp-integration.md:86-91` vs `tidus/mcp/server.py:59-65` — Docs claim output includes `tier: 2` but server also returns undocumented `score`. Docs show `tidus_get_budget_status` output without `has_policy` but server sets `limit_usd: null` when no policy.

### Low
- `tidus/cli/setup_wizard.py:144` — Banner reads "Tidus v1.0.0" but `main.py:118` is `1.1.0`.
- `tidus/api/v1/complete.py:131-133` — Raises `HTTPException(500, "Selected model not found")` for data-consistency bug; 503 more diagnosable.
- `tidus/main.py:219-221` — `/dash` → `/dashboard/index.html`, `/` → `/dashboard/` — two aliases, minor OpenAPI clutter.
- `tidus/mcp/server.py` — No health/ping tool; integrators can't test connectivity from Claude Desktop without logs.

---

## 6. Documentation

### Critical (users misled)
- **SSO/RBAC marked "Roadmap"** — `docs/index.md:38-40`, `docs/enterprise/sso.md:2`, `docs/enterprise/rbac.md:2`, `docs/pricing.md:96` say "Roadmap" but `tidus/auth/` is fully implemented, README Phase 8 Complete, `docs/troubleshooting.md:33-50` + `docs/deployment.md:67-70` say OIDC REQUIRED in production.
- **Pricing tiers contradictory** — `README.md:174-177` advertises only Community (free <1000 users) + Enterprise, but `docs/pricing.md:10-14` sells Pro ($99/100K req) and Business ($499/1M req). Customer can't tell what's actually for sale.
- **Model count + verification date drift** — `README.md:149` says "53 total across 11 vendors, verified 2026-04-03"; `docs/pricing.md:24` says "53 models, 12 vendors, verified 2026-04-17"; `docs/pricing.md:89` says "53-model, 11 vendors, 2026-04-03". Actual `config/models.yaml` has **55 models (45 enabled), 13 vendors, all `last_price_check: 2026-04-17`**.
- **New models undocumented** — YAML has `claude-opus-4-7`, `grok-4`, `deepseek-v4`, `gemini-nano`. None in README table, `docs/adapters.md`, `docs/pricing-sync.md:210-221`, or `docs/architecture.md:211` tier examples.

### High
- `docs/quickstart.md:146`, `first-15-minutes.md:129`, `savings-report.md:131` reference "Saved vs Baseline" KPI; `docs/dashboard.md:19-31` only lists 6 KPI cards, doesn't include it.
- `docs/dashboard.md:16` — "No login required in v0.1 (authentication is a Phase 8 feature)" — project is v1.1.0 and Phase 8 is complete.
- `docs/troubleshooting.md:162` — Says confidential tasks fail "Stage 2"; in code privacy is Stage 1 check 1.4 per `selection-algorithm.md:134-144`.
- `CHANGELOG.md` ends at v1.1.0 (2026-04-06); no "Unreleased"/v1.2.0 entry for in-progress auto-classification.

### Medium
- `docs/selection-algorithm.md:16` — Hardcodes "All 53 models" (now 55).
- `docs/quickstart.md` — Two `## 8.` sections (duplicate numbering).
- `docs/index.md:21-40` — Missing entries for `how-to-use.md`, `selection-algorithm.md`, `pricing-model.md`, `pricing-sync.md`, all `runbooks/*`.
- `docs/api-reference.md:76` — Says `model_disabled` fires for `deprecated=true`; `selector.py:220-224` says deprecated models are NOT rejected (just +0.15 penalty).
- `docs/architecture.md:181-184` — Describes v0.1 price sync; missing v1.1 multi-source consensus documented in `pricing-sync.md`.

### Low
- `docs/architecture.md:123` + `adapters.md:12` — Lists Anthropic `count_tokens` endpoint (deprecated per Anthropic).
- `docs/pricing-sync.md:51` — "41 models" in HardcodedSource possibly stale.
- `docs/deployment.md:34` — Docker `ollama pull llama4` doesn't match YAML `llama4-scout-ollama`/`llama4-maverick-ollama` tags.

---

## 7. Test suite

### Test run
- **487 passed, 0 failed, 0 skipped** (1 conditional skip in `test_reports_api.py`), ~37s wall
- **Coverage: 61% overall**, 6000 stmts, 2329 missed
- CI-safe: all DB tests use `sqlite+aiosqlite:///:memory:`; no real network

### Critical gaps
- **`tidus/cache/semantic_cache.py`** — **0% coverage** on Layer-2 cache. Invalidation, threshold, TTL entirely untested.
- **`tidus/reporting/pricing_report.py`, `landing_updater.py`** — **0% coverage** (605 stmts). Reports API tested only shallowly.
- **`tidus/sync/price_sync.py`** — **0%** on the sync entrypoint.
- **All 8 adapters: 23–31% coverage.** No error-path tests (auth failures, 429, malformed responses, timeouts).
- **Fallback chain** (`complete.py:156-171`) — Only one "fail once, succeed" case. No tests for: fallback also fails, multi-hop, fallback disabled, fallback budget reject, fallback tier ceiling.
- **`tidus/auth/oidc.py` — 21%**; `tidus/cli/setup_wizard.py` — **0%**.
- **Drift engine: 32%**, **scheduler: 29%**.
- No test combining ExactCache overflow eviction + TTL-expired entries (potential off-by-one in `_evict_ten_percent`).

### High (flaky/unreliable)
- `tests/unit/test_telemetry_writer.py:97-105` — **broken mock**: `coroutine ... was never awaited` warning; test passes trivially without exercising DB-error path.
- `tests/integration/test_cache_eviction_and_load.py:85,159`, `test_cache_correctness.py:134-164` — Mutate private `cache._store[...]` and monkey-patch module-level `time` — any refactor silently breaks these.

### Medium
- `tests/unit/test_selector.py:347-399` — Imports private `_score_and_pick`; impl-coupled.
- `tests/unit/test_selector.py:197-199` — `assert chosen != "premium"` too weak.
- `tests/integration/test_error_recovery.py:173-195` — Asserts on a test-local string constant, not the production error constant from `complete.py`.
- `tests/integration/test_complete_endpoint.py:22-27` — Module-scope `client` fixture shares `SpendCounter` across tests; fragile.
- `tests/integration/test_selector_real_registry.py:103` — `assert len(registry) >= 55` — brittle to weekly vendor churn.

### Low
- Deprecation warnings: fastapi 422/413, `google.generativeai` import path.
- Five `patch("asyncio.sleep")` in `test_pricing_sources.py` — correct but module-global side-effect.

---

## 8. Magazine + public HTML + dashboard

### Source of truth
- **`D:\dev\tidus\index.html`** — THE magazine (has `<!-- TIDUS:MAG-CARD:END -->` marker, `tidus.magazine` branding)
- **`D:\dev\tidus\www\index.html`** — Stale first-iteration SaaS marketing from 2026-04-02, never updated. **Contradicts current product on every axis.**
- **`D:\dev\tidus\tidus\dashboard\static\index.html`** — Live `/dashboard/` page; minimal shell, dynamic values from API

### Critical (user-facing wrong)
- **`index.html:871,:873`** — Claims "8 production adapters running today (Anthropic, OpenAI, Google, Mistral, DeepSeek, xAI, **Groq, Together AI**)". **Groq and Together AI adapters do not exist** in `tidus/adapters/`. Actual 8: anthropic, deepseek, google, mistral, **moonshot, ollama**, openai, xai. Moonshot/Ollama shipped but unadvertised; Groq/Together advertised but absent.
- **`index.html:741,:749,:1088,:1178,:1277,:1722,:1883,:1927`** — "53 models". Config has **55 (45 enabled)**; JS MODELS array has 42. Same page also says "43 Models Tracked" (`:1288`) — internally inconsistent.
- **`index.html:1107,:1117,:1137`** — Anthropic "3 models" (actual 4), Google DeepMind "5" (actual 6), xAI "2" (actual 3). Vendor cards not updated after this week's additions.
- **`index.html:1102,:750`** — "12 Vendors" (config has 13, including Ollama which is omitted from the vendor grid entirely).
- **`reports/pricing-2026-04-17.md:21`** — Enum serialization bug: `"Tier ModelTier.mid"` instead of `"2"`/`"mid"`. Only in "New Models" block for grok-4.

### High (stale)
- **`www/index.html`** — Entire file: SaaS pitch ($99/mo tier at :573, 10K/1M req caps, `api.tidus.ai/v1` at :481, `hello@tidus.ai` at :594, `github.com/tidusai/tidus` at :403, `twitter.com/tidusai` at :649, "87–96% savings" at :413). Real project is open-source, self-hosted, `github.com/kensterinvest/tidus`, `lapkei01@gmail.com`, 60–80% claim. **Recommend deletion or stub-redirect.**
- `www/index.html:658-661` — `// TODO: Replace with confirmed endpoint from Kenny`, posts to `/api/signup` (doesn't exist; real is `/api/v1/subscribe`). Silent localStorage fallback.
- `index.html:1929` — `claude-opus-4-7` in JS MODELS but narrative callouts at `:797,:1737,:1756,:1835,:1850` still cite `claude-opus-4-6` as premium baseline.

### Medium
- `tidus/dashboard/static/index.html:66` — Placeholder `claude-opus-4-6`; backend default `tidus/api/v1/reports.py:38` `_DEFAULT_BASELINE = "claude-opus-4-6"` should bump to 4-7.
- `tidus/__init__.py:8` — `__version__ = "1.0.0"` while `pyproject.toml:7`, `main.py:118,161` say `1.1.0`.
- `index.html:722-731` — Two nav entries both read "How It Works" (duplicate label, different anchors).
- `index.html:1138` — "Grok-3-Fast is the highest-cost routed model at $5/1M" — blended is $15/1M. Misleading.

### Low
- `www/sitemap.xml` — Not wired to anything canonical; should go with `www/`.
- `index.html` — Three different headline savings numbers: 60–80% (`:7`), 70% (`:743`), 62% (`:1363` ROI default).
- `index.html:933` — "xAI / Groq" combined pill inconsistent with separate cards at `:1136,:1141`.
- Subscribe form at `index.html:1884` — wired correctly to `/api/v1/subscribe`; works.

Pricing spot-check vs `hardcoded_source.py`: claude-haiku-4-5 $1/$5, deepseek-r1 $0.70/$2.50, kimi-k2.5 $0.60/$2.50, gpt-oss-120b $0.039/$0.10, claude-opus-4-7 $5/$25 — **all match report and magazine table** except the enum-serialization bug.

---

## Recommended fix order

### Phase 1 — Stop bleeding (this week)
1. Wire fallback path through full selector (`complete.py:156-190`)
2. Fix budget race (atomic check-and-deduct) (`enforcer.py:83-96`)
3. Wire `AgentGuard`/`SessionStore` into `/complete` (don't trust client `agent_depth`)
4. Add team filter to dashboard cost records + guardrails session + budgets write (4 cross-tenant leaks)
5. Fix RBAC hierarchy (`rbac.py:30-54`)
6. Fix MCP singleton rebuild (cache at module level) (`mcp/server.py`)

### Phase 2 — Data correctness (next 2 weeks)
7. Add retired-model path to registry pipeline (`pipeline.py:173`)
8. Implement the claimed recency tie-breaker in consensus (`consensus.py:134`)
9. Fix workflow budget reset keying (`enforcer.py:169-171`)
10. Move SpendCounter to Redis-backed impl for multi-worker
11. Wire cache layers into `/complete` with privacy guard (or remove docstring lie)
12. Fix `x-titus-user-id` header impersonation (`metering/middleware.py:56`)

### Phase 3 — Hygiene (ongoing)
13. Update `config/model_descriptions.yaml` for 4-7/grok-4/qwen-flash + fix stale Haiku/DeepSeek prices
14. Sync magazine (`index.html`) to 55 models + correct 8 adapters (anthropic/deepseek/google/mistral/moonshot/ollama/openai/xai; NOT Groq/Together)
15. Fix SSO/RBAC "Roadmap" mislabel in 4 doc files
16. Reconcile pricing tiers: Community/Pro/Business/Enterprise in README + pricing.md
17. Delete `www/` directory (or stub-redirect to root)
18. Add adapter timeouts + distinguished exception types + retry/backoff
19. Add tests for fallback chain, semantic cache, adapter error paths
20. Add audit + cost log on error paths in `/complete`

### Phase 4 — v1.2.0 blockers
21. Integrate auto-classification per existing `plan.md`/`revise.md`
22. Update CHANGELOG and README with v1.2.0 preview
23. Create `docs/auto-classification.md` + `/api/v1/classify` reference
