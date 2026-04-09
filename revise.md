# Project: Tidus
# Version: 1.1.0
# Plan Date: 2026-04-06
# Source: Claude Code implementation plan
# Description: Multi-Source Self-Healing Registry — upgrade from editable YAML to a
#              layered, versioned, self-healing model catalog with MAD-based consensus
#              pricing, drift detection, billing reconciliation, and Prometheus observability.

---

# Tidus v1.1.0 — Multi-Source Self-Healing Registry

## Deliverables

| Artifact | Path | Purpose |
|---|---|---|
| Claude plan file | `C:\Users\OWNER\.claude\plans\cozy-marinating-flame.md` | Internal planning reference |
| **Project plan** | **`D:\dev\tidus\revise.md`** | **External-facing plan for passing to other AI agents, CI systems, or documentation tools** |

`revise.md` is generated as the **first action after plan approval**, before any code is written.
It contains a front-matter header (project, version, date) followed by the full plan content.

---

## Context

Tidus v1.0.0 routes AI requests using a single `config/models.yaml` file loaded once at startup.
That design was appropriate for MVP but creates production risk: a bad edit breaks routing, pricing
errors are undetectable, health data is lost on restart, and there is no governance trail for changes.

v1.1.0 upgrades the registry from "editable YAML" to a **layered, versioned, self-healing catalog**:
- Pricing changes create audited DB revisions — never silent in-memory mutations
- Health/telemetry persists across restarts
- Scoped, RBAC-controlled overrides replace direct YAML edits
- Multi-source consensus pricing with MAD-based outlier detection
- Drift detectors auto-disable models whose behaviour diverges from the catalog
- Billing reconciliation flags cost leakage between registry prices and provider invoices
- Prometheus gauges expose staleness before it causes routing quality degradation

---

## Addressing the "Centralized Pricing API" Question

> Instead of local pricing sync, host calculations on Kenny's servers so clients get pricing
> automatically.

**Do not do this.** Three hard blockers:

1. **Privacy wall.** Token counting requires the actual messages. Sending messages to a third-party
   server is an instant disqualifier for HIPAA, SOC 2 Type II, GDPR, and air-gapped enterprise
   deployments. The self-hosted, on-prem positioning is the core enterprise selling point.

2. **Availability dependency.** Every AI routing decision would have a hard dependency on Kenny's
   server uptime. 99.9% uptime = 8.7 hours/year of degraded routing. Enterprise SLAs are 99.99%.

3. **Trust model.** Enterprises do not want a third party to know their AI usage patterns, team
   structures, or cost allocation strategy.

**The right design (included in this plan):**
A public **pricing feed** — a simple GET endpoint that returns `{model_id, input_price,
output_price, updated_at, confidence}` tuples. **No customer data. No messages. No team IDs.**
Clients pull it periodically and apply prices locally. Routing computation always stays on-prem.
This is the npm-registry model: data centralized, execution local. Off by default
(`TIDUS_PRICING_FEED_URL` env var enables it).

---

## Architecture: Three-Layer Merge

At request time the router reads an **EffectiveModelSpec** computed as:

```
EffectiveModelSpec = merge(
    Layer 1: base catalog  (DB, sync-generated, never hand-edited),
    Layer 2: overrides     (DB, RBAC-protected, human-controlled, scoped),
    Layer 3: telemetry     (DB, health probe output, measured runtime facts)
)
```

The existing `ModelSelector`, `CapabilityMatcher`, and all route endpoints see the same
`ModelRegistry` interface (`get()`, `list_all()`, `list_enabled()`) — they do not change.
`EffectiveRegistry` is a drop-in replacement that implements that interface.

**Merge precedence (priority order — highest wins):**

| Priority | Field / Override | Winner |
|---|---|---|
| **0 (highest)** | `emergency_freeze_revision` override | **Supersedes everything.** While active, `EffectiveModelSpec` returns the base catalog unchanged — no overrides applied, no telemetry applied, no mutations of any kind. Only lifting the freeze restores normal merge behaviour. |
| 1 | `hard_disable_model` override | Forces `enabled=false`; immune to telemetry re-enable; subordinate only to emergency freeze |
| 2 | `enabled=false` in base catalog | Base wins; telemetry cannot re-enable |
| 3 | `force_local_only`, `force_tier_ceiling` overrides | Win over corresponding base fields |
| 4 | `price_multiplier` override | Scales base input/output price at merge time |
| 5 | `latency_p50_ms` | Telemetry wins if measured within staleness window (see Telemetry Staleness Policy); reverts to base if stale |
| 6 (base) | `input_price`, `output_price`, `cache_read_price`, `cache_write_price` | Base wins |
| 6 (base) | `deprecated`, `capabilities` | Base always wins; capabilities can only be restricted by overrides |

**Override conflict rules:**
`emergency_freeze > hard_disable > force_local_only > force_tier_ceiling > price_multiplier`.
Conflicts detected at creation time, returned in API response as `conflicts: list[str]`, written
to audit log. Overlapping overrides are allowed — interaction is fully predictable via the table.

**Revision promotion state machine:**
```
PENDING → (Tier 1+2 validation) → VALIDATING → (Tier 3 canary with retries) → ACTIVE
ACTIVE → (new revision promoted) → SUPERSEDED
VALIDATING or PENDING → (any failure) → FAILED
ACTIVE → (emergency_freeze_revision override active) → promotion blocked
```
Only one revision is `ACTIVE`. Promotion is atomic (two-phase write — see Critical Safety
Mechanisms). `FAILED` revisions are never auto-retried. Admin can force-promote with justification
(bypasses Tier 3 only, not Tier 1/2). Rollback: admin re-promotes a `SUPERSEDED` revision
(skips Tier 3, already verified). `SUPERSEDED` revisions retained for 90 days (configurable).

**Schema Evolution Policy:**

All `spec_json` entries carry a `schema_version: int` (starting at 1). Rules:

| Rule | Detail |
|---|---|
| New fields require defaults | Any field added in a `schema_version` bump must default to a valid value so older readers can parse newer entries without error |
| Old revisions readable for 90 days | `REGISTRY_REVISION_RETENTION_DAYS=90`; within this window, routers can roll back to any retained revision |
| Routers tolerate unknown fields | `ModelSpec` loaded with `model_config = ConfigDict(extra='ignore')` — unknown fields in newer schema versions are silently dropped |
| No forward compatibility | A router running older code encountering a higher `schema_version` loads only known fields and logs `registry_schema_version_ahead` warning |
| Schema changes require a version bump | Any field removal or type change must increment `schema_version` and include an upgrade path in `RegistrySeeder` |
| Backward compat only | Routers are always upgraded before old revisions are retired; never the reverse |

**Router compatibility matrix:** Each `schema_version` bump ships with a new row in `docs/registry.md`:

| schema_version | Minimum router version | New fields / breaking change |
|---|---|---|
| 1 | 1.0.0 | Initial schema (all v1.1.0 fields) |

Routers encountering a `schema_version` higher than their maximum known version log
`registry_schema_version_ahead` and load only known fields (`extra='ignore'`). This matrix is the
canonical upgrade-dependency document — before retiring an old revision, confirm all replicas are
running ≥ the minimum router version for that revision's schema_version.

**Model Lifecycle:**

Models in the registry follow a one-way state progression:

| State | Routing behaviour | How to enter | How to exit |
|---|---|---|---|
| **Active** | Routed normally | Default; or lift `hard_disable` override | Deprecate or retire |
| **Deprecated** | Still routed; selector applies a small score penalty; logged as `routing_deprecated_model` | Set `deprecated=true` in base catalog (pipeline revision) | Retire or re-activate |
| **Retired** | Excluded from routing (same as `enabled=false`); still returned by `GET /api/v1/models` with `status='retired'` | `POST /api/v1/models/{id}/retire` (admin only); sets `retired_at` + `retirement_reason` in next revision | Cannot be un-retired (use a new model entry instead) |
| **Removed** | Not present in the ACTIVE revision; invisible to routing; exists in SUPERSEDED revisions for audit | Omit from a future pipeline revision | N/A — historical only |

`docs/registry.md` documents this lifecycle with the progression diagram and examples of each
transition. Retired models are preserved in revision history indefinitely (not subject to the
90-day `SUPERSEDED` revision retention window).

---

## Critical Safety Mechanisms

These address the seven high-risk gaps identified in design review. Each is tied to a specific
phase for implementation.

### 1. Atomic Publish with Two-Phase Write (Phase 3)

**Risk:** Routers reading `active_revision_id` while a revision is being published can see partial
state if entries and the status flip happen in separate transactions.

**Fix — two-phase write protocol in `RegistryPipeline`:**

```
Phase A (writes entries, safe to read — router ignores PENDING revisions):
  BEGIN;
  INSERT INTO model_catalog_revisions (revision_id, status='pending', ...)
  INSERT INTO model_catalog_entries (revision_id, model_id, spec_json, ...) [N rows]
  COMMIT;

Phase B (single-statement atomic flip after validation passes):
  BEGIN;
  UPDATE model_catalog_revisions SET status='superseded' WHERE status='active';
  UPDATE model_catalog_revisions SET status='active', activated_at=NOW() WHERE revision_id=$1;
  COMMIT;
```

Routers reading the DB between Phase A and Phase B continue to see the old `ACTIVE` revision
(they never read `PENDING` entries). The flip is a two-row update in one transaction — atomic.

### 2. Distributed Publisher Lock (Phase 3)

**Risk:** In a k8s 3-replica deployment, each replica runs its own APScheduler, causing 3 price
sync jobs to run simultaneously and creating 3 competing revision promotions.

**Fix — PostgreSQL advisory lock in `RegistryPipeline`:**

```python
SYNC_LOCK_KEY = 1_234_567_891  # stable integer key for this job

async def _acquire_lock(session) -> bool:
    result = await session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": SYNC_LOCK_KEY})
    return result.scalar()

async def _release_lock(session) -> None:
    await session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": SYNC_LOCK_KEY})
```

`run_price_sync_cycle()` acquires the lock at entry and releases it in a `finally` block.
If `pg_try_advisory_lock` returns false (another replica holds it), the job logs
`price_sync_skipped_lock_held` and returns — no error. For SQLite (dev mode): skip the
advisory lock call (detect via `DATABASE_URL` prefix check).

### 3. Revision-Aware Cache Invalidation (Phase 2)

**Risk:** `EffectiveRegistry` holds an in-memory cache with a 60-second TTL. During those 60
seconds after a new revision is promoted, different replicas serve different model specs.

**Fix — acceptable staleness window + revision key:**

The in-memory cache is keyed on `(active_revision_id, override_generation)`. When `refresh()`
detects that `active_revision_id` changed, it rebuilds the cache immediately regardless of TTL.
Each `refresh()` call (60-second interval) queries `SELECT revision_id, activated_at FROM
model_catalog_revisions WHERE status='active' LIMIT 1` — a cheap single-row indexed read.

For pricing data, up to 60 seconds of inter-replica inconsistency is acceptable: it means a
request may route using prices that are up to 60 seconds stale, not a safety issue. Document
this in deployment guides. Push invalidation (webhook/pub-sub) is deferred to post-v1.1.0.

### 4. Canary Probe Retries + Auditable Results (Phase 3)

**Risk:** Network blips or provider rate limits cause Tier 3 canary probes to fail and block
promotions unnecessarily. Single-pass 2/3 success has no retry logic.

**Fix — retry window with stored provenance:**

```python
class CanaryProbeResult:
    model_id: str
    attempts: int
    successes: int
    failure_reasons: list[str]
    verdict: Literal["pass", "fail", "skip"]  # skip = adapter not available/disabled
```

`CanaryProbe.run()` probes 3 randomly sampled models. Per model: up to `canary_max_attempts`
retries (default 3) with `canary_retry_delay_seconds` delay (default 30). A model passes if any
attempt succeeds. Overall revision passes if ≥ `canary_pass_rate` (default 0.67) of models pass.

All `CanaryProbeResult` objects are serialized to JSON and stored in
`model_catalog_revisions.canary_results` (new JSON column added in Phase 3 migration).

**Manual promotion override:** Admin can call `POST /api/v1/registry/revisions/{id}/force-activate`
with a mandatory `justification` field. This bypasses Tier 3 only (not Tier 1/2), writes an audit
log entry with `action='registry.force_promote'`, and completes the two-phase write.

### 5. MAD-Based Outlier Detection (Phase 3)

**Risk:** Naive percentage-from-median threshold rejects legitimate large vendor price changes
(e.g., a 50% price cut) or accepts coordinated bad data from two low-quality sources.

**Fix — Modified Z-Score using Median Absolute Deviation:**

```
For each model with N quotes:
  median_price = median(all quotes' input_price)
  MAD = median(|price_i - median_price| for each quote i)
  modified_z_score(i) = 0.6745 * |price_i - median_price| / MAD
  
  Reject quote if modified_z_score > 3.5 (configurable: consensus.outlier_z_threshold)
  If MAD == 0 (all sources agree exactly): no rejection, all quotes are consistent
```

Additional rules:
- If only one source: accept it but mark `single_source=True`, lower the effective confidence
  by 0.2, and increase drift alert sensitivity for that model.
- If all sources are rejected as outliers: reject the revision and alert (indicates a systemic
  data quality problem).
- Source confidence is used as a weight when choosing among non-outlier quotes (same model, two
  non-outlier sources → pick the higher-confidence one).

### 6. Override Expiry Enforcement Job (Phase 2)

**Risk:** `expires_at` fields exist in the schema but expired overrides remain active if nothing
enforces them, creating phantom restrictions.

**Fix — scheduled deactivation job:**

`OverrideExpiryJob` runs every 15 minutes (configurable). Executes:
```sql
UPDATE model_overrides
SET is_active=false, deactivated_at=NOW(), deactivated_by='system_expiry'
WHERE is_active=true AND expires_at IS NOT NULL AND expires_at < NOW()
RETURNING override_id, model_id, owner_team_id
```

Each deactivated row is written to `audit_logs` with `action='registry.override_expired'`.
After the batch, triggers `EffectiveRegistry.refresh()` so stale overrides are immediately
removed from the merge layer.

**Conflict detection at creation:** Before inserting a new override, `OverrideManager` queries
for existing active overrides on the same `(model_id, override_type, scope_id)` combination.
If one exists: warn in response (`conflicts: list[str]`) but do not block (the new override
coexists via deterministic precedence rules). Conflicts are also written to `audit_logs`.

### 7. Probe Sampling to Control Cost (Phase 4)

**Risk:** 53 models × 5-minute interval × live completion calls = expensive probe traffic that
inflates provider costs and may trigger rate limits.

**Fix — three-tier sampling with synthetic-first probing:**

Each health probe cycle classifies models into tiers:
- **Tier A (always probe this cycle):** models with `consecutive_failures > 0` or
  `drift_status='warning'` in recent telemetry.
- **Tier B (probe if not probed in last 30 min):** models with no recent probe or no telemetry.
- **Tier C (sample with 10% probability):** healthy models probed recently.

For Tier B and Tier C: run a **synthetic probe first** — call `adapter.count_tokens(model_id,
["hi"])` which is free on most vendors (no LLM call, just tokenization). If the synthetic probe
fails, escalate to a live `adapter.health_check()` call. Only Tier A always starts with a live
`health_check()`.

Track probe costs: `TelemetryWriter` records `probe_type: synthetic|live` per telemetry row.
The `MetricsUpdater` aggregates and exposes `tidus_probe_live_calls_total` and
`tidus_probe_synthetic_calls_total` as Counters.

---

## Six Phases

Each phase is independently deployable and leaves the system in a working state.

---

### Phase 1 — Foundation: DB Schema, Alembic Formalization, YAML Seeding

**Goal:** All tables under Alembic management. Five new registry tables created. YAML seeded into
DB as revision 0 on first run. System still behaves identically to v1.0.0.

#### Migration Strategy (critical)

Current state: baseline migration `f7ee6ab5176b` only creates `audit_logs`. The other 5 tables
(`cost_records`, `budget_policies`, `price_change_log`, `routing_decisions`, `ai_user_events`)
are created by `create_tables()` at startup outside Alembic's awareness.

**Two-migration catch-up strategy:**

- **Migration 2** (`catchup_existing_tables`): Uses `op.create_table(..., checkfirst=True)` for
  the 5 pre-existing tables. On a v1.0.0 production DB these are no-ops. On a fresh DB they
  create the tables. `downgrade()` is a no-op — never drop production data.
- **Migration 3** (`add_registry_tables`): Adds the 5 new registry tables (see below).
  `downgrade()` drops all 5 in reverse dependency order.

After deployment: `uv run alembic upgrade head`.

Note on `spec_json` schema versioning: each `model_catalog_entries.spec_json` object includes a
`schema_version: int` field (starting at 1). When `ModelSpec` gains new fields in future versions,
`EffectiveRegistry` reads `schema_version` and applies field defaults for missing keys. This allows
older revisions to be read by newer routers during a rolling rollout.

#### New Files

| File | Purpose |
|---|---|
| `tidus/db/registry_orm.py` | 5 new ORM classes sharing `Base` from `engine.py` |
| `tidus/registry/__init__.py` | Package marker |
| `tidus/registry/seeder.py` | `RegistrySeeder.seed_from_yaml()` — idempotent YAML → DB import |
| `alembic/versions/<hash>_catchup_existing_tables.py` | Catch-up migration |
| `alembic/versions/<hash>_add_registry_tables.py` | New registry tables |

#### New DB Tables (in `registry_orm.py`)

**`model_catalog_revisions`**
```
revision_id (PK String), created_at (DateTime server_default), activated_at (DateTime nullable),
source (String: yaml_seed|price_sync|manual), signature_hash (String),
status (String: pending|validating|active|superseded|failed), failure_reason (Text nullable),
canary_results (JSON nullable)   ← populated in Phase 3
```
Index on `status`.

**`model_catalog_entries`**
```
id (PK String), revision_id (FK→revisions, NOT NULL), model_id (String NOT NULL),
spec_json (JSON NOT NULL), schema_version (Integer default=1), created_at (DateTime server_default)
```
Unique composite index on `(revision_id, model_id)`.

**`model_overrides`**
```
override_id (PK), override_type (String), scope (global|team), scope_id (nullable),
model_id (nullable), payload (JSON), owner_team_id, justification (Text NOT NULL),
created_by, created_at, expires_at (nullable), is_active (Boolean default=true),
deactivated_at (nullable), deactivated_by (nullable)
```
Index on `(model_id, is_active)`. Index on `(scope_id, is_active)`.

**`model_telemetry`**
```
id (PK), model_id (String), measured_at (DateTime), latency_p50_ms (Integer nullable),
is_healthy (Boolean), consecutive_failures (Integer default=0),
context_exceeded_rate (Float nullable), token_delta_pct (Float nullable),
source (String: health_probe|request_log), probe_type (String: synthetic|live nullable)
```
Index on `(model_id, measured_at)`.

**`model_drift_events`**
```
id (PK), model_id (String), drift_type (String: context|latency|tokenization|price),
severity (String: warning|critical), detected_at (DateTime server_default),
metric_value (Float), threshold_value (Float),
drift_status (String: open|auto_resolved|manually_resolved),
resolved_at (nullable), audit_record_id (nullable FK→audit_logs.id),
active_revision_id (nullable FK→model_catalog_revisions.revision_id)
  ← the revision that was ACTIVE when this event was detected; useful for debugging
    whether a registry change caused the drift; populated by DriftEngine at detection time
```
Index on `(model_id, drift_status)`.

#### Modified Files

| File | Change |
|---|---|
| `tidus/db/engine.py` | Add `from tidus.db.registry_orm import *  # noqa` so Alembic autogenerate picks up new tables |
| `tidus/main.py` | After `build_singletons()`, call `await RegistrySeeder().seed_from_yaml(...)` |
| `tidus/settings.py` | Add `tidus_pricing_feed_url: str = ""` and `registry_revision_retention_days: int = 90` |
| `tidus/models/model_registry.py` | Add `cache_read_price: float = 0.0`, `cache_write_price: float = 0.0`, `retired_at: datetime \| None = None`, `retirement_reason: str \| None = None` to `ModelSpec`; add `POST /api/v1/models/{model_id}/retire` endpoint (admin only); retired models are excluded from routing (same as `enabled=false`) but preserved in revision history |

#### Tests

- `tests/unit/test_registry_seeder.py` — seeding creates 1 revision + N entries; idempotent on
  second call; transactional (no partial rows on commit failure); schema_version=1 on seed rows
- `tests/integration/test_migration_chain.py` — fresh DB: `alembic upgrade head` creates all 11
  tables; existing v1.0.0 DB: catch-up migration is a no-op

#### Docs

- `docs/configuration.md` — document `TIDUS_PRICING_FEED_URL`, `REGISTRY_REVISION_RETENTION_DAYS`
- `docs/architecture.md` — add "Registry Architecture" section with layer diagram

#### Verification

1. `uv run alembic upgrade head` on fresh SQLite — all 11 tables exist, no errors
2. Same command on existing 6-table DB — succeeds, no duplicate table errors
3. Server startup → `model_catalog_revisions` has 1 row, `model_catalog_entries` has 53 rows
4. Second restart → no duplicate revision rows (idempotency)
5. All 260 existing tests pass

---

### Phase 2 — Layered Registry: EffectiveRegistry, Override Engine, API

**Goal:** Replace in-memory `ModelRegistry` with `EffectiveRegistry` as the selector's backing
store. Add override CRUD API. Add override expiry enforcement job. The selector, capability
matcher, and all route handlers are unchanged.

#### Key Design: `build_singletons()` → `async`

`EffectiveRegistry.build()` requires a DB session (async). Change:
- `tidus/api/deps.py`: `def build_singletons()` → `async def build_singletons()`
- `tidus/main.py`: `build_singletons()` → `await build_singletons()`

`EffectiveRegistry` startup: reads active revision from DB; falls back to YAML load if no active
revision exists. Cached in-memory with a 60-second refresh interval via the scheduler.
Cache is keyed on `(active_revision_id, override_generation)` — invalidated immediately when
either changes (not just on TTL expiry).

#### New Files

| File | Purpose |
|---|---|
| `tidus/registry/effective_registry.py` | `EffectiveRegistry` — drop-in replacement with merge layer, revision-aware cache key |
| `tidus/registry/merge.py` | Pure merge functions: `merge_spec()`, `apply_price_multiplier()` |
| `tidus/registry/override_manager.py` | `OverrideManager` CRUD + RBAC + conflict detection |
| `tidus/registry/telemetry_reader.py` | `TelemetryReader.get_latest_snapshot()` with staleness window |
| `tidus/models/registry_models.py` | `ModelOverride`, `TelemetrySnapshot`, `CreateOverrideRequest`, `RevisionSummary` |
| `tidus/api/v1/registry.py` | Registry API router (see endpoints below) |
| `tidus/db/repositories/registry_repo.py` | Thin async SQLAlchemy repo for registry reads/writes |
| `tidus/sync/override_expiry.py` | `OverrideExpiryJob.run()` — deactivates expired overrides, writes audit entries |

#### Registry API Endpoints (new router, prefix `/api/v1/registry`)

| Method | Path | Role | Purpose |
|---|---|---|---|
| GET | `/revisions` | read_only+ | List revisions (summary) |
| GET | `/revisions/{id}` | read_only+ | Revision detail + entry count + canary_results |
| POST | `/revisions/{id}/activate` | admin | Rollback: re-promote SUPERSEDED revision |
| POST | `/revisions/{id}/force-activate` | admin | Bypass Tier 3 with mandatory justification |
| GET | `/overrides` | team_manager scoped, admin all | List active overrides |
| POST | `/overrides` | team_manager (own team), admin | Create override; returns conflicts in response |
| DELETE | `/overrides/{id}` | team_manager (own), admin | Deactivate override |
| GET | `/overrides/export` | admin | Export as HMAC-SHA256 signed YAML bundle |
| GET | `/drift` | read_only+ | List open drift events |
| POST | `/drift/{id}/resolve` | developer+ | Mark drift event manually resolved |
| GET | `/revisions/{id}/diff` | read_only+ | Field-level diff between two revisions (`?base={other_id}`); returns `{model_id, changed_fields: {field: {from, to}}}` per changed model |
| GET | `/revisions/{id}/preview` | read_only+ | Full merged `EffectiveModelSpec` for each model as it would appear **if this revision were promoted to ACTIVE** — combines the revision's base entries with current active overrides + current telemetry. Distinct from `/diff`: diff shows what changed vs another revision; preview shows the full merged result an operator would observe after promotion. Useful for validating a SUPERSEDED or PENDING revision before force-activating. |

RBAC: `team_manager` can only create/delete overrides where `scope_id == actor.team_id`.
All override create/delete/expiry actions written to `audit_logs`.

**Override types:**

| Type | Payload | Effect at merge |
|---|---|---|
| `price_multiplier` | `{multiplier: float}` | Scales input/output price at cost estimate |
| `hard_disable_model` | `{}` | Forces `enabled=false`, immune to telemetry |
| `force_tier_ceiling` | `{max_tier: int}` | Caps effective tier |
| `force_local_only` | `{}` | Filters non-local models for this team |
| `pin_provider` | `{vendor: str}` | Biases scoring toward specified vendor |
| `emergency_freeze_revision` | `{}` | Blocks all revision promotions until lifted |

#### Override Expiry Job

`OverrideExpiryJob.run(session_factory, registry)` runs every 15 minutes (new scheduler job).
Executes single batch UPDATE via raw SQL for atomicity (see Critical Safety Mechanisms §6).
Each deactivated row gets an `audit_logs` entry. After the batch: triggers `registry.refresh()`.

#### Modified Files

| File | Change |
|---|---|
| `tidus/api/deps.py` | `build_singletons()` → async; `EffectiveRegistry.build()` replaces `ModelRegistry.load()`; add `get_override_manager()` getter |
| `tidus/main.py` | `await build_singletons()`; register `registry` router; add override expiry job to scheduler |
| `tidus/sync/scheduler.py` | Add registry refresh (60s), override expiry (15min) jobs |

#### Tests

- `tests/unit/test_merge.py` — 9 tests: all precedence rules, non-mutation, conflict resolution
- `tests/unit/test_effective_registry.py` — YAML fallback, DB path, cache invalidates on revision change (not just TTL), refresh picks up new override
- `tests/unit/test_override_manager.py` — RBAC (team_manager own vs other), conflict detection returns warning not error, deactivate sets fields
- `tests/unit/test_override_expiry.py` — expired override deactivated by job, non-expired untouched, audit entry written
- `tests/integration/test_registry_api.py` — hard_disable removes model from routing within 60s; delete override re-enables; expired overrides not returned; force-activate requires justification; preview returns merged EffectiveModelSpec incorporating current overrides
- `tests/concurrency/test_revision_cache.py` — concurrent refresh calls do not produce duplicate DB queries; revision change detected in ≤ 1 refresh cycle

#### Docs

- `docs/registry.md` (new) — layered model, revision lifecycle, override types, rollback procedure, GitOps export format, conflict resolution rules
- `docs/enterprise/rbac.md` — document team_manager override scope permission
- `docs/architecture.md` — update request lifecycle diagram

#### Verification

1. `GET /api/v1/models` returns identical 53 models as v1.0.0 (selector interface unchanged)
2. POST hard_disable override → model absent from `list_enabled()` within 60s
3. Expired override → deactivated by expiry job within 15 min, model re-enabled
4. Two conflicting overrides on same model → both created, response contains `conflicts` field
5. Full integration test suite passes

---

### Phase 3 — Multi-Source Pricing + Validation Pipeline

**Goal:** Replace hardcoded `_KNOWN_PRICES` dict with pluggable `PricingSource` abstraction and
MAD-based consensus pricing. Price changes produce versioned DB revisions with full source
provenance. Three-tier validation with retry gates promotion. Atomic two-phase publish.

#### New Package: `tidus/sync/pricing/`

| File | Class | Purpose |
|---|---|---|
| `base.py` | `PricingSource` (ABC), `PriceQuote` | Source interface |
| `hardcoded_source.py` | `HardcodedSource` | Wraps `_KNOWN_PRICES` dict; `confidence=0.7` |
| `feed_source.py` | `TidusPricingFeedSource` | Pulls from `TIDUS_PRICING_FEED_URL`; disabled if URL empty; `confidence=0.85`; exponential backoff on failure (max 3 retries, 2s/4s/8s) |
| `consensus.py` | `PriceConsensus` | MAD outlier detection (Modified Z-Score); confidence weighting; exposes `single_source_models` |

`PriceQuote` fields: `model_id`, `input_price`, `output_price`, `cache_read_price`,
`cache_write_price`, `currency`, `effective_date`, `retrieved_at`, `source_name`,
`source_confidence`, `evidence_url (nullable)`.

`TidusPricingFeedSource`: sends only `GET {url}/prices?schema_version=1` — no customer data.
Response format: `{"prices": [{model_id, input_price, output_price, updated_at, confidence}]}`.
Failure modes: HTTP error, timeout (10s), malformed JSON → log warning, return `[]`.
Exponential backoff: uses `asyncio.sleep` with jitter; max 3 retries; records attempt count.

#### New Table: `pricing_ingestion_runs` (source provenance)

Added to `registry_orm.py` and a new Alembic migration:
```
run_id (PK), started_at, completed_at, source_name, status (success|failed|partial),
raw_payload (JSON),  ← full response from the source for audit/debug
model_count (Integer), quotes_valid (Integer), quotes_rejected (Integer),
rejection_reasons (JSON nullable),  ← {model_id: reason} for each rejected quote
error_message (Text nullable), revision_id_created (FK→revisions nullable)
```
`RegistryPipeline` writes one row per source per sync cycle.

#### New File: `tidus/registry/validators.py`

- **Tier 1 — SchemaValidator**: Pydantic validation. Required fields, price ≥ 0,
  context_window > 0, tier in 1–4, tokenizer in known enum.
- **Tier 2 — InvariantValidator**: Cross-field checks:
  - `min_complexity` order ≤ `max_complexity` order
  - Local models: `input_price=0` AND `output_price=0`
  - If confidential domain required: `is_local=True` or explicit override present
  - Adapter capabilities subset check (registry cannot tag multimodal if adapter is chat-only)
- **Tier 3 — CanaryProbe**: Up to 3 randomly sampled models. Per model: `canary_max_attempts`
  retries (default 3) with `canary_retry_delay_seconds` (default 30). Model passes if any
  attempt succeeds. Revision passes if ≥ `canary_pass_rate` (0.67) models pass. Results stored
  in `model_catalog_revisions.canary_results` JSON. Manual force-activate bypasses this tier.

#### New File: `tidus/registry/pipeline.py`

`RegistryPipeline.run_price_sync_cycle()`:
1. **Acquire distributed lock** via PostgreSQL advisory lock (skip on SQLite)
2. **Ingest**: Call all `is_available` sources concurrently via `asyncio.gather`; write one
   `pricing_ingestion_runs` row per source
3. **Consensus**: Run `PriceConsensus.resolve()` with MAD outlier detection
4. **Normalize**: Compute updated specs for models where price changed ≥ `change_threshold` (5%)
5. If no changes: release lock, return `None`
6. **Tier 1 + Tier 2 validation**: fail → set `status='failed'`, write audit entry, return `None`
7. **Phase A write**: Insert revision (`status='pending'`) + all entries in one transaction
8. **Tier 3 canary**: retry logic, store results in `canary_results`; fail → set `status='failed'`,
   release lock, return `None`
9. **Phase B atomic flip**: single transaction flips old `ACTIVE` → `SUPERSEDED` and new
   `PENDING` → `ACTIVE` (the two-phase write from Critical Safety Mechanisms §1)
10. Write `PriceChangeRecord` rows (backward compat), write audit entry
11. Trigger `EffectiveRegistry.refresh()`
12. Release lock, return `revision_id`

`RegistryPipeline.force_activate(revision_id, actor)`:
- Validates revision exists and is `SUPERSEDED` or `PENDING`
- Runs Tier 1 + Tier 2 only (skips Tier 3)
- Executes Phase B atomic flip
- Writes audit entry with `action='registry.force_promote'` and actor's justification

#### Pricing Feed Integrity

Feed responses must include `X-Tidus-Signature: hmac-sha256=<hex>`.
The HMAC is `HMAC-SHA256(raw_response_body, TIDUS_PRICING_FEED_SIGNING_KEY)`.

`TidusPricingFeedSource` verifies before parsing:
- `TIDUS_PRICING_FEED_SIGNING_KEY` set: reject missing or invalid signatures (log `pricing_feed_invalid_signature`, return `[]`)
- `TIDUS_PRICING_FEED_SIGNING_KEY` unset: accept unsigned but log `pricing_feed_unsigned` warning (operator must explicitly accept this risk)

New `settings.py` field: `tidus_pricing_feed_signing_key: str = ""`.
Ed25519 asymmetric signing (eliminates shared-secret distribution) is post-v1.1.0.

Rate guard: `TidusPricingFeedSource` enforces a minimum call interval of
`pricing_sync.min_feed_interval_seconds` (default 3600) — if the scheduler fires more
frequently, returns the last cached response to avoid accidental DDoS of the feed endpoint.

**Circuit breaker:** `TidusPricingFeedSource` implements CLOSED → OPEN → HALF-OPEN states:

| State | Behaviour | Transition |
|---|---|---|
| **CLOSED** | Normal — requests flow through; consecutive failure count increments on each error | After `pricing_feed_failure_threshold` (default 5) consecutive failures → OPEN |
| **OPEN** | Short-circuit — all requests return `[]` immediately, no network call; logs `pricing_feed_circuit_open` | After `pricing_feed_reset_timeout_seconds` (default 300) → HALF-OPEN |
| **HALF-OPEN** | One probe request allowed; success → CLOSED (reset counter); failure → back to OPEN | On probe result |

State is in-process only (not persisted to DB) — resets to CLOSED on restart, which is safe
because the circuit is a latency/cost guard, not a correctness guard (HardcodedSource always
provides a fallback). New settings fields: `pricing_feed_failure_threshold: int = 5` and
`pricing_feed_reset_timeout_seconds: int = 300`.

#### Dry-Run Mode

`RegistryPipeline.run_price_sync_cycle()` accepts `dry_run: bool = False`.
When `dry_run=True`: all validation and consensus run, but Phase A and Phase B writes are
skipped. Returns a `DryRunResult(would_change: list[dict], validation_errors: list[str])`.

`POST /api/v1/sync/prices?dry_run=true` exposes this. Useful for testing consensus logic and
debugging validation failures without creating a new revision.

#### Pipeline Rollback Guarantees

| State | What happens | Recovery |
|---|---|---|
| Phase A write fails mid-insert | Revision stays PENDING; entries may be partial | Next cycle's `_cleanup_stale_pending()` deletes entries and marks revision FAILED (PENDING > 1h) |
| Tier 1/2 validation fails | Revision set to FAILED; old ACTIVE untouched | No action needed |
| Tier 3 canary fails | Revision set to FAILED; old ACTIVE untouched | No action needed; admin can force-activate |
| Phase B flip fails (DB error) | DB transaction rolls back; old ACTIVE remains | No action needed — ACID guarantees |
| Force-activate fails | Revision stays in current state | No action needed |

`pricing_ingestion_runs` rows are always written, including for failed runs (`status='failed'`,
`error_message` populated). Complete audit trail regardless of pipeline outcome.

`_cleanup_stale_pending()`: called at start of each sync cycle. Sets `status='failed'` for any
PENDING revision older than 1 hour and deletes its `model_catalog_entries` rows.

#### Modified Files

| File | Change |
|---|---|
| `tidus/sync/price_sync.py` | Refactor to ~20 lines delegating to `HardcodedSource + PriceConsensus + RegistryPipeline`; same function signature |
| `tidus/sync/scheduler.py` | `_run_price_sync` assembles all available sources; passes to pipeline |
| `tidus/api/v1/sync.py` | Response includes `revision_id`, `sources_used`, `single_source_models`, `ingestion_run_ids`; old `changes` key preserved; `?dry_run=true` support |
| `tidus/settings.py` | Add `tidus_pricing_feed_signing_key: str = ""`, `pricing_feed_failure_threshold: int = 5`, `pricing_feed_reset_timeout_seconds: int = 300` |
| `pyproject.toml` | Add `respx>=0.21.0` to dev extras |

#### Tests

- `tests/unit/test_price_consensus.py` — MAD outlier rejected; legitimate large price change not
  rejected (MAD adjusts); single-source lowers confidence; all-sources-rejected → revision fails
- `tests/unit/test_validators.py` — 8 tests covering all invariants and all failure modes
- `tests/unit/test_pipeline.py` — 6 tests: new revision on change, no revision on identical prices,
  Tier 1/2 failure → FAILED status, Tier 3 failure → FAILED + old revision stays ACTIVE,
  force-activate skips Tier 3, advisory lock prevents concurrent sync
- `tests/integration/test_pricing_sources.py` — hardcoded source; feed disabled when URL empty;
  feed with mocked HTTP (respx); malformed feed response returns empty quotes; feed timeout returns
  empty (does not crash pipeline); delayed response still succeeds within timeout; circuit breaker
  opens after 5 consecutive failures (returns [] without network call); resets to HALF-OPEN after
  timeout; HALF-OPEN probe success → CLOSED; HALF-OPEN probe failure → OPEN again
- `tests/concurrency/test_pipeline_concurrent.py` — two simultaneous sync calls: only one creates
  a revision (advisory lock prevents the second from duplicating)

#### Docs

- `docs/pricing-sync.md` — complete rewrite: `PricingSource` hierarchy, MAD algorithm explanation,
  `TidusPricingFeedSource` (states explicitly: pricing data only, no customer data sent),
  how to add a custom `PricingSource`, provenance table schema
- `docs/registry.md` — pipeline state machine, validation tier descriptions, force-activate runbook

#### Verification

1. `POST /api/v1/sync/prices` → revision `status='active'` or `'failed'`, never `'pending'`
2. `TIDUS_PRICING_FEED_URL` unset → feed source excluded from consensus
3. Price change >5% → new ACTIVE revision, old SUPERSEDED
4. Identical prices → no new revision (idempotent)
5. Advisory lock test: concurrent sync → exactly 1 new revision
6. All Phase 1–2 tests pass

---

### Phase 4 — Drift Detection + Telemetry Persistence

**Goal:** Persist health probe output to DB (survives restarts). Intelligent probe sampling.
Four drift detectors. Auto-disable on critical drift.

#### New Package: `tidus/sync/drift/`

| File | Class | Drift type |
|---|---|---|
| `detectors.py` | `ContextDriftDetector` | `context_exceeded_rate` in recent `cost_records` vs declared `max_context` |
| `detectors.py` | `LatencyDriftDetector` | Measured P50 vs catalog P50 ratio exceeds threshold |
| `detectors.py` | `TokenizationDriftDetector` | Avg `(actual − estimated) / estimated` over 7d > 25% |
| `detectors.py` | `PriceDriftDetector` | >3 price changes in 30d, or >15% deviation from latest `price_change_log` |
| `engine.py` | `DriftEngine` | Runs all detectors concurrently; writes `ModelDriftEventORM`; auto-disables on critical |

All thresholds in `config/policies.yaml` under `drift:` section.

#### Telemetry Staleness Policy

`TelemetryReader` applies a 3-tier staleness window:

| Staleness | Treatment |
|---|---|
| < 24h | Use in merge (normal operation) |
| 24h – 72h | Treat as **unknown** — do NOT override base layer fields; log `telemetry_stale_warning`; merge returns base values |
| > 72h | Treat as **expired** — excluded from merge entirely; log `telemetry_expired_error`; increment `tidus_telemetry_stale_models_count` gauge |

When telemetry is unknown or expired:
- `EffectiveModelSpec` falls back to Layer 1 (base) for `latency_p50_ms` and health fields
- Models are **not auto-disabled** due to missing telemetry — only active probe failures trigger disable
- Drift detection **ignores** stale telemetry to prevent false positives during telemetry pipeline outages

This means a telemetry outage cannot cascade into mass model disablement.

**Note on exponential decay (deferred to post-v1.1.0):** The binary 24h/72h tiers are
conservative and easy to audit. A future improvement would apply exponential decay —
`confidence(t) = initial_confidence × e^(−λt)` — letting telemetry contributions degrade
smoothly rather than cutting off hard at 24h. Deferred because binary tiers are simpler to
reason about in incident postmortems and the 24h window is already generous for a 5-minute
probe interval.

#### Drift Remediation Policy

`DriftEngine` applies automated remediation, closing the detection→action loop:

| Severity | Automated Action |
|---|---|
| Warning | Escalate model to Tier A probe priority (always probed every cycle); no override created |
| Critical | Auto-create `hard_disable_model` override (`created_by='drift_engine'`, no expiry, justification contains drift type + metric value) |
| Recovery (3 consecutive healthy cycles after critical) | Auto-deactivate the drift_engine override; set `drift_event.drift_status='auto_resolved'` |

Auto-created overrides are distinguishable (`created_by='drift_engine'`) and visible in the
override list. Audit entry written with `action='registry.override_auto_created'`.
Admins can manually deactivate them at any time. Recovery detection: `DriftEngine` tracks last 3
probe results per critical model; on 3 consecutive healthy results, calls `OverrideManager.deactivate_override()`.

`DriftEngine` reads `EffectiveRegistry.active_revision_id` at detection time and writes it into
`model_drift_events.active_revision_id`. This creates a direct link from each drift event to the
registry state in effect when the drift was observed — critical for postmortems where a pricing
revision or a capability change may have caused the behavioural divergence.

#### New File: `tidus/sync/telemetry_writer.py`

`TelemetryWriter.write(session_factory, model_id, is_healthy, latency_ms, consecutive_failures,
probe_type)` — inserts one `ModelTelemetryORM` row. Non-fatal (exceptions logged, never raised).

#### Probe Sampling Strategy (in `health_probe.py`)

Each `run_once()` call classifies all models into three tiers (see Critical Safety Mechanisms §7):
- **Tier A:** always probe live
- **Tier B:** synthetic first (count_tokens), escalate to live only if synthetic fails
- **Tier C:** 10% sample, synthetic first

The `probe_type` field ('synthetic' or 'live') is passed to `TelemetryWriter` for cost tracking.

#### Modified Files

| File | Change |
|---|---|
| `tidus/sync/health_probe.py` | Add `session_factory=None` to `__init__`; implement 3-tier sampling; call `TelemetryWriter` after each probe; track `probe_type` |
| `tidus/sync/scheduler.py` | Pass `session_factory` to `HealthProbe`; add `DriftEngine` job (5-min interval) after health probe |
| `tidus/registry/effective_registry.py` | `refresh()` reads latest `model_telemetry` via `TelemetryReader`; includes in merge |
| `config/policies.yaml` | Add `drift:` section; add `health.synthetic_probe_first: true` |

#### Tests

- `tests/unit/test_drift_detectors.py` — 8 tests: each detector below/warning/critical threshold; drift event carries correct `active_revision_id` matching the registry state at detection time
- `tests/unit/test_telemetry_writer.py` — writes ORM row; non-fatal on DB error; probe_type stored
- `tests/unit/test_probe_sampling.py` — Tier A always probed; Tier B uses synthetic first; Tier C sampled at ~10%; probe_type correctly set per tier
- `tests/integration/test_health_probe_persistence.py` — probe writes telemetry; EffectiveRegistry refresh reads new telemetry; critical drift disables model; restart re-reads persisted telemetry; synthetic probe logged without live call charge

#### Docs

- `docs/registry.md` — Drift Detection section: threshold table, `model_drift_events` schema
- `docs/troubleshooting.md` — "Why did my model get auto-disabled?" runbook

#### Verification

1. `HealthProbe.run_once()` with session factory → rows in `model_telemetry` with `probe_type`
2. 3 simulated consecutive failures → `ModelDriftEventORM` + model disabled in-memory
3. Server restart → model stays disabled (telemetry persisted)
4. Tier C model at 10% sampling — confirm roughly 1-in-10 probe cycles produce a telemetry row
5. All Phase 1–3 tests pass

---

### Phase 5 — Billing Reconciliation

**Goal:** Detect cost leakage between registry prices and provider invoices. Basic CSV ingestion
connector + nightly reconciliation job + report endpoint.

#### Design

Enterprise customers receive provider billing exports (CSV/JSON per vendor). Tidus also tracks
actual spend in `cost_records`. Phase 5 ingests a **normalized billing CSV** and flags mismatches.

Scope for v1.1.0: normalized input format (user converts from raw provider CSV). Raw provider
parsing adapters (OpenAI, Anthropic, etc.) are post-v1.1.0.

**Normalized billing CSV format:**
```csv
model_id,date,provider_cost_usd
claude-opus-4-6,2026-04-01,127.45
gpt-4o,2026-04-01,89.20
```

#### New DB Table: `billing_reconciliations`

Added via new Alembic migration:
```
id (PK), reconciliation_date (Date), uploaded_at (DateTime server_default),
uploaded_by (String), model_id (String), tidus_cost_usd (Float),
provider_cost_usd (Float), variance_usd (Float), variance_pct (Float),
status (String: matched|warning|critical),
  matched:  |variance_pct| ≤ 0.05
  warning:  0.05 < |variance_pct| ≤ 0.25
  critical: |variance_pct| > 0.25
notes (Text nullable)
```
Composite index on `(model_id, reconciliation_date)`.

#### New Files

| File | Purpose |
|---|---|
| `tidus/billing/__init__.py` | Package marker |
| `tidus/billing/reconciler.py` | `BillingReconciler.reconcile(csv_rows, date_from, date_to, session_factory)` |
| `tidus/billing/csv_parser.py` | Validates and parses normalized CSV format |
| `tidus/api/v1/billing.py` | Billing API router (see endpoints below) |

#### Billing API Endpoints (new router, prefix `/api/v1/billing`)

| Method | Path | Role | Purpose |
|---|---|---|---|
| POST | `/reconcile` | team_manager, admin | Upload normalized billing CSV + trigger reconciliation |
| GET | `/reconciliations` | team_manager scoped, admin all | List reconciliation results |
| GET | `/reconciliations/summary` | team_manager scoped, admin all | Aggregate: total variance, critical models |

`POST /billing/reconcile` body: multipart form with `file` (CSV), `date_from`, `date_to`.
Returns: `{reconciliation_count, warnings, criticals, total_variance_usd}`.

`BillingReconciler.reconcile()`:
1. Parse CSV rows via `BillingCsvParser.parse()`
2. Query `cost_records` aggregated by `(model_id, date)` for the date range
3. For each (model_id, date) pair: compute `variance_pct = (tidus_cost - provider_cost) / provider_cost`
4. Write `billing_reconciliations` rows
5. Write audit entry (`action='billing.reconcile'`, metadata includes summary stats)
6. Return summary

#### Modified Files

| File | Change |
|---|---|
| `tidus/main.py` | Register `billing` router |

#### Tests

- `tests/unit/test_billing_reconciler.py` — matched case; warning case; critical case; zero provider cost edge case; missing model in cost_records; date range filtering
- `tests/unit/test_billing_csv_parser.py` — valid CSV; missing columns; invalid floats; empty file; encoding edge cases (UTF-8 BOM)
- `tests/integration/test_billing_api.py` — upload CSV → reconciliation rows in DB; GET returns scoped results; team_manager cannot see other teams' reconciliations

#### Docs

- `docs/billing-reconciliation.md` (new) — purpose, normalized CSV format spec, interpretation of warning/critical statuses, how to convert raw provider exports to normalized format (manual instructions for OpenAI, Anthropic, Google as the three most common)

#### Verification

1. Upload a CSV with one critical variance model → `status='critical'` row in DB
2. Summary endpoint shows `criticals=1`
3. Team A's upload not visible to Team B's team_manager
4. Zero variance across all models → all rows `status='matched'`
5. All Phase 1–4 tests pass

---

### Phase 6 — Prometheus Metrics, Alerting Rules, Runbooks, Release

**Goal:** 9 custom Prometheus gauges/counters. Alerting rules. Updated Grafana dashboard.
Runbooks. Version bump to 1.1.0.

#### New Package: `tidus/observability/`

**`registry_metrics.py`** — all metrics using explicit `prometheus_client`:

```
# Gauges
tidus_registry_last_successful_sync_timestamp        (no labels)
tidus_registry_active_revision_activated_timestamp   (no labels)
tidus_registry_model_last_price_update_timestamp     (labels: model_id)
tidus_registry_model_confidence                      (labels: model_id)
tidus_registry_active_revision_id                    (no labels — deterministic int hash of UUID)
tidus_registry_models_stale_count                    (no labels)

# Counters
tidus_probe_live_calls_total                         (labels: model_id, result: success|fail)
tidus_probe_synthetic_calls_total                    (labels: model_id, result: success|fail)
tidus_registry_drift_events_total                    (labels: model_id, drift_type, severity)
```

Note on `active_revision_id`: `int(revision_uuid.replace('-',''), 16) % (2**53)` — deterministic
integer that changes every revision promotion, sufficient for "revision changed" alerting.

**`metrics_updater.py`** — `MetricsUpdater.update(registry, session_factory)`:
Called every 5 minutes. Staleness definition: `last_price_check` older than 8 days.
Also reads `pricing_ingestion_runs` to set `last_successful_sync_timestamp`.

#### New File: `monitoring/alerting-rules.yaml`

```yaml
groups:
  - name: tidus-registry
    rules:
      - alert: TidusRegistrySyncStale
        expr: (time() - tidus_registry_last_successful_sync_timestamp) > 172800
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "Registry sync has not completed in >2 days"
      - alert: TidusRegistrySyncCriticallyStale
        expr: (time() - tidus_registry_last_successful_sync_timestamp) > 604800
        for: 5m
        labels: { severity: critical }
        annotations:
          summary: "Registry sync has not completed in >7 days"
      - alert: TidusRegistryStaleModelCount
        expr: tidus_registry_models_stale_count > 10
        for: 30m
        labels: { severity: warning }
        annotations:
          summary: "{{ $value }} models have stale price data (>8 days)"
      - alert: TidusActiveRevisionAged
        expr: (time() - tidus_registry_active_revision_activated_timestamp) > 2592000
        for: 1h
        labels: { severity: info }
        annotations:
          summary: "Active revision is older than 30 days"
      - alert: TidusProbeHighFailureRate
        expr: rate(tidus_probe_live_calls_total{result="fail"}[10m]) > 0.5
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "Health probe failure rate >50% for 5 minutes"
```

#### New File: `docs/runbooks/emergency-freeze.md`

Emergency registry freeze procedure:
1. Call `POST /api/v1/registry/overrides` with `override_type=emergency_freeze_revision` (admin)
2. Confirm `GET /api/v1/registry/overrides` shows freeze active
3. Investigate root cause (check `model_drift_events`, `pricing_ingestion_runs`)
4. Rollback if needed: `POST /api/v1/registry/revisions/{superseded_id}/activate`
5. Remove freeze: `DELETE /api/v1/registry/overrides/{freeze_override_id}`
6. Write post-mortem to audit log

#### New File: `docs/runbooks/drift-incident.md`

Drift incident runbook:
1. Alert fires → check `GET /api/v1/registry/drift` for open events
2. Identify drift type: context/latency/tokenization/price
3. For critical drift: model already auto-disabled → verify via `GET /api/v1/models`
4. Investigate: query `model_telemetry` for trend; query `cost_records` for context overflow rate
5. Options: wait for auto-recovery (latency spike), adjust threshold in `policies.yaml`
   (if false positive), or create `hard_disable_model` override (if confirmed regression)
6. Resolve: `POST /api/v1/registry/drift/{id}/resolve` with resolution notes

#### New File: `docs/runbooks/override-abuse.md`

Override abuse response:
1. Detect: unusual `audit_logs` entries (`action` like `registry.override.*`)
2. Revoke: `DELETE /api/v1/registry/overrides/{id}` (admin)
3. Review: query `audit_logs WHERE actor_sub=$suspect AND action LIKE 'registry.%'`
4. If API key compromised: rotate key via SSO provider; review all overrides created by that actor
5. Escalate to security team if cross-team scope violations detected

#### Modified Files

| File | Change |
|---|---|
| `tidus/sync/scheduler.py` | Add `MetricsUpdater` job (5-min interval) |
| `tidus/main.py` | Call `await MetricsUpdater().update(...)` at startup; register billing router |
| `tidus/registry/pipeline.py` | On successful sync: update sync timestamp + per-model gauges |
| `monitoring/grafana-dashboard.json` | Add "Registry Health" panel group: 4 panels (sync age, stale count, confidence distribution, active revision age) + 2 probe panels (live vs synthetic call rate, failure rate) |
| `pyproject.toml` | Explicit `prometheus-client>=0.20.0`; bump `version="1.1.0"` |
| `CHANGELOG.md` | Full v1.1.0 entry |

#### Tests

- `tests/unit/test_metrics_updater.py` — all 9 metrics set; stale count correct; revision hash changes on new revision; probe counters increment
- `tests/unit/test_registry_metrics.py` — metric names correct; hash function stable and deterministic

#### Docs

- `docs/deployment.md` — "Registry Metrics" section: all 9 metrics, labels, recommended scrape interval
- `monitoring/README.md` — alerting rules setup (`rule_files:` stanza), Grafana import procedure

#### Verification

1. `GET /metrics` contains all 9 `tidus_registry_*` / `tidus_probe_*` metric names after startup
2. `POST /api/v1/sync/prices` → `tidus_registry_last_successful_sync_timestamp` updates
3. Grafana dashboard imports without errors; 6 new panels display data
4. `promtool check rules monitoring/alerting-rules.yaml` exits 0
5. All 350+ tests pass (full suite)

---

## Files Summary

### New Files (33 total)

```
tidus/db/registry_orm.py
tidus/registry/__init__.py
tidus/registry/seeder.py
tidus/registry/effective_registry.py
tidus/registry/merge.py
tidus/registry/override_manager.py
tidus/registry/telemetry_reader.py
tidus/registry/pipeline.py
tidus/registry/validators.py
tidus/models/registry_models.py
tidus/sync/pricing/__init__.py
tidus/sync/pricing/base.py
tidus/sync/pricing/hardcoded_source.py
tidus/sync/pricing/feed_source.py
tidus/sync/pricing/consensus.py
tidus/sync/drift/__init__.py
tidus/sync/drift/detectors.py
tidus/sync/drift/engine.py
tidus/sync/telemetry_writer.py
tidus/sync/override_expiry.py
tidus/db/repositories/registry_repo.py
tidus/api/v1/registry.py
tidus/billing/__init__.py
tidus/billing/reconciler.py
tidus/billing/csv_parser.py
tidus/api/v1/billing.py
tidus/observability/__init__.py
tidus/observability/registry_metrics.py
tidus/observability/metrics_updater.py
alembic/versions/<hash>_catchup_existing_tables.py
alembic/versions/<hash>_add_registry_tables.py
alembic/versions/<hash>_add_pricing_ingestion_runs.py
alembic/versions/<hash>_add_billing_reconciliations.py
monitoring/alerting-rules.yaml
monitoring/README.md
docs/runbooks/emergency-freeze.md
docs/runbooks/drift-incident.md
docs/runbooks/override-abuse.md
```

### Modified Files (14 total)

```
tidus/db/engine.py                 (import registry_orm for Alembic)
tidus/models/model_registry.py    (add cache_read_price, cache_write_price to ModelSpec)
tidus/api/deps.py                  (async build_singletons; EffectiveRegistry; override_manager getter)
tidus/main.py                      (await build_singletons; seed call; new routers; metrics init)
tidus/sync/scheduler.py            (registry refresh, override expiry, drift engine, metrics updater jobs)
tidus/sync/health_probe.py         (session_factory; 3-tier sampling; TelemetryWriter; probe_type tracking)
tidus/sync/price_sync.py           (refactor to delegate to pricing package)
tidus/api/v1/sync.py               (richer response, backward-compatible)
tidus/settings.py                  (5 new settings: pricing_feed_url, revision_retention_days, pricing_feed_signing_key, pricing_feed_failure_threshold, pricing_feed_reset_timeout_seconds)
config/policies.yaml               (drift: section; health.synthetic_probe_first)
monitoring/grafana-dashboard.json  (6 new panels)
pyproject.toml                     (version bump; prometheus-client; respx dev dep)
CHANGELOG.md                       (v1.1.0 entry)
```

---

## Additional Test Matrix

Beyond the per-phase tests listed above, the following cross-cutting test files are required:

| Test File | Covers |
|---|---|
| `tests/concurrency/test_pipeline_concurrent.py` | Advisory lock prevents duplicate revisions from simultaneous sync calls |
| `tests/concurrency/test_revision_cache.py` | Concurrent refresh calls don't double-query DB; revision change detected within 1 cycle |
| `tests/integration/test_revision_rollback.py` | Promote SUPERSEDED revision via activate endpoint; router serves new spec within 60s |
| `tests/integration/test_canary_retry.py` | Canary passes on 2nd attempt after initial failure (mocked adapter fails once then succeeds) |
| `tests/integration/test_billing_api.py` | Full CSV upload → reconciliation → GET reconciliations flow |
| `tests/unit/test_mad_outlier.py` | MAD rejects statistical outlier; legitimate large price cut not rejected; all-agree MAD=0 edge case |
| `tests/unit/test_probe_sampling.py` | Tier classification; Tier C sampled at ~10%; synthetic-first escalation |
| `tests/unit/test_emergency_freeze.py` | Emergency freeze suppresses all override and telemetry merging; lifting freeze restores normal behaviour |
| `tests/unit/test_schema_evolution.py` | Unknown fields in spec_json ignored; schema_version mismatch logs warning; missing fields use defaults |
| `tests/unit/test_telemetry_staleness.py` | <24h telemetry applied; 24-72h treated as unknown (base fallback); >72h excluded; model NOT auto-disabled on stale telemetry |
| `tests/unit/test_drift_remediation.py` | Critical drift creates hard_disable override; 3 healthy cycles auto-deactivates it; warning drift escalates probe tier |
| `tests/unit/test_pipeline_dry_run.py` | Dry-run returns would-change list; no DB writes occur; validation errors returned correctly |
| `tests/unit/test_feed_signature.py` | Valid HMAC accepted; tampered body rejected; missing signature with key set rejected; no key → accept with warning |
| `tests/integration/test_revision_diff.py` | Diff between two revisions shows only changed models and fields |
| `tests/integration/test_model_retirement.py` | Retired model excluded from routing; preserved in revision history |
| `tests/unit/test_circuit_breaker.py` | Feed source opens after threshold failures; HALF-OPEN probe resets on success; resets to OPEN on HALF-OPEN failure; state does not persist across restart |
| `tests/integration/test_revision_preview.py` | Preview of a SUPERSEDED revision applies current overrides + current telemetry on top of the revision's base entries; preview of a PENDING revision before promotion shows expected merged state |
| `tests/unit/test_drift_revision_link.py` | Drift events carry `active_revision_id` of the registry state at detection; a revision promotion does not retroactively change existing event's revision_id |

---

## Backward Compatibility Guarantees

- `ModelRegistry.load()` continues to work (used as seeder input; nothing removes it)
- `ModelSelector.__init__` signature unchanged
- `CapabilityMatcher` unchanged
- `GET /api/v1/models` response schema unchanged
- `POST /api/v1/sync/prices` still works; response is a strict superset of v1.0.0
- `HealthProbe.__init__` gains one optional kwarg (`session_factory=None`) — non-breaking
- `price_change_log` table continues to be populated (existing audit queries unaffected)
- `ModelSpec.cache_read_price` and `cache_write_price` default to 0.0 — existing specs unaffected

---

## Post-v1.1.0 Deferred Items

These are explicitly out of scope for v1.1.0. They are listed here so they are not forgotten.

| Item | Why Deferred |
|---|---|
| KMS/HSM for provider API keys (envelope encryption) | Requires key management infrastructure; high effort |
| Automatic key rotation | Requires KMS first |
| Mutual TLS for hosted API; IP allowlists | Infrastructure work; out of scope for app-layer v1.1.0 |
| PostgreSQL row-level security policies | v1.1.0 RBAC + query scoping is sufficient; RLS is hardening |
| Full SBOM and SCA scanning in CI | CI tooling work; independent of registry feature |
| Pen test schedule and threat model document | Requires external security team engagement |
| Regional endpoints / data residency options | Infrastructure/hosting decision |
| Raw provider CSV parsers (OpenAI, Anthropic, Google) for billing | High effort per vendor; normalized CSV is v1.1.0 |
| Billing ingestion via S3 / BigQuery / Cloud Billing API | Connectors work; higher effort |
| Push invalidation for EffectiveRegistry (webhook/pub-sub) | 60s staleness is acceptable for pricing; push is optimization |
| Signed bundle import (`POST /registry/overrides/import`) | Export-only is sufficient for v1.1.0 GitOps story |
| Chaos tests and full pen test | Post-hardening work |
| **Multi-region / HA database strategy** | **Budget-deferred.** Advisory locks work correctly for single-region PostgreSQL. When multi-region is needed: price sync publisher runs primary-region only; telemetry writes must be idempotent across regions; canary probes run in one region only to avoid cost explosion; read replicas need consistency annotations. Design to be added when infrastructure budget permits. |
| Ed25519 asymmetric signing for pricing feed | Upgrade path from HMAC-SHA256; eliminates shared-secret distribution to each Tidus instance |
| Exponential decay for telemetry staleness | Binary 24h/72h tiers are simpler to audit; `confidence(t) = C₀ × e^(−λt)` model deferred until operational value is proven |

---

## End-to-End Verification (full suite)

1. `uv run alembic upgrade head` on fresh DB → 13 tables, no errors
2. `uv run alembic upgrade head` on existing v1.0.0 DB → succeeds, no duplicate tables
3. Server startup → 1 revision row, 53 entry rows seeded from YAML
4. `POST /api/v1/sync/prices` → new revision in DB with `status='active'` or `'failed'`
5. Concurrent sync calls → exactly 1 new revision (advisory lock test)
6. Create `hard_disable_model` override → model absent from routing within 60s
7. Wait for expiry job → expired override deactivated, model re-enabled within 15 min
8. Simulate 3 probe failures → `model_drift_events` row, model disabled in routing
9. Server restart → disabled model stays disabled (telemetry persisted)
10. Upload billing CSV with critical variance → `status='critical'` row in DB
11. `GET /metrics` → all 9 `tidus_registry_*` / `tidus_probe_*` metrics present
12. `promtool check rules monitoring/alerting-rules.yaml` → exit 0
13. Emergency freeze override → `EffectiveModelSpec` for any model equals base catalog only (no overrides or telemetry applied)
14. `POST /api/v1/sync/prices?dry_run=true` → returns would-change list, no new revision in DB
15. Pricing feed with invalid HMAC → source returns `[]`, pipeline does not use feed data
16. `GET /api/v1/registry/revisions/{new_id}/diff?base={old_id}` → lists only changed models/fields
17. `POST /api/v1/models/{model_id}/retire` → model excluded from routing, visible in revision history
18. Feed source: 5 consecutive mock failures → circuit OPEN (no network calls); after 300s reset → HALF-OPEN → one probe succeeds → CLOSED
19. `GET /api/v1/registry/revisions/{superseded_id}/preview` → returns merged EffectiveModelSpec reflecting current overrides, not ACTIVE revision's base entries
20. `model_drift_events.active_revision_id` matches `model_catalog_revisions.revision_id` for the revision active at detection time
21. `uv run pytest tests/ -x` → all tests pass (target: 400+ with all new tests)
