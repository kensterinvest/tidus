"""ORM models for the v1.1.0 self-healing model registry.

Phase 1 tables:
  model_catalog_revisions  — versioned snapshots of the full model catalog
  model_catalog_entries    — per-model spec within a revision (spec_json column)
  model_overrides          — scoped, RBAC-controlled overrides applied at merge time
  model_telemetry          — persisted health-probe measurements (survives restarts)
  model_drift_events       — behavioural divergence detections with auto-remediation

Phase 3 tables:
  pricing_ingestion_runs   — per-source audit trail for each price sync cycle
  model_price_snapshots    — weekly full-catalog price snapshot for time-series graphs

All classes share the Base declared in tidus.db.engine so they are picked up
by Alembic autogenerate when engine.py imports this module.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

from tidus.db.engine import Base


class ModelCatalogRevisionORM(Base):
    """One row per versioned catalog snapshot.

    status lifecycle: pending → (validating) → active → superseded
                                              ↘ failed (at any point before active)
    Only one revision is ACTIVE at a time.
    """

    __tablename__ = "model_catalog_revisions"

    revision_id = Column(String, primary_key=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    activated_at = Column(DateTime, nullable=True)
    source = Column(String, nullable=False)  # yaml_seed | price_sync | manual
    signature_hash = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, default="pending")  # pending|validating|active|superseded|failed
    failure_reason = Column(Text, nullable=True)
    canary_results = Column(JSON, nullable=True)  # populated in Phase 3

    __table_args__ = (
        Index("ix_model_catalog_revisions_status", "status"),
    )


class ModelCatalogEntryORM(Base):
    """One row per model per revision.

    spec_json stores the full ModelSpec as a dict including schema_version.
    Readers use extra='ignore' on ModelSpec so unknown fields in newer
    schema versions are silently dropped during a rolling upgrade.
    """

    __tablename__ = "model_catalog_entries"

    id = Column(String, primary_key=True)
    revision_id = Column(
        String,
        ForeignKey("model_catalog_revisions.revision_id"),
        nullable=False,
        index=True,
    )
    model_id = Column(String, nullable=False)
    spec_json = Column(JSON, nullable=False)
    schema_version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("revision_id", "model_id", name="uq_catalog_entry_rev_model"),
    )


class ModelOverrideORM(Base):
    """RBAC-controlled scoped override applied at merge time.

    override_type controls which layer-2 merge rule fires.
    Expired overrides (expires_at < now) are deactivated by OverrideExpiryJob.
    """

    __tablename__ = "model_overrides"

    override_id = Column(String, primary_key=True)
    override_type = Column(String, nullable=False)  # price_multiplier|hard_disable_model|...
    scope = Column(String, nullable=False, default="global")  # global | team
    scope_id = Column(String, nullable=True)
    model_id = Column(String, nullable=True)
    payload = Column(JSON, nullable=False, default=dict)
    owner_team_id = Column(String, nullable=False)
    justification = Column(Text, nullable=False)
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    deactivated_at = Column(DateTime, nullable=True)
    deactivated_by = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_model_overrides_model_active", "model_id", "is_active"),
        Index("ix_model_overrides_scope_active", "scope_id", "is_active"),
    )


class ModelTelemetryORM(Base):
    """One row per health probe observation.

    Persisted so that model health state survives server restarts.
    EffectiveRegistry reads the most recent row per model_id with a
    freshness check (< 24h = use, 24–72h = base fallback, > 72h = exclude).
    """

    __tablename__ = "model_telemetry"

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    measured_at = Column(DateTime, server_default=func.now(), nullable=False)
    latency_p50_ms = Column(Integer, nullable=True)
    is_healthy = Column(Boolean, nullable=False)
    consecutive_failures = Column(Integer, default=0, nullable=False)
    context_exceeded_rate = Column(Float, nullable=True)
    token_delta_pct = Column(Float, nullable=True)
    source = Column(String, nullable=False, default="health_probe")  # health_probe | request_log
    probe_type = Column(String, nullable=True)  # synthetic | live

    __table_args__ = (
        Index("ix_model_telemetry_model_measured", "model_id", "measured_at"),
    )


class ModelDriftEventORM(Base):
    """One row per detected behavioural divergence.

    active_revision_id links the event to the catalog revision that was ACTIVE
    at detection time — useful in postmortems to correlate drift with a pricing
    or capability change that happened in a recent revision promotion.
    """

    __tablename__ = "model_drift_events"

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    drift_type = Column(String, nullable=False)  # context | latency | tokenization | price
    severity = Column(String, nullable=False)  # warning | critical
    detected_at = Column(DateTime, server_default=func.now(), nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    drift_status = Column(String, nullable=False, default="open")  # open | auto_resolved | manually_resolved
    resolved_at = Column(DateTime, nullable=True)
    audit_record_id = Column(String, ForeignKey("audit_logs.id"), nullable=True)
    active_revision_id = Column(
        String,
        ForeignKey("model_catalog_revisions.revision_id"),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_model_drift_events_model_status", "model_id", "drift_status"),
    )


class PricingIngestionRunORM(Base):
    """Audit record for a single pricing source fetch within a sync cycle.

    One row is written per source per sync cycle regardless of outcome.
    This provides a complete provenance trail for every price change.
    """

    __tablename__ = "pricing_ingestion_runs"

    run_id = Column(String, primary_key=True)
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    source_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="success")  # success | failed | partial
    raw_payload = Column(JSON, nullable=True)       # full response for audit/debug
    model_count = Column(Integer, nullable=False, default=0)
    quotes_valid = Column(Integer, nullable=False, default=0)
    quotes_rejected = Column(Integer, nullable=False, default=0)
    rejection_reasons = Column(JSON, nullable=True)  # {model_id: reason}
    error_message = Column(Text, nullable=True)
    revision_id_created = Column(
        String,
        ForeignKey("model_catalog_revisions.revision_id"),
        nullable=True,
    )


class PriceMarketHistoryORM(Base):
    """Running log of every price change event with narrative context.

    Written by RegistryPipeline after each successful sync.
    Provides the data layer for:
      - the Tidus AI Model Latest Pricing Report
      - per-model price history queries
      - market trend analysis across all vendors

    The `reason` field stores a human-readable explanation of why the price
    changed (e.g., "Alibaba cut Qwen prices ~90% in Q1 2026 — competitive
    response to DeepSeek pricing"). Initially populated by the report generator;
    operators can amend via the API.
    """

    __tablename__ = "price_market_history"

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False, index=True)
    vendor = Column(String, nullable=False)
    event_date = Column(Date, nullable=False)
    recorded_at = Column(DateTime, server_default=func.now(), nullable=False)
    field = Column(String, nullable=False)        # input_price | output_price | cache_read_price | cache_write_price
    old_value_usd_1m = Column(Float, nullable=False)  # $/1M tokens
    new_value_usd_1m = Column(Float, nullable=False)  # $/1M tokens
    delta_pct = Column(Float, nullable=False)     # (new - old) / old × 100
    source = Column(String, nullable=False)       # hardcoded | feed | manual
    revision_id = Column(
        String,
        ForeignKey("model_catalog_revisions.revision_id"),
        nullable=True,
        index=True,
    )
    reason = Column(Text, nullable=True)          # narrative / analyst note

    __table_args__ = (
        Index("ix_price_market_history_model_date", "model_id", "event_date"),
        Index("ix_price_market_history_vendor_date", "vendor", "event_date"),
    )


class ModelPriceSnapshotORM(Base):
    """Weekly full-catalog price snapshot — one row per model per sync cycle.

    Written every Sunday by RegistryPipeline.write_weekly_snapshot() regardless
    of whether prices changed. Provides the clean time-series data for graphing
    price trends over time.

    Query pattern for a price history graph:
        SELECT snapshot_date, model_id, input_usd_1m, output_usd_1m
        FROM model_price_snapshots
        WHERE model_id = 'gpt-4o'
        ORDER BY snapshot_date
    """

    __tablename__ = "model_price_snapshots"

    id = Column(String, primary_key=True)
    snapshot_date = Column(Date, nullable=False)
    snapshot_at = Column(DateTime, server_default=func.now(), nullable=False)
    model_id = Column(String, nullable=False)
    vendor = Column(String, nullable=False)
    input_usd_1m = Column(Float, nullable=False)      # $/1M input tokens
    output_usd_1m = Column(Float, nullable=False)     # $/1M output tokens
    cache_read_usd_1m = Column(Float, nullable=False, default=0.0)
    cache_write_usd_1m = Column(Float, nullable=False, default=0.0)
    revision_id = Column(
        String,
        ForeignKey("model_catalog_revisions.revision_id"),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_model_price_snapshots_model_date", "model_id", "snapshot_date"),
        Index("ix_model_price_snapshots_date", "snapshot_date"),
        UniqueConstraint("model_id", "snapshot_date", name="uq_model_price_snapshot"),
    )


class BillingReconciliationORM(Base):
    """One row per (model_id, reconciliation_date) per upload.

    Compares Tidus cost_records against the provider's actual invoice to detect
    cost leakage. Status thresholds:
      matched:  |variance_pct| ≤ 0.05
      warning:  0.05 < |variance_pct| ≤ 0.25
      critical: |variance_pct| > 0.25
    """

    __tablename__ = "billing_reconciliations"

    id = Column(String, primary_key=True)
    reconciliation_date = Column(Date, nullable=False)
    uploaded_at = Column(DateTime, server_default=func.now(), nullable=False)
    uploaded_by = Column(String, nullable=False)   # actor sub from JWT
    team_id = Column(String, nullable=False, index=True)  # scoping for team_manager
    model_id = Column(String, nullable=False)
    tidus_cost_usd = Column(Float, nullable=False)
    provider_cost_usd = Column(Float, nullable=False)
    variance_usd = Column(Float, nullable=False)   # provider - tidus
    variance_pct = Column(Float, nullable=False)   # (provider - tidus) / provider
    status = Column(String, nullable=False)        # matched | warning | critical
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_billing_reconciliations_model_date", "model_id", "reconciliation_date"),
    )
