"""Prometheus metrics for the Tidus self-healing registry.

Nine custom metrics exposed at /metrics alongside the standard FastAPI
instrumentator metrics.

Gauges (6):
  tidus_registry_last_successful_sync_timestamp        — Unix ts of last successful price sync
  tidus_registry_active_revision_activated_timestamp   — Unix ts when active revision was promoted
  tidus_registry_model_last_price_update_timestamp     — per-model last_price_check as Unix ts
  tidus_registry_model_confidence                      — per-model source confidence (0-1)
  tidus_registry_active_revision_id                    — deterministic int hash of active UUID
  tidus_registry_models_stale_count                    — models with price data > 8 days old

Counters (3):
  tidus_probe_live_calls_total           — live health_check calls by model and result
  tidus_probe_synthetic_calls_total      — synthetic count_tokens calls by model and result
  tidus_registry_drift_events_total      — drift detections by model, type, severity

Usage:
    from tidus.observability.registry_metrics import PROBE_LIVE_CALLS
    PROBE_LIVE_CALLS.labels(model_id="gpt-4o", result="success").inc()
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge

# ── Gauges ────────────────────────────────────────────────────────────────────

REGISTRY_LAST_SYNC_TS = Gauge(
    "tidus_registry_last_successful_sync_timestamp",
    "Unix timestamp of the last successfully completed price sync cycle",
)

REGISTRY_ACTIVE_REVISION_TS = Gauge(
    "tidus_registry_active_revision_activated_timestamp",
    "Unix timestamp when the currently active catalog revision was promoted",
)

REGISTRY_MODEL_PRICE_UPDATE_TS = Gauge(
    "tidus_registry_model_last_price_update_timestamp",
    "Unix timestamp of the last price update for each model",
    labelnames=["model_id"],
)

REGISTRY_MODEL_CONFIDENCE = Gauge(
    "tidus_registry_model_confidence",
    "Source confidence score (0–1) for each model's price data; drops below 1.0 when stale",
    labelnames=["model_id"],
)

REGISTRY_ACTIVE_REVISION_ID = Gauge(
    "tidus_registry_active_revision_id",
    "Deterministic integer hash of the active revision UUID — changes on every promotion",
)

REGISTRY_STALE_MODEL_COUNT = Gauge(
    "tidus_registry_models_stale_count",
    "Number of models whose price data is more than 8 days old",
)

# ── Counters ──────────────────────────────────────────────────────────────────

PROBE_LIVE_CALLS = Counter(
    "tidus_probe_live_calls_total",
    "Total live health_check probe calls by model and result",
    labelnames=["model_id", "result"],
)

PROBE_SYNTHETIC_CALLS = Counter(
    "tidus_probe_synthetic_calls_total",
    "Total synthetic count_tokens probe calls by model and result",
    labelnames=["model_id", "result"],
)

DRIFT_EVENTS = Counter(
    "tidus_registry_drift_events_total",
    "Total drift detection events by model, type, and severity",
    labelnames=["model_id", "drift_type", "severity"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def revision_id_to_int(revision_id: str) -> int:
    """Convert a revision UUID to a deterministic Prometheus-safe integer.

    Uses the lower 53 bits of the hex digest to stay within JavaScript's
    safe integer range (2^53 - 1), which matters for Grafana annotation tooltips.
    Deterministic and collision-resistant for practical revision counts.
    """
    return int(revision_id.replace("-", ""), 16) % (2 ** 53)
