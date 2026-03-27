"""Price sync — weekly job that detects vendor price changes.

Compares current pricing in the model registry against a known-prices
reference (hardcoded from last verified check). When a delta exceeds
the configured threshold (default: 5%), the registry is updated in-memory
and a PriceChangeRecord is written to the database for audit.

In production this would call vendor pricing APIs. For v0.1 we maintain
a verified-prices dict updated on each release and use it as the source.

Prices verified: 2026-03-26
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog

from tidus.router.registry import ModelRegistry
from tidus.utils.yaml_loader import load_yaml

log = structlog.get_logger(__name__)

# Last-verified prices (USD per 1K tokens). Update on each release.
# Source: official vendor pricing pages, verified 2026-03-26.
_KNOWN_PRICES: dict[str, dict[str, float]] = {
    "o3":                    {"input": 0.010,     "output": 0.040},
    "o4-mini":               {"input": 0.001,     "output": 0.004},
    "gpt-4.1":               {"input": 0.002,     "output": 0.008},
    "gpt-4o-mini":           {"input": 0.00015,   "output": 0.00060},
    "gpt-oss-120b":          {"input": 0.000039,  "output": 0.0001},
    "gpt-5-codex":           {"input": 0.00125,   "output": 0.010},
    "codex-mini-latest":     {"input": 0.00075,   "output": 0.003},
    "claude-opus-4-6":       {"input": 0.005,     "output": 0.025},
    "claude-sonnet-4-6":     {"input": 0.003,     "output": 0.015},
    "claude-haiku-4-5":      {"input": 0.001,     "output": 0.005},
    "gemini-3.1-pro":        {"input": 0.002,     "output": 0.012},
    "gemini-3.1-flash":      {"input": 0.00025,   "output": 0.0015},
    "gemini-nano":            {"input": 0.0,       "output": 0.0},
    "mistral-large-3":       {"input": 0.0005,    "output": 0.0015},
    "mistral-medium":        {"input": 0.0004,    "output": 0.002},
    "mistral-small":         {"input": 0.00007,   "output": 0.0002},
    "codestral":             {"input": 0.0002,    "output": 0.0006},
    "devstral":              {"input": 0.0004,    "output": 0.002},
    "deepseek-r1":           {"input": 0.00055,   "output": 0.00219},
    "deepseek-v3":           {"input": 0.000014,  "output": 0.000028},
    "grok-3":                {"input": 0.003,     "output": 0.015},
    "kimi-k2.5":             {"input": 0.0006,    "output": 0.0025},
}


async def run_price_sync(
    registry: ModelRegistry,
    policies_path: str = "config/policies.yaml",
    session_factory=None,
) -> list[dict]:
    """Compare registry prices against known prices. Return list of changes detected.

    Args:
        registry:      The live model registry.
        policies_path: Path to policies.yaml (reads change_threshold).
        session_factory: SQLAlchemy session factory for writing PriceChangeRecords.

    Returns:
        List of change dicts: {model_id, field, old_value, new_value, delta_pct}
    """
    raw = load_yaml(policies_path)
    threshold = raw.get("pricing_sync", {}).get("change_threshold", 0.05)

    changes = []
    now = datetime.now(timezone.utc)

    for spec in registry.list_all():
        known = _KNOWN_PRICES.get(spec.model_id)
        if known is None:
            continue

        for field, known_value in [
            ("input_price", known["input"]),
            ("output_price", known["output"]),
        ]:
            current_value = getattr(spec, field)
            if known_value == 0 and current_value == 0:
                continue
            if known_value == 0:
                delta_pct = 1.0
            else:
                delta_pct = abs(current_value - known_value) / known_value

            if delta_pct >= threshold:
                change = {
                    "model_id": spec.model_id,
                    "field": field,
                    "old_value": current_value,
                    "new_value": known_value,
                    "delta_pct": round(delta_pct * 100, 2),
                    "detected_at": now,
                }
                changes.append(change)

                # Update registry in-memory
                setattr(spec, field, known_value)
                log.warning(
                    "price_change_detected",
                    model_id=spec.model_id,
                    field=field,
                    old=current_value,
                    new=known_value,
                    delta_pct=round(delta_pct * 100, 1),
                )

                # Persist to DB if session factory provided
                if session_factory:
                    await _write_price_record(session_factory, spec.model_id, spec.vendor, change)

    if not changes:
        log.info("price_sync_complete", changes=0)
    else:
        log.info("price_sync_complete", changes=len(changes))

    return changes


async def _write_price_record(session_factory, model_id: str, vendor: str, change: dict) -> None:
    from tidus.models.cost import PriceChangeRecord
    from tidus.db.engine import PriceChangeLogORM

    record = PriceChangeRecord(
        id=str(uuid.uuid4()),
        model_id=model_id,
        vendor=vendor,
        field_changed=change["field"],
        old_value=change["old_value"],
        new_value=change["new_value"],
        delta_pct=change["delta_pct"] / 100,
        detected_at=change["detected_at"],
        source="weekly_sync",
    )
    try:
        async with session_factory() as session:
            orm = PriceChangeLogORM(
                id=record.id,
                model_id=record.model_id,
                vendor=record.vendor,
                field_changed=record.field_changed,
                old_value=record.old_value,
                new_value=record.new_value,
                delta_pct=record.delta_pct,
                detected_at=record.detected_at,
                source=record.source,
            )
            session.add(orm)
            await session.commit()
    except Exception as exc:
        log.error("price_record_write_failed", model_id=model_id, error=str(exc))
