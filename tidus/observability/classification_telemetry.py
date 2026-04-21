"""Stage B — PII-safe classification telemetry.

Emits one structured log record per classification via structlog. Schema
follows plan.md §Stage B line 547:

    request_id           unique per request (uuid4)
    tenant_id            from TokenPayload.tenant_id (JWT claim / header / team fallback)
    ts                   ISO-8601 UTC timestamp
    embedding_reduced_64d list[float] — PCA(384->64) of the encoder's embedding
    presidio_entities    list[str] — entity TYPES only, never values
    regex_hits           list[str] — pattern IDs only, never matched strings
    tier_decided         str — classification_tier from ClassificationResult
    classification       {"domain", "complexity", "privacy"}
    model_routed         str | None — chosen model_id when known (None from /classify)
    latency_ms           int — wall-clock classification latency

PII safety invariants:
    * Never log raw prompts, rationale text, or any substring of user input.
    * Embeddings are dim-reduced before logging; the raw 384-d can recover
      the prompt via nearest-neighbour lookup against public corpora.
    * entity/regex lists are type-names only (PresidioResult.entity_types,
      Tier1Signals.regex_hits — both are frozen to types/IDs by construction
      in their producer modules).

Configuration:
    * `settings.classify_telemetry_enabled` — master toggle (default True).
    * `settings.classify_pca_path` — path to the joblib PCA artifact. When
      missing, we log a warning and omit `embedding_reduced_64d` from the
      record; other fields still emit.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import structlog

from tidus.classification.models import (
    ClassificationResult,
    EncoderResult,
    PresidioResult,
    Tier1Signals,
)

log = structlog.get_logger("tidus.classification.telemetry")
_load_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_pca = None
_pca_load_failed = False
_pca_lock = Lock()


def _load_pca(pca_path: str):
    """Load and cache the PCA artifact. Thread-safe. Returns None on failure
    (missing file, joblib error). Caller must fall back gracefully."""
    global _pca, _pca_load_failed
    if _pca is not None:
        return _pca
    if _pca_load_failed:
        return None
    with _pca_lock:
        if _pca is not None:
            return _pca
        if _pca_load_failed:
            return None
        p = Path(pca_path)
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        if not p.is_file():
            _load_log.warning(
                "PCA artifact missing at %s — Stage B telemetry will omit "
                "embedding_reduced_64d. Run: uv run python scripts/fit_pca_64d.py",
                p,
            )
            _pca_load_failed = True
            return None
        try:
            _pca = joblib.load(p)
        except Exception as exc:  # noqa: BLE001 — joblib raises varied types
            _load_log.warning("PCA artifact load failed at %s: %s", p, exc)
            _pca_load_failed = True
            return None
    return _pca


def _reset_cache_for_tests() -> None:
    """Let tests exercise repeated load attempts on the same artifact path
    without module-reload gymnastics."""
    global _pca, _pca_load_failed
    with _pca_lock:
        _pca = None
        _pca_load_failed = False


def _reduce_embedding(embedding: list[float] | None, pca_path: str) -> list[float] | None:
    """Dim-reduce a 384-d embedding via the cached PCA artifact. Returns
    None when either the embedding or the PCA is unavailable — emit
    continues without the field rather than failing the request."""
    if embedding is None:
        return None
    pca = _load_pca(pca_path)
    if pca is None:
        return None
    arr = np.asarray([embedding], dtype=np.float32)
    reduced = pca.transform(arr)[0]
    return reduced.astype(float).tolist()


def emit_classification_telemetry(
    *,
    tenant_id: str | None,
    result: ClassificationResult,
    signals: Tier1Signals | None,
    encoder: EncoderResult | None,
    presidio: PresidioResult | None,
    model_routed: str | None,
    latency_ms: int,
    pca_path: str,
    request_id: str | None = None,
) -> None:
    """Emit one Stage B log record. NEVER raises — telemetry emission must
    not fail the request path. Logs at info level via structlog under the
    event name `classification`.

    Args mirror the record schema above. Fields nullable to reflect the
    actual data availability: e.g., `encoder` can be None when encoder
    failed to load, `presidio` when T2b is disabled.
    """
    try:
        record = {
            "request_id": request_id or str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "ts": datetime.now(UTC).isoformat(),
            "presidio_entities": list(presidio.entity_types) if presidio else [],
            "regex_hits": list(signals.regex_hits) if signals else [],
            "tier_decided": result.classification_tier,
            "classification": {
                "domain": result.domain,
                "complexity": result.complexity,
                "privacy": result.privacy,
            },
            "model_routed": model_routed,
            "latency_ms": latency_ms,
        }
        reduced = _reduce_embedding(
            encoder.embedding if encoder else None,
            pca_path,
        )
        if reduced is not None:
            record["embedding_reduced_64d"] = reduced
        log.info("classification", **record)
    except Exception as exc:  # noqa: BLE001 — never fail the request path
        _load_log.warning("classification telemetry emit failed: %s", exc)
