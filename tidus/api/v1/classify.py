"""POST /api/v1/classify — classify a message without executing a completion.

Callers use this endpoint to see what the T0→T5 cascade would return for a
given message *without* actually calling a downstream model. Useful for:
  * Offline routing previews / debugging
  * UI surfaces that want to show "Tidus thinks this is confidential"
  * Integration tests that validate client expectations

When `auto_classify_enabled=false`, this endpoint returns 503. Use `/health`
to check the classifier's per-tier readiness before relying on it.

This module also exports `enrich_task_fields()` — used by `/complete` and
`/route` to auto-classify when callers omit `complexity`/`domain`/`privacy`/
`estimated_input_tokens`. That's v1.3.0's actual value prop: callers can drop
the classification bookkeeping and Tidus handles it internally.
"""
from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from tidus.api.deps import get_classifier_optional
from tidus.auth.middleware import TokenPayload, get_current_user
from tidus.classification import ClassificationResult, TaskClassifier
from tidus.observability.classification_telemetry import (
    emit_classification_telemetry,
)
from tidus.settings import get_settings

log = structlog.get_logger(__name__)

router = APIRouter(tags=["classify"])


# Fields that the classifier can fill in. If ANY are None on the inbound
# request, we call the classifier to produce them. Caller-provided values
# always win via the existing caller_override merge rule.
_CLASSIFIABLE_FIELDS = ("complexity", "domain", "privacy", "estimated_input_tokens")


async def enrich_task_fields(
    classifier: TaskClassifier | None,
    complexity: str | None,
    domain: str | None,
    privacy: str | None,
    estimated_input_tokens: int | None,
    messages: list[dict],
    telemetry_observer=None,
) -> dict[str, Any]:
    """Fill in missing classification fields by running the classifier.

    Returns a dict with all four fields populated. Caller-provided values
    are preserved — for complexity/domain/privacy via caller_override merge
    (asymmetric safety still applies), for estimated_input_tokens by
    straight passthrough.

    Behaviour:
      * All four fields provided → no-op, returns them as-is.
      * Any field missing + classifier available → run classify_async, merge.
      * Any field missing + classifier unavailable → 422 (caller must supply).
      * No user message in `messages` → 400 (nothing to classify).
    """
    provided = {
        "complexity": complexity,
        "domain": domain,
        "privacy": privacy,
        "estimated_input_tokens": estimated_input_tokens,
    }
    missing = [k for k, v in provided.items() if v is None]
    if not missing:
        return provided

    if classifier is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "classification_fields_required",
                "missing": missing,
                "message": (
                    "Auto-classification is disabled (auto_classify_enabled=false). "
                    "Callers must supply complexity, domain, privacy, and "
                    "estimated_input_tokens explicitly."
                ),
            },
        )

    text = _extract_user_text(messages)

    # Only the three taxonomy axes go into caller_override — the classifier
    # doesn't reason about estimated_input_tokens (that's a T1 token estimate).
    caller_override = {
        k: v for k, v in provided.items()
        if v is not None and k in ("complexity", "domain", "privacy")
    } or None

    result = await classifier.classify_async(
        text=text,
        caller_override=caller_override,
        telemetry_observer=telemetry_observer,
    )

    return {
        "complexity": result.complexity,
        "domain": result.domain,
        "privacy": result.privacy,
        # Caller's estimate wins if supplied; otherwise the T1 char-based estimate.
        "estimated_input_tokens": (
            estimated_input_tokens if estimated_input_tokens is not None
            else result.estimated_input_tokens
        ),
    }


# ── Request / Response models ─────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    messages: list[dict] = Field(
        ...,
        description="OpenAI-style messages list. The classifier uses the LAST "
                    "user message's `content` as the text to classify.",
        min_length=1,
    )
    team_id: str = Field(..., description="Requesting team's identifier.")
    # Optional caller-supplied override. If all three dimensions are provided,
    # T0 short-circuits the cascade and returns the caller's values as-is.
    caller_override: dict | None = Field(
        default=None,
        description="Optional partial/complete classification hints from the "
                    "caller. Keys: domain, complexity, privacy.",
    )
    include_debug: bool = Field(
        default=False,
        description="Include per-tier debug details (Tier 1 signals, encoder "
                    "output, Presidio entities, T5 rationale) in the response. "
                    "Off by default — debug payloads can be large.",
    )


class ClassifyResponse(BaseModel):
    result: ClassificationResult
    classifier_health: dict = Field(
        ...,
        description="Per-tier load state at time of classification. Lets the "
                    "caller correlate confidence_warning with specific degraded "
                    "tiers (e.g., Presidio unavailable, LLM endpoint down).",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_user_text(messages: list[dict]) -> str:
    """Return the content of the LAST user message in the thread.

    Classification is per-message, not per-thread — we don't concatenate
    history because the privacy axis is about what's IN the prompt, not
    what was said in prior turns.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    raise HTTPException(
        status_code=400,
        detail="No user message with string content found in `messages`.",
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(
    body: ClassifyRequest,
    classifier: Annotated[TaskClassifier | None, Depends(get_classifier_optional)],
    auth: Annotated[TokenPayload, Depends(get_current_user)],
    # RBAC: anyone with a valid team token can classify — it's a read-only
    # operation that doesn't mutate state or consume external budget.
) -> ClassifyResponse:
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Auto-classification is disabled (auto_classify_enabled=false). "
                   "Callers must supply complexity/domain/privacy explicitly.",
        )

    text = _extract_user_text(body.messages)
    settings = get_settings()

    capture = make_telemetry_capture(
        enabled=settings.classify_telemetry_enabled,
        tenant_id=auth.tenant_id,
        pca_path=settings.classify_pca_path,
    )

    result = await classifier.classify_async(
        text=text,
        caller_override=body.caller_override,
        include_debug=body.include_debug,
        telemetry_observer=capture.observer,
    )
    # /classify is a dry run — no downstream model routed. Still emit so the
    # Stage B record is complete for this path.
    capture.emit(model_routed=None)

    return ClassifyResponse(
        result=result,
        classifier_health=classifier.healthy,
    )


class _TelemetryCapture:
    """Buffer classification intermediates so the endpoint can emit the
    Stage B record AFTER routing completes. The observer callback fires
    inside classify_async (before routing). If we emitted there, `model_routed`
    would always be None on /complete and /route — defeating plan.md's
    self-improvement feedback-loop design."""

    def __init__(
        self, *,
        enabled: bool,
        tenant_id: str | None,
        pca_path: str,
    ) -> None:
        self._enabled = enabled
        self._tenant_id = tenant_id
        self._pca_path = pca_path
        self._started = time.perf_counter()
        self._request_id = str(uuid.uuid4())
        self._signals = None
        self._encoder = None
        self._presidio = None
        self._result = None
        self._captured = False

    @property
    def observer(self):
        """Return the callback to pass to classify_async. None when telemetry
        is disabled so the classifier can skip the observer machinery entirely."""
        if not self._enabled:
            return None

        def _obs(*, signals, encoder, presidio, result):
            self._signals = signals
            self._encoder = encoder
            self._presidio = presidio
            self._result = result
            self._captured = True

        return _obs

    def emit(self, *, model_routed: str | None) -> None:
        """Emit the Stage B log record with the now-known model_routed.
        Safe to call multiple times (e.g. once on rejection, once on
        success) — second call is a no-op since we reset `_captured`."""
        if not self._captured:
            return
        latency_ms = int((time.perf_counter() - self._started) * 1000)
        emit_classification_telemetry(
            tenant_id=self._tenant_id,
            result=self._result,
            signals=self._signals,
            encoder=self._encoder,
            presidio=self._presidio,
            model_routed=model_routed,
            latency_ms=latency_ms,
            pca_path=self._pca_path,
            request_id=self._request_id,
        )
        self._captured = False


def make_telemetry_capture(
    *,
    enabled: bool,
    tenant_id: str | None,
    pca_path: str,
) -> _TelemetryCapture:
    """Factory — endpoints call this, pass `capture.observer` into
    `classify_async`, then call `capture.emit(model_routed=...)` once the
    routing decision is known (or once a rejection is handled, to still
    emit with model_routed=None)."""
    return _TelemetryCapture(
        enabled=enabled,
        tenant_id=tenant_id,
        pca_path=pca_path,
    )
