"""Data models for the auto-classification cascade (T0–T5).

Types are Pydantic BaseModel for clean JSON serialization on the
`/api/v1/classify` endpoint and for request logging. Internal
orchestration uses these same types — no duplicate internal-vs-API
schemas.

See plan.md for the taxonomy and tier boundaries.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal[
    "chat", "code", "reasoning", "extraction",
    "classification", "summarization", "creative",
]
Complexity = Literal["simple", "moderate", "complex", "critical"]
Privacy = Literal["public", "internal", "confidential"]
ClassificationTier = Literal[
    "caller_override",  # T0
    "heuristic",        # T1 fired at least one regex or keyword signal
    "default",          # T1 ran but no signals; safe defaults returned (T2 pending)
    "encoder",          # T2(+T2b) terminal
    "llm",              # T3/T5 escalation
    "cached",           # T5 cache hit
]


class Tier1Signals(BaseModel):
    """Output of Tier 1 heuristic fast-path.

    `regex_hits` and `secret_types` hold *pattern IDs* / type names, never
    matched values — required for the PII-safe telemetry schema in plan.md.
    """
    regex_hits: list[str] = Field(default_factory=list)
    secret_types: list[str] = Field(default_factory=list)
    keyword_hits: list[str] = Field(default_factory=list)
    has_code_fence: bool = False
    estimated_input_tokens: int = 0
    any_hit: bool = False  # convenience: true iff any regex/secret/keyword fired


class EncoderResult(BaseModel):
    """Output of Tier 2 trained encoder (Recipe B: MiniLM + LR heads).

    `embedding` holds the raw 384-dim sentence embedding used to produce the
    predictions, surfaced so Stage B telemetry can dim-reduce it to 64-d
    (PCA) before logging. Optional — None outside telemetry paths, or when
    encoder wasn't the one that produced the result (e.g., stub tests).
    """
    domain: Domain
    complexity: Complexity
    privacy: Privacy
    confidence: dict[str, float]  # keys: "domain", "complexity", "privacy"
    embedding: list[float] | None = None


class LLMResult(BaseModel):
    """Output of Tier 5 local LLM escalation (Ollama-hosted)."""
    domain: Domain
    complexity: Complexity
    privacy: Privacy
    confidence: dict[str, float]
    rationale: str | None = None  # one-sentence reasoning from the LLM


class PresidioResult(BaseModel):
    """Output of Tier 2b Presidio NER. Types only, never entity values.

    `entity_scores` holds max Presidio confidence per entity type (PII-safe —
    scores, not matched strings). Useful for debug surfacing and future
    uncertainty-aware merge logic. Not plumbed into the current merge rule;
    `detected_person` + `entity_types` remain the decision signals (task #48).
    """
    entity_types: list[str] = Field(default_factory=list)
    entity_scores: dict[str, float] = Field(default_factory=dict)
    detected_person: bool = False


class ClassificationResult(BaseModel):
    """The terminal classification returned to the router.

    Returned from `TaskClassifier.classify(...)`. Serves both:
      - internal router input (routing.confidential_require_local etc.)
      - `POST /api/v1/classify` response body
    """
    domain: Domain
    complexity: Complexity
    privacy: Privacy
    estimated_input_tokens: int
    classification_tier: ClassificationTier
    confidence: dict[str, float]
    confidence_warning: bool = False  # set when T5 was needed but unavailable
    debug: dict | None = None  # optional — populated only when requested


class ClassificationError(Exception):
    """Base error for classification failures that must not leak to callers."""


class EncoderLoadError(ClassificationError):
    """Recipe B weights missing, malformed, or label mapping mismatch."""


class LLMUnavailableError(ClassificationError):
    """Ollama endpoint unreachable or model not pulled. Triggers graceful
    degradation to encoder output + confidence_warning=True."""
