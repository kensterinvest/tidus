"""Tidus auto-classification package — T0 → T5 cascade.

Architecture: plan.md §Architecture.
Hardware spec: docs/hardware-requirements.md.
Training entry point: scripts/train_encoder_recipe_b.py.

Stage A milestones:
    A.1 (shipped)  Config, data models, Tier 1 heuristic fast-path, TaskClassifier T0+T1
    A.2 (shipped)  Tier 2 — Recipe B encoder runtime (MiniLM + LR heads)
    A.3 (shipped)  Tier 2b — Presidio NER parallel to T2 via asyncio.gather
    A.4 (shipped)  Tier 5 — Ollama LLM escalation (Enterprise SKU, requires GPU)
    A.5 (shipped)  POST /api/v1/classify + DI singleton + lifespan wiring +
                   per-tier async locks + /ready classifier health
"""
from tidus.classification.classifier import TaskClassifier
from tidus.classification.llm_classifier import LLMClassifier
from tidus.classification.models import (
    ClassificationError,
    ClassificationResult,
    ClassificationTier,
    Complexity,
    Domain,
    EncoderLoadError,
    EncoderResult,
    LLMResult,
    LLMUnavailableError,
    PresidioResult,
    Privacy,
    Tier1Signals,
)
from tidus.classification.presidio_wrapper import (
    HIGH_TRUST_ENTITIES,
    PresidioWrapper,
)

__all__ = [
    "TaskClassifier",
    "ClassificationResult",
    "ClassificationTier",
    "Tier1Signals",
    "EncoderResult",
    "PresidioResult",
    "LLMResult",
    "Domain",
    "Complexity",
    "Privacy",
    "ClassificationError",
    "EncoderLoadError",
    "LLMUnavailableError",
    "PresidioWrapper",
    "HIGH_TRUST_ENTITIES",
    "LLMClassifier",
]
