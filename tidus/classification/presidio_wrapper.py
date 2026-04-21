"""Tier 2b — Presidio NER wrapper.

Presidio detects PII entities that our Tier 1 regex set can't cover well:
non-US national IDs, international phone/passport formats, and PERSON names.
It runs in parallel with Tier 2 encoder via asyncio.gather in the classifier's
async path — total privacy-voter latency is max(T2, T2b), not sum.

Design:
    * Entities split into HIGH_TRUST (always → confidential if detected) and
      CONTEXTUAL (PERSON only, gated by E1/E2 rule at the merge step).
    * We DON'T remove SpacyRecognizer — we need it for PERSON detection, which
      is the topic-bearing signal for 6/12 IRR confidentials (findings.md §3).
    * Input is capped at `max_chars` to keep spaCy NER bounded on pathological
      prompts; plan.md Phase 0.5 benchmark showed p95 ~25ms at this cap.

Thread safety: Presidio's AnalyzerEngine wraps spaCy, which has per-Doc state.
Under concurrent FastAPI load (two asyncio.to_thread calls at once), results
can interleave. A.5's endpoint wiring must use a lock or instance pool — see
backlog task #43.
"""
from __future__ import annotations

import logging

from tidus.classification.models import PresidioResult

log = logging.getLogger(__name__)

# Entities where a single detection is enough to force confidential privacy.
# Every one of these is a Presidio built-in recognizer with strong precision
# (pattern-based, not NER). PERSON is intentionally absent — it has a PERSON-
# name false-positive rate of ~30-50% on capitalized common words, which is
# why E1/E2 rules gate it separately.
#
# DATE_TIME is INTENTIONALLY ABSENT: the merge rule treats any high-trust
# entity as a terminal confidential vote via asymmetric-safety OR. A single
# "tomorrow" or "next Tuesday" mention would therefore inflate the
# confidential flag rate with no real privacy signal. Presidio's DATE_TIME
# is useful for entity extraction but not as a privacy-class vote.
HIGH_TRUST_ENTITIES: frozenset[str] = frozenset({
    "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "IP_ADDRESS",
    "UK_NHS", "CREDIT_CARD", "US_PASSPORT", "MEDICAL_LICENSE",
    "IBAN_CODE", "CRYPTO", "AU_ABN", "AU_ACN", "AU_TFN",
    "AU_MEDICARE", "ES_NIF", "IT_DRIVER_LICENSE", "IT_FISCAL_CODE",
    "IT_VAT_CODE", "IT_PASSPORT", "IT_IDENTITY_CARD",
    "SG_NRIC_FIN", "SG_UEN", "PL_PESEL", "KR_RRN", "IN_AADHAAR",
    "IN_VEHICLE_REGISTRATION", "IN_VOTER", "IN_PASSPORT", "IN_PAN",
    "FI_PERSONAL_IDENTITY_CODE", "NG_NIN",
})

# A narrow allow-list of what we ask Presidio to score. Limiting `entities=` at
# query time is faster than scoring the full default set and filtering after.
# LOCATION and ORGANIZATION are queried for debugging/telemetry visibility
# but are NOT in HIGH_TRUST — they never vote confidential alone.
_QUERY_ENTITIES: list[str] = sorted(HIGH_TRUST_ENTITIES | {"PERSON", "LOCATION", "ORGANIZATION"})


class PresidioWrapper:
    """Thin Presidio AnalyzerEngine wrapper returning a `PresidioResult`.

    Lifecycle:
        1. `__init__` — cheap
        2. `load()` — expensive (~1-2s), downloads spaCy pipeline if missing
        3. `analyze(text) -> PresidioResult` — fast (~5-25ms per plan.md Phase 0.5)
    """

    def __init__(self, max_chars: int = 5000) -> None:
        self._max_chars = max_chars
        self._analyzer = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        # Lazy imports to keep the bare `from presidio_wrapper import ...` cheap.
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        log.info("Loading Presidio AnalyzerEngine with en_core_web_sm")
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }).create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self._loaded = True

    def analyze(self, text: str) -> PresidioResult:
        if not self._loaded or self._analyzer is None:
            raise RuntimeError("PresidioWrapper.analyze() called before load()")

        snippet = text[: self._max_chars]
        results = self._analyzer.analyze(
            text=snippet,
            entities=_QUERY_ENTITIES,
            language="en",
        )
        entity_types = sorted({r.entity_type for r in results})
        # Max score per entity type — Presidio can emit multiple recognizers
        # for the same type on one span (e.g. PhoneRecognizer + pattern fallback).
        # Max keeps the strongest signal without inflating via duplicates.
        entity_scores: dict[str, float] = {}
        for r in results:
            existing = entity_scores.get(r.entity_type, 0.0)
            if r.score > existing:
                entity_scores[r.entity_type] = float(r.score)
        return PresidioResult(
            entity_types=entity_types,
            entity_scores=entity_scores,
            detected_person="PERSON" in entity_types,
        )


def has_high_trust_hit(presidio: PresidioResult) -> bool:
    """True iff Presidio detected any high-trust PII entity."""
    return any(e in HIGH_TRUST_ENTITIES for e in presidio.entity_types)
