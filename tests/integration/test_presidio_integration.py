"""Integration test — real Presidio AnalyzerEngine on real prompts.

Loads spaCy en_core_web_sm + Presidio (~1-2s cold), so it sits in
tests/integration/. Skipped automatically when Presidio or spaCy
model isn't available in the environment.

These tests verify:
    1. PresidioWrapper loads and runs on realistic prompts
    2. PERSON detection fires for clear named entities
    3. High-trust patterns (EMAIL, PHONE) fire as expected
    4. Full cascade (TaskClassifier + real Presidio + real encoder) produces
       consistent outputs
"""
from __future__ import annotations

import importlib.util

import pytest

from tidus.classification import TaskClassifier
from tidus.classification.encoder import Encoder, resolve_weights_dir
from tidus.classification.presidio_wrapper import (
    HIGH_TRUST_ENTITIES,
    PresidioWrapper,
)

_PRESIDIO_AVAILABLE = importlib.util.find_spec("presidio_analyzer") is not None
_SPACY_MODEL_AVAILABLE = importlib.util.find_spec("en_core_web_sm") is not None
_ENV_READY = _PRESIDIO_AVAILABLE and _SPACY_MODEL_AVAILABLE

needs_presidio = pytest.mark.skipif(
    not _ENV_READY,
    reason="presidio-analyzer + en_core_web_sm required; run `uv sync` and "
           "`uv run python -m spacy download en_core_web_sm`",
)

_WEIGHTS_DIR = resolve_weights_dir("tidus/classification/weights_b")
_WEIGHTS_READY = (_WEIGHTS_DIR / "label_mappings.json").is_file()


@pytest.fixture(scope="session")
def loaded_presidio() -> PresidioWrapper:
    wrapper = PresidioWrapper()
    wrapper.load()
    return wrapper


@needs_presidio
class TestPresidioWrapperDetections:
    def test_person_detected(self, loaded_presidio: PresidioWrapper):
        r = loaded_presidio.analyze(
            "Dr. Sarah Chen at Mount Sinai reviewed the X-ray results.",
        )
        assert r.detected_person
        assert "PERSON" in r.entity_types

    def test_email_detected_as_high_trust(self, loaded_presidio: PresidioWrapper):
        r = loaded_presidio.analyze("email me at jane.doe@example.com please")
        assert "EMAIL_ADDRESS" in r.entity_types
        from tidus.classification.presidio_wrapper import has_high_trust_hit
        assert has_high_trust_hit(r)

    def test_phone_number_detected(self, loaded_presidio: PresidioWrapper):
        r = loaded_presidio.analyze("call the office at (415) 555-0199")
        assert "PHONE_NUMBER" in r.entity_types

    def test_benign_text_has_no_high_trust(self, loaded_presidio: PresidioWrapper):
        r = loaded_presidio.analyze("what's the weather forecast for next week")
        # May detect LOCATION or DATE_TIME but not hard-PII entities.
        # DATE_TIME is in HIGH_TRUST — assert not the specifically-private ones.
        strict = r.entity_types and any(
            e in HIGH_TRUST_ENTITIES - {"DATE_TIME"} for e in r.entity_types
        )
        assert not strict

    def test_load_is_idempotent(self, loaded_presidio: PresidioWrapper):
        loaded_presidio.load()  # second call — no-op
        assert loaded_presidio.loaded

    def test_entity_scores_populated_per_detected_type(
        self, loaded_presidio: PresidioWrapper,
    ):
        """entity_scores must hold a max-per-type float for every entity
        listed in entity_types — guards against future refactors dropping
        the per-entity plumbing (task #48)."""
        r = loaded_presidio.analyze("email me at jane.doe@example.com please")
        assert "EMAIL_ADDRESS" in r.entity_types
        assert "EMAIL_ADDRESS" in r.entity_scores
        # Presidio scores are in [0, 1]; EMAIL_ADDRESS is a strong pattern.
        assert 0.0 < r.entity_scores["EMAIL_ADDRESS"] <= 1.0
        # Every entity_types value must have a score; they're parallel fields.
        assert set(r.entity_scores.keys()) == set(r.entity_types)


@needs_presidio
class TestFullCascadeWithPresidio:
    """TaskClassifier with real Presidio (+ real encoder if available)."""

    def test_presidio_catches_person_encoder_missed(
        self, loaded_presidio: PresidioWrapper,
    ):
        """Case from findings.md §3: topic-bearing confidential the encoder
        might label non-confidential — Presidio PERSON detection catches it
        via the E1 rule, asymmetric-safety OR wins.

        Verifies that T2b was the *deciding* voter — not T1 regex accidentally
        firing on the prompt text.
        """
        prompt = (
            "Draft a reference letter for Sarah Mitchell, my former "
            "colleague at Acme Corp where she was senior architect."
        )
        clf = TaskClassifier(presidio=loaded_presidio)  # no encoder — T1 + T2b only
        r = clf.classify(prompt, include_debug=True)

        assert r.privacy == "confidential"
        # Verify T1 did NOT fire any confidential-triggering regex on this prompt.
        # If T1 had fired, the test wouldn't be proving "Presidio catches what
        # encoder/T1 missed" — it would be proving "T1 regex caught it."
        from tidus.classification.heuristics import _CONFIDENTIAL_PATTERN_IDS
        t1_hits: list[str] = r.debug["tier1_signals"]["regex_hits"]
        t1_confidential_hits = [h for h in t1_hits if h in _CONFIDENTIAL_PATTERN_IDS]
        assert t1_confidential_hits == [], (
            f"T1 regex unexpectedly fired confidential: {t1_confidential_hits}. "
            f"This test needs a different prompt to isolate Presidio's contribution."
        )
        # Verify T2b was the vote source (PERSON detected).
        assert r.debug["tier2b_presidio"] is not None
        assert r.debug["tier2b_presidio"]["detected_person"]

    @pytest.mark.skipif(not _WEIGHTS_READY, reason="encoder weights not trained")
    def test_full_cascade_all_tiers(self, loaded_presidio: PresidioWrapper):
        enc = Encoder(weights_dir=str(_WEIGHTS_DIR))
        enc.load()
        clf = TaskClassifier(encoder=enc, presidio=loaded_presidio)

        # A clear confidential prompt — multiple signals should converge.
        r = clf.classify(
            "Review my resume: Jane Smith, Senior Engineer at OpenAI, "
            "jane.smith@example.com, looking for staff roles.",
        )
        assert r.privacy == "confidential"
        # Multiple voters agreeing → confidence should be high (>=0.9).
        assert r.confidence["privacy"] >= 0.9


class TestPresidioErrorPaths:
    def test_analyze_before_load_raises(self):
        wrapper = PresidioWrapper()
        with pytest.raises(RuntimeError, match="before load"):
            wrapper.analyze("anything")
