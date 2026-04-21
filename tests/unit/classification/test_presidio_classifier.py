"""Stage A.3 tests — TaskClassifier with T2b Presidio voter.

Uses a StubPresidio to exercise every branch of the three-way merge
(T1 regex, T2 encoder, T2b Presidio) and the E1/E2 rule switch.

Integration test with real Presidio (slow, pulls spaCy) lives in
tests/integration/test_presidio_integration.py.
"""
from __future__ import annotations

import pytest

from tidus.classification import TaskClassifier
from tidus.classification.encoder import Encoder
from tidus.classification.models import EncoderResult, PresidioResult
from tidus.classification.presidio_wrapper import PresidioWrapper
from tidus.settings import get_settings


class StubEncoder(Encoder):
    def __init__(self, result: EncoderResult) -> None:  # noqa: D401
        self._result = result
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:  # type: ignore[override]
        return None

    def classify(self, text: str) -> EncoderResult:  # type: ignore[override]
        return self._result


class StubPresidio(PresidioWrapper):
    def __init__(self, result: PresidioResult) -> None:  # noqa: D401
        self._result = result
        self._loaded = True
        self._max_chars = 5000

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:  # type: ignore[override]
        return None

    def analyze(self, text: str) -> PresidioResult:  # type: ignore[override]
        return self._result


def _enc(
    domain: str = "chat",
    complexity: str = "moderate",
    privacy: str = "internal",
    dom_c: float = 0.70,
    cmp_c: float = 0.65,
    prv_c: float = 0.60,
) -> EncoderResult:
    return EncoderResult(
        domain=domain,
        complexity=complexity,
        privacy=privacy,
        confidence={"domain": dom_c, "complexity": cmp_c, "privacy": prv_c},
    )


def _ps(entity_types: list[str], detected_person: bool = False) -> PresidioResult:
    # If PERSON in entity_types, set detected_person automatically.
    if "PERSON" in entity_types and not detected_person:
        detected_person = True
    return PresidioResult(entity_types=entity_types, detected_person=detected_person)


class TestT2bHighTrustVote:
    def test_high_trust_entity_forces_confidential(self):
        """Presidio detects PHONE_NUMBER → confidential vote at 0.90."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.95)),
            presidio=StubPresidio(_ps(["PHONE_NUMBER"])),
        )
        r = clf.classify("the number is (555) 123-4567")
        assert r.privacy == "confidential"
        # T2b high-trust vote at 0.90; encoder voted public (not confidential);
        # no T1 regex — so 1 voter → 0.90 confidence.
        assert r.confidence["privacy"] == 0.90

    def test_high_trust_plus_encoder_confidential_climbs(self):
        """Both T2b high-trust and T2 encoder agree → +0.05 bump."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="confidential", prv_c=0.80)),
            presidio=StubPresidio(_ps(["EMAIL_ADDRESS", "CREDIT_CARD"])),
        )
        r = clf.classify("my card is 4111-1111-1111-1111 at me@ex.com")
        assert r.privacy == "confidential"
        # T2b (0.90) + T2 (0.80) + T1 regex CREDIT_CARD (0.90, valid Luhn+BIN) =
        # 3 voters; base=max(0.90,0.80,0.90)=0.90; +0.05*2=1.0 capped.
        assert r.confidence["privacy"] == 1.0


class TestE1Rule:
    """Default E1: PERSON alone triggers confidential."""

    def test_e1_person_alone_triggers_confidential(self, monkeypatch):
        # Default rule is E1 per settings
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.95)),
            presidio=StubPresidio(_ps(["PERSON"])),
        )
        r = clf.classify("Hello, my friend Alice Johnson works at...")
        assert r.privacy == "confidential"
        # T2b PERSON-only vote at 0.70, encoder voted public (no vote)
        assert r.confidence["privacy"] == 0.70

    def test_e1_no_person_no_high_trust_no_vote(self):
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.95)),
            presidio=StubPresidio(_ps(["LOCATION"])),  # LOCATION is not high-trust
        )
        r = clf.classify("the hotel is in downtown Toronto")
        # No confidential vote → encoder's public at 0.95 (above floor)
        assert r.privacy == "public"


class TestE2Rule:
    """E2: PERSON only triggers confidential when encoder also says non-public."""

    def _clf_e2(self, encoder, presidio) -> TaskClassifier:
        settings = get_settings()
        settings.classify_presidio_rule = "E2"
        return TaskClassifier(settings=settings, encoder=encoder, presidio=presidio)

    def test_e2_person_plus_public_encoder_no_vote(self):
        """E2: PERSON present but encoder says public → no confidential vote."""
        clf = self._clf_e2(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.95)),
            presidio=StubPresidio(_ps(["PERSON"])),
        )
        r = clf.classify("celebrity Alice Johnson is touring")
        # T2b PERSON alone would fire in E1, but E2 requires non-public encoder.
        # No vote; encoder public at 0.95 (above floor) → public
        assert r.privacy == "public"

    def test_e2_person_plus_internal_encoder_fires(self):
        """E2: PERSON + encoder non-public → confidential."""
        clf = self._clf_e2(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.75)),
            presidio=StubPresidio(_ps(["PERSON"])),
        )
        r = clf.classify("HR review of John Smith")
        assert r.privacy == "confidential"

    def test_e2_high_trust_always_fires_regardless_of_rule(self):
        """High-trust entities bypass the E1/E2 gate — SSN is SSN."""
        clf = self._clf_e2(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.95)),
            presidio=StubPresidio(_ps(["US_SSN"])),
        )
        r = clf.classify("...")
        assert r.privacy == "confidential"

    def teardown_method(self, method):
        """Reset settings so E2 doesn't leak to other tests."""
        settings = get_settings()
        settings.classify_presidio_rule = "E1"


class TestThreeWayDisagreement:
    """Privacy merge when T1, T2, T2b disagree."""

    def test_t1_regex_alone_vs_encoder_public(self):
        """T1 regex fires SSN → confidential, encoder says public. T1 wins via asymmetric safety."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.95)),
            presidio=StubPresidio(_ps([])),  # no Presidio hits
        )
        r = clf.classify("my SSN is 123-45-6789")
        assert r.privacy == "confidential"
        assert r.confidence["privacy"] == 0.90  # T1 alone

    def test_t2b_person_alone_vs_encoder_public_t1_clear(self):
        """Only T2b PERSON votes confidential; T2 says public at high conf; T1 empty.
        Asymmetric safety still wins for PERSON (E1 rule)."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.98)),
            presidio=StubPresidio(_ps(["PERSON"])),
        )
        r = clf.classify("Dr Martinez reviewed the case notes")
        assert r.privacy == "confidential"
        assert r.confidence["privacy"] == 0.70  # PERSON-only confidence

    def test_all_agree_public(self):
        """No tier flags confidential; encoder public above floor."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.92)),
            presidio=StubPresidio(_ps([])),
        )
        r = clf.classify("what time is it")
        assert r.privacy == "public"
        assert r.confidence["privacy"] == 0.92


class TestTierLabelWithPresidio:
    def test_presidio_only_labels_encoder_tier(self):
        """Label is 'encoder' when any model-based tier (T2 or T2b) ran,
        since that carries strictly more info than T1 regex alone."""
        clf = TaskClassifier(presidio=StubPresidio(_ps(["PERSON"])))
        r = clf.classify("Alice Johnson")
        assert r.classification_tier == "encoder"

    def test_no_presidio_no_encoder_safe_defaults_default_tier(self):
        clf = TaskClassifier()
        r = clf.classify("hello")
        assert r.classification_tier == "default"


class TestAsyncParallelPath:
    @pytest.mark.asyncio
    async def test_classify_async_runs_both_in_parallel(self):
        """classify_async() produces the same result as classify() — we just
        assert behavioural equivalence, not wall-clock latency (that's
        verified manually during integration benchmarks)."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="confidential", prv_c=0.80)),
            presidio=StubPresidio(_ps(["EMAIL_ADDRESS"])),
        )
        sync_result = clf.classify("test@example.com")
        async_result = await clf.classify_async("test@example.com")

        assert sync_result.privacy == async_result.privacy
        assert sync_result.confidence == async_result.confidence
        assert sync_result.classification_tier == async_result.classification_tier

    @pytest.mark.asyncio
    async def test_t2_and_t2b_actually_run_in_parallel(self):
        """Wall-clock proof: if each tier sleeps 200ms and they ran serially,
        total would be ~400ms. Parallel via asyncio.gather collapses to ~200ms.
        Budget is 350ms — serialized execution blows it by 50ms.

        Advisor A.5 coverage gap #1: protects against a future refactor that
        accidentally introduces a shared lock and silently regresses
        parallelism.
        """
        import time

        class SlowEncoder(StubEncoder):
            def classify(self, text):  # type: ignore[override]
                time.sleep(0.2)
                return super().classify(text)

        class SlowPresidio(StubPresidio):
            def analyze(self, text):  # type: ignore[override]
                time.sleep(0.2)
                return super().analyze(text)

        clf = TaskClassifier(
            encoder=SlowEncoder(_enc()),
            presidio=SlowPresidio(_ps([])),
        )
        start = time.perf_counter()
        await clf.classify_async("anything")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.35, (
            f"T2 || T2b ran in {elapsed:.3f}s — expected < 0.35s for parallel "
            f"execution of two 200ms tiers. If > 0.35s, tiers are serializing "
            f"(possibly a shared lock was introduced)."
        )
