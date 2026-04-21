"""Stage A.2 tests — TaskClassifier cascade with encoder wired via stub.

The encoder's own tests (loading real weights_b/, running inference) live
in tests/integration/test_encoder_integration.py since they pay the
SentenceTransformer load cost (~2-5s) and depend on trained artefacts.

Here we inject a StubEncoder so the cascade merge logic is verified
without loading any ML model — fast, hermetic, and exercises every branch
of the merge rule independently of model quality.
"""
from __future__ import annotations

from tidus.classification import TaskClassifier
from tidus.classification.encoder import Encoder
from tidus.classification.models import EncoderResult


class StubEncoder(Encoder):
    """Pre-loaded encoder that always returns a fixed result. Skips I/O."""

    def __init__(self, result: EncoderResult) -> None:  # noqa: D401 — stub
        self._result = result
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:  # type: ignore[override]
        return None

    def classify(self, text: str) -> EncoderResult:  # type: ignore[override]
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


class TestEncoderTierLabel:
    def test_encoder_fired_labels_tier_encoder(self):
        clf = TaskClassifier(encoder=StubEncoder(_enc()))
        r = clf.classify("just some text")
        assert r.classification_tier == "encoder"

    def test_no_encoder_benign_text_stays_default(self):
        clf = TaskClassifier()  # no encoder
        r = clf.classify("just some text")
        assert r.classification_tier == "default"


class TestT2DomainMerge:
    def test_encoder_domain_wins_over_t1_code_fence(self):
        # T1 heuristic would infer domain=code from the fence; encoder says
        # reasoning. Encoder wins because it's the actual trained model.
        clf = TaskClassifier(encoder=StubEncoder(_enc(domain="reasoning", dom_c=0.85)))
        r = clf.classify("```\nprint('x')\n```")
        assert r.domain == "reasoning"
        assert r.confidence["domain"] == 0.85

    def test_caller_override_wins_over_encoder_domain(self):
        clf = TaskClassifier(encoder=StubEncoder(_enc(domain="reasoning")))
        r = clf.classify("anything", caller_override={"domain": "creative"})
        assert r.domain == "creative"
        assert r.confidence["domain"] == 1.0


class TestT2ComplexityMerge:
    def test_encoder_complexity_used_when_no_keyword_veto(self):
        clf = TaskClassifier(encoder=StubEncoder(_enc(complexity="complex", cmp_c=0.72)))
        r = clf.classify("a benign question without keywords")
        assert r.complexity == "complex"
        assert r.confidence["complexity"] == 0.72

    def test_keyword_veto_raises_encoder_complexity(self):
        # Encoder says "simple"; medical keyword veto forces critical.
        # The floor rule can only raise, never lower.
        clf = TaskClassifier(encoder=StubEncoder(_enc(complexity="simple", cmp_c=0.90)))
        r = clf.classify("please diagnose my symptoms")
        assert r.complexity == "critical"
        assert r.confidence["complexity"] == 0.90  # keyword-veto confidence

    def test_keyword_does_not_lower_encoder_complexity(self):
        # Encoder says "critical"; HR keyword would force "complex".
        # Floor rule must NOT lower critical → complex.
        clf = TaskClassifier(encoder=StubEncoder(_enc(complexity="critical", cmp_c=0.88)))
        r = clf.classify("employment complaint about harassment")
        assert r.complexity == "critical"  # not lowered

    def test_caller_override_wins_over_encoder_complexity(self):
        clf = TaskClassifier(encoder=StubEncoder(_enc(complexity="critical")))
        r = clf.classify("anything", caller_override={"complexity": "simple"})
        assert r.complexity == "simple"


class TestT2PrivacyMerge:
    def test_encoder_public_plus_ssn_regex_forces_confidential(self):
        """Asymmetric safety: T1 regex = confidential wins even if encoder says public."""
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="public", prv_c=0.95)))
        r = clf.classify("my SSN is 123-45-6789 please help")
        assert r.privacy == "confidential"

    def test_encoder_confidential_plus_benign_text(self):
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="confidential", prv_c=0.82)))
        r = clf.classify("innocent-looking question")
        assert r.privacy == "confidential"
        assert r.confidence["privacy"] == 0.82

    def test_both_tiers_agree_confidential_boosts_confidence(self):
        """T1 regex (0.90) + encoder (0.80) both vote confidential →
        base(max)=0.90, +0.05 for the second voter → 0.95."""
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="confidential", prv_c=0.80)))
        r = clf.classify("my SSN 123-45-6789")
        assert r.privacy == "confidential"
        assert r.confidence["privacy"] == 0.95

    def test_encoder_public_stays_public_when_confident(self):
        """No T1 regex, encoder says public at 0.91 (above 0.70 floor) — accept."""
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="public", prv_c=0.91)))
        r = clf.classify("what's the weather like today")
        assert r.privacy == "public"
        assert r.confidence["privacy"] == 0.91

    def test_encoder_weak_public_downgrades_to_internal(self):
        """Encoder says public at 0.45 (below floor) — merge rule demotes to
        internal per plan.md's never-default-to-public rule.

        The low confidence still propagates so telemetry reflects the
        uncertainty rather than claiming high confidence in a default.
        """
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="public", prv_c=0.45)))
        r = clf.classify("neutral-looking business prose")
        assert r.privacy == "internal"
        assert r.confidence["privacy"] == 0.45

    def test_encoder_weak_internal_stays_internal(self):
        """Internal at low confidence is already the safe floor — no change."""
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="internal", prv_c=0.48)))
        r = clf.classify("neutral-looking business prose")
        assert r.privacy == "internal"
        assert r.confidence["privacy"] == 0.48

    def test_three_way_confidential_agreement_reaches_1(self):
        """Caller override + T1 regex + encoder all confidential → 1.0 confidence."""
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="confidential", prv_c=0.80)))
        r = clf.classify(
            "SSN 123-45-6789",
            caller_override={"privacy": "confidential"},
        )
        assert r.privacy == "confidential"
        assert r.confidence["privacy"] == 1.0


class TestEncoderLoadFailureFallsBackToT1Only:
    def test_classifier_without_encoder_degrades_to_t1_only(self):
        """If encoder weights are missing, startup() sets self._encoder=None.

        The cascade must still return a usable ClassificationResult.
        """
        clf = TaskClassifier()  # encoder=None; startup() not called
        r = clf.classify("my SSN is 123-45-6789")
        assert r.privacy == "confidential"  # T1 still catches this
        assert r.classification_tier == "heuristic"  # not "encoder"


class TestT0StillShortCircuits:
    def test_full_override_bypasses_encoder(self):
        # Encoder should never be consulted when caller provided everything.
        calls = {"count": 0}

        class CountingStub(StubEncoder):
            def classify(self, text):  # type: ignore[override]
                calls["count"] += 1
                return super().classify(text)

        clf = TaskClassifier(encoder=CountingStub(_enc()))
        clf.classify(
            "anything",
            caller_override={"domain": "code", "complexity": "complex", "privacy": "public"},
        )
        assert calls["count"] == 0
