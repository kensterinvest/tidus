"""Stage A.4 tests — TaskClassifier with T5 LLM in the cascade.

Uses a StubLLM to exercise trigger logic + merge without hitting any real
Ollama. LLMClassifier's own HTTP/cache/rate-limit is tested in
test_llm_classifier.py with respx.
"""
from __future__ import annotations

import pytest

from tidus.classification import TaskClassifier
from tidus.classification.encoder import Encoder
from tidus.classification.llm_classifier import LLMClassifier
from tidus.classification.models import EncoderResult, LLMResult


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


class StubLLM(LLMClassifier):
    """Test stub: returns a fixed LLMResult (or None to simulate failure)."""

    def __init__(self, result: LLMResult | None) -> None:  # noqa: D401
        self._result = result
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return True

    async def startup(self) -> None:  # type: ignore[override]
        return None

    async def classify(self, text: str) -> LLMResult | None:  # type: ignore[override]
        return self._result


def _enc(
    domain: str = "chat",
    complexity: str = "moderate",
    privacy: str = "internal",
    dom_c: float = 0.70,
    cmp_c: float = 0.65,
    prv_c: float = 0.70,
) -> EncoderResult:
    return EncoderResult(
        domain=domain,
        complexity=complexity,
        privacy=privacy,
        confidence={"domain": dom_c, "complexity": cmp_c, "privacy": prv_c},
    )


def _llm(
    domain: str = "chat",
    complexity: str = "moderate",
    privacy: str = "confidential",
    rationale: str | None = None,
) -> LLMResult:
    return LLMResult(
        domain=domain,
        complexity=complexity,
        privacy=privacy,
        confidence={"domain": 0.95, "complexity": 0.95, "privacy": 0.95},
        rationale=rationale,
    )


class TestT5TriggerLogic:
    @pytest.mark.asyncio
    async def test_fires_when_topic_keyword_and_uncertain_encoder(self):
        """T5's target case: topic keyword + encoder uncertain (below threshold)."""
        # Encoder privacy=internal at 0.60 — below the 0.75 threshold, so T5 fires.
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.60)),
            llm=StubLLM(_llm(privacy="confidential")),
        )
        r = await clf.classify_async("I was just laid off, can't afford rent")
        assert r.privacy == "confidential"
        assert r.classification_tier == "llm"

    @pytest.mark.asyncio
    async def test_skipped_when_encoder_confident_non_public(self):
        """Bug #2 from advisor: routine medical question with confident encoder
        should NOT burn GPU on T5. Encoder at 0.85 ≥ 0.75 threshold and
        verdict is internal (non-public) → T5 doesn't escalate."""
        calls = {"count": 0}

        class Counting(StubLLM):
            async def classify(self, text):  # type: ignore[override]
                calls["count"] += 1
                return await super().classify(text)

        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.85)),
            llm=Counting(_llm(privacy="confidential")),
        )
        r = await clf.classify_async("what are common flu symptoms")  # medical kw
        assert r.privacy == "internal"  # encoder verdict stands
        assert r.classification_tier != "llm"
        assert calls["count"] == 0

    @pytest.mark.asyncio
    async def test_fires_when_encoder_leans_public_with_topic_keyword(self):
        """Encoder public + medical keyword = potential-miss class → T5 fires.
        This is the critical miss pattern from findings.md §3."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="public", prv_c=0.92)),  # confident public
            llm=StubLLM(_llm(privacy="confidential")),
        )
        r = await clf.classify_async("my doctor diagnosed me with depression")
        # Even a confident "public" on a medical prompt → T5 must re-check,
        # because this is the exact miss class T5 exists to catch.
        assert r.classification_tier == "llm"
        assert r.privacy == "confidential"

    @pytest.mark.asyncio
    async def test_fires_when_no_encoder_cpu_only_fallback(self):
        """CPU-only path: no encoder loaded, topic keyword → T5 runs on keyword alone."""
        clf = TaskClassifier(llm=StubLLM(_llm(privacy="confidential")))  # no encoder
        r = await clf.classify_async("need an attorney for my NDA")
        assert r.classification_tier == "llm"

    @pytest.mark.asyncio
    async def test_skipped_when_already_confidential(self):
        """If T1+T2+T2b already said confidential, skip T5 (save GPU)."""
        calls = {"count": 0}

        class Counting(StubLLM):
            async def classify(self, text):  # type: ignore[override]
                calls["count"] += 1
                return await super().classify(text)

        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="confidential", prv_c=0.90)),
            llm=Counting(_llm(privacy="confidential")),
        )
        r = await clf.classify_async("my SSN is 123-45-6789 and I want legal help")
        assert r.privacy == "confidential"
        # T5 was NOT called despite the legal keyword — confidential already settled.
        assert calls["count"] == 0

    @pytest.mark.asyncio
    async def test_skipped_when_no_topic_keyword(self):
        """Without a topic keyword, T5 shouldn't fire (no signal it'd help)."""
        calls = {"count": 0}

        class Counting(StubLLM):
            async def classify(self, text):  # type: ignore[override]
                calls["count"] += 1
                return await super().classify(text)

        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal")),
            llm=Counting(_llm(privacy="confidential")),
        )
        r = await clf.classify_async("what's 2 + 2")  # no keyword
        assert r.privacy == "internal"
        assert calls["count"] == 0

    @pytest.mark.asyncio
    async def test_skipped_when_no_llm_configured(self):
        """CPU-only SKU — no LLM passed, T5 never runs."""
        clf = TaskClassifier(encoder=StubEncoder(_enc(privacy="internal")))
        r = await clf.classify_async("my attorney mentioned the NDA")
        assert r.classification_tier != "llm"


class TestT5AsymmetricSafety:
    @pytest.mark.asyncio
    async def test_t5_flips_to_confidential(self):
        """T5 says confidential → final is confidential."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.72)),
            llm=StubLLM(_llm(privacy="confidential")),
        )
        r = await clf.classify_async("I can't afford my rent anymore")
        assert r.privacy == "confidential"
        assert r.confidence["privacy"] == 0.95  # max(0.72, 0.95)

    @pytest.mark.asyncio
    async def test_t5_agrees_non_confidential_keeps_prior_tier_label(self):
        """T5 says public — asymmetric safety keeps prior 'internal'. AND
        tier label stays at the prior decisive tier, not 'llm' — T5 didn't
        actually drive the verdict (advisor A.4 Semantic #2)."""
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.60)),  # below threshold → T5 runs
            llm=StubLLM(_llm(privacy="public")),
        )
        r = await clf.classify_async("medical question about vitamins")
        assert r.privacy == "internal"
        assert r.confidence["privacy"] == 0.60  # prior confidence unchanged
        # Tier label must NOT be "llm" — T5 was consulted but didn't flip.
        # Prior tier (encoder) was the decisive voter.
        assert r.classification_tier == "encoder"

    @pytest.mark.asyncio
    async def test_t5_disagreement_internal_to_public_blocked(self):
        """T5 saying public when prior was internal → never lower (asymmetric safety)."""
        # Use low encoder confidence so T5 actually fires (above threshold → skipped)
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.60)),
            llm=StubLLM(_llm(privacy="public")),
        )
        r = await clf.classify_async("HR complaint about manager")
        # Privacy stays internal; T5 cannot demote
        assert r.privacy == "internal"


class TestT5FailureHandling:
    @pytest.mark.asyncio
    async def test_t5_returns_none_sets_confidence_warning(self):
        """LLM unavailable mid-request → confidence_warning=True on result.

        This is the Enterprise SKU's graceful-degradation signal. CPU-only
        SKU wouldn't even attempt T5 (no warning).
        """
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.60)),  # below threshold
            llm=StubLLM(None),  # simulate failure
        )
        r = await clf.classify_async("need an attorney for my HR case")
        assert r.confidence_warning is True
        # Prior verdict preserved
        assert r.privacy == "internal"

    @pytest.mark.asyncio
    async def test_no_warning_when_t5_disabled_cpu_sku(self):
        """CPU-only SKU — no LLM configured, no warning IF the encoder is
        confident across all axes. The threshold-driven warning fires
        independently of T5 state when any axis is below threshold.
        """
        clf = TaskClassifier(
            encoder=StubEncoder(
                # All confidences clearly above their thresholds
                # (privacy>=0.75, domain>=0.70, complexity>=0.65)
                _enc(privacy="internal", prv_c=0.90, dom_c=0.85, cmp_c=0.80)
            ),
        )
        r = await clf.classify_async("need an attorney for my HR case")
        assert r.confidence_warning is False


class TestT5DebugPayload:
    @pytest.mark.asyncio
    async def test_debug_includes_tier5_when_fired(self):
        clf = TaskClassifier(
            encoder=StubEncoder(_enc(privacy="internal", prv_c=0.60)),  # below threshold
            llm=StubLLM(_llm(privacy="confidential", rationale="personal disclosure")),
        )
        r = await clf.classify_async("I'm struggling financially", include_debug=True)
        assert r.debug is not None
        assert "tier5_llm" in r.debug
        assert r.debug["tier5_llm"]["privacy"] == "confidential"
        assert r.debug["tier5_llm"]["rationale"] == "personal disclosure"
