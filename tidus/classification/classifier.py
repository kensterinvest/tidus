"""TaskClassifier — orchestrates the T0 → T5 cascade.

Stage A milestones wired so far:
    A.1  T0 (caller override) + T1 (heuristic fast-path)
    A.2  T2 (Recipe B encoder runtime)
    A.3  T2b (Presidio NER) parallel to T2 via asyncio.gather
    A.4  T5 (Ollama LLM escalation, Enterprise SKU)

Invariant: `classify()`/`classify_async()` always return a valid
`ClassificationResult`. Classification must never raise to the caller —
a router that can't classify degrades (safe defaults + confidence_warning),
it does not fail.

Merge rule (updated in A.3):
    * Privacy: asymmetric-safety OR across 4 voters (caller, T1 regex, T2
      encoder, T2b Presidio). ANY confidential vote → confidential.
      Confidence scales with agreement: 1-vote keeps that voter's confidence,
      2+-vote agreement bumps by +0.05 per extra vote, capped at 1.0.
      The T2b PERSON vote is gated by the E1/E2 rule (settings.classify_presidio_rule).
    * Domain:    T2 encoder wins over T1 inference. Caller override tops.
    * Complexity: T1 keyword veto raises the floor (critical/complex).
      Otherwise T2 wins. Caller override tops.

Sync vs async paths: `classify()` runs T2 then T2b serially (simpler, fine
for scripts and tests). `classify_async()` runs them in parallel via
asyncio.gather — total latency becomes max(T2, T2b) instead of T2 + T2b,
which matches plan.md §Tier-2b's "NOT in series" requirement.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from tidus.classification import heuristics, keywords, presidio_wrapper
from tidus.classification.encoder import Encoder, EncoderProtocol
from tidus.classification.llm_classifier import LLMClassifier
from tidus.classification.models import (
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
)
from tidus.classification.presidio_wrapper import PresidioWrapper
from tidus.observability.classification_metrics import T5_FLIPS_TOTAL
from tidus.settings import Settings, get_settings

log = logging.getLogger(__name__)

_EXPECTED_OVERRIDE_KEYS = ("domain", "complexity", "privacy")
_COMPLEXITY_ORDER = {"simple": 0, "moderate": 1, "complex": 2, "critical": 3}

# Per-voter default privacy-confidence when casting a confidential vote.
_CONF_T1_REGEX = 0.90   # pattern-based, high precision
_CONF_T2B_HIGH_TRUST = 0.90  # Presidio built-in recognizers (PHONE/SSN/etc.)
_CONF_T2B_PERSON_ONLY = 0.70  # PERSON has higher FP rate (see findings.md)
_CONF_CALLER = 1.0


class TaskClassifier:
    """Orchestrator.

    NOT thread-safe under concurrent async load — Encoder and Presidio hold
    non-thread-safe state (torch + spaCy). A.5 endpoint wiring adds a lock
    or pool (backlog task #43). Until then, construct one instance per
    process and rely on FastAPI's request-scoping to serialize access.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        encoder: EncoderProtocol | None = None,
        presidio: PresidioWrapper | None = None,
        llm: LLMClassifier | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._encoder = encoder
        self._presidio = presidio
        self._llm = llm
        # Per-tier async locks for thread-safety under concurrent FastAPI load.
        # Encoder (torch) and Presidio (spaCy) hold non-thread-safe state; two
        # requests hitting asyncio.to_thread concurrently can interleave or
        # crash. Per-tier locks preserve in-request T2‖T2b parallelism while
        # serializing across requests (advisor A.2 Bug #1, task #43).
        self._encoder_lock = asyncio.Lock()
        self._presidio_lock = asyncio.Lock()

    async def startup(self) -> None:
        """Load T2 + T2b + T5 artefacts concurrently. Failures degrade gracefully.

        * Encoder load failure → T2 disabled, T1-only fallback.
        * Presidio load failure → T2b disabled, T1+T2 proceed.
        * T5 LLM failure → Enterprise features disabled, CPU-only SKU baseline.

        Loads run in parallel via `asyncio.gather` — cuts ~half off startup
        time for K8s liveness readiness (advisor A.5 Bug #2). Encoder + Presidio
        are sync loads wrapped in `asyncio.to_thread`; T5 LLM startup is
        already async.
        """
        if self._encoder is None:
            self._encoder = Encoder(
                weights_dir=self._settings.classify_encoder_dir,
                max_chars=self._settings.classify_encoder_max_chars,
            )
        if self._settings.classify_presidio_enabled and self._presidio is None:
            self._presidio = PresidioWrapper(
                max_chars=self._settings.classify_presidio_max_chars,
            )
        if self._settings.classify_tier5_enabled and self._llm is None:
            self._llm = LLMClassifier(
                model=self._settings.classify_tier5_model,
                endpoint=self._settings.ollama_base_url,
                rate_limit_per_minute=self._settings.classify_tier5_rate_limit_per_minute,
                cache_ttl_seconds=self._settings.classify_cache_ttl_seconds,
                cache_max_entries=self._settings.classify_cache_max_entries,
            )

        results = await asyncio.gather(
            self._safe_load_encoder(),
            self._safe_load_presidio(),
            self._safe_load_llm(),
            return_exceptions=False,  # helpers catch + log internally
        )
        # Helper return values indicate which tier(s) failed — we use those to
        # null out the instance attribute so later classify() calls see
        # `loaded=False`. Order matches the gather order above.
        if not results[0]:
            self._encoder = None
        if not results[1]:
            self._presidio = None
        if not results[2]:
            self._llm = None

    async def _safe_load_encoder(self) -> bool:
        if self._encoder is None:
            return False
        try:
            await asyncio.to_thread(self._encoder.load)
            return True
        except EncoderLoadError as exc:
            log.warning("Encoder unavailable, falling back to T1-only: %s", exc)
            return False

    async def _safe_load_presidio(self) -> bool:
        if self._presidio is None:
            return False
        try:
            await asyncio.to_thread(self._presidio.load)
            return True
        except Exception as exc:  # noqa: BLE001 — Presidio raises varied types
            log.warning("Presidio unavailable, falling back to T1+T2: %s", exc)
            return False

    async def _safe_load_llm(self) -> bool:
        if self._llm is None:
            return False
        try:
            await self._llm.startup()
            return True
        except LLMUnavailableError as exc:
            log.warning(
                "T5 LLM unavailable; running as CPU-only SKU (89.2%% recall baseline). "
                "Enterprise deployments: check Ollama / GPU. %s",
                exc,
            )
            return False

    # ── Sync path (tests, CLI, offline backtests) ────────────────────

    def classify(
        self,
        text: str,
        caller_override: dict | None = None,
        include_debug: bool = False,
    ) -> ClassificationResult:
        """Serial cascade. T2 and T2b run one after the other."""
        if self._is_complete_override(caller_override):
            return self._from_override(text, caller_override, include_debug)

        kw_hits = keywords.match(text)
        signals = heuristics.run_tier1(text, keyword_hits=keywords.flatten(kw_hits))

        t2 = self._encoder.classify(text) if self._encoder and self._encoder.loaded else None
        t2b = self._presidio.analyze(text) if self._presidio and self._presidio.loaded else None

        return self._build_result(signals, kw_hits, t2, t2b, caller_override, include_debug)

    # ── Async path (FastAPI endpoint, A.5) ───────────────────────────

    async def classify_async(
        self,
        text: str,
        caller_override: dict | None = None,
        include_debug: bool = False,
        telemetry_observer: Callable[..., None] | None = None,
    ) -> ClassificationResult:
        """Parallel cascade. T2 and T2b run via `asyncio.gather` — latency
        collapses to max(T2, T2b) instead of T2 + T2b. Matches plan.md
        §Tier-2b "NOT in series" requirement.

        T5 runs sequentially AFTER T1+T2+T2b merge, gated by the topic-
        keyword + non-confidential trigger (plan.md §Stage-A line 526).

        `telemetry_observer`, when provided, is invoked with the per-tier
        intermediates (signals / encoder / presidio / result) after the
        cascade completes. Stage B telemetry uses this to log a PII-safe
        record without coupling the classifier to observability. Observer
        exceptions are swallowed — telemetry never fails the request path.
        """
        if self._is_complete_override(caller_override):
            result = self._from_override(text, caller_override, include_debug)
            self._invoke_observer(telemetry_observer, None, None, None, result)
            return result

        kw_hits = keywords.match(text)
        signals = heuristics.run_tier1(text, keyword_hits=keywords.flatten(kw_hits))

        t2_coro = self._run_encoder_locked(text) if self._encoder and self._encoder.loaded else None
        t2b_coro = self._run_presidio_locked(text) if self._presidio and self._presidio.loaded else None

        if t2_coro and t2b_coro:
            t2, t2b = await asyncio.gather(t2_coro, t2b_coro)
        elif t2_coro:
            t2 = await t2_coro
            t2b = None
        elif t2b_coro:
            t2 = None
            t2b = await t2b_coro
        else:
            t2 = t2b = None

        result = self._build_result(signals, kw_hits, t2, t2b, caller_override, include_debug)

        # T5 escalation — topic-bearing confidential last-chance catch.
        if self._should_escalate_to_t5(result, kw_hits, t2):
            assert self._llm is not None  # narrowed by _should_escalate_to_t5
            t5 = await self._llm.classify(text)
            result = self._apply_t5(result, t5, include_debug)

        self._invoke_observer(telemetry_observer, signals, t2, t2b, result)
        return result

    @staticmethod
    def _invoke_observer(
        observer: Callable[..., None] | None,
        signals,
        encoder: EncoderResult | None,
        presidio: PresidioResult | None,
        result: ClassificationResult,
    ) -> None:
        if observer is None:
            return
        try:
            observer(
                signals=signals,
                encoder=encoder,
                presidio=presidio,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001 — telemetry must never fail requests
            log.warning("classification telemetry observer raised: %s", exc)

    # ── Shared cascade merge ─────────────────────────────────────────

    def _build_result(
        self,
        signals,
        kw_hits: dict,
        t2: EncoderResult | None,
        t2b: PresidioResult | None,
        caller_override: dict | None,
        include_debug: bool,
    ) -> ClassificationResult:
        t1_priv, t1_priv_conf = self._derive_privacy_from_t1(signals)
        t1_domain, t1_domain_conf = self._derive_domain_from_t1(signals)
        t1_cmplx, t1_cmplx_conf = self._derive_complexity_from_t1(kw_hits)
        cmplx_floor = keywords.complexity_veto(kw_hits)

        domain, domain_conf = self._merge_domain(t1_domain, t1_domain_conf, t2, caller_override)
        complexity, cmplx_conf = self._merge_complexity(
            t1_cmplx, t1_cmplx_conf, cmplx_floor, t2, caller_override,
        )
        privacy, privacy_conf = self._merge_privacy(
            t1_priv, t1_priv_conf, t2, t2b, caller_override,
            public_floor=self._settings.classify_privacy_public_floor,
            presidio_rule=self._settings.classify_presidio_rule,
        )

        tier = self._pick_tier(
            t2_fired=t2 is not None,
            t2b_fired=t2b is not None,
            t1_any_hit=signals.any_hit,
            had_override=bool(caller_override),
        )

        debug = None
        if include_debug:
            debug = {
                "tier1_signals": signals.model_dump(),
                "keyword_categories": {k: v for k, v in kw_hits.items()},
                "tier2_encoder": t2.model_dump() if t2 else None,
                "tier2b_presidio": t2b.model_dump() if t2b else None,
                "presidio_rule": self._settings.classify_presidio_rule,
                "caller_override_applied": bool(caller_override),
            }

        return ClassificationResult(
            domain=domain,
            complexity=complexity,
            privacy=privacy,
            estimated_input_tokens=signals.estimated_input_tokens,
            classification_tier=tier,
            confidence={
                "domain": domain_conf,
                "complexity": cmplx_conf,
                "privacy": privacy_conf,
            },
            confidence_warning=False,  # A.4 sets on T5-unavailable fallback
            debug=debug,
        )

    # ── Override handling ──────────────────────────────────────────────

    @staticmethod
    def _is_complete_override(override: dict | None) -> bool:
        return bool(override) and all(k in override for k in _EXPECTED_OVERRIDE_KEYS)

    @staticmethod
    def _from_override(
        text: str, override: dict, include_debug: bool,
    ) -> ClassificationResult:
        tokens = override.get("estimated_input_tokens") or heuristics.estimate_tokens(text)
        return ClassificationResult(
            domain=override["domain"],
            complexity=override["complexity"],
            privacy=override["privacy"],
            estimated_input_tokens=tokens,
            classification_tier="caller_override",
            confidence={"domain": 1.0, "complexity": 1.0, "privacy": 1.0},
            debug={"caller_override_applied": True} if include_debug else None,
        )

    # ── T1 inference ──────────────────────────────────────────────────

    @staticmethod
    def _derive_privacy_from_t1(signals) -> tuple[Privacy, float]:
        if heuristics.any_confidential_regex(signals):
            return "confidential", _CONF_T1_REGEX
        return "internal", 0.50

    @staticmethod
    def _derive_domain_from_t1(signals) -> tuple[Domain, float]:
        if signals.has_code_fence:
            return "code", 0.80
        return "chat", 0.30

    # Keyword-veto confidence values below are NOT calibrated probabilities —
    # they are voter strengths for a RULE-TRIGGERED FLOOR applied in
    # `_merge_complexity`. Semantics:
    #   * When a `critical` keyword fires (e.g. medical-treatment ask), the
    #     cascade treats the prompt as critical regardless of encoder output.
    #   * When a `complex` keyword fires (legal / financial / HR), same idea
    #     at the `complex` tier.
    # The values are "how much weight does this rule carry when it wins the
    # floor comparison" — they appear in `ClassificationResult.confidence`
    # only when the floor actually overrides the encoder. A measured value
    # would be per-vote agreement accuracy on IRR-adjudicated ground truth;
    # we don't have that measurement yet (backlog task #45).
    _CONF_CMPLX_CRITICAL_RULE = 0.90
    _CONF_CMPLX_COMPLEX_RULE = 0.80
    _CONF_CMPLX_DEFAULT = 0.30  # no keyword hit; encoder's value usually wins anyway

    @staticmethod
    def _derive_complexity_from_t1(kw_hits: dict) -> tuple[Complexity, float]:
        floor = keywords.complexity_veto(kw_hits)
        if floor == "critical":
            return "critical", TaskClassifier._CONF_CMPLX_CRITICAL_RULE
        if floor == "complex":
            return "complex", TaskClassifier._CONF_CMPLX_COMPLEX_RULE
        return "moderate", TaskClassifier._CONF_CMPLX_DEFAULT

    # ── Merge rules ───────────────────────────────────────────────────

    @staticmethod
    def _merge_domain(
        t1_domain: Domain, t1_conf: float,
        t2: EncoderResult | None,
        caller_override: dict | None,
    ) -> tuple[Domain, float]:
        if caller_override and "domain" in caller_override:
            return caller_override["domain"], 1.0
        if t2 is not None:
            return t2.domain, t2.confidence["domain"]
        return t1_domain, t1_conf

    @staticmethod
    def _merge_complexity(
        t1_cmplx: Complexity, t1_conf: float,
        t1_floor: str | None,
        t2: EncoderResult | None,
        caller_override: dict | None,
    ) -> tuple[Complexity, float]:
        if caller_override and "complexity" in caller_override:
            return caller_override["complexity"], 1.0

        base_cmplx: Complexity = t2.complexity if t2 else t1_cmplx
        base_conf = t2.confidence["complexity"] if t2 else t1_conf

        if t1_floor is not None:
            if _COMPLEXITY_ORDER[t1_floor] > _COMPLEXITY_ORDER[base_cmplx]:
                return t1_floor, t1_conf
        return base_cmplx, base_conf

    @staticmethod
    def _presidio_votes_confidential(
        t2b: PresidioResult,
        t2: EncoderResult | None,
        rule: str,
    ) -> tuple[bool, float]:
        """Does T2b's output cast a confidential vote? Returns (votes, confidence).

        High-trust entities (SSN/PHONE/CREDIT_CARD/IBAN/...) are an immediate
        confidential vote regardless of rule. PERSON goes through the E1/E2 gate:
          * E1: PERSON alone → vote confidential at PERSON confidence.
          * E2: PERSON → vote ONLY if encoder says non-public. Matches the
                precision-preferred tenant's deployment.

        The E2 definition is "PERSON + Encoder-non-public" per findings.md §3
        (measured at 83.1% recall, 19% flag rate). High-trust entities fire
        standalone in BOTH E1 and E2 — findings.md's 83.1% was measured with
        that behaviour. Broader "encoder-corroborates-all" variants are out
        of scope until a new IRR measurement justifies them.
        """
        if presidio_wrapper.has_high_trust_hit(t2b):
            return True, _CONF_T2B_HIGH_TRUST
        if t2b.detected_person:
            if rule == "E1":
                return True, _CONF_T2B_PERSON_ONLY
            if rule == "E2":
                # E2 requires encoder corroboration: non-public.
                if t2 is not None and t2.privacy != "public":
                    return True, _CONF_T2B_PERSON_ONLY
        return False, 0.0

    @staticmethod
    def _merge_privacy(
        t1_priv: Privacy, t1_conf: float,
        t2: EncoderResult | None,
        t2b: PresidioResult | None,
        caller_override: dict | None,
        *,
        public_floor: float,
        presidio_rule: str,
    ) -> tuple[Privacy, float]:
        """Asymmetric-safety OR across up to 4 voters."""
        caller_priv = caller_override.get("privacy") if caller_override else None

        t2b_vote, t2b_conf = (
            TaskClassifier._presidio_votes_confidential(t2b, t2, presidio_rule)
            if t2b is not None else (False, 0.0)
        )

        votes = (
            t1_priv == "confidential",                       # T1
            t2 is not None and t2.privacy == "confidential",  # T2
            t2b_vote,                                         # T2b
            caller_priv == "confidential",                    # caller
        )
        vote_confs = (
            t1_conf if votes[0] else 0.0,
            t2.confidence["privacy"] if (votes[1] and t2) else 0.0,
            t2b_conf if votes[2] else 0.0,
            _CONF_CALLER if votes[3] else 0.0,
        )

        n_votes = sum(votes)
        if n_votes:
            base = max(c for v, c in zip(votes, vote_confs) if v)
            # +0.05 per additional agreeing voter, capped at 1.0.
            # Round to 4 decimals so downstream equality checks don't trip
            # on 0.9 + 0.05 = 0.9500000000000001 float drift.
            conf = round(min(1.0, base + 0.05 * (n_votes - 1)), 4)
            return "confidential", conf

        # No confidential vote: caller > encoder > T1.
        if caller_priv is not None:
            return caller_priv, 1.0
        if t2 is not None:
            # Demote weak "public" per never-default-to-public rule.
            if t2.privacy == "public" and t2.confidence["privacy"] < public_floor:
                return "internal", t2.confidence["privacy"]
            return t2.privacy, t2.confidence["privacy"]
        return t1_priv, t1_conf

    # ── Locked tier runners ───────────────────────────────────────────

    async def _run_encoder_locked(self, text: str) -> EncoderResult:
        async with self._encoder_lock:
            return await asyncio.to_thread(self._encoder.classify, text)

    async def _run_presidio_locked(self, text: str) -> PresidioResult:
        async with self._presidio_lock:
            return await asyncio.to_thread(self._presidio.analyze, text)

    # ── Health ────────────────────────────────────────────────────────

    @property
    def healthy(self) -> dict:
        """Per-tier load state for /health and observability. A degraded
        classifier (e.g., Presidio failed to load) still answers requests —
        but dashboards can flag the reduced recall (task #44)."""
        return {
            "encoder_loaded": self._encoder is not None and self._encoder.loaded,
            "presidio_loaded": self._presidio is not None and self._presidio.loaded,
            "llm_loaded": self._llm is not None and self._llm.loaded,
            "sku": "enterprise" if (self._llm is not None and self._llm.loaded) else "cpu-only",
        }

    # ── Tier 5 escalation ─────────────────────────────────────────────

    def _should_escalate_to_t5(
        self,
        result: ClassificationResult,
        kw_hits: dict,
        t2: EncoderResult | None,
    ) -> bool:
        """T5 trigger per plan.md §Stage-A line 526.

        Fire when ALL of:
          * T5 is configured + loaded (Enterprise SKU, GPU present)
          * Current merged verdict is NOT already confidential (routing is
            already safe; no reason to burn GPU cycles confirming)
          * At least one topic keyword matched — topic-bearing confidentials
            are T5's target class per findings.md §3 (6/12 IRR flips)
          * The encoder ISN'T confident in a non-public verdict. A confident
            "internal" from the encoder on a benign medical question ("what
            are flu symptoms") shouldn't burn GPU cycles. We still fire when
            the encoder leaned "public" — that's the potential-miss class —
            or when the encoder was missing (CPU-only-SKU fallback path).

        Bug #2 from advisor A.4 review: broad `bool(kw_hits)` gate fired T5
        on every medical/legal/HR question regardless of encoder confidence,
        blowing up the Enterprise SKU's GPU cost envelope.
        """
        if self._llm is None or not self._llm.loaded:
            return False
        if result.privacy == "confidential":
            return False
        if not kw_hits:
            return False
        if t2 is None:
            # No encoder signal — T5 is our only other vote. Escalate.
            return True
        if t2.privacy == "public":
            # Encoder leaned public on a topic-keyword prompt — the exact
            # miss class T5 exists to catch. Escalate.
            return True
        # Encoder said internal/confidential — trust if confident enough.
        threshold = self._settings.classify_privacy_threshold
        if t2.confidence["privacy"] >= threshold:
            return False
        return True

    def _apply_t5(
        self,
        prior: ClassificationResult,
        t5: LLMResult | None,
        include_debug: bool,
    ) -> ClassificationResult:
        """Merge T5 verdict into the pre-T5 result.

        T5 is asymmetric-safety only: it can flip privacy to `confidential`
        but never lower. Domain/complexity are NOT overwritten — T5 was
        invoked for the privacy decision, not to re-adjudicate the other
        axes. If T5 was intended but returned None, emit `confidence_warning`
        so downstream consumers can flag the classification for review.
        """
        if t5 is None:
            # LLM was configured + loaded but the request failed (rate limit,
            # timeout, parse error). Caller should treat this as "Enterprise
            # SKU couldn't fully classify — degrade gracefully."
            new_debug = prior.debug
            if include_debug:
                new_debug = {**(prior.debug or {}), "tier5_llm": None}
            return prior.model_copy(update={
                "confidence_warning": True,
                "debug": new_debug,
            })

        flipped = t5.privacy == "confidential" and prior.privacy != "confidential"

        new_privacy: Privacy = prior.privacy
        new_privacy_conf: float = prior.confidence["privacy"]
        if flipped:
            new_privacy = "confidential"
            new_privacy_conf = max(prior.confidence["privacy"], t5.confidence["privacy"])
            T5_FLIPS_TOTAL.labels(from_privacy=prior.privacy).inc()

        new_confidence = dict(prior.confidence)
        new_confidence["privacy"] = new_privacy_conf

        # Tier label only moves to "llm" when T5 actually changed the verdict.
        # When T5 was consulted but agreed non-confidential, the prior tier
        # (e.g., "encoder") was the decisive voter — the label should reflect
        # that, not imply LLM was decisive. Advisor A.4 review Semantic #2.
        new_tier = "llm" if flipped else prior.classification_tier

        new_debug = prior.debug
        if include_debug:
            new_debug = {**(prior.debug or {}), "tier5_llm": t5.model_dump()}

        return prior.model_copy(update={
            "privacy": new_privacy,
            "classification_tier": new_tier,
            "confidence": new_confidence,
            "debug": new_debug,
        })

    # ── Tier labelling ────────────────────────────────────────────────

    @staticmethod
    def _pick_tier(
        *,
        t2_fired: bool,
        t2b_fired: bool,
        t1_any_hit: bool,
        had_override: bool,
    ) -> ClassificationTier:
        # Encoder tier wins the label when any model-based tier contributed,
        # since T2/T2b outputs carry more information than T1 heuristics.
        if t2_fired or t2b_fired:
            return "encoder"
        if t1_any_hit or had_override:
            return "heuristic"
        return "default"
