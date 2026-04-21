"""Unit tests for TaskClassifier — Stage A.1 (T0 + T1 paths only).

T2 encoder, T2b Presidio, and T5 LLM escalation are not yet wired;
once they are, tests marked `@pytest.mark.skipif(no_gpu)` will cover
the T5 paths per plan.md shipping plan (Stage A scope B).
"""
from __future__ import annotations

from tidus.classification import TaskClassifier


class TestT0CallerOverride:
    def test_full_override_short_circuits(self):
        clf = TaskClassifier()
        result = clf.classify(
            "anything at all",
            caller_override={"domain": "code", "complexity": "complex", "privacy": "public"},
        )
        assert result.classification_tier == "caller_override"
        assert result.domain == "code"
        assert result.complexity == "complex"
        assert result.privacy == "public"
        assert result.confidence == {"domain": 1.0, "complexity": 1.0, "privacy": 1.0}

    def test_partial_override_is_not_t0(self):
        clf = TaskClassifier()
        # Missing complexity -> cannot short-circuit T0, must run T1
        result = clf.classify(
            "```python\nprint('hi')\n```",
            caller_override={"domain": "code", "privacy": "public"},
        )
        assert result.classification_tier == "heuristic"

    def test_no_override_no_signal_falls_to_default_tier(self):
        clf = TaskClassifier()
        result = clf.classify("hello there")
        # No T1 signal fired and no override — "default" (not "heuristic") to
        # give observability an honest picture of the safe-default class.
        assert result.classification_tier == "default"

    def test_no_override_with_t1_signal_uses_heuristic_tier(self):
        clf = TaskClassifier()
        result = clf.classify("my SSN is 123-45-6789")  # regex hits
        assert result.classification_tier == "heuristic"


class TestT1PrivacyShortCircuit:
    def test_ssn_forces_confidential(self):
        clf = TaskClassifier()
        r = clf.classify("my SSN is 123-45-6789 please help")
        assert r.privacy == "confidential"
        # 0.90 (not 1.0) leaves room for T2b Presidio / T5 to correct false
        # positives from SSN-shaped but non-SSN strings. Asymmetric-safety OR
        # merge still forces confidential if any downstream tier agrees.
        assert r.confidence["privacy"] == 0.90

    def test_aws_key_forces_confidential(self):
        clf = TaskClassifier()
        r = clf.classify("use this AKIAIOSFODNN7EXAMPLE to auth")
        assert r.privacy == "confidential"

    def test_benign_text_defaults_to_internal_not_public(self):
        """Plan.md §What-NOT-to-do: never default privacy to public."""
        clf = TaskClassifier()
        r = clf.classify("what's the weather today")
        assert r.privacy == "internal"
        assert r.confidence["privacy"] == 0.50


class TestT1DomainInference:
    def test_code_fence_infers_code(self):
        clf = TaskClassifier()
        r = clf.classify("```python\ndef add(a, b): return a + b\n```")
        assert r.domain == "code"
        assert r.confidence["domain"] == 0.80

    def test_python_def_infers_code(self):
        clf = TaskClassifier()
        r = clf.classify("def my_func(x):\n    return x * 2")
        assert r.domain == "code"

    def test_prose_defaults_to_chat(self):
        clf = TaskClassifier()
        r = clf.classify("can you help me with a question?")
        assert r.domain == "chat"
        assert r.confidence["domain"] == 0.30


class TestT1ComplexityVeto:
    def test_medical_keyword_forces_critical(self):
        clf = TaskClassifier()
        r = clf.classify("can you diagnose these symptoms for me?")
        assert r.complexity == "critical"
        assert r.confidence["complexity"] == 0.90

    def test_legal_keyword_forces_complex(self):
        clf = TaskClassifier()
        r = clf.classify("need to review this NDA with my attorney")
        assert r.complexity == "complex"

    def test_financial_keyword_forces_complex(self):
        clf = TaskClassifier()
        r = clf.classify("explain my W-2 earnings for tax return prep")
        assert r.complexity == "complex"

    def test_benign_text_defaults_to_moderate(self):
        clf = TaskClassifier()
        r = clf.classify("what's 2 + 2")
        assert r.complexity == "moderate"


class TestAsymmetricSafetyMerge:
    def test_caller_public_with_ssn_becomes_confidential(self):
        """Caller says public; message contains SSN. Asymmetric safety wins."""
        clf = TaskClassifier()
        r = clf.classify(
            "here's my SSN 123-45-6789",
            caller_override={"domain": "chat", "privacy": "public"},  # partial
        )
        assert r.privacy == "confidential"

    def test_caller_confidential_stays_confidential(self):
        clf = TaskClassifier()
        r = clf.classify(
            "totally benign prose",
            caller_override={"privacy": "confidential"},
        )
        assert r.privacy == "confidential"

    def test_caller_domain_overrides_t1_domain(self):
        clf = TaskClassifier()
        r = clf.classify(
            "```python\nprint('x')\n```",  # T1 would infer code
            caller_override={"domain": "creative"},  # caller insists it's creative
        )
        assert r.domain == "creative"


class TestDebugPayload:
    def test_debug_omitted_by_default(self):
        clf = TaskClassifier()
        r = clf.classify("hello")
        assert r.debug is None

    def test_debug_included_when_requested(self):
        clf = TaskClassifier()
        r = clf.classify("my SSN 123-45-6789", include_debug=True)
        assert r.debug is not None
        assert "tier1_signals" in r.debug
        assert "SSN_US" in r.debug["tier1_signals"]["regex_hits"]


class TestTokenEstimation:
    def test_estimated_tokens_populated(self):
        clf = TaskClassifier()
        r = clf.classify("a" * 40)
        # 40 chars / 4 chars-per-token = 10
        assert r.estimated_input_tokens == 10
