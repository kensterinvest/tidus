"""Unit tests for tidus.classification.heuristics — Tier 1 fast-path."""
from __future__ import annotations

from tidus.classification.heuristics import (
    _luhn_valid,
    any_confidential_regex,
    estimate_tokens,
    run_tier1,
)


class TestLuhn:
    def test_valid_visa(self):
        # Test Visa number from PCI DSS sample docs (not a real card)
        assert _luhn_valid("4111111111111111")

    def test_valid_mastercard(self):
        assert _luhn_valid("5555555555554444")

    def test_invalid_wrong_checksum(self):
        assert not _luhn_valid("4111111111111112")

    def test_rejects_too_short(self):
        assert not _luhn_valid("411111111")  # 9 digits

    def test_rejects_too_long(self):
        assert not _luhn_valid("4" * 20)

    def test_strips_separators(self):
        assert _luhn_valid("4111-1111-1111-1111")
        assert _luhn_valid("4111 1111 1111 1111")


class TestRegexPatterns:
    def test_ssn_us_detected(self):
        s = run_tier1("My SSN is 123-45-6789 please don't share")
        assert "SSN_US" in s.regex_hits
        assert any_confidential_regex(s)

    def test_aws_access_key_detected(self):
        s = run_tier1("Key: AKIAIOSFODNN7EXAMPLE")
        assert "AWS_ACCESS_KEY" in s.regex_hits
        assert any_confidential_regex(s)

    def test_github_token_detected(self):
        s = run_tier1("token ghp_" + "a" * 36)
        assert "GITHUB_TOKEN" in s.regex_hits
        assert any_confidential_regex(s)

    def test_openai_key_detected(self):
        s = run_tier1("export OPENAI_API_KEY=sk-" + "a" * 30)
        assert "OPENAI_KEY" in s.regex_hits
        assert any_confidential_regex(s)

    def test_email_is_signal_not_confidential(self):
        s = run_tier1("contact us at support@example.com")
        assert "EMAIL" in s.regex_hits
        # EMAIL alone must NOT force confidential — encoder/Presidio decide
        assert not any_confidential_regex(s)

    def test_credit_card_luhn_valid_with_bin_detected(self):
        # Visa BIN (4), Luhn-valid
        s = run_tier1("card 4111 1111 1111 1111")
        assert "CREDIT_CARD" in s.regex_hits
        assert any_confidential_regex(s)

    def test_credit_card_luhn_invalid_rejected(self):
        # A 16-digit string that isn't a valid Luhn checksum
        s = run_tier1("order number 1234567890123456")
        assert "CREDIT_CARD" not in s.regex_hits

    def test_credit_card_rejects_bad_bin_even_when_luhn_valid(self):
        # All zeros passes Luhn (sum = 0 mod 10) but starts with 0 — not a real
        # BIN. Our pattern requires first digit in [3-6].
        s = run_tier1("reference 0000000000000000")
        assert "CREDIT_CARD" not in s.regex_hits

    def test_credit_card_rejects_serial_starting_with_1_or_2(self):
        # 16-digit serial that would Luhn-fail but let's make sure BIN
        # gate rejects it even before Luhn runs.
        s = run_tier1("serial 1234567812345670")  # starts with 1
        assert "CREDIT_CARD" not in s.regex_hits

    def test_private_key_header_detected(self):
        s = run_tier1("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...")
        assert "PRIVATE_KEY_HEADER" in s.regex_hits
        assert any_confidential_regex(s)

    def test_no_hits_on_benign_text(self):
        s = run_tier1("hello, what's the weather like today?")
        assert s.regex_hits == []
        assert not any_confidential_regex(s)


class TestStructuralSignals:
    def test_code_fence_detected(self):
        s = run_tier1("```python\nprint('hi')\n```")
        assert s.has_code_fence

    def test_python_def_detected(self):
        s = run_tier1("def add(a, b):\n    return a + b")
        assert s.has_code_fence

    def test_prose_has_no_code_fence(self):
        s = run_tier1("the quick brown fox jumps over the lazy dog")
        assert not s.has_code_fence


class TestTokenEstimation:
    def test_empty_returns_one(self):
        assert estimate_tokens("") == 1

    def test_rough_4_chars_per_token(self):
        assert estimate_tokens("a" * 40) == 10

    def test_small_text(self):
        # "hello world" = 11 chars, floor = 2 tokens; floored min = 1
        assert estimate_tokens("hi") == 1


class TestAnyHit:
    def test_any_hit_true_when_regex_matched(self):
        s = run_tier1("my SSN is 123-45-6789")
        assert s.any_hit

    def test_any_hit_true_when_keyword_matched(self):
        s = run_tier1("normal text", keyword_hits=["medical:diagnose"])
        assert s.any_hit

    def test_any_hit_false_on_pure_prose(self):
        s = run_tier1("just chatting about the weather")
        assert not s.any_hit
