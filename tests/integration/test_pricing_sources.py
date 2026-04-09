"""Integration tests for the PricingSource hierarchy.

Covers:
  - HardcodedSource: returns all 18 known-price quotes, always available
  - TidusPricingFeedSource: disabled (is_available=False) when URL is empty
  - Feed with mocked HTTP response: parses quotes correctly (via respx)
  - Malformed feed response: returns [] without raising
  - HMAC signature verification: valid sig accepted; tampered body rejected; missing sig rejected
  - Circuit breaker: CLOSED → OPEN after threshold failures; no network call while OPEN
  - Circuit breaker: OPEN → HALF-OPEN after reset_timeout; success → CLOSED; failure → OPEN
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import patch

import pytest
import respx
from httpx import Response

from tidus.sync.pricing.feed_source import TidusPricingFeedSource, _CBState
from tidus.sync.pricing.hardcoded_source import HardcodedSource, _KNOWN_PRICES


# ── HardcodedSource ───────────────────────────────────────────────────────────

class TestHardcodedSource:
    @pytest.mark.asyncio
    async def test_returns_all_known_models(self):
        quotes = await HardcodedSource().fetch_quotes()
        model_ids = {q.model_id for q in quotes}
        assert model_ids == set(_KNOWN_PRICES.keys())

    @pytest.mark.asyncio
    async def test_all_quotes_have_correct_source_name(self):
        quotes = await HardcodedSource().fetch_quotes()
        assert all(q.source_name == "hardcoded" for q in quotes)

    @pytest.mark.asyncio
    async def test_prices_are_positive(self):
        quotes = await HardcodedSource().fetch_quotes()
        assert all(q.input_price > 0 and q.output_price > 0 for q in quotes)

    def test_is_always_available(self):
        assert HardcodedSource().is_available is True

    def test_confidence_is_0_7(self):
        assert HardcodedSource().confidence == 0.7


# ── TidusPricingFeedSource: availability ─────────────────────────────────────

class TestFeedSourceAvailability:
    def test_disabled_when_url_is_empty(self):
        feed = TidusPricingFeedSource(feed_url="")
        assert feed.is_available is False

    def test_enabled_when_url_is_set(self):
        feed = TidusPricingFeedSource(feed_url="https://example.com")
        assert feed.is_available is True

    @pytest.mark.asyncio
    async def test_fetch_returns_empty_when_not_available(self):
        feed = TidusPricingFeedSource(feed_url="")
        quotes = await feed.fetch_quotes()
        assert quotes == []


# ── TidusPricingFeedSource: HTTP fetch ───────────────────────────────────────

FEED_URL = "https://pricing.tidus.example.com"
_VALID_PAYLOAD = {
    "prices": [
        {
            "model_id": "gpt-4o",
            "input_price": 0.0025,
            "output_price": 0.01,
            "confidence": 0.85,
        },
        {
            "model_id": "claude-opus-4-6",
            "input_price": 0.015,
            "output_price": 0.075,
            "confidence": 0.85,
        },
    ]
}


def _make_signature(body: bytes, key: str) -> str:
    digest = hmac.new(key.encode(), body, hashlib.sha256).hexdigest()
    return f"hmac-sha256={digest}"


class TestFeedSourceHTTP:
    @pytest.mark.asyncio
    async def test_parses_valid_response_into_quotes(self):
        body = json.dumps(_VALID_PAYLOAD).encode()

        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(
                return_value=Response(200, content=body, headers={"content-type": "application/json"})
            )
            feed = TidusPricingFeedSource(feed_url=FEED_URL, signing_key="")
            quotes = await feed.fetch_quotes()

        assert len(quotes) == 2
        model_ids = {q.model_id for q in quotes}
        assert "gpt-4o" in model_ids
        assert "claude-opus-4-6" in model_ids

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self):
        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(
                return_value=Response(200, content=b"not-json", headers={"content-type": "text/plain"})
            )
            feed = TidusPricingFeedSource(feed_url=FEED_URL, signing_key="")
            quotes = await feed.fetch_quotes()

        assert quotes == []

    @pytest.mark.asyncio
    async def test_http_500_returns_empty_after_retries(self):
        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(return_value=Response(500))
            # Retries are fast in test; patch sleep to avoid delays
            with patch("asyncio.sleep"):
                feed = TidusPricingFeedSource(feed_url=FEED_URL, signing_key="", max_retries=2)
                quotes = await feed.fetch_quotes()

        assert quotes == []

    @pytest.mark.asyncio
    async def test_valid_hmac_signature_accepted(self):
        key = "super-secret"
        body = json.dumps(_VALID_PAYLOAD).encode()
        sig = _make_signature(body, key)

        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(
                return_value=Response(
                    200, content=body,
                    headers={"content-type": "application/json", "X-Tidus-Signature": sig},
                )
            )
            feed = TidusPricingFeedSource(feed_url=FEED_URL, signing_key=key)
            quotes = await feed.fetch_quotes()

        assert len(quotes) == 2

    @pytest.mark.asyncio
    async def test_tampered_body_signature_rejected(self):
        key = "super-secret"
        # Sign a valid body, but serve different content
        body = json.dumps(_VALID_PAYLOAD).encode()
        sig = _make_signature(body, key)
        tampered = b'{"prices": []}'  # different content, same signature

        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(
                return_value=Response(
                    200, content=tampered,
                    headers={"content-type": "application/json", "X-Tidus-Signature": sig},
                )
            )
            feed = TidusPricingFeedSource(feed_url=FEED_URL, signing_key=key)
            quotes = await feed.fetch_quotes()

        assert quotes == []

    @pytest.mark.asyncio
    async def test_missing_signature_header_with_key_configured_rejected(self):
        body = json.dumps(_VALID_PAYLOAD).encode()

        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(
                return_value=Response(200, content=body, headers={"content-type": "application/json"})
                # No X-Tidus-Signature header
            )
            feed = TidusPricingFeedSource(feed_url=FEED_URL, signing_key="my-key")
            quotes = await feed.fetch_quotes()

        assert quotes == []


# ── TidusPricingFeedSource: circuit breaker ──────────────────────────────────

class TestCircuitBreaker:
    def _make_feed(self, threshold: int = 3) -> TidusPricingFeedSource:
        return TidusPricingFeedSource(
            feed_url=FEED_URL,
            signing_key="",
            failure_threshold=threshold,
            reset_timeout_seconds=300,
            min_interval_seconds=0,  # disable rate guard for testing
        )

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self):
        feed = self._make_feed(threshold=3)
        assert feed._cb_state == _CBState.CLOSED

        # Simulate 3 consecutive failures
        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(return_value=Response(500))
            with patch("asyncio.sleep"):
                for _ in range(3):
                    await feed.fetch_quotes()

        assert feed._cb_state == _CBState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_returns_empty_without_network_call(self):
        feed = self._make_feed(threshold=1)

        # Trip the circuit open
        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(return_value=Response(500))
            with patch("asyncio.sleep"):
                await feed.fetch_quotes()

        assert feed._cb_state == _CBState.OPEN

        # Next call should return [] without any HTTP call
        with respx.mock:
            # No routes registered — any HTTP call would raise an error
            quotes = await feed.fetch_quotes()

        assert quotes == []

    @pytest.mark.asyncio
    async def test_open_transitions_to_half_open_after_timeout(self):
        feed = self._make_feed(threshold=1)

        # Trip to OPEN
        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(return_value=Response(500))
            with patch("asyncio.sleep"):
                await feed.fetch_quotes()
        assert feed._cb_state == _CBState.OPEN

        # Simulate timeout elapsed by backdating _open_since
        feed._open_since = time.monotonic() - 400  # past reset_timeout_seconds=300

        # Checking is_available transitions to HALF_OPEN
        assert feed.is_available is True
        assert feed._cb_state == _CBState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        feed = self._make_feed(threshold=1)
        feed._cb_state = _CBState.HALF_OPEN
        feed._consecutive_failures = 1

        body = json.dumps(_VALID_PAYLOAD).encode()
        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(
                return_value=Response(200, content=body, headers={"content-type": "application/json"})
            )
            quotes = await feed.fetch_quotes()

        assert feed._cb_state == _CBState.CLOSED
        assert feed._consecutive_failures == 0
        assert len(quotes) == 2

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        feed = self._make_feed(threshold=1)
        feed._cb_state = _CBState.HALF_OPEN

        with respx.mock:
            respx.get(f"{FEED_URL}/prices").mock(return_value=Response(500))
            with patch("asyncio.sleep"):
                quotes = await feed.fetch_quotes()

        assert feed._cb_state == _CBState.OPEN
        assert quotes == []
