"""TidusPricingFeedSource — optional remote pricing feed with circuit breaker.

Protocol:
  GET {url}/prices?schema_version=1
  Response: {"prices": [{model_id, input_price, output_price, updated_at, confidence}]}

Security:
  When TIDUS_PRICING_FEED_SIGNING_KEY is set, the response body must include
  an X-Tidus-Signature: hmac-sha256=<hex> header. Unsigned responses are
  accepted (with a warning) only when the key is not configured.

Circuit breaker:
  CLOSED → OPEN after `failure_threshold` consecutive failures.
  OPEN → HALF-OPEN after `reset_timeout_seconds`.
  HALF-OPEN → CLOSED on probe success; → OPEN on probe failure.
  State is in-process only (resets to CLOSED on restart — safe because
  HardcodedSource always provides a fallback).

Rate guard:
  Enforces minimum `min_interval_seconds` between real HTTP calls.
  Returns the cached last response if called sooner.
"""

from __future__ import annotations

import hashlib
import hmac
import random
import time
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

import structlog

from tidus.sync.pricing.base import PriceQuote, PricingSource

log = structlog.get_logger(__name__)

_SCHEMA_VERSION = 1
_REQUEST_TIMEOUT_S = 10.0
_RETRY_BASE_DELAY_S = 2.0  # delays: 2, 4, 8 seconds


class _CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TidusPricingFeedSource(PricingSource):
    """Pulls price data from a Tidus-operated public pricing feed.

    Only GET /prices is called — no customer data is sent.
    Disabled automatically when feed_url is empty.
    """

    def __init__(
        self,
        feed_url: str,
        signing_key: str = "",
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 300,
        min_interval_seconds: int = 3600,
        max_retries: int = 3,
    ) -> None:
        self._feed_url = feed_url.rstrip("/")
        self._signing_key = signing_key
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries

        # Circuit breaker state
        self._cb_state = _CBState.CLOSED
        self._consecutive_failures = 0
        self._open_since: float = 0.0

        # Rate guard
        self._last_call_ts: float = 0.0
        self._last_response: list[PriceQuote] = []

    @property
    def source_name(self) -> str:
        return "tidus_feed"

    @property
    def confidence(self) -> float:
        return 0.85

    @property
    def is_available(self) -> bool:
        if not self._feed_url:
            return False
        # OPEN state short-circuits — no requests until reset_timeout elapses
        if self._cb_state == _CBState.OPEN:
            if time.monotonic() - self._open_since >= self._reset_timeout:
                self._cb_state = _CBState.HALF_OPEN
                log.info("pricing_feed_circuit_half_open")
            else:
                return False
        return True

    async def fetch_quotes(self) -> list[PriceQuote]:
        """Fetch quotes from the remote feed. Returns [] on any failure."""
        if not self.is_available:
            log.info("pricing_feed_circuit_open")
            return []

        # Rate guard — return cached response if called too soon
        now_ts = time.monotonic()
        if now_ts - self._last_call_ts < self._min_interval and self._last_response:
            log.debug("pricing_feed_rate_guard_cached")
            return self._last_response

        quotes = await self._fetch_with_retry()
        if quotes is not None:
            self._on_success()
            self._last_call_ts = time.monotonic()
            self._last_response = quotes
            return quotes
        else:
            self._on_failure()
            return []

    async def _fetch_with_retry(self) -> list[PriceQuote] | None:
        """Attempt the HTTP request up to max_retries times with exponential backoff."""
        import asyncio

        import httpx

        url = f"{self._feed_url}/prices?schema_version={_SCHEMA_VERSION}"
        attempt_count = 0
        failure_reasons: list[str] = []

        for attempt in range(self._max_retries):
            attempt_count += 1
            try:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
                    response = await client.get(url)
                    response.raise_for_status()
            except Exception as exc:
                reason = str(exc)
                failure_reasons.append(reason)
                log.warning("pricing_feed_request_failed", attempt=attempt + 1, error=reason)
                if attempt < self._max_retries - 1:
                    # Exponential backoff with ±25% jitter
                    delay = _RETRY_BASE_DELAY_S * (2 ** attempt) * (0.75 + random.random() * 0.5)
                    await asyncio.sleep(delay)
                continue

            # Verify HMAC signature if key is configured
            if self._signing_key:
                sig_header = response.headers.get("X-Tidus-Signature", "")
                if not sig_header.startswith("hmac-sha256="):
                    log.error("pricing_feed_invalid_signature", reason="missing_header")
                    return None
                expected = hmac.new(
                    self._signing_key.encode(),
                    response.content,
                    hashlib.sha256,
                ).hexdigest()
                provided = sig_header.removeprefix("hmac-sha256=")
                if not hmac.compare_digest(expected, provided):
                    log.error("pricing_feed_invalid_signature", reason="digest_mismatch")
                    return None
            else:
                log.warning("pricing_feed_unsigned", reason="TIDUS_PRICING_FEED_SIGNING_KEY not set")

            try:
                data = response.json()
                return self._parse_response(data, response.headers)
            except Exception as exc:
                log.error("pricing_feed_parse_failed", error=str(exc))
                return None

        log.error("pricing_feed_all_retries_failed", attempts=attempt_count, reasons=failure_reasons)
        return None

    def _parse_response(self, data: dict[str, Any], headers) -> list[PriceQuote]:
        now = datetime.now(UTC)
        today = now.date()
        quotes = []
        for item in data.get("prices", []):
            try:
                quotes.append(PriceQuote(
                    model_id=item["model_id"],
                    input_price=float(item["input_price"]),
                    output_price=float(item["output_price"]),
                    cache_read_price=float(item.get("cache_read_price", 0.0)),
                    cache_write_price=float(item.get("cache_write_price", 0.0)),
                    currency=item.get("currency", "USD"),
                    effective_date=today,
                    retrieved_at=now,
                    source_name=self.source_name,
                    source_confidence=float(item.get("confidence", self.confidence)),
                    evidence_url=item.get("evidence_url"),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                log.warning("pricing_feed_item_parse_failed", item=item, error=str(exc))
        return quotes

    def _on_success(self) -> None:
        if self._cb_state == _CBState.HALF_OPEN:
            self._cb_state = _CBState.CLOSED
            self._consecutive_failures = 0
            log.info("pricing_feed_circuit_closed")
        else:
            self._consecutive_failures = 0

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        if self._cb_state == _CBState.HALF_OPEN:
            # Probe failed → back to OPEN
            self._cb_state = _CBState.OPEN
            self._open_since = time.monotonic()
            log.warning("pricing_feed_circuit_open", reason="half_open_probe_failed")
        elif self._consecutive_failures >= self._failure_threshold:
            self._cb_state = _CBState.OPEN
            self._open_since = time.monotonic()
            log.warning(
                "pricing_feed_circuit_open",
                reason="failure_threshold_reached",
                failures=self._consecutive_failures,
            )
