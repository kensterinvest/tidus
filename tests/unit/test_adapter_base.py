"""Unit tests for the Fix 18 adapter exception hierarchy + with_retry helper."""

from __future__ import annotations

import asyncio

import pytest

from tidus.adapters.base import (
    AdapterAuthError,
    AdapterClientError,
    AdapterError,
    AdapterRateLimitError,
    AdapterServerError,
    AdapterTimeoutError,
    translate_vendor_exception,
    with_retry,
)

# ── Exception hierarchy ───────────────────────────────────────────────────────

class TestExceptionHierarchy:
    def test_all_adapter_errors_are_adapter_error(self):
        for cls in (
            AdapterAuthError,
            AdapterRateLimitError,
            AdapterTimeoutError,
            AdapterServerError,
            AdapterClientError,
        ):
            assert issubclass(cls, AdapterError), f"{cls.__name__} must subclass AdapterError"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(AdapterRateLimitError):
            raise AdapterRateLimitError("test")
        with pytest.raises(AdapterError):
            raise AdapterAuthError("test")


# ── with_retry: happy path ────────────────────────────────────────────────────

class TestWithRetrySuccess:
    async def test_returns_result_on_first_attempt(self):
        async def ok():
            return 42
        result = await with_retry(ok, timeout_seconds=1.0)
        assert result == 42

    async def test_returns_result_after_transient_retry(self):
        calls = {"n": 0}
        async def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise AdapterRateLimitError("once")
            return "ok"
        result = await with_retry(
            flaky, timeout_seconds=1.0, max_attempts=3, base_delay_seconds=0.001,
        )
        assert result == "ok"
        assert calls["n"] == 2


# ── with_retry: transient failures ────────────────────────────────────────────

class TestWithRetryTransient:
    async def test_gives_up_after_max_attempts_on_rate_limit(self):
        calls = {"n": 0}
        async def always_rate_limited():
            calls["n"] += 1
            raise AdapterRateLimitError("nope")
        with pytest.raises(AdapterRateLimitError):
            await with_retry(
                always_rate_limited,
                timeout_seconds=1.0,
                max_attempts=3,
                base_delay_seconds=0.001,
            )
        assert calls["n"] == 3

    async def test_server_error_retried_then_raised(self):
        async def always_5xx():
            raise AdapterServerError("500")
        with pytest.raises(AdapterServerError):
            await with_retry(
                always_5xx, timeout_seconds=1.0, max_attempts=2, base_delay_seconds=0.001,
            )


# ── with_retry: timeout ───────────────────────────────────────────────────────

class TestWithRetryTimeout:
    async def test_slow_call_is_cancelled_and_raised_as_timeout(self):
        async def slow():
            await asyncio.sleep(0.1)
            return "never"
        with pytest.raises(AdapterTimeoutError):
            await with_retry(
                slow, timeout_seconds=0.01,
                max_attempts=1, base_delay_seconds=0.001,
            )

    async def test_timeout_is_retried_as_transient(self):
        calls = {"n": 0}
        async def first_slow_then_fast():
            calls["n"] += 1
            if calls["n"] == 1:
                await asyncio.sleep(0.1)
            return "ok"
        # First attempt times out; second succeeds quickly.
        result = await with_retry(
            first_slow_then_fast,
            timeout_seconds=0.02,
            max_attempts=3,
            base_delay_seconds=0.001,
        )
        assert result == "ok"
        assert calls["n"] == 2


# ── with_retry: non-retryable ─────────────────────────────────────────────────

class TestWithRetryNonRetryable:
    async def test_auth_error_not_retried(self):
        calls = {"n": 0}
        async def bad_key():
            calls["n"] += 1
            raise AdapterAuthError("invalid API key")
        with pytest.raises(AdapterAuthError):
            await with_retry(
                bad_key, timeout_seconds=1.0, max_attempts=5, base_delay_seconds=0.001,
            )
        assert calls["n"] == 1, "Auth errors must fail fast — never retried"

    async def test_client_error_not_retried(self):
        calls = {"n": 0}
        async def bad_request():
            calls["n"] += 1
            raise AdapterClientError("malformed")
        with pytest.raises(AdapterClientError):
            await with_retry(
                bad_request, timeout_seconds=1.0, max_attempts=5, base_delay_seconds=0.001,
            )
        assert calls["n"] == 1

    async def test_unknown_exception_propagates_unretried(self):
        """Non-AdapterError exceptions surface unretried so unknown failures are visible."""
        calls = {"n": 0}
        async def mystery():
            calls["n"] += 1
            raise ValueError("unmapped vendor exception")
        with pytest.raises(ValueError):
            await with_retry(
                mystery, timeout_seconds=1.0, max_attempts=3, base_delay_seconds=0.001,
            )
        assert calls["n"] == 1


# ── translate_vendor_exception ────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _StatusExc(Exception):
    def __init__(self, status_code: int, msg: str = "vendor err"):
        super().__init__(msg)
        self.status_code = status_code


class _ResponseExc(Exception):
    def __init__(self, status_code: int):
        super().__init__("with response")
        self.response = _FakeResp(status_code)


class TestTranslateVendorException:
    def test_401_maps_to_auth(self):
        assert isinstance(translate_vendor_exception(_StatusExc(401)), AdapterAuthError)

    def test_403_maps_to_auth(self):
        assert isinstance(translate_vendor_exception(_StatusExc(403)), AdapterAuthError)

    def test_429_maps_to_rate_limit(self):
        assert isinstance(translate_vendor_exception(_StatusExc(429)), AdapterRateLimitError)

    def test_500_maps_to_server(self):
        assert isinstance(translate_vendor_exception(_StatusExc(503)), AdapterServerError)

    def test_400_maps_to_client(self):
        assert isinstance(translate_vendor_exception(_StatusExc(400)), AdapterClientError)

    def test_response_status_attribute_recognised(self):
        assert isinstance(translate_vendor_exception(_ResponseExc(429)), AdapterRateLimitError)

    def test_timeout_error_maps_to_timeout(self):
        assert isinstance(translate_vendor_exception(TimeoutError("slow")), AdapterTimeoutError)

    def test_class_name_authentication_error(self):
        class AuthenticationError(Exception):
            pass
        assert isinstance(translate_vendor_exception(AuthenticationError("bad key")), AdapterAuthError)

    def test_class_name_rate_limit_error(self):
        class RateLimitError(Exception):
            pass
        assert isinstance(translate_vendor_exception(RateLimitError("slow down")), AdapterRateLimitError)

    def test_unknown_exception_maps_to_base_adapter_error(self):
        result = translate_vendor_exception(ValueError("mystery"))
        assert isinstance(result, AdapterError)
        # Not a transient subclass — will not be retried
        assert not isinstance(result, (AdapterRateLimitError, AdapterServerError, AdapterTimeoutError))
