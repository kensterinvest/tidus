"""Integration-style tests for the Fix 18 adapter retry + translate pipeline.

Verifies that when a vendor SDK raises a status-code-bearing exception inside
an adapter's ``complete``, it is translated to the correct AdapterError
subclass and the retry logic in ``with_retry`` behaves as expected.

Tests mock the SDK client returned by ``_get_client`` so we exercise the real
adapter code paths without any network calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tidus.adapters.base import (
    AdapterAuthError,
    AdapterRateLimitError,
    AdapterServerError,
)


class _StatusCodeError(Exception):
    """Mimics the shape of openai.APIStatusError and similar SDK classes."""

    def __init__(self, status_code: int, message: str = "vendor error"):
        super().__init__(message)
        self.status_code = status_code


def _task_stub(messages=None):
    return SimpleNamespace(
        messages=messages or [{"role": "user", "content": "hi"}],
        estimated_output_tokens=32,
    )


@pytest.fixture
def low_retry_settings():
    """Override adapter settings so tests complete in milliseconds, not seconds."""
    with patch(
        "tidus.adapters.openai_adapter.get_settings",
        return_value=SimpleNamespace(
            openai_api_key="test",
            adapter_timeout_seconds=1.0,
            adapter_max_retries=3,
            adapter_base_delay_seconds=0.001,
        ),
    ):
        yield


# ── OpenAI adapter retry + translate pipeline ────────────────────────────────

class TestOpenAIAdapterRetry:
    async def test_rate_limit_retried_and_surfaced_as_adapter_error(self, low_retry_settings):
        from tidus.adapters.openai_adapter import OpenAIAdapter
        call_count = {"n": 0}

        async def always_429(**kwargs):
            call_count["n"] += 1
            raise _StatusCodeError(429, "too many requests")

        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=always_429)),
        )
        with patch("tidus.adapters.openai_adapter._get_client", return_value=mock_client):
            with pytest.raises(AdapterRateLimitError):
                await OpenAIAdapter().complete("gpt-4.1-nano", _task_stub())
        assert call_count["n"] == 3, "Expected 3 attempts (1 initial + 2 retries)"

    async def test_auth_error_not_retried(self, low_retry_settings):
        from tidus.adapters.openai_adapter import OpenAIAdapter
        call_count = {"n": 0}

        async def bad_key(**kwargs):
            call_count["n"] += 1
            raise _StatusCodeError(401, "invalid API key")

        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=bad_key)),
        )
        with patch("tidus.adapters.openai_adapter._get_client", return_value=mock_client):
            with pytest.raises(AdapterAuthError):
                await OpenAIAdapter().complete("gpt-4.1-nano", _task_stub())
        assert call_count["n"] == 1, "Auth errors must fail fast — no retries"

    async def test_server_error_retried_succeeds_on_second_attempt(self, low_retry_settings):
        from tidus.adapters.openai_adapter import OpenAIAdapter

        async def flaky(**kwargs):
            flaky.n += 1  # type: ignore[attr-defined]
            if flaky.n == 1:  # type: ignore[attr-defined]
                raise _StatusCodeError(503, "service unavailable")
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="Hello back"),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
                model_dump=lambda: {"mock": True},
            )
        flaky.n = 0  # type: ignore[attr-defined]

        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=flaky)),
        )
        with patch("tidus.adapters.openai_adapter._get_client", return_value=mock_client):
            response = await OpenAIAdapter().complete("gpt-4.1-nano", _task_stub())

        assert response.content == "Hello back"
        assert response.input_tokens == 5
        assert response.output_tokens == 3
        assert flaky.n == 2  # type: ignore[attr-defined]

    async def test_server_error_escalates_to_adapter_server_error_after_max_attempts(
        self, low_retry_settings
    ):
        from tidus.adapters.openai_adapter import OpenAIAdapter

        async def always_500(**kwargs):
            raise _StatusCodeError(500, "internal server error")

        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=always_500)),
        )
        with patch("tidus.adapters.openai_adapter._get_client", return_value=mock_client):
            with pytest.raises(AdapterServerError):
                await OpenAIAdapter().complete("gpt-4.1-nano", _task_stub())
