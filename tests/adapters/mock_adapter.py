"""Reusable mock vendor adapter for integration tests.

Simulates a vendor API without making any network calls. Returned
from a pytest fixture so tests can exercise the full HTTP pipeline
(routing → selection → adapter → cost logging) with zero API keys.

Usage:
    from tests.adapters.mock_adapter import MockAdapter, FailingAdapter

    adapter = MockAdapter()
    # or to simulate a vendor outage:
    adapter = FailingAdapter(error_msg="Service unavailable")
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from tidus.adapters.base import AbstractModelAdapter, AdapterResponse


class MockAdapter(AbstractModelAdapter):
    """Returns a fixed canned response — use for happy-path endpoint tests."""

    vendor = "mock"

    def __init__(
        self,
        content: str = "Mock vendor response.",
        input_tokens: int = 42,
        output_tokens: int = 18,
        latency_ms: float = 55.0,
        finish_reason: str = "stop",
    ) -> None:
        self._content = content
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._latency_ms = latency_ms
        self._finish_reason = finish_reason

    async def complete(self, model_id: str, task) -> AdapterResponse:
        return AdapterResponse(
            model_id=model_id,
            content=self._content,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=self._latency_ms,
            finish_reason=self._finish_reason,
        )

    async def stream_complete(self, model_id: str, task) -> AsyncIterator[str]:  # type: ignore[override]
        yield self._content

    async def health_check(self, model_id: str) -> bool:
        return True

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        return self._input_tokens


class FailingAdapter(AbstractModelAdapter):
    """Raises on complete() — use to test fallback and error-handling paths."""

    vendor = "mock"

    def __init__(self, error_msg: str = "Simulated vendor outage") -> None:
        self._error_msg = error_msg

    async def complete(self, model_id: str, task) -> AdapterResponse:
        raise RuntimeError(self._error_msg)

    async def stream_complete(self, model_id: str, task) -> AsyncIterator[str]:  # type: ignore[override]
        raise RuntimeError(self._error_msg)
        yield  # make it a generator  # noqa

    async def health_check(self, model_id: str) -> bool:
        return False

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        return 1
