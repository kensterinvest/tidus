"""Abstract model adapter interface.

Every vendor adapter inherits AbstractModelAdapter and registers itself via
@register_adapter so the AdapterFactory can dispatch by vendor name.

Fix 18 additions:
  - Structured exception hierarchy (``AdapterError`` and subclasses) so the
    router can distinguish auth/rate-limit/timeout/server/client failures
    instead of collapsing every upstream problem to bare ``Exception``.
  - ``with_retry`` helper wraps a coroutine with per-call timeout and
    exponential backoff on transient errors only — auth/client errors are
    raised immediately so callers do not pay retry latency for problems that
    will never succeed.

Example (defining a new adapter):
    @register_adapter
    class MyAdapter(AbstractModelAdapter):
        vendor = "myvendor"
        supported_model_ids = ["my-model-v1"]

        async def complete(self, model_id, task) -> AdapterResponse: ...
        async def stream_complete(self, model_id, task): ...
        async def health_check(self, model_id) -> bool: ...
        async def count_tokens(self, model_id, messages) -> int: ...
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)


# ── Exception hierarchy ───────────────────────────────────────────────────────

class AdapterError(Exception):
    """Base class for all adapter-originated errors.

    The router catches ``AdapterError`` at the /complete boundary and can
    distinguish transient from permanent failure via the concrete subclass.
    Vendor-specific SDK exceptions must be mapped into one of the subclasses
    in each adapter's ``complete`` implementation.
    """


class AdapterAuthError(AdapterError):
    """401/403 from vendor — bad API key or missing entitlement.

    Never retried — no amount of waiting will make a bad key work.
    """


class AdapterRateLimitError(AdapterError):
    """429 / Too Many Requests. Retried with exponential backoff."""


class AdapterTimeoutError(AdapterError):
    """Request exceeded ``adapter_timeout_seconds``. Retried."""


class AdapterServerError(AdapterError):
    """5xx from vendor. Retried — assume transient infrastructure hiccup."""


class AdapterClientError(AdapterError):
    """4xx (other than 401/403/429) — malformed request, bad model id, etc.

    Never retried — the caller's request is at fault.
    """


_TRANSIENT = (AdapterRateLimitError, AdapterServerError, AdapterTimeoutError)


# ── Retry helper ──────────────────────────────────────────────────────────────

async def with_retry[T](
    coro_fn: Callable[[], Awaitable[T]],
    *,
    timeout_seconds: float,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
) -> T:
    """Call ``coro_fn`` with a per-attempt timeout and exponential backoff.

    Args:
        coro_fn: Zero-argument async callable. Produces a fresh coroutine per
            attempt — must not be a single awaited coroutine.
        timeout_seconds: Per-attempt hard wall-clock limit. On timeout the
            attempt is cancelled and counted as a transient failure.
        max_attempts: Total attempts including the first. 3 = one retry.
        base_delay_seconds: Delay before the first retry. Doubles each attempt.

    Raises:
        AdapterTimeoutError / AdapterRateLimitError / AdapterServerError on
        transient failure after ``max_attempts``; AdapterAuthError /
        AdapterClientError immediately on the first occurrence.
    """
    last_exc: AdapterError | None = None
    for attempt in range(max_attempts):
        try:
            return await asyncio.wait_for(coro_fn(), timeout=timeout_seconds)
        except TimeoutError as exc:
            last_exc = AdapterTimeoutError(
                f"Adapter call exceeded {timeout_seconds}s timeout"
            )
            last_exc.__cause__ = exc
        except _TRANSIENT as exc:
            last_exc = exc
        except AdapterError:
            raise  # auth/client — do not retry

        if attempt == max_attempts - 1:
            break
        await asyncio.sleep(base_delay_seconds * (2 ** attempt))
        log.warning(
            "adapter_retry",
            attempt=attempt + 1,
            max_attempts=max_attempts,
            error=type(last_exc).__name__,
        )

    assert last_exc is not None  # loop body always assigns on failure
    raise last_exc


# ── Generic vendor-exception translator ───────────────────────────────────────

def translate_vendor_exception(exc: BaseException) -> AdapterError:
    """Best-effort mapping of a vendor-SDK exception to the AdapterError hierarchy.

    Inspects common status-code attributes first (``status_code``,
    ``response.status_code``) and falls back to class-name heuristics for SDKs
    that raise custom classes without a status code. Unknown exceptions map
    to the base ``AdapterError`` which the router treats as non-retryable.

    Adapters call this from inside their ``complete`` wrapper so that every
    vendor call emits an ``AdapterError`` subclass to the router.
    """
    if isinstance(exc, TimeoutError):
        return AdapterTimeoutError(str(exc))

    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)

    if status is not None:
        if status in (401, 403):
            return AdapterAuthError(str(exc))
        if status == 408:
            return AdapterTimeoutError(str(exc))
        if status == 429:
            return AdapterRateLimitError(str(exc))
        if 500 <= status < 600:
            return AdapterServerError(str(exc))
        if 400 <= status < 500:
            return AdapterClientError(str(exc))

    name = type(exc).__name__.lower()
    if "authentication" in name or "permission" in name or "unauthorized" in name:
        return AdapterAuthError(str(exc))
    if "ratelimit" in name or "rate_limit" in name or "toomany" in name:
        return AdapterRateLimitError(str(exc))
    if "timeout" in name:
        return AdapterTimeoutError(str(exc))
    if "apiconnection" in name or "serverunavailable" in name or "service_unavailable" in name:
        return AdapterServerError(str(exc))

    return AdapterError(f"{type(exc).__name__}: {exc}")


@dataclass
class AdapterResponse:
    """Normalised response returned by every adapter."""

    model_id: str
    content: str                       # generated text
    input_tokens: int                  # actual tokens consumed (from vendor)
    output_tokens: int
    latency_ms: float
    finish_reason: str = "stop"        # "stop" | "length" | "content_filter"
    raw: dict = field(default_factory=dict)  # raw vendor response for debugging


class AbstractModelAdapter(ABC):
    """Base class all vendor adapters must implement."""

    vendor: str                         # class-level constant, e.g. "anthropic"
    supported_model_ids: list[str] = [] # informational; not used for routing

    @abstractmethod
    async def complete(self, model_id: str, task) -> AdapterResponse:
        """Execute a non-streaming completion."""

    @abstractmethod
    async def stream_complete(
        self, model_id: str, task
    ) -> AsyncIterator[str]:
        """Execute a streaming completion, yielding text chunks."""

    @abstractmethod
    async def health_check(self, model_id: str) -> bool:
        """Return True if the model is reachable and responding."""

    @abstractmethod
    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        """Return the token count for the given messages."""


# ── Registry ──────────────────────────────────────────────────────────────────

_ADAPTER_REGISTRY: dict[str, AbstractModelAdapter] = {}


def register_adapter(cls: type[AbstractModelAdapter]) -> type[AbstractModelAdapter]:
    """Class decorator — instantiates the adapter and registers it by vendor."""
    instance = cls()
    _ADAPTER_REGISTRY[cls.vendor] = instance
    return cls


def get_adapter(vendor: str) -> AbstractModelAdapter:
    """Return the adapter for a vendor name.

    Raises:
        KeyError: if no adapter is registered for the vendor.
    """
    if vendor not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"No adapter registered for vendor {vendor!r}. "
            f"Available: {sorted(_ADAPTER_REGISTRY)}"
        )
    return _ADAPTER_REGISTRY[vendor]


def list_adapters() -> list[str]:
    """Return all registered vendor names."""
    return sorted(_ADAPTER_REGISTRY.keys())
