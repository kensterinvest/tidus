"""Agent guardrail checker — depth/retry/token-per-step policy enforcement.

Works with SessionStore to validate each agent step against the active
GuardrailPolicy before routing proceeds.

Example:
    guard = AgentGuard(policy, store)
    result = await guard.check_and_advance("session-123", input_tokens=500)
    if not result.allowed:
        raise HTTPException(429, result.reason)
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from tidus.guardrails.session_store import SessionStore
from tidus.models.guardrails import AgentSession, GuardrailPolicy

log = structlog.get_logger(__name__)


@dataclass
class GuardResult:
    """Result of a guardrail check."""

    allowed: bool
    reason: str | None
    session: AgentSession | None


class AgentGuard:
    """Validates agent steps against guardrail policy limits."""

    def __init__(self, policy: GuardrailPolicy, store: SessionStore) -> None:
        self._policy = policy
        self._store = store

    async def check_and_advance(
        self,
        session_id: str,
        input_tokens: int,
    ) -> GuardResult:
        """Check limits and increment depth if allowed.

        Checks (in order):
          1. Session exists
          2. max_agent_depth not exceeded
          3. max_tokens_per_step not exceeded
          4. max_retries_per_task not exceeded (increments on each call)

        Returns GuardResult(allowed=True) and increments depth on success.
        Returns GuardResult(allowed=False, reason=...) without incrementing on failure.

        Example:
            result = await guard.check_and_advance("sess-abc", input_tokens=1200)
        """
        session = await self._store.get(session_id)
        if session is None:
            return GuardResult(allowed=False, reason=f"Session {session_id!r} not found", session=None)

        # Depth check (current_depth is depth BEFORE this step)
        next_depth = session.current_depth + 1
        if next_depth > self._policy.max_agent_depth:
            log.warning(
                "agent_depth_exceeded",
                session_id=session_id,
                depth=next_depth,
                max_depth=self._policy.max_agent_depth,
            )
            return GuardResult(
                allowed=False,
                reason=f"Agent depth {next_depth} exceeds maximum {self._policy.max_agent_depth}",
                session=session,
            )

        # Token-per-step check
        if input_tokens > self._policy.max_tokens_per_step:
            log.warning(
                "tokens_per_step_exceeded",
                session_id=session_id,
                input_tokens=input_tokens,
                max_tokens=self._policy.max_tokens_per_step,
            )
            return GuardResult(
                allowed=False,
                reason=f"Input tokens {input_tokens} exceeds per-step limit {self._policy.max_tokens_per_step}",
                session=session,
            )

        # Retry check
        if session.retry_count >= self._policy.max_retries_per_task:
            log.warning(
                "max_retries_exceeded",
                session_id=session_id,
                retry_count=session.retry_count,
                max_retries=self._policy.max_retries_per_task,
            )
            return GuardResult(
                allowed=False,
                reason=f"Retry count {session.retry_count} has reached maximum {self._policy.max_retries_per_task}",
                session=session,
            )

        # All checks passed — advance depth
        updated = await self._store.increment_depth(session_id)
        await self._store.add_tokens(session_id, input_tokens)

        log.info(
            "agent_step_allowed",
            session_id=session_id,
            depth=next_depth,
            input_tokens=input_tokens,
        )
        return GuardResult(allowed=True, reason=None, session=updated)
