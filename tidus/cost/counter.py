"""Atomic per-team/workflow spend counters.

Two interchangeable backends (same structural interface):

- :class:`SpendCounter` — in-memory, asyncio-lock protected. Used in tests and
  single-worker deployments.
- :class:`RedisSpendCounter` — Redis-backed with a Lua script for atomic
  check-and-add. Used for multi-worker / multi-pod deployments so budget
  state is globally consistent.

Pick the backend in ``tidus.api.deps.build_singletons`` based on
``settings.redis_url``.

Example:
    counter = SpendCounter()
    await counter.add("team-engineering", None, 0.0025)
    total = await counter.get("team-engineering", None)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis


class SpendCounter:
    """Thread-safe, in-process atomic spend counters keyed by (team_id, workflow_id).

    All amounts are in USD. The counter accumulates indefinitely — the
    BudgetEnforcer is responsible for comparing against limits and resetting
    at period boundaries.
    """

    def __init__(self) -> None:
        # Key: (team_id, workflow_id or None)
        self._totals: dict[tuple[str, str | None], float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def add(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
    ) -> float:
        """Atomically add amount_usd to the counter for the given scope.

        Returns the new total after the addition.
        """
        key = (team_id, workflow_id)
        async with self._lock:
            self._totals[key] += amount_usd
            return self._totals[key]

    async def get(
        self,
        team_id: str,
        workflow_id: str | None,
    ) -> float:
        """Return the current total spend for the given scope."""
        key = (team_id, workflow_id)
        async with self._lock:
            return self._totals[key]

    async def reset(
        self,
        team_id: str,
        workflow_id: str | None,
    ) -> None:
        """Reset the counter for the given scope to zero (called at period boundary)."""
        key = (team_id, workflow_id)
        async with self._lock:
            self._totals[key] = 0.0

    async def reset_workflow(self, workflow_id: str) -> int:
        """Reset all (team_id, workflow_id) counters matching workflow_id.

        Used when a workflow-scoped BudgetPolicy hits its period boundary —
        every team's counter for that workflow is zeroed in a single atomic
        pass. Returns the number of counters reset.
        """
        reset_count = 0
        async with self._lock:
            for key in list(self._totals.keys()):
                if key[1] == workflow_id:
                    self._totals[key] = 0.0
                    reset_count += 1
        return reset_count

    async def check_and_add(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
        limit_usd: float,
    ) -> tuple[bool, float]:
        """Atomically check the limit and add amount if within budget.

        Holds the lock across both the comparison and the increment so that
        concurrent requests cannot both pass the check and then both deduct,
        overrunning the limit.

        Returns:
            (allowed, new_total) — allowed is False if adding amount_usd would
            exceed limit_usd; new_total is the value after addition (if allowed).
        """
        key = (team_id, workflow_id)
        async with self._lock:
            current = self._totals[key]
            if current + amount_usd > limit_usd:
                return False, current
            self._totals[key] += amount_usd
            return True, self._totals[key]

    async def get_all(self) -> dict[tuple[str, str | None], float]:
        """Return a snapshot of all counters. Used by the budget status endpoint."""
        async with self._lock:
            return dict(self._totals)


# ── Redis-backed implementation ───────────────────────────────────────────────

# Atomically check current spend against a limit and increment if the total
# would stay within the limit. Returns [allowed, new_total] — new_total is the
# value AFTER addition on success, or the unchanged current value on reject.
_CHECK_AND_ADD_LUA = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
if current + amount > limit then
    return {0, tostring(current)}
end
local new_total = redis.call('INCRBYFLOAT', KEYS[1], amount)
return {1, tostring(new_total)}
"""


class RedisSpendCounter:
    """Redis-backed SpendCounter for multi-worker / multi-pod deployments.

    Uses ``INCRBYFLOAT`` for adds (atomic in Redis single-threaded execution)
    and a Lua script for ``check_and_add`` (the critical atomic gate for the
    reserve pattern — Redis runs the whole script as one atomic unit).

    Key layout (prefix defaults to ``tidus:spend``):

        {prefix}:team:{team_id}:total           → team-level counter
        {prefix}:team:{team_id}:wf:{workflow_id} → workflow-level counter

    ``reset_workflow`` and ``get_all`` use ``SCAN`` (non-blocking, production-safe).
    """

    def __init__(self, redis: Redis, prefix: str = "tidus:spend") -> None:
        self._redis = redis
        self._prefix = prefix
        # Register the Lua script — Redis caches compiled scripts by SHA.
        self._check_and_add = self._redis.register_script(_CHECK_AND_ADD_LUA)

    def _key(self, team_id: str, workflow_id: str | None) -> str:
        if workflow_id is None:
            return f"{self._prefix}:team:{team_id}:total"
        return f"{self._prefix}:team:{team_id}:wf:{workflow_id}"

    async def add(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
    ) -> float:
        """Atomically add amount to the counter; returns the new total."""
        key = self._key(team_id, workflow_id)
        new_total = await self._redis.incrbyfloat(key, amount_usd)
        return float(new_total)

    async def get(
        self,
        team_id: str,
        workflow_id: str | None,
    ) -> float:
        """Return the current total spend for the scope (0.0 if key missing)."""
        key = self._key(team_id, workflow_id)
        value = await self._redis.get(key)
        return float(value) if value is not None else 0.0

    async def reset(
        self,
        team_id: str,
        workflow_id: str | None,
    ) -> None:
        """Delete the counter key (SET 0 would leak keys over time)."""
        key = self._key(team_id, workflow_id)
        await self._redis.delete(key)

    async def reset_workflow(self, workflow_id: str) -> int:
        """Delete every (team, workflow) counter keyed on workflow_id.

        Uses SCAN so it's safe against large key-spaces in production.
        Returns the number of keys deleted.
        """
        pattern = f"{self._prefix}:team:*:wf:{workflow_id}"
        deleted = 0
        async for key in self._redis.scan_iter(match=pattern, count=500):
            await self._redis.delete(key)
            deleted += 1
        return deleted

    async def check_and_add(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
        limit_usd: float,
    ) -> tuple[bool, float]:
        """Atomically check the limit and add amount if within budget.

        Implemented via Lua script so the read + compare + write are a single
        atomic operation on the Redis server — concurrent callers cannot race
        between the check and the increment.
        """
        key = self._key(team_id, workflow_id)
        allowed_raw, total_raw = await self._check_and_add(
            keys=[key], args=[str(amount_usd), str(limit_usd)]
        )
        return (bool(int(allowed_raw)), float(total_raw))

    async def get_all(self) -> dict[tuple[str, str | None], float]:
        """Snapshot of every counter under the prefix."""
        result: dict[tuple[str, str | None], float] = {}
        pattern = f"{self._prefix}:team:*"
        async for key in self._redis.scan_iter(match=pattern, count=500):
            key_str = key.decode() if isinstance(key, (bytes, bytearray)) else key
            # Parse key: {prefix}:team:{team_id}:total  or  :wf:{wf_id}
            parts = key_str.removeprefix(f"{self._prefix}:team:").split(":")
            if len(parts) == 2 and parts[1] == "total":
                scope_key: tuple[str, str | None] = (parts[0], None)
            elif len(parts) >= 3 and parts[1] == "wf":
                # workflow_id may contain ':' if caller allowed it — rejoin
                scope_key = (parts[0], ":".join(parts[2:]))
            else:
                continue  # skip malformed keys
            value = await self._redis.get(key)
            result[scope_key] = float(value) if value is not None else 0.0
        return result
