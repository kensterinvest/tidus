"""Atomic per-team/workflow spend counters.

In-memory implementation using asyncio locks for thread safety within a single
process. The interface is designed for a Redis swap-in: replace the class body
with INCRBYFLOAT + TTL keys and the rest of the system is unchanged.

Example:
    counter = SpendCounter()
    await counter.add("team-engineering", None, 0.0025)
    total = await counter.get("team-engineering", None)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


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
