"""Usage metering service — tracks unique AI users per rolling 30-day window.

Implements the metering spec from TIT-32:
  - Caller identity resolution: X-Titus-User-Id header > API key owner > hash(IP+UA)
  - Rolling 30-day unique caller count per org
  - Alert stage thresholds: yellow (800+), orange (950+), threshold (1000+)
  - Grace period enforcement: enterprise features deactivated at 60 days post-threshold;
    core routing is NEVER stopped.

Usage:
    service = MeteringService(session_factory)
    await service.record_event(caller_id="alice", caller_source="header", team_id="eng")
    status = await service.get_status()
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from tidus.db.engine import AiUserEventORM

log = structlog.get_logger(__name__)

# ── Thresholds (from TIT-32 spec) ────────────────────────────────────────────

_YELLOW_THRESHOLD = 800
_ORANGE_THRESHOLD = 950
_RED_THRESHOLD = 1_000
_ROLLING_WINDOW_DAYS = 30


class MeteringStage(str, Enum):
    normal = "normal"         # < 800 unique AI users
    yellow = "yellow"         # 800-949 — dashboard banner + enterprise CTA
    orange = "orange"         # 950-999 — escalated banner + email to org admin
    threshold = "threshold"   # 1000+ — threshold crossed, grace period begins
    enforcing = "enforcing"   # 60+ days post-threshold, enterprise features deactivated


class MeteringStatus:
    """Snapshot of the current metering state."""

    def __init__(
        self,
        active_user_count: int,
        stage: MeteringStage,
        window_start: datetime,
        window_end: datetime,
        trend_7d: list[int],
    ) -> None:
        self.active_user_count = active_user_count
        self.stage = stage
        self.window_start = window_start
        self.window_end = window_end
        self.trend_7d = trend_7d  # daily unique user counts for last 7 days (oldest → newest)

    def to_dict(self) -> dict:
        return {
            "active_user_count": self.active_user_count,
            "threshold": _RED_THRESHOLD,
            "stage": self.stage.value,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "trend_7d": self.trend_7d,
        }


# ── Caller identity resolution ────────────────────────────────────────────────

def resolve_caller_id(
    user_id_header: str | None,
    api_key_sub: str | None,
    client_ip: str | None,
    user_agent: str | None,
) -> tuple[str, str]:
    """Resolve caller identity per TIT-32 spec.

    Resolution rules (security-hardened in Fix 12):
      1. If a JWT sub is present, it is ALWAYS the identity. A mismatching
         X-Titus-User-Id header is ignored (and logged) — the header cannot
         be used to impersonate other users.
      2. If no JWT sub is present, X-Titus-User-Id is honored (unauthenticated
         deployments rely on the header for identity).
      3. Otherwise fall back to a SHA-256(IP + UA) anonymous fingerprint.

    Returns:
        (caller_id, caller_source) where source is "header", "api_key", or "ip_hash"
    """
    has_auth = bool(api_key_sub) and api_key_sub not in ("dev", "")

    if has_auth:
        header_norm = user_id_header.strip() if user_id_header else None
        if header_norm and header_norm != api_key_sub:
            log.warning(
                "metering_header_impersonation_attempt",
                claimed=header_norm,
                actual=api_key_sub,
            )
        return api_key_sub, "api_key"

    if user_id_header:
        return user_id_header.strip(), "header"

    # Anonymous fingerprint — best-effort; not PII since it's a hash
    raw = f"{client_ip or 'unknown'}:{user_agent or 'unknown'}"
    fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return fingerprint, "ip_hash"


# ── Metering service ──────────────────────────────────────────────────────────

class MeteringService:
    """Records AI user events and computes rolling-window unique caller counts.

    Designed to be a singleton (stored in deps.py), initialized once at startup.
    All DB operations are non-fatal — metering failures never block routing.
    """

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._sf = session_factory

    async def record_event(
        self,
        caller_id: str,
        caller_source: str,
        team_id: str | None = None,
        path: str | None = None,
    ) -> None:
        """Persist one AI user event. Non-fatal — logs and swallows DB errors."""
        try:
            async with self._sf() as session:
                event = AiUserEventORM(
                    id=str(uuid.uuid4()),
                    caller_id=caller_id,
                    caller_source=caller_source,
                    team_id=team_id,
                    path=path,
                )
                session.add(event)
                await session.commit()
        except Exception:
            log.exception("metering_record_failed", caller_id=caller_id)

    async def get_active_user_count(self) -> int:
        """Count unique caller_ids in the rolling 30-day window."""
        cutoff = datetime.now(UTC) - timedelta(days=_ROLLING_WINDOW_DAYS)
        try:
            async with self._sf() as session:
                result = await session.execute(
                    select(func.count(func.distinct(AiUserEventORM.caller_id))).where(
                        AiUserEventORM.timestamp >= cutoff
                    )
                )
                return result.scalar_one() or 0
        except Exception:
            log.exception("metering_count_failed")
            return 0

    async def get_trend_7d(self) -> list[int]:
        """Return daily unique user counts for each of the last 7 days (oldest first)."""
        now = datetime.now(UTC)
        counts: list[int] = []
        try:
            async with self._sf() as session:
                for day_offset in range(6, -1, -1):  # 6 days ago → today
                    day_start = (now - timedelta(days=day_offset)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    day_end = day_start + timedelta(days=1)
                    result = await session.execute(
                        select(func.count(func.distinct(AiUserEventORM.caller_id))).where(
                            AiUserEventORM.timestamp >= day_start,
                            AiUserEventORM.timestamp < day_end,
                        )
                    )
                    counts.append(result.scalar_one() or 0)
        except Exception:
            log.exception("metering_trend_failed")
            counts = [0] * 7
        return counts

    async def get_status(self) -> MeteringStatus:
        """Return a full metering snapshot including count, stage, and 7-day trend."""
        now = datetime.now(UTC)
        count, trend = await self.get_active_user_count(), await self.get_trend_7d()

        if count >= _RED_THRESHOLD:
            stage = MeteringStage.threshold
        elif count >= _ORANGE_THRESHOLD:
            stage = MeteringStage.orange
        elif count >= _YELLOW_THRESHOLD:
            stage = MeteringStage.yellow
        else:
            stage = MeteringStage.normal

        return MeteringStatus(
            active_user_count=count,
            stage=stage,
            window_start=now - timedelta(days=_ROLLING_WINDOW_DAYS),
            window_end=now,
            trend_7d=trend,
        )
