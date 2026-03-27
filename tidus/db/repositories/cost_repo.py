"""Cost record repository — thin async SQLAlchemy wrapper."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tidus.db.engine import CostRecordORM
from tidus.models.cost import CostRecord


class CostRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(self, record: CostRecord) -> None:
        orm = CostRecordORM(
            id=record.id,
            task_id=record.task_id,
            team_id=record.team_id,
            workflow_id=record.workflow_id,
            agent_session_id=record.agent_session_id,
            agent_depth=record.agent_depth,
            routing_decision_id=record.routing_decision_id,
            model_id=record.model_id,
            vendor=record.vendor,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cost_usd=record.cost_usd,
            latency_ms=record.latency_ms,
            timestamp=record.timestamp,
            fallback_used=record.fallback_used,
            fallback_from=record.fallback_from,
        )
        self._session.add(orm)
        await self._session.commit()

    async def list_by_team(self, team_id: str, limit: int = 100) -> list[CostRecord]:
        result = await self._session.execute(
            select(CostRecordORM)
            .where(CostRecordORM.team_id == team_id)
            .order_by(CostRecordORM.timestamp.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [_to_model(r) for r in rows]


def _to_model(orm: CostRecordORM) -> CostRecord:
    return CostRecord(
        id=orm.id,
        task_id=orm.task_id,
        team_id=orm.team_id,
        workflow_id=orm.workflow_id,
        agent_session_id=orm.agent_session_id,
        agent_depth=orm.agent_depth or 0,
        routing_decision_id=orm.routing_decision_id,
        model_id=orm.model_id,
        vendor=orm.vendor,
        input_tokens=orm.input_tokens,
        output_tokens=orm.output_tokens,
        cost_usd=orm.cost_usd,
        latency_ms=orm.latency_ms,
        timestamp=orm.timestamp,
        fallback_used=orm.fallback_used or False,
        fallback_from=orm.fallback_from,
    )
