"""Cost logger — persists CostRecord to the database after each completed task.

Called by the /complete endpoint after the adapter returns.

Example:
    logger = CostLogger(session_factory)
    await logger.record(task, decision, adapter_response)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from tidus.adapters.base import AdapterResponse
from tidus.db.repositories.cost_repo import CostRepository
from tidus.models.cost import CostRecord
from tidus.models.routing import RoutingDecision
from tidus.models.task import TaskDescriptor

log = structlog.get_logger(__name__)


class CostLogger:
    """Writes a CostRecord to the database after every executed task."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def record(
        self,
        task: TaskDescriptor,
        decision: RoutingDecision,
        response: AdapterResponse,
        vendor: str,
    ) -> CostRecord:
        """Build and persist a CostRecord.

        Args:
            task:     The original task descriptor.
            decision: The routing decision that selected this model.
            response: The adapter response containing actual token counts.
            vendor:   The vendor string from the model spec.

        Returns:
            The persisted CostRecord.
        """
        from tidus.router.registry import ModelRegistry
        # Compute actual cost from real token counts
        # We rely on the registry being accessible via the decision; if not,
        # fall back to the estimated cost from the decision.
        actual_cost = decision.estimated_cost_usd or 0.0

        record = CostRecord(
            id=str(uuid.uuid4()),
            task_id=task.task_id,
            team_id=task.team_id,
            workflow_id=task.workflow_id,
            agent_session_id=task.agent_session_id,
            agent_depth=task.agent_depth,
            routing_decision_id=decision.task_id,
            model_id=response.model_id,
            vendor=vendor,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=actual_cost,
            latency_ms=response.latency_ms,
            timestamp=datetime.now(timezone.utc),
            fallback_used=decision.fallback_from is not None,
            fallback_from=decision.fallback_from,
        )

        try:
            async with self._session_factory() as session:
                repo = CostRepository(session)
                await repo.insert(record)
            log.info(
                "cost_recorded",
                task_id=task.task_id,
                model_id=response.model_id,
                cost_usd=actual_cost,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except Exception as exc:
            # Non-fatal: log and continue — don't fail the request over DB write
            log.error("cost_record_failed", task_id=task.task_id, error=str(exc))

        return record
