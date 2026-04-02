"""GET /api/v1/metering/status — AI user metering status.

Returns the current active AI user count, alert stage, rolling window dates,
and 7-day trend for display in the dashboard and for ops monitoring.

Stages (from TIT-32 spec):
  normal    < 800 unique AI users
  yellow    800–949   — in-dashboard banner + enterprise contact CTA
  orange    950–999   — escalated banner + email to org admin
  threshold 1000+     — grace period begins; enterprise features warned
  enforcing 60+ days post-threshold — enterprise-gated features deactivated
             (core routing is NEVER stopped)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from tidus.api.deps import get_metering
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.metering.service import MeteringService

router = APIRouter(prefix="/metering", tags=["Metering"])


class MeteringStatusResponse(BaseModel):
    active_user_count: int
    threshold: int
    stage: str
    window_start: str
    window_end: str
    trend_7d: list[int]


@router.get(
    "/status",
    response_model=MeteringStatusResponse,
    summary="AI user metering status",
    response_description=(
        "Current rolling-30-day unique AI user count, alert stage, and 7-day trend"
    ),
)
async def metering_status(
    metering: Annotated[MeteringService, Depends(get_metering)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.team_manager, Role.admin,
    ))],
) -> MeteringStatusResponse:
    """Return the current AI user count and alert stage.

    Only `team_manager` and `admin` roles can view metering data.
    """
    status = await metering.get_status()
    return MeteringStatusResponse(**status.to_dict())
