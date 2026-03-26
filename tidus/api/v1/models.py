"""GET /api/v1/models — List and manage the model registry.

Endpoints:
    GET  /api/v1/models           — List all models (optionally filter by enabled/tier)
    PATCH /api/v1/models/{id}     — Enable/disable a model or update its latency
    GET  /api/v1/models/{id}      — Get a single model spec

Example curl:
    curl http://localhost:8000/api/v1/models
    curl -X PATCH http://localhost:8000/api/v1/models/gpt-4o-mini \\
      -H "Content-Type: application/json" \\
      -d '{"enabled": false}'
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from tidus.api.deps import get_registry
from tidus.models.model_registry import ModelSpec
from tidus.router.registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["Models"])


# ── Request / Response models ─────────────────────────────────────────────────

class ModelPatchRequest(BaseModel):
    enabled: bool | None = None
    latency_p50_ms: int | None = None


# ── Route handlers ────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=list[ModelSpec],
    summary="List all models in the registry",
)
async def list_models(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    enabled_only: bool = False,
    tier: int | None = None,
) -> list[ModelSpec]:
    """Return all model specs. Filter by enabled status or tier if specified."""
    models = registry.list_all() if not enabled_only else registry.list_enabled()
    if tier is not None:
        models = [m for m in models if m.tier.value == tier]
    return models


@router.get(
    "/{model_id}",
    response_model=ModelSpec,
    summary="Get a single model by ID",
)
async def get_model(
    model_id: str,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> ModelSpec:
    spec = registry.get(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return spec


@router.patch(
    "/{model_id}",
    response_model=ModelSpec,
    summary="Update a model's enabled status or latency baseline",
)
async def patch_model(
    model_id: str,
    body: ModelPatchRequest,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> ModelSpec:
    """Enable/disable a model or update its measured latency.

    Changes are applied to the in-memory registry immediately and affect
    all subsequent routing decisions. They are NOT persisted to models.yaml
    (use the price_sync job for durable changes).
    """
    spec = registry.get(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if body.enabled is not None:
        registry.set_enabled(model_id, body.enabled)
    if body.latency_p50_ms is not None:
        registry.update_latency(model_id, body.latency_p50_ms)

    return registry.get(model_id)  # type: ignore[return-value]
