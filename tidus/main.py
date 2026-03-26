"""Tidus — Enterprise AI Router Service

Entry point for the FastAPI application. Handles startup (DB creation,
config loading, scheduler start) and shutdown (scheduler stop) via lifespan.

Run with:
    uvicorn tidus.main:app --reload
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tidus.db.engine import create_tables
from tidus.settings import get_settings
from tidus.utils.logging import configure_logging

log = structlog.get_logger("tidus.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)

    log.info("tidus_starting", environment=settings.environment, tier=settings.tidus_tier)

    # Create DB tables (idempotent)
    await create_tables()
    log.info("database_ready")

    # Future: start APScheduler (health probes + price sync)
    # scheduler.start()

    yield

    # Future: scheduler.shutdown()
    log.info("tidus_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Tidus — Enterprise AI Router",
        description=(
            "Vendor-agnostic, cost-aware AI model router and governance system. "
            "Routes every AI request to the cheapest capable model, enforces budgets, "
            "and prevents runaway multi-agent loops."
        ),
        version="0.1.0",
        contact={"name": "Kenny Wong", "email": "lapkei01@gmail.com"},
        license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health endpoints ──────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"], summary="Liveness check")
    async def health():
        return {"status": "ok", "service": "tidus"}

    @app.get("/ready", tags=["Health"], summary="Readiness check")
    async def ready():
        # TODO Phase 3+: check DB + adapter connectivity
        return {"status": "ready"}

    # ── API routers (registered per phase) ───────────────────────────────────
    # Phase 3: from tidus.api.v1 import route, complete, models, budgets, usage
    # app.include_router(route.router, prefix="/api/v1")
    # ...

    # ── Dashboard static files ────────────────────────────────────────────────
    # Phase 5: app.mount("/dashboard", StaticFiles(...), name="dashboard")

    return app


app = create_app()


def run():
    """CLI entry point: tidus"""
    import uvicorn
    uvicorn.run("tidus.main:app", host="0.0.0.0", port=8000, reload=True)
