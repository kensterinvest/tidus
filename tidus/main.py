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
from prometheus_fastapi_instrumentator import Instrumentator

from tidus.api.deps import build_singletons, get_metering, get_registry, get_session_factory
from tidus.api.v1 import audit, budgets, complete, dashboard, guardrails, metering, models, route, sync, usage
from tidus.metering.middleware import MeteringMiddleware
from tidus.db.engine import create_tables
from tidus.settings import get_settings
from tidus.sync.scheduler import TidusScheduler
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

    # Build shared singletons: registry, selector, enforcer, guardrails, cost logger
    build_singletons()
    log.info("singletons_ready")

    # Start background scheduler (health probes every 5 min + weekly price sync)
    scheduler = TidusScheduler(
        registry=get_registry(),
        session_factory=get_session_factory(),
    )
    scheduler.start()

    yield

    scheduler.shutdown()
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
        version="1.0.0",
        contact={"name": "Kenny Wong", "email": "lapkei01@gmail.com"},
        license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    _cors_origins = (
        ["*"] if settings.environment == "development"
        else [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=bool(_cors_origins and _cors_origins != ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metering middleware — records AI user events on /route and /complete.
    # The MeteringService singleton is resolved lazily via get_metering() on the
    # first request; this avoids a circular dependency with the lifespan startup.
    app.add_middleware(MeteringMiddleware, metering_getter=get_metering)

    # ── Root ──────────────────────────────────────────────────────────────────
    from fastapi.requests import Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    @app.get("/", include_in_schema=False)
    async def root(request: Request):
        """Browser → dashboard redirect; API clients get a JSON index."""
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return RedirectResponse("/dashboard/")
        return {
            "service": "tidus",
            "version": "1.0.0",
            "description": "Enterprise AI Router — routes every request to the cheapest capable model",
            "links": {
                "dashboard": "/dashboard/",
                "docs":      "/docs",
                "redoc":     "/redoc",
                "health":    "/health",
                "route":     "/api/v1/route",
                "complete":  "/api/v1/complete",
                "models":    "/api/v1/models",
                "budgets":   "/api/v1/budgets",
                "usage":     "/api/v1/usage/summary",
                "metering":  "/api/v1/metering/status",
            },
        }

    # ── Health endpoints ──────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"], summary="Liveness check")
    async def health():
        return {"status": "ok", "service": "tidus"}

    @app.get("/ready", tags=["Health"], summary="Readiness check")
    async def ready():
        return {"status": "ready"}

    # ── API v1 routers ────────────────────────────────────────────────────────
    app.include_router(route.router, prefix="/api/v1")
    app.include_router(complete.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(budgets.router, prefix="/api/v1")
    app.include_router(usage.router, prefix="/api/v1")
    app.include_router(guardrails.router, prefix="/api/v1")
    app.include_router(sync.router, prefix="/api/v1")
    app.include_router(dashboard.router, prefix="/api/v1")
    app.include_router(audit.router, prefix="/api/v1")
    app.include_router(metering.router, prefix="/api/v1")

    # ── Prometheus metrics ────────────────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/ready", "/metrics"],
    ).instrument(app).expose(app, include_in_schema=False, tags=["Observability"])

    # ── Dashboard static files ────────────────────────────────────────────────
    import pathlib
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = pathlib.Path(__file__).parent / "dashboard" / "static"
    app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="dashboard")

    @app.get("/dash", include_in_schema=False)
    async def dash_redirect():
        return RedirectResponse("/dashboard/index.html")

    return app


app = create_app()


def run():
    """CLI entry point: tidus"""
    import uvicorn
    uvicorn.run("tidus.main:app", host="0.0.0.0", port=8000, reload=True)
