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

from tidus.api.deps import (
    build_singletons,
    get_enforcer,
    get_metering,
    get_registry,
    get_session_factory,
)
from tidus.registry.seeder import RegistrySeeder
from tidus.api.v1 import (
    audit,
    billing,
    budgets,
    complete,
    dashboard,
    guardrails,
    metering,
    models,
    registry,
    reports,
    route,
    sync,
    usage,
)
from tidus.db.engine import create_tables
from tidus.metering.middleware import MeteringMiddleware
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

    # Build shared singletons: EffectiveRegistry (DB-backed), selector, enforcer, etc.
    await build_singletons()
    log.info("singletons_ready")

    # Seed the model catalog DB from models.yaml if no active revision exists.
    # Idempotent: no-op when a revision is already present (e.g. after the first run).
    await RegistrySeeder().seed_from_yaml(
        get_session_factory(),
        settings.models_config_path,
    )
    log.info("registry_seeder_complete")

    # Populate Prometheus Gauges before the first scrape so dashboards don't
    # start with empty series.  Counters (probe calls, drift events) are
    # incremented at the point of each operation — no startup seeding needed.
    try:
        from tidus.observability.metrics_updater import MetricsUpdater
        await MetricsUpdater().update(get_registry(), get_session_factory())
        log.info("metrics_initialized")
    except Exception as exc:
        log.warning("metrics_init_failed", error=str(exc))

    # Start background scheduler (health probes, price sync, budget reset,
    # registry refresh, override expiry)
    scheduler = TidusScheduler(
        registry=get_registry(),
        session_factory=get_session_factory(),
        enforcer=get_enforcer(),
    )
    scheduler.start()

    yield

    scheduler.shutdown()
    log.info("tidus_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    # Per-IP rate limiting (non-fatal import — slowapi is an optional dep)
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address
        _limiter = Limiter(key_func=get_remote_address)
    except ImportError:  # pragma: no cover
        _limiter = None

    app = FastAPI(
        title="Tidus — Enterprise AI Router",
        description=(
            "Vendor-agnostic, cost-aware AI model router and governance system. "
            "Routes every AI request to the cheapest capable model, enforces budgets, "
            "and prevents runaway multi-agent loops."
        ),
        version="1.1.0",
        contact={"name": "Kenny Wong", "email": "lapkei01@gmail.com"},
        license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    if _limiter is not None:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.state.limiter = _limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    async def root(request: Request):
        """Browser → dashboard redirect; API clients get a JSON index."""
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return RedirectResponse("/dashboard/")
        return {
            "service": "tidus",
            "version": "1.1.0",
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
    app.include_router(reports.router, prefix="/api/v1")
    app.include_router(registry.router, prefix="/api/v1")
    app.include_router(billing.router, prefix="/api/v1")

    # ── Prometheus metrics ────────────────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/ready", "/metrics"],
    ).instrument(app).expose(app, include_in_schema=False, tags=["Observability"])

    # ── Dashboard static files ────────────────────────────────────────────────
    import pathlib

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
