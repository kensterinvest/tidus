"""Tidus — public web frontend.

Minimal FastAPI app that exposes ONLY the subscribe endpoints. Used to serve
the landing-page subscribe form at ai-router.z-tidus.com without exposing the
full router / classify / complete / metrics surface.

Run with:
    uv run uvicorn tidus.web_main:app --host 127.0.0.1 --port 9000
"""

from __future__ import annotations

from collections import defaultdict
from time import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tidus.api.v1.subscribe import router as subscribe_router

_RATE_WINDOW_SEC = 60.0
_RATE_MAX_PER_IP = 5
_buckets: dict[str, list[float]] = defaultdict(list)


app = FastAPI(
    title="Tidus — Public Web",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def rate_limit_subscribe(request: Request, call_next):
    if request.method == "POST" and request.url.path.endswith("/subscribe"):
        # Caddy sets X-Forwarded-For. Trust it because the only upstream
        # is 127.0.0.1 (loopback-bound uvicorn) which is unreachable
        # from outside the VPS.
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip() if fwd else (
            request.client.host if request.client else "unknown"
        )
        now = time()
        bucket = [t for t in _buckets[ip] if t > now - _RATE_WINDOW_SEC]
        if len(bucket) >= _RATE_MAX_PER_IP:
            return JSONResponse(
                {"detail": "Too many subscription attempts. Try again in a minute."},
                status_code=429,
            )
        bucket.append(now)
        _buckets[ip] = bucket
    return await call_next(request)


app.include_router(subscribe_router, prefix="/api/v1")
app.include_router(subscribe_router)


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}
