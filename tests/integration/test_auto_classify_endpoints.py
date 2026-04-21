"""Integration tests — v1.3.0 auto-classification on /complete and /route.

Proves the value prop: callers can omit complexity/domain/privacy/
estimated_input_tokens on existing endpoints, and Tidus fills them in
via the T0→T5 cascade.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tidus.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def preserve_classifier_singleton():
    """Save/restore deps._classifier around tests that build a fresh app with
    auto_classify_enabled=False — otherwise that fresh build overwrites the
    shared module-level singleton with None and later tests see 503.

    Advisor A.5 Bug #1: don't leave module-level state leaked across tests.
    """
    from tidus.api import deps as deps_module
    saved = deps_module._classifier
    yield
    deps_module._classifier = saved


def _fake_adapter_response(model_id: str = "llama4-scout-ollama"):
    from tidus.adapters.base import AdapterResponse
    return AdapterResponse(
        model_id=model_id,
        content="mocked output",
        input_tokens=10,
        output_tokens=5,
        latency_ms=10.0,
        finish_reason="stop",
    )


@contextmanager
def _mock_adapter():
    fake = MagicMock()
    fake.complete = AsyncMock(return_value=_fake_adapter_response())
    with patch("tidus.api.v1.complete.get_adapter", return_value=fake), \
         patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=15)):
        yield fake


# ── /route — routing without execution ────────────────────────────────────────

class TestRouteAutoClassify:
    def test_route_with_all_fields_omitted(self, client: TestClient):
        """Callers can drop all four classification fields; classifier fills in.
        Uses a simple chat prompt for reliable domain=chat classification."""
        resp = client.post("/api/v1/route", json={
            "team_id": "team-dev",
            "messages": [{"role": "user", "content": "hello, how are you"}],
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["accepted"] is True
        assert body["chosen_model_id"] is not None

    def test_route_with_partial_caller_override(self, client: TestClient):
        """Caller says privacy=confidential + domain=chat; classifier fills rest.
        Caller values preserved via caller_override merge."""
        resp = client.post("/api/v1/route", json={
            "team_id": "team-dev",
            "privacy": "confidential",
            "domain": "chat",
            "messages": [{"role": "user", "content": "What's the weather"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] is True

    def test_route_backward_compat_all_fields(self, client: TestClient):
        """Existing callers that provide everything still work unchanged."""
        resp = client.post("/api/v1/route", json={
            "team_id": "team-dev",
            "complexity": "simple",
            "domain": "chat",
            "privacy": "internal",
            "estimated_input_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200


# ── /complete — routing + execution ───────────────────────────────────────────

class TestCompleteAutoClassify:
    def test_complete_with_all_fields_omitted(self, client: TestClient):
        """v1.3.0 value prop: caller provides only messages + team_id.

        Uses a simple greeting that reliably classifies as domain=chat, which
        every model in the test registry supports. We're testing the
        classification PLUMBING here, not the encoder's label choices.
        """
        with _mock_adapter():
            resp = client.post("/api/v1/complete", json={
                "team_id": "team-dev",
                "messages": [{"role": "user", "content": "hello, how are you"}],
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chosen_model_id"] is not None
        assert body["content"] == "mocked output"

    def test_complete_ssn_forces_confidential_routing(self, client: TestClient):
        """Auto-classify catches SSN → confidential → routes to local-only model.

        Pins domain AND complexity so the routing decision depends only on
        privacy (the value-prop claim we're actually verifying). Without
        these pins, test registry's model-eligibility matrix fluctuates
        with the classifier's other per-axis outputs.
        """
        with _mock_adapter():
            resp = client.post("/api/v1/complete", json={
                "team_id": "team-dev",
                "domain": "chat",
                "complexity": "simple",
                "messages": [{
                    "role": "user",
                    "content": "my SSN is 123-45-6789",
                }],
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chosen_model_id"] is not None

    def test_complete_backward_compat_all_fields(self, client: TestClient):
        with _mock_adapter():
            resp = client.post("/api/v1/complete", json={
                "team_id": "team-dev",
                "complexity": "simple",
                "domain": "chat",
                "privacy": "public",
                "estimated_input_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 200


class TestAutoClassifyDisabled:
    """When auto_classify_enabled=False, omitting fields must return 422."""

    def test_omitted_fields_rejected_when_classifier_disabled(
        self, monkeypatch: pytest.MonkeyPatch, preserve_classifier_singleton,
    ):
        from tidus.settings import Settings
        # Build a fresh Settings with classifier disabled; pass it so the
        # build_singletons path sees it and skips classifier init.
        # Advisor A.5 Bug #1: avoid mutating the shared module-level singleton.
        fresh_settings = Settings()
        fresh_settings.auto_classify_enabled = False

        # Patch get_settings to return our fresh Settings for this app.
        monkeypatch.setattr(
            "tidus.api.deps.get_settings",
            lambda: fresh_settings,
        )
        monkeypatch.setattr(
            "tidus.main.get_settings",
            lambda: fresh_settings,
        )

        app = create_app()
        with TestClient(app) as c:
            resp = c.post("/api/v1/route", json={
                "team_id": "team-dev",
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status_code == 422
            detail = resp.json()["detail"]
            assert detail["error"] == "classification_fields_required"
            assert "complexity" in detail["missing"]


# ── Concurrency + parallelism (advisor A.5 coverage gaps #1 and #2) ───────────

class TestConcurrentClassifyRequests:
    """Advisor coverage gap #2: prove thread-safety locks work under
    simultaneous /classify requests."""

    @pytest.mark.asyncio
    async def test_five_concurrent_classifications_all_succeed(self):
        """Five parallel POSTs — all 200, all produce valid classifications.
        Exercises the per-tier asyncio.Lock added for encoder + Presidio.

        Uses httpx.AsyncClient with ASGITransport so the test runs against
        the real app via asyncio (not TestClient's sync threads, which
        aren't safe to share across tasks)."""
        from httpx import ASGITransport, AsyncClient

        app = create_app()
        # Explicit lifespan context so startup completes before requests fire.
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            # Run lifespan via a dummy GET /health (triggers startup if not yet)
            await c.get("/health")

            prompts = [
                "what's the weather",
                "my SSN is 111-22-3333",
                "write a Python function",
                "summarize this article",
                "I need legal advice about an NDA",
            ]

            async def one(prompt: str):
                return await c.post("/api/v1/classify", json={
                    "messages": [{"role": "user", "content": prompt}],
                    "team_id": "team-dev",
                })

            responses = await asyncio.gather(*(one(p) for p in prompts))
            statuses = [r.status_code for r in responses]
            assert all(s == 200 for s in statuses), f"Got {statuses}"


class TestReadyShape:
    """Advisor coverage gap #3: /ready response must not include classifier
    block when auto_classify_enabled=False."""

    def test_ready_omits_classifier_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, preserve_classifier_singleton,
    ):
        from tidus.settings import Settings
        fresh_settings = Settings()
        fresh_settings.auto_classify_enabled = False
        monkeypatch.setattr("tidus.api.deps.get_settings", lambda: fresh_settings)
        monkeypatch.setattr("tidus.main.get_settings", lambda: fresh_settings)

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ready"
            assert "classifier" not in body
