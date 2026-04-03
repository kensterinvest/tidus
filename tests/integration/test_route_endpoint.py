"""Integration tests for POST /api/v1/route.

Uses FastAPI's TestClient against the real app (with real registry + policies)
but mocks the tokenizer so no vendor API keys are required.

Run with:
    uv run pytest tests/integration/test_route_endpoint.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tidus.main import create_app

# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient with app lifespan (builds singletons once per module)."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# ── Helper ────────────────────────────────────────────────────────────────────

def _route(client, **kwargs) -> dict:
    defaults = {
        "team_id": "team-engineering",
        "complexity": "simple",
        "domain": "chat",
        "estimated_input_tokens": 500,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        resp = client.post("/api/v1/route", json=defaults)
    return resp


# ── Happy-path tests ──────────────────────────────────────────────────────────

def test_simple_chat_routes_successfully(client):
    resp = _route(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["chosen_model_id"] is not None
    assert body["estimated_cost_usd"] is not None


def test_route_returns_task_id(client):
    resp = _route(client)
    body = resp.json()
    assert "task_id" in body
    assert len(body["task_id"]) > 0


def test_critical_reasoning_selects_tier1(client):
    resp = _route(client, complexity="critical", domain="reasoning")
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    # The model registry only has tier-1 models eligible for critical tasks
    assert body["chosen_model_id"] is not None


def test_confidential_task_routes_to_local(client):
    resp = _route(client, privacy="confidential", complexity="simple", domain="chat")
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    # Local models have model_ids ending in -ollama or are gemini-nano
    model_id = body["chosen_model_id"]
    assert "ollama" in model_id or model_id == "gemini-nano"


def test_preferred_model_respected(client):
    resp = _route(
        client,
        complexity="simple",
        domain="chat",
        preferred_model_id="claude-haiku-4-5",
    )
    assert resp.status_code == 200
    assert resp.json()["chosen_model_id"] == "claude-haiku-4-5"


def test_all_domains_route_successfully(client):
    """Every supported domain should produce a successful routing decision."""
    for domain in ("chat", "code", "reasoning", "extraction", "classification", "summarization"):
        resp = _route(client, domain=domain, complexity="simple")
        assert resp.status_code == 200, f"Domain '{domain}' failed: {resp.json()}"
        assert resp.json()["accepted"] is True


# ── Error / rejection tests ───────────────────────────────────────────────────

def test_agent_depth_over_limit_returns_422(client):
    resp = _route(client, agent_depth=6)
    assert resp.status_code == 422


def test_missing_required_field_returns_422(client):
    """Omitting team_id (required) should trigger Pydantic validation error."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        resp = client.post("/api/v1/route", json={"complexity": "simple", "domain": "chat"})
    assert resp.status_code == 422


def test_unknown_complexity_returns_422(client):
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        resp = client.post(
            "/api/v1/route",
            json={
                "team_id": "team-eng",
                "complexity": "ultra-critical",
                "domain": "chat",
                "estimated_input_tokens": 500,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 422


# ── Health checks ─────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_endpoint(client):
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_openapi_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
