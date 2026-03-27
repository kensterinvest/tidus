"""Integration tests for POST /api/v1/complete.

Uses FastAPI's TestClient against the real app with real registry + policies.
The vendor adapter's complete() is mocked so no API keys are required.

Run with:
    uv run pytest tests/integration/test_complete_endpoint.py -v
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tidus.main import create_app


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient with app lifespan (builds singletons once per module)."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _fake_adapter_response(model_id: str = "llama4-scout-ollama"):
    """Returns a mock AdapterResponse matching the real dataclass shape."""
    from tidus.adapters.base import AdapterResponse
    return AdapterResponse(
        model_id=model_id,
        content="Test response from mocked adapter",
        input_tokens=15,
        output_tokens=8,
        latency_ms=120.0,
        finish_reason="stop",
    )


@contextmanager
def _mock_tokenizer(token_count: int = 15):
    """Patch the tokenizer where it's imported so vendor API calls are not needed."""
    # Patch at the usage site (engine.py already imported the function)
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=token_count)):
        yield


def _complete(client, token_count: int = 15, **kwargs) -> dict:
    defaults = {
        "team_id": "team-engineering",
        "complexity": "simple",
        "domain": "chat",
        "estimated_input_tokens": 300,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    with _mock_tokenizer(token_count):
        resp = client.post("/api/v1/complete", json=defaults)
    return resp


# ── Core happy path ────────────────────────────────────────────────────────────

class TestCompleteHappyPath:
    def test_complete_returns_200_and_content(self, client):
        """Basic complete request returns 200 with content field."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(return_value=_fake_adapter_response())
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=15)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(client)

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Test response from mocked adapter"
        assert "chosen_model_id" in data
        assert "cost_usd" in data
        assert data["cost_usd"] >= 0.0

    def test_complete_response_has_required_fields(self, client):
        """Response includes all documented fields."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(return_value=_fake_adapter_response())
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=15)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(client)

        assert resp.status_code == 200
        data = resp.json()
        for field in ("task_id", "chosen_model_id", "vendor", "content",
                      "input_tokens", "output_tokens", "cost_usd",
                      "latency_ms", "finish_reason"):
            assert field in data, f"Missing field: {field}"

    def test_simple_task_routes_to_cheap_model(self, client):
        """A simple/chat task should route to a tier 3 or 4 model (not premium)."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(
            side_effect=lambda model_id, task: _fake_adapter_response(model_id)
        )
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(client, complexity="simple", domain="chat",
                             estimated_input_tokens=100)

        assert resp.status_code == 200
        data = resp.json()
        # Simple tasks must not pick tier 1 models (claude-opus, gpt-5, etc.)
        assert data["cost_usd"] < 0.01, (
            f"Simple task cost {data['cost_usd']:.6f} is too high — should route cheaply"
        )


# ── Budget enforcement ─────────────────────────────────────────────────────────

class TestBudgetEnforcement:
    def test_complete_respects_max_cost_usd(self, client):
        """Setting max_cost_usd=0.000001 should cause a 429 budget rejection."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(return_value=_fake_adapter_response())
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(
                client,
                max_cost_usd=0.000001,
                estimated_input_tokens=50000,
                complexity="complex",
                domain="reasoning",
            )

        # All models cost more than $0.000001 for 50K tokens → budget_exceeded → 422
        assert resp.status_code == 422

    def test_complete_zero_cost_local_model_bypasses_budget(self, client):
        """Local Ollama models (cost=0.0) should pass even with a tiny budget."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(
            side_effect=lambda model_id, task: _fake_adapter_response(model_id)
        )
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=5)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(
                client,
                max_cost_usd=0.000001,  # tiny budget
                complexity="simple",
                domain="chat",
                estimated_input_tokens=100,
            )

        # Free local model should win
        assert resp.status_code == 200
        assert resp.json()["cost_usd"] == 0.0


# ── Privacy routing ────────────────────────────────────────────────────────────

class TestPrivacyRouting:
    def test_confidential_task_routes_to_local_model(self, client):
        """Confidential privacy must only route to is_local=True models."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(
            side_effect=lambda model_id, task: _fake_adapter_response(model_id)
        )
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(
                client,
                privacy="confidential",
                complexity="simple",
                domain="chat",
                estimated_input_tokens=200,
            )

        assert resp.status_code == 200
        data = resp.json()
        # Confidential tasks must land on a local/on-prem model (is_local=True)
        from tidus.api.deps import get_registry as _get_reg
        reg = _get_reg()
        spec = reg.get(data["chosen_model_id"])
        assert spec is not None and spec.is_local, (
            f"Confidential task routed to {data['chosen_model_id']} (vendor={data['vendor']}), "
            "expected an is_local=True model"
        )


# ── Adapter failure + fallback ─────────────────────────────────────────────────

class TestAdapterFallback:
    def test_adapter_error_triggers_fallback(self, client):
        """If the primary adapter raises, the endpoint should try fallbacks."""
        call_count = 0

        async def fail_then_succeed(model_id, task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Primary model unavailable")
            return _fake_adapter_response(model_id)

        mock_adapter = MagicMock()
        mock_adapter.complete = fail_then_succeed
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(client, complexity="simple", domain="chat",
                             estimated_input_tokens=200)

        # Should succeed via fallback
        assert resp.status_code == 200
        assert resp.json()["fallback_from"] is not None


# ── Sync admin endpoints ────────────────────────────────────────────────────────

class TestSyncEndpoints:
    def test_health_probe_endpoint_returns_results(self, client):
        """POST /api/v1/sync/health returns health results dict."""
        mock_adapter = MagicMock()
        mock_adapter.health_check = AsyncMock(return_value=True)

        with patch("tidus.adapters.adapter_factory.get_adapter", return_value=mock_adapter):
            resp = client.post("/api/v1/sync/health")

        assert resp.status_code == 200
        data = resp.json()
        assert "probed" in data
        assert "healthy" in data
        assert "results" in data

    def test_price_sync_endpoint_returns_changes(self, client):
        """POST /api/v1/sync/prices returns changes list (may be empty)."""
        resp = client.post("/api/v1/sync/prices")
        assert resp.status_code == 200
        data = resp.json()
        assert "changes_detected" in data
        assert isinstance(data["changes"], list)
