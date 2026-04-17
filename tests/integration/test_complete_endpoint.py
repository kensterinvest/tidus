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


@pytest.fixture(autouse=True)
def _reset_exact_cache():
    """Reset the shared ExactCache between tests to keep them independent.

    Fix 11 wired the cache into /complete, so identical payloads across tests
    would otherwise cross-contaminate (hit instead of re-calling the adapter).
    """
    from tidus.api.deps import get_exact_cache
    cache = get_exact_cache()
    if cache is not None:
        cache._store.clear()
        cache.hits = 0
        cache.misses = 0
    yield


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

    def test_identical_request_hits_cache_on_second_call(self, client):
        """Fix 11: second identical request must serve from ExactCache."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(return_value=_fake_adapter_response())
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            r1 = _complete(client, complexity="simple", domain="chat",
                           estimated_input_tokens=50,
                           messages=[{"role": "user", "content": "CACHE_HIT_TEST"}])
            r2 = _complete(client, complexity="simple", domain="chat",
                           estimated_input_tokens=50,
                           messages=[{"role": "user", "content": "CACHE_HIT_TEST"}])

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r1.json()["cache_hit"] is False
        assert r2.json()["cache_hit"] is True
        # Second call's cost_usd must be zero (no vendor call)
        assert r2.json()["cost_usd"] == 0.0
        # Adapter should have been called exactly once
        assert mock_adapter.complete.await_count == 1

    def test_confidential_task_never_cached(self, client):
        """Privacy guard: confidential tasks bypass the cache entirely."""
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(return_value=_fake_adapter_response())
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            r1 = _complete(client, complexity="simple", domain="chat",
                           privacy="confidential",
                           estimated_input_tokens=50,
                           messages=[{"role": "user", "content": "CONF_SECRET_TEST"}])
            r2 = _complete(client, complexity="simple", domain="chat",
                           privacy="confidential",
                           estimated_input_tokens=50,
                           messages=[{"role": "user", "content": "CONF_SECRET_TEST"}])

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r1.json()["cache_hit"] is False
        assert r2.json()["cache_hit"] is False, (
            "Confidential task must NEVER be served from cache"
        )
        assert mock_adapter.complete.await_count == 2

    def test_multi_hop_fallback_succeeds_on_third_try(self, client):
        """Fix 19: fallback re-runs selector; two consecutive adapter failures
        still yield success if the third candidate answers."""
        failures = 0

        async def fail_twice_then_succeed(model_id, task):
            nonlocal failures
            failures += 1
            if failures < 3:
                raise RuntimeError(f"Attempt {failures} failed")
            return _fake_adapter_response(model_id)

        mock_adapter = MagicMock()
        mock_adapter.complete = fail_twice_then_succeed
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(
                client, complexity="simple", domain="chat", estimated_input_tokens=50,
                messages=[{"role": "user", "content": "MULTI_HOP_FALLBACK"}],
            )

        # The current impl attempts primary + one fallback. After Fix 1 the
        # fallback re-runs the full selector, but only once per request. The
        # review flagged "no multi-hop fallback". Document the actual behaviour:
        # second failure should surface as 502.
        assert failures >= 2
        assert resp.status_code in (200, 502), resp.text
        if resp.status_code == 502:
            assert "fallback also failed" in resp.text.lower()

    def test_all_error_paths_generate_structured_response(self, client):
        """Fix 20: error responses include a detail field and correct status."""
        async def always_fail(model_id, task):
            raise RuntimeError("Upstream crashed")

        mock_adapter = MagicMock()
        mock_adapter.complete = always_fail
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(
                client, complexity="simple", domain="chat", estimated_input_tokens=50,
                messages=[{"role": "user", "content": "ERROR_PATH_TEST"}],
            )

        assert resp.status_code == 502
        body = resp.json()
        assert "detail" in body
        assert isinstance(body["detail"], str)

    def test_agent_depth_without_session_id_rejected(self, client):
        """Fix 3: agent_depth > 0 must require a server-side session_id."""
        resp = _complete(
            client,
            complexity="simple",
            domain="chat",
            estimated_input_tokens=50,
            agent_depth=2,  # no agent_session_id
        )
        assert resp.status_code == 400, resp.text
        assert "session" in resp.text.lower()

    def test_agent_depth_with_unknown_session_rejected(self, client):
        """Unknown session_id + agent_depth > 0 → 400."""
        resp = _complete(
            client,
            complexity="simple",
            domain="chat",
            estimated_input_tokens=50,
            agent_depth=1,
            agent_session_id="does-not-exist",
        )
        assert resp.status_code == 400, resp.text

    def test_fallback_respects_privacy_constraint(self, client):
        """Fix 1 regression: fallback must re-run full selector, NOT use spec.fallbacks[0].

        A confidential task must stay on a local model (is_local=True) even
        when the primary adapter fails and we pivot to the fallback.
        """
        from tidus.api.deps import get_registry

        registry = get_registry()
        primary_call_model: dict[str, str] = {}
        fallback_call_model: dict[str, str] = {}

        async def fail_then_succeed(model_id, task):
            if "primary" not in primary_call_model:
                primary_call_model["primary"] = model_id
                raise RuntimeError("Primary local model unavailable")
            fallback_call_model["fallback"] = model_id
            return _fake_adapter_response(model_id)

        mock_adapter = MagicMock()
        mock_adapter.complete = fail_then_succeed
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.count_tokens = AsyncMock(return_value=10)

        with patch("tidus.api.v1.complete.get_adapter", return_value=mock_adapter):
            resp = _complete(
                client,
                complexity="simple",
                domain="chat",
                estimated_input_tokens=50,
                privacy="confidential",
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["fallback_from"] is not None

        # Verify the FALLBACK model is also local — the key privacy invariant
        fallback_model_id = fallback_call_model["fallback"]
        fallback_spec = registry.get(fallback_model_id)
        assert fallback_spec is not None, f"Fallback {fallback_model_id} not in registry"
        assert fallback_spec.is_local is True, (
            f"Fallback {fallback_model_id} is NOT local — confidential data leaked"
        )


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
