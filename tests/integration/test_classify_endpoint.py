"""Integration tests for POST /api/v1/classify.

Uses FastAPI's TestClient against the real app with the full classifier
cascade wired (encoder + Presidio + T5 LLM when available). Covers:
  * Happy-path classification (no caller override)
  * Caller override (T0 short-circuit)
  * Invalid request (no user message)
  * Classifier disabled (503)
  * /ready endpoint reports classifier health
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tidus.main import create_app


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient — builds singletons (and loads encoder +
    Presidio) once per test module. First request pays ~5s; subsequent
    calls are fast."""
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def preserve_classifier_singleton():
    """See test_auto_classify_endpoints.py for rationale."""
    from tidus.api import deps as deps_module
    saved = deps_module._classifier
    yield
    deps_module._classifier = saved


class TestHappyPath:
    def test_benign_prompt_returns_200(self, client: TestClient):
        resp = client.post("/api/v1/classify", json={
            "messages": [{"role": "user", "content": "What's the weather in Toronto?"}],
            "team_id": "team-dev",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        assert "classifier_health" in body
        r = body["result"]
        assert r["domain"] in {
            "chat", "code", "reasoning", "extraction",
            "classification", "summarization", "creative",
        }
        assert r["privacy"] in {"public", "internal", "confidential"}
        assert r["classification_tier"] in {
            "caller_override", "heuristic", "default", "encoder", "llm", "cached",
        }

    def test_ssn_prompt_flags_confidential(self, client: TestClient):
        resp = client.post("/api/v1/classify", json={
            "messages": [{"role": "user", "content": "My SSN is 123-45-6789 please help"}],
            "team_id": "team-dev",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["privacy"] == "confidential"

    def test_include_debug_populates_payload(self, client: TestClient):
        resp = client.post("/api/v1/classify", json={
            "messages": [{"role": "user", "content": "my SSN is 123-45-6789"}],
            "team_id": "team-dev",
            "include_debug": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        debug = body["result"]["debug"]
        assert debug is not None
        assert "tier1_signals" in debug
        assert "SSN_US" in debug["tier1_signals"]["regex_hits"]


class TestCallerOverride:
    def test_complete_override_short_circuits(self, client: TestClient):
        resp = client.post("/api/v1/classify", json={
            "messages": [{"role": "user", "content": "anything"}],
            "team_id": "team-dev",
            "caller_override": {
                "domain": "code", "complexity": "complex", "privacy": "public",
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["classification_tier"] == "caller_override"
        assert body["result"]["domain"] == "code"

    def test_partial_override_merges_with_t1(self, client: TestClient):
        """Caller says public but SSN in text → asymmetric safety wins."""
        resp = client.post("/api/v1/classify", json={
            "messages": [{"role": "user", "content": "my SSN 123-45-6789"}],
            "team_id": "team-dev",
            "caller_override": {"privacy": "public"},  # partial
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["privacy"] == "confidential"


class TestRequestValidation:
    def test_empty_messages_rejected(self, client: TestClient):
        resp = client.post("/api/v1/classify", json={
            "messages": [],
            "team_id": "team-dev",
        })
        assert resp.status_code == 422  # Pydantic min_length=1

    def test_no_user_message_returns_400(self, client: TestClient):
        resp = client.post("/api/v1/classify", json={
            "messages": [{"role": "assistant", "content": "hi"}],
            "team_id": "team-dev",
        })
        assert resp.status_code == 400
        assert "No user message" in resp.json()["detail"]

    def test_missing_team_id_rejected(self, client: TestClient):
        resp = client.post("/api/v1/classify", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 422


class TestReadyEndpoint:
    def test_ready_includes_classifier_health(self, client: TestClient):
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        # Since auto_classify_enabled defaults to True, classifier should be
        # present. Each tier's loaded state reflects the real environment.
        assert "classifier" in body
        ch = body["classifier"]
        assert "encoder_loaded" in ch
        assert "presidio_loaded" in ch
        assert "llm_loaded" in ch
        assert "sku" in ch
        # On test hardware without a GPU + Ollama server active, we expect
        # the CPU-only SKU.
        assert ch["sku"] in {"cpu-only", "enterprise"}


class TestClassifierDisabled:
    def test_503_when_classifier_disabled(
        self, monkeypatch: pytest.MonkeyPatch, preserve_classifier_singleton,
    ):
        """When auto_classify_enabled=False, /classify returns 503.

        Advisor A.5 Bug #1: inject a FRESH Settings instance via get_settings
        patch rather than mutating the shared module-level singleton. Prior
        implementation set `_classifier = None` on the shared deps module,
        risking cross-test state leaks.
        """
        from tidus.settings import Settings
        fresh_settings = Settings()
        fresh_settings.auto_classify_enabled = False
        monkeypatch.setattr("tidus.api.deps.get_settings", lambda: fresh_settings)
        monkeypatch.setattr("tidus.main.get_settings", lambda: fresh_settings)

        app = create_app()
        with TestClient(app) as c:
            resp = c.post("/api/v1/classify", json={
                "messages": [{"role": "user", "content": "hi"}],
                "team_id": "team-dev",
            })
            assert resp.status_code == 503
            assert "disabled" in resp.json()["detail"].lower()
