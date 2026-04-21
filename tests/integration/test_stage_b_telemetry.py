"""Integration — Stage B telemetry emits at the /classify endpoint.

Runs through the full classifier cascade with a real TaskClassifier + real
encoder + real Presidio, and asserts the structured log record lands with
every plan.md line 547 field populated.

This is the end-to-end proof that the observer callback is wired through
from the endpoint into TaskClassifier.classify_async and back out to the
telemetry emitter.

Capture strategy: we monkeypatch the telemetry module's `log` with a
MagicMock. Tidus configures structlog with `cache_logger_on_first_use=True`
(utils/logging.py), which makes `structlog.testing.capture_logs()`
unreliable once the app boots. Patching the reference inside the module
is deterministic regardless of app-init order.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tidus.main import create_app
from tidus.observability.classification_telemetry import _reset_cache_for_tests


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_pca_cache():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.fixture
def captured_log(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(
        "tidus.observability.classification_telemetry.log", mock,
    )

    class _Helper:
        @property
        def records(self) -> list[dict]:
            out = []
            for call in mock.info.call_args_list:
                args, kwargs = call
                event = args[0] if args else kwargs.get("event")
                out.append({"event": event, **kwargs})
            return out

    return _Helper()


class TestStageBTelemetry:
    def test_classify_emits_stage_b_record(
        self, client: TestClient, captured_log,
    ):
        """Hit /classify with a prompt that triggers a Presidio entity hit.
        Verify the emitted record's schema."""
        resp = client.post("/api/v1/classify", json={
            "messages": [{
                "role": "user",
                "content": "my SSN is 123-45-6789 please help",
            }],
            "team_id": "team-dev",
        })
        assert resp.status_code == 200

        records = [r for r in captured_log.records if r.get("event") == "classification"]
        assert len(records) >= 1
        record = records[-1]

        for field in (
            "request_id", "tenant_id", "ts",
            "presidio_entities", "regex_hits",
            "tier_decided", "classification",
            "model_routed", "latency_ms",
        ):
            assert field in record, f"missing field: {field}"

        # Tenant falls through to team_id in dev mode
        assert record["tenant_id"] == "team-dev"
        # /classify is a dry-run — no downstream model routed
        assert record["model_routed"] is None
        # Latency is reasonable (non-negative int)
        assert isinstance(record["latency_ms"], int)
        assert record["latency_ms"] >= 0
        # Classification axes present
        assert set(record["classification"].keys()) == {"domain", "complexity", "privacy"}
        # Regex fired for SSN pattern
        assert "SSN_US" in record["regex_hits"]
        # Embedding reduction worked (PCA artifact must exist in the repo)
        assert "embedding_reduced_64d" in record
        assert len(record["embedding_reduced_64d"]) == 64

    def test_no_raw_prompt_in_record(
        self, client: TestClient, captured_log,
    ):
        """Explicitly assert the prompt text is not anywhere in the record.
        plan.md §Stage B: 'features only, never raw prompts.'"""
        secret = "super-unique-secret-xyzzy-PII-123-45-6789"
        client.post("/api/v1/classify", json={
            "messages": [{"role": "user", "content": secret}],
            "team_id": "team-dev",
        })

        records = [r for r in captured_log.records if r.get("event") == "classification"]
        assert records
        for r in records:
            for v in r.values():
                flat = str(v).lower()
                assert secret.lower() not in flat, (
                    f"raw prompt leaked into Stage B record via field: {v}"
                )

    def test_tenant_header_propagates_into_record(
        self, client: TestClient, captured_log,
    ):
        """X-Tenant-ID header overrides the team_id fallback in dev mode."""
        client.post(
            "/api/v1/classify",
            headers={"X-Tenant-ID": "tenant-acme"},
            json={
                "messages": [{"role": "user", "content": "hello, how are you"}],
                "team_id": "team-dev",
            },
        )
        records = [r for r in captured_log.records if r.get("event") == "classification"]
        assert records
        assert records[-1]["tenant_id"] == "tenant-acme"


def _fake_adapter_response():
    from tidus.adapters.base import AdapterResponse
    return AdapterResponse(
        model_id="llama4-scout-ollama",
        content="mocked",
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


class TestStageBModelRouted:
    """Advisor-flagged bug: model_routed was always None on /complete and
    /route because the observer fired before routing. These tests lock in
    the fix — model_routed must be populated with the chosen model id."""

    def test_complete_emits_with_model_routed(
        self, client: TestClient, captured_log,
    ):
        """The bug the advisor caught: previously `model_routed` was always
        None because the observer fired inside `classify_async` (before
        routing). After the fix, `model_routed` must be the selector's
        `decision.chosen_model_id` — a real model, not None.

        Note: we don't compare against the response's `chosen_model_id`
        because /complete returns the adapter's actual `response.model_id`
        (which can differ from the selected model after fallback or vendor-
        side routing). The Stage B telemetry intentionally records the
        ROUTER's decision, not the vendor's execution.
        """
        with _mock_adapter():
            resp = client.post("/api/v1/complete", json={
                "team_id": "team-dev",
                "messages": [{"role": "user", "content": "hello, how are you"}],
            })
        assert resp.status_code == 200, resp.text

        records = [r for r in captured_log.records if r.get("event") == "classification"]
        assert records, "no Stage B record emitted from /complete"
        record = records[-1]
        assert isinstance(record["model_routed"], str), (
            f"expected model_routed to be a non-None model id, got "
            f"{record['model_routed']!r}"
        )
        assert record["model_routed"]  # non-empty

    def test_route_emits_with_model_routed(
        self, client: TestClient, captured_log,
    ):
        """Same bug fix, but on /route (which doesn't execute a model and
        so has no adapter response to disambiguate from — the response's
        `chosen_model_id` is `decision.chosen_model_id` directly)."""
        resp = client.post("/api/v1/route", json={
            "team_id": "team-dev",
            "messages": [{"role": "user", "content": "hello, how are you"}],
        })
        assert resp.status_code == 200

        records = [r for r in captured_log.records if r.get("event") == "classification"]
        assert records
        record = records[-1]
        assert record["model_routed"] is not None
        assert record["model_routed"] == resp.json()["chosen_model_id"]
