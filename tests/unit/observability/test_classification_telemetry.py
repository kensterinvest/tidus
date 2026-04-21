"""Unit tests for the Stage B PII-safe classification telemetry emitter.

We verify:
  * Happy-path record has every field in the plan.md line 547 schema.
  * `embedding_reduced_64d` is emitted and has length 64.
  * When PCA artifact is missing, emission succeeds and the field is omitted
    (never block the request path).
  * Emission NEVER raises, even when inputs are partially None (encoder
    failure mode).
  * `rationale` from T5 is NOT present anywhere in the record (plan.md:
    "never raw prompts").

Capture strategy: we monkeypatch the module-level `log` with a MagicMock
instead of using `structlog.testing.capture_logs()`. Tidus configures
structlog with `cache_logger_on_first_use=True` — once the bound logger
is warmed by any other code (e.g., a `create_app()` in a sibling test
module), `capture_logs` can no longer interpose. Patching the reference
keeps tests deterministic regardless of prior app setup.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest

from tidus.classification.models import (
    ClassificationResult,
    EncoderResult,
    PresidioResult,
    Tier1Signals,
)
from tidus.observability.classification_telemetry import (
    _reset_cache_for_tests,
    emit_classification_telemetry,
)


def _make_result(**overrides) -> ClassificationResult:
    base = {
        "domain": "chat",
        "complexity": "simple",
        "privacy": "public",
        "estimated_input_tokens": 10,
        "classification_tier": "encoder",
        "confidence": {"domain": 0.8, "complexity": 0.7, "privacy": 0.9},
    }
    base.update(overrides)
    return ClassificationResult(**base)


def _make_encoder_result(embedding: list[float] | None = None) -> EncoderResult:
    return EncoderResult(
        domain="chat", complexity="simple", privacy="public",
        confidence={"domain": 0.8, "complexity": 0.7, "privacy": 0.9},
        embedding=embedding,
    )


@pytest.fixture(autouse=True)
def clear_pca_cache():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.fixture
def captured_log(monkeypatch):
    """Replace the telemetry module's `log` with a MagicMock. Returns the
    mock so tests can inspect `.info.call_args_list` for emitted records.

    Returns a small helper that extracts the last emitted record as
    (event_name, dict_of_kwargs). Any structlog configuration done by
    app init is bypassed — we look at the call directly."""
    mock = MagicMock()
    monkeypatch.setattr(
        "tidus.observability.classification_telemetry.log", mock,
    )

    class _Helper:
        @property
        def records(self) -> list[dict]:
            # Each .info(event, **kwargs) call becomes one record: {event, **kwargs}
            out = []
            for call in mock.info.call_args_list:
                args, kwargs = call
                event = args[0] if args else kwargs.get("event")
                rec = {"event": event, **kwargs}
                out.append(rec)
            return out

        @property
        def info_calls(self) -> int:
            return mock.info.call_count

    return _Helper()


def test_happy_path_has_all_schema_fields(tmp_path: Path, captured_log):
    # Build a tiny PCA artifact: input_dim=8 -> output_dim=4 for speed.
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(42)
    fake_corpus = rng.standard_normal((32, 8)).astype(np.float32)
    pca = PCA(n_components=4, random_state=42).fit(fake_corpus)
    pca_path = tmp_path / "pca_tiny.joblib"
    joblib.dump(pca, pca_path)

    encoder = _make_encoder_result(embedding=[0.1] * 8)
    presidio = PresidioResult(entity_types=["PERSON", "EMAIL_ADDRESS"])
    signals = Tier1Signals(regex_hits=["SSN_US", "CREDIT_CARD"], any_hit=True)
    result = _make_result(privacy="confidential", classification_tier="encoder")

    emit_classification_telemetry(
        tenant_id="t-42",
        result=result,
        signals=signals,
        encoder=encoder,
        presidio=presidio,
        model_routed="haiku-4-5",
        latency_ms=37,
        pca_path=str(pca_path),
        request_id="req-abc",
    )

    assert captured_log.info_calls == 1
    record = captured_log.records[0]
    assert record["event"] == "classification"
    assert record["request_id"] == "req-abc"
    assert record["tenant_id"] == "t-42"
    assert "ts" in record and record["ts"].endswith("+00:00")
    assert record["presidio_entities"] == ["PERSON", "EMAIL_ADDRESS"]
    assert record["regex_hits"] == ["SSN_US", "CREDIT_CARD"]
    assert record["tier_decided"] == "encoder"
    assert record["classification"] == {
        "domain": "chat", "complexity": "simple", "privacy": "confidential",
    }
    assert record["model_routed"] == "haiku-4-5"
    assert record["latency_ms"] == 37
    assert "embedding_reduced_64d" in record
    assert len(record["embedding_reduced_64d"]) == 4  # tiny PCA


def test_missing_pca_artifact_still_emits_without_embedding(tmp_path: Path, captured_log):
    missing = tmp_path / "does_not_exist.joblib"
    encoder = _make_encoder_result(embedding=[0.1] * 384)

    emit_classification_telemetry(
        tenant_id="t-1",
        result=_make_result(),
        signals=Tier1Signals(),
        encoder=encoder,
        presidio=PresidioResult(),
        model_routed=None,
        latency_ms=5,
        pca_path=str(missing),
    )

    assert captured_log.info_calls == 1
    record = captured_log.records[0]
    # Other fields still present — emission didn't short-circuit.
    assert record["event"] == "classification"
    assert "embedding_reduced_64d" not in record


def test_none_encoder_omits_embedding_field(tmp_path: Path, captured_log):
    # With no encoder (e.g. encoder load failed), we still emit the record
    # but the embedding field is absent. No raising, no errors logged.
    emit_classification_telemetry(
        tenant_id="t-1",
        result=_make_result(),
        signals=Tier1Signals(),
        encoder=None,
        presidio=None,
        model_routed=None,
        latency_ms=1,
        pca_path=str(tmp_path / "nope.joblib"),
    )
    assert captured_log.info_calls == 1
    assert "embedding_reduced_64d" not in captured_log.records[0]


def test_emit_never_raises_even_if_pca_transform_errors(
    tmp_path: Path, captured_log,
):
    """Defensive: if transform blows up (dimension mismatch etc), emission
    must still succeed — other fields go to the log or nothing logs, but
    no exception escapes."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(42)
    pca = PCA(n_components=4, random_state=42).fit(rng.standard_normal((32, 8)).astype(np.float32))
    pca_path = tmp_path / "pca.joblib"
    joblib.dump(pca, pca_path)

    # Sabotage: wrong-dim embedding triggers sklearn ValueError during transform.
    encoder = _make_encoder_result(embedding=[0.1] * 999)  # expected 8, got 999

    # Must not raise.
    emit_classification_telemetry(
        tenant_id="t-1",
        result=_make_result(),
        signals=None,
        encoder=encoder,
        presidio=None,
        model_routed=None,
        latency_ms=7,
        pca_path=str(pca_path),
    )
    # Either no record (all errors swallowed in the outer try) OR record
    # with no embedding. Both satisfy the "never raises + never logs raw"
    # invariant.
    if captured_log.records:
        assert "embedding_reduced_64d" not in captured_log.records[0]


def test_signals_and_presidio_contain_only_type_names(tmp_path: Path, captured_log):
    """PII-safety invariant: the record must never include values, only types.
    Tier1Signals.regex_hits and PresidioResult.entity_types are type-names by
    construction — this test documents and guards that invariant."""
    signals = Tier1Signals(regex_hits=["EMAIL", "SSN_US"])
    presidio = PresidioResult(entity_types=["PERSON", "PHONE_NUMBER"])

    emit_classification_telemetry(
        tenant_id=None,
        result=_make_result(),
        signals=signals,
        encoder=None,
        presidio=presidio,
        model_routed=None,
        latency_ms=10,
        pca_path=str(tmp_path / "pca.joblib"),
    )

    record = captured_log.records[0]
    # Types only; no "123-45-6789", "foo@bar.com" etc.
    assert record["regex_hits"] == ["EMAIL", "SSN_US"]
    assert record["presidio_entities"] == ["PERSON", "PHONE_NUMBER"]


def test_pca_cache_reused_across_calls(tmp_path: Path, captured_log):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(42)
    pca = PCA(n_components=4, random_state=42).fit(rng.standard_normal((32, 8)).astype(np.float32))
    pca_path = tmp_path / "pca.joblib"
    joblib.dump(pca, pca_path)

    mock_load = MagicMock(wraps=joblib.load)
    import joblib as joblib_mod
    original_load = joblib_mod.load
    joblib_mod.load = mock_load
    try:
        _reset_cache_for_tests()
        encoder = _make_encoder_result(embedding=[0.1] * 8)

        for _ in range(5):
            emit_classification_telemetry(
                tenant_id="t", result=_make_result(), signals=None,
                encoder=encoder, presidio=None, model_routed=None,
                latency_ms=1, pca_path=str(pca_path),
            )

        # Artifact loaded once, not five times.
        assert mock_load.call_count == 1
    finally:
        joblib_mod.load = original_load
        _reset_cache_for_tests()


def test_rationale_is_never_in_record(tmp_path: Path, captured_log):
    """T5's LLMResult.rationale can paraphrase the prompt verbatim. plan.md
    says the telemetry record must never contain raw prompts. This test
    locks the invariant in — we can relax it only by explicit decision."""
    encoder = _make_encoder_result(embedding=[0.1] * 8)
    presidio = PresidioResult(entity_types=["PERSON"])
    signals = Tier1Signals()
    result = _make_result(privacy="confidential")

    emit_classification_telemetry(
        tenant_id="t",
        result=result,
        signals=signals,
        encoder=encoder,
        presidio=presidio,
        model_routed=None,
        latency_ms=1,
        pca_path=str(tmp_path / "nope.joblib"),
    )

    record = captured_log.records[0]
    # Just in case future refactor adds a rationale field, catch it.
    assert "rationale" not in record
    # And nothing else that could carry prompt text.
    assert "prompt" not in record
    assert "text" not in record
