"""Unit tests for ClaudeAnomalyVerifier and build_anomalies_from_changes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tidus.sync.ai_verifier import (
    Anomaly,
    ClaudeAnomalyVerifier,
    build_anomalies_from_changes,
)

# ── Change → Anomaly filtering ────────────────────────────────────────────────

class TestBuildAnomalies:
    def test_filters_below_threshold(self):
        changes = [
            {"model_id": "gpt-4o", "field": "input_price",
             "old_value": 0.005, "new_value": 0.004, "delta_pct": -20.0},
            {"model_id": "claude-opus-4-7", "field": "input_price",
             "old_value": 0.005, "new_value": 0.001, "delta_pct": -80.0},
        ]
        anomalies = build_anomalies_from_changes(changes, threshold_pct=50.0)
        assert len(anomalies) == 1
        assert anomalies[0].model_id == "claude-opus-4-7"
        assert anomalies[0].delta_pct == -80.0

    def test_threshold_inclusive(self):
        changes = [
            {"model_id": "exact", "field": "input_price",
             "old_value": 1.0, "new_value": 2.0, "delta_pct": 50.0},
        ]
        anomalies = build_anomalies_from_changes(changes, threshold_pct=50.0)
        assert len(anomalies) == 1

    def test_vendor_read_from_modelspec_not_pricequote(self):
        """Regression test for the advisor-caught bug: vendor must come from
        ModelSpec.vendor (the actual vendor name like 'openai'), NOT from
        PriceQuote.source_name (which is the pricing-source provenance like
        'openrouter')."""
        from tidus.models.model_registry import ModelSpec

        spec = ModelSpec.model_validate({
            "model_id":      "gpt-4o",
            "vendor":        "openai",   # <-- this is what should reach Claude
            "tier":          2,
            "max_context":   128000,
            "input_price":   0.0025,
            "output_price":  0.01,
            "tokenizer":     "tiktoken_o200k",
            "capabilities":  ["chat"],
            "min_complexity": "simple",
            "max_complexity": "complex",
        })
        changes = [
            {"model_id": "gpt-4o", "field": "input_price",
             "old_value": 0.005, "new_value": 0.001, "delta_pct": -80.0},
        ]
        anomalies = build_anomalies_from_changes(
            changes, threshold_pct=50.0, specs_by_id={"gpt-4o": spec},
        )
        assert len(anomalies) == 1
        assert anomalies[0].vendor == "openai"  # NOT "openrouter"

    def test_vendor_blank_when_specs_by_id_missing(self):
        changes = [
            {"model_id": "gpt-4o", "field": "input_price",
             "old_value": 0.005, "new_value": 0.001, "delta_pct": -80.0},
        ]
        anomalies = build_anomalies_from_changes(changes, threshold_pct=50.0)
        assert anomalies[0].vendor == ""

    def test_drops_new_model_and_retired(self):
        changes = [
            {"model_id": "newbie", "field": "new_model",
             "old_value": 0.0, "new_value": 0.005, "delta_pct": 100.0},
            {"model_id": "oldie", "field": "retired",
             "old_value": 0.005, "new_value": 0.0, "delta_pct": -100.0},
            {"model_id": "real", "field": "output_price",
             "old_value": 0.01, "new_value": 0.001, "delta_pct": -90.0},
        ]
        anomalies = build_anomalies_from_changes(changes, threshold_pct=50.0)
        assert {a.model_id for a in anomalies} == {"real"}


# ── Verifier behaviour ────────────────────────────────────────────────────────

class TestVerifierAvailability:
    def test_disabled_when_enabled_false(self):
        v = ClaudeAnomalyVerifier(api_key="sk-x", enabled=False)
        assert v.is_available is False

    def test_disabled_when_no_key(self):
        v = ClaudeAnomalyVerifier(api_key="", enabled=True)
        assert v.is_available is False

    def test_enabled_when_key_and_flag(self):
        v = ClaudeAnomalyVerifier(api_key="sk-x", enabled=True)
        assert v.is_available is True


def _anomaly(model_id="gpt-4o", field="input_price", delta=-80.0) -> Anomaly:
    return Anomaly(
        model_id=model_id,
        vendor="openai",
        field=field,
        old_value_per_1k=0.005,
        new_value_per_1k=0.001,
        delta_pct=delta,
    )


@pytest.mark.asyncio
async def test_disabled_verifier_accepts_all_without_api_call():
    v = ClaudeAnomalyVerifier(api_key="sk-x", enabled=False)
    result = await v.verify([_anomaly()])
    assert result.skipped is True
    assert len(result.accepted) == 1
    assert result.rejected == []


@pytest.mark.asyncio
async def test_empty_anomaly_list_returns_empty_result():
    v = ClaudeAnomalyVerifier(api_key="sk-x")
    result = await v.verify([])
    assert result.accepted == []
    assert result.rejected == []
    assert result.skipped is False


def _mock_anthropic_response(verdicts: list[dict]) -> MagicMock:
    """Build a fake AsyncAnthropic response object with the given verdicts."""
    import json as _json
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = _json.dumps({"verdicts": verdicts})
    response = MagicMock()
    response.content = [text_block]
    return response


@pytest.mark.asyncio
async def test_accepted_and_rejected_split_correctly():
    anomalies = [
        _anomaly(model_id="gpt-4o", field="input_price", delta=-80.0),
        _anomaly(model_id="claude-opus-4-7", field="output_price", delta=-95.0),
    ]
    fake_response = _mock_anthropic_response([
        {"model_id": "gpt-4o", "field": "input_price",
         "decision": "accept", "reasoning": "OpenAI cut prices in May 2026."},
        {"model_id": "claude-opus-4-7", "field": "output_price",
         "decision": "reject", "reasoning": "95% drop on a flagship model is implausible."},
    ])

    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        v = ClaudeAnomalyVerifier(api_key="sk-x")
        result = await v.verify(anomalies)

    assert [a.model_id for a in result.accepted] == ["gpt-4o"]
    assert [r.anomaly.model_id for r in result.rejected] == ["claude-opus-4-7"]
    assert "implausible" in result.rejected[0].reasoning


@pytest.mark.asyncio
async def test_missing_verdict_fails_open_accept():
    anomalies = [
        _anomaly(model_id="a", field="input_price"),
        _anomaly(model_id="b", field="input_price"),
    ]
    fake_response = _mock_anthropic_response([
        {"model_id": "a", "field": "input_price",
         "decision": "reject", "reasoning": "Implausible."},
        # "b" omitted entirely — fail-open accepts it
    ])

    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        v = ClaudeAnomalyVerifier(api_key="sk-x")
        result = await v.verify(anomalies)

    accepted_ids = {a.model_id for a in result.accepted}
    rejected_ids = {r.anomaly.model_id for r in result.rejected}
    assert accepted_ids == {"b"}
    assert rejected_ids == {"a"}


@pytest.mark.asyncio
async def test_api_error_fails_open():
    anomalies = [_anomaly()]
    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(side_effect=RuntimeError("network down"))
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        v = ClaudeAnomalyVerifier(api_key="sk-x")
        result = await v.verify(anomalies)

    assert result.skipped is True
    assert "network down" in result.skipped_reason
    assert len(result.accepted) == 1   # fail-open
    assert result.rejected == []


@pytest.mark.asyncio
async def test_malformed_json_fails_open():
    anomalies = [_anomaly()]
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "not-json{{{"
    fake_response = MagicMock()
    fake_response.content = [text_block]

    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        v = ClaudeAnomalyVerifier(api_key="sk-x")
        result = await v.verify(anomalies)

    assert result.skipped is True
    assert result.skipped_reason == "json_parse_failed"
    assert len(result.accepted) == 1


@pytest.mark.asyncio
async def test_uses_configured_model_and_caching():
    anomalies = [_anomaly()]
    fake_response = _mock_anthropic_response([
        {"model_id": "gpt-4o", "field": "input_price",
         "decision": "accept", "reasoning": "ok"},
    ])

    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        v = ClaudeAnomalyVerifier(api_key="sk-x", model="claude-opus-4-7")
        await v.verify(anomalies)

    call_kwargs = mock_messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-7"
    # System prompt is cache-controlled
    system = call_kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # Output is constrained by json_schema
    assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
