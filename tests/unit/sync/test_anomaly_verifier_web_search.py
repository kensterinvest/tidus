"""Unit tests for the web-search upgrade to ClaudeAnomalyVerifier (Task 6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tidus.sync.ai_verifier import Anomaly, ClaudeAnomalyVerifier
from tidus.sync.anthropic_client import SyncTokenLedger


class _Capture:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kw):
        self.kwargs = kw
        import json
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        {
                            "verdicts": [
                                {
                                    "model_id": "m",
                                    "field": "input_price",
                                    "decision": "accept",
                                    "reasoning": "ok",
                                }
                            ]
                        }
                    ),
                )
            ],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


@pytest.mark.asyncio
async def test_anomaly_verifier_passes_web_search_tool():
    cap = _Capture()
    v = ClaudeAnomalyVerifier(api_key="k", ledger=SyncTokenLedger(), use_web_search=True)
    v._client_override = SimpleNamespace(messages=cap)  # test seam
    a = Anomaly(
        model_id="m",
        vendor="acme",
        field="input_price",
        old_value_per_1k=0.01,
        new_value_per_1k=0.001,
        delta_pct=-90.0,
    )
    res = await v.verify([a])
    assert cap.kwargs is not None
    assert any(t.get("type", "").startswith("web_search") for t in cap.kwargs.get("tools", []))
    assert [x.model_id for x in res.accepted] == ["m"]


@pytest.mark.asyncio
async def test_anomaly_verifier_records_ledger_usage():
    cap = _Capture()
    ledger = SyncTokenLedger()
    v = ClaudeAnomalyVerifier(api_key="k", ledger=ledger, use_web_search=True)
    v._client_override = SimpleNamespace(messages=cap)
    a = Anomaly(
        model_id="m",
        vendor="acme",
        field="input_price",
        old_value_per_1k=0.01,
        new_value_per_1k=0.001,
        delta_pct=-90.0,
    )
    await v.verify([a])
    assert "anomaly" in ledger.stages
    assert ledger.stages["anomaly"].input_tokens == 1


@pytest.mark.asyncio
async def test_anomaly_verifier_omits_web_search_tool_when_disabled():
    cap = _Capture()
    v = ClaudeAnomalyVerifier(api_key="k", use_web_search=False)
    v._client_override = SimpleNamespace(messages=cap)
    a = Anomaly(
        model_id="m",
        vendor="acme",
        field="input_price",
        old_value_per_1k=0.01,
        new_value_per_1k=0.001,
        delta_pct=-90.0,
    )
    await v.verify([a])
    assert cap.kwargs is not None
    assert "tools" not in cap.kwargs
