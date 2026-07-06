import json
from types import SimpleNamespace

import pytest

from tidus.sync.anthropic_client import SyncTokenLedger
from tidus.sync.discovery.claude_market import ClaudeMarketDiscoverySource


class _FakeMessages:
    def __init__(self, payload, raise_exc=None):
        self._payload, self._raise = payload, raise_exc

    async def create(self, **kw):
        if self._raise:
            raise self._raise
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(self._payload))],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


class _FakeClient:
    def __init__(self, payload, raise_exc=None):
        self.messages = _FakeMessages(payload, raise_exc)


@pytest.mark.asyncio
async def test_parses_candidates_and_divides_price_to_per_1k():
    payload = {"models": [{
        "model_id": "acme-ultra-2", "vendor": "acme",
        "input_usd_per_1m": 5.0, "output_usd_per_1m": 20.0,
        "purpose": "frontier reasoning", "positioning": "flagship",
        "sources": ["https://acme.ai/pricing"],
    }]}
    src = ClaudeMarketDiscoverySource(
        client=_FakeClient(payload), ledger=SyncTokenLedger(),
        model="claude-sonnet-5", last_sync_date="2026-07-01")
    out = await src.list_models()
    assert len(out) == 1
    m = out[0]
    assert m.model_id == "acme-ultra-2"
    assert m.raw_metadata["claude_sourced"] is True
    assert m.raw_metadata["price_in_per_1k"] == 0.005
    assert m.raw_metadata["price_out_per_1k"] == 0.020
    assert m.raw_metadata["sources"] == ["https://acme.ai/pricing"]


@pytest.mark.asyncio
async def test_unavailable_when_no_client():
    src = ClaudeMarketDiscoverySource(client=None, ledger=SyncTokenLedger(),
                                      model="claude-sonnet-5", last_sync_date="2026-07-01")
    assert src.is_available is False
    assert await src.list_models() == []


@pytest.mark.asyncio
async def test_fail_open_on_api_error():
    src = ClaudeMarketDiscoverySource(
        client=_FakeClient({}, raise_exc=RuntimeError("boom")),
        ledger=SyncTokenLedger(), model="claude-sonnet-5", last_sync_date="2026-07-01")
    assert await src.list_models() == []
