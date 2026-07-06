import json
from types import SimpleNamespace

import pytest

from tidus.sync.ai_verifier import ClaudeMarketPriceVerifier, MarketCandidate
from tidus.sync.anthropic_client import SyncTokenLedger


class _FakeMsgs:
    def __init__(self, payload):
        self._p = payload

    async def create(self, **kw):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self._p))],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


class _FakeClient:
    def __init__(self, payload):
        self.messages = _FakeMsgs(payload)


def _c(mid):
    return MarketCandidate(
        model_id=mid,
        vendor="acme",
        input_price_per_1k=0.005,
        output_price_per_1k=0.02,
        sources=["https://acme.ai"],
    )


@pytest.mark.asyncio
async def test_accepts_and_rejects_by_verdict():
    payload = {
        "verdicts": [
            {"model_id": "good", "decision": "accept", "reasoning": "matches page"},
            {"model_id": "bad", "decision": "reject", "reasoning": "no such price"},
        ]
    }
    v = ClaudeMarketPriceVerifier(
        client=_FakeClient(payload), ledger=SyncTokenLedger(), model="claude-sonnet-5"
    )
    res = await v.verify([_c("good"), _c("bad")])
    assert [c.model_id for c in res.accepted] == ["good"]
    assert res.rejected[0][0].model_id == "bad"


@pytest.mark.asyncio
async def test_skips_when_no_client():
    v = ClaudeMarketPriceVerifier(client=None, ledger=SyncTokenLedger(), model="claude-sonnet-5")
    res = await v.verify([_c("x")])
    assert res.skipped is True and [c.model_id for c in res.accepted] == ["x"]
