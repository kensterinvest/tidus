from types import SimpleNamespace

import pytest

from tidus.reporting.market_intelligence import render_cost_footer, render_market_intelligence
from tidus.sync.anthropic_client import SyncTokenLedger


def test_cost_footer_reports_spend():
    led = SyncTokenLedger()
    led.record(
        "discovery",
        SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        web_searches=2,
    )
    footer = render_cost_footer(led)
    assert "AI cost this issue" in footer
    assert "2 searches" in footer


@pytest.mark.asyncio
async def test_market_section_fails_open_without_client():
    md = await render_market_intelligence(
        client=None,
        ledger=SyncTokenLedger(),
        model="claude-sonnet-5",
        discoveries=[],
        price_moves=[],
    )
    assert isinstance(md, str)  # minimal fallback, never raises
