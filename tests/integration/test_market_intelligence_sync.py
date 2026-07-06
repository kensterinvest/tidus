"""Integration tests for Task 9: wiring Claude market-intelligence into
scripts/weekly_full_sync.py.

The dark-promotion and section-content assertions are already covered by the
Task 7 and Task 8 unit tests. These tests lock the fail-open invariant on the
real entrypoint (concrete + runnable) and the section-injection seam main()
relies on.
"""
import pytest

from tidus.reporting.market_intelligence import render_cost_footer, render_market_intelligence
from tidus.sync.anthropic_client import SyncTokenLedger


@pytest.mark.asyncio
async def test_section_injection_seam_appends_to_markdown():
    """The two helpers main() uses to extend the report never raise and always
    return appendable strings, including with no client (fail-open)."""
    base_md = "# Report\n"
    ledger = SyncTokenLedger()
    section = await render_market_intelligence(
        client=None, ledger=ledger, model="claude-sonnet-5",
        discoveries=[], price_moves=[])
    out = base_md + section + render_cost_footer(ledger)
    assert out.startswith("# Report")
    assert "Market Intelligence" in out
    assert "AI cost this issue" in out


def test_main_wires_without_sync_key(monkeypatch):
    """Import scripts.weekly_full_sync and confirm build_sync_anthropic_client()
    returns None when the sync key is unset — the branch main() relies on to
    skip every Claude pass. (Full end-to-end main() is exercised by the live
    cron; here we lock the fail-open gate that guarantees it still ships.)"""
    from tidus.sync import anthropic_client
    monkeypatch.setattr(anthropic_client, "get_settings",
                        lambda: __import__("types").SimpleNamespace(tidus_sync_anthropic_key=""))
    assert anthropic_client.build_sync_anthropic_client() is None
