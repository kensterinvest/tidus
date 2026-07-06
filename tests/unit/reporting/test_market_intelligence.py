from types import SimpleNamespace

import pytest

from tidus.reporting.market_intelligence import (
    enrich_report_in_place,
    render_cost_footer,
    render_market_intelligence,
    render_section_html,
)
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


def test_render_section_html_handles_headings_bullets_and_links():
    section = (
        "## 🌐 Market Intelligence\n\n"
        "### Spotlight\n\n"
        "- **NewModel** — cheap and fast, see [source](http://y)\n"
    )
    out = render_section_html(section)
    assert "<h2>" in out
    assert "<h3>" in out
    assert "<li>" in out
    assert '<a href="http://y">source</a>' in out
    assert "<strong>NewModel</strong>" in out


def test_render_section_html_escapes_raw_markup():
    out = render_section_html("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_section_html_empty_input():
    assert render_section_html("") == ""
    assert render_section_html("   \n\n") == ""


def test_enrich_report_in_place_updates_markdown_and_html():
    report = SimpleNamespace(markdown="# R\n", html="<html><body>hi</body></html>")
    section = "## 🌐 Market Intelligence\n\nSome narrative.\n"
    footer = "\n\n---\n_AI cost this issue: $0.01_\n"

    enrich_report_in_place(report, section, footer)

    assert "## 🌐 Market Intelligence" in report.markdown
    assert "AI cost this issue" in report.markdown

    assert "Market Intelligence" in report.html
    body_close_idx = report.html.index("</body>")
    section_idx = report.html.index("Market Intelligence")
    assert section_idx < body_close_idx
