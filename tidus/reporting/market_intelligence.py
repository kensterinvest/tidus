"""Claude-authored 'Market Intelligence' magazine section + AI-cost footer.

Both entry points are fail-open: a missing client or a failed API call
falls back to a minimal factual section rather than breaking the magazine
build. `render_market_intelligence` does NOT construct its own Anthropic
client — the client is always injected by the caller (see Task 9 wiring).
"""
from __future__ import annotations

import html as html_module
import json
import re

import structlog

log = structlog.get_logger(__name__)

_SYSTEM = (
    "You write the 'Market Intelligence' section of the Tidus AI pricing "
    "magazine. Given this week's newly discovered models and notable price "
    "moves, write a concise, readable market brief (markdown): a spotlight per "
    "notable new model (purpose, positioning, price), then a short 'market "
    "movements' paragraph. Cite source URLs inline. No preamble."
)


def render_cost_footer(ledger) -> str:
    s = ledger.summary()
    return (
        f"\n\n---\n_🤖 AI cost this issue: ${s['estimated_usd']:.2f} "
        f"({s['web_searches']} searches, "
        f"{s['total_input_tokens'] + s['total_output_tokens']:,} tokens)_\n"
    )


async def render_market_intelligence(*, client, ledger, model, discoveries, price_moves) -> str:
    header = "## 🌐 Market Intelligence\n\n"
    if client is None:
        # Fail-open: minimal factual fallback, no narrative, no API call.
        if not discoveries:
            return header + "_No new models discovered this cycle._\n"
        lines = [
            f"- **{d.model_id}** ({d.vendor}) — {d.raw_metadata.get('purpose', '')}"
            for d in discoveries
        ]
        return header + "\n".join(lines) + "\n"

    facts = {
        "new_models": [
            {
                "model_id": d.model_id,
                "vendor": d.vendor,
                "purpose": d.raw_metadata.get("purpose", ""),
                "positioning": d.raw_metadata.get("positioning", ""),
                "input_usd_per_1m": round(d.raw_metadata.get("price_in_per_1k", 0) * 1000, 3),
                "output_usd_per_1m": round(d.raw_metadata.get("price_out_per_1k", 0) * 1000, 3),
                "sources": d.raw_metadata.get("sources", []),
            }
            for d in discoveries
        ],
        "price_moves": price_moves,
    }
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[
                {
                    "role": "user",
                    "content": "Write the section from these facts:\n" + json.dumps(facts, indent=2),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 — fail-open, any API error falls back
        log.warning("market_intelligence_failed", error=str(exc))
        return header + "_Market narrative unavailable this cycle._\n"
    ledger.record("magazine", getattr(resp, "usage", None))
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    return header + (text or "_Market narrative unavailable this cycle._\n")


_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def _inline_html(text: str) -> str:
    """Escape then apply the narrow set of inline markdown we render."""
    escaped = html_module.escape(text)
    escaped = _LINK_RE.sub(r'<a href="\2">\1</a>', escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    return escaped


def render_section_html(section_markdown: str) -> str:
    """Convert the narrow markdown subset the narrative uses into an HTML fragment.

    Handles: ## / ### headings, "- " bullets (grouped into <ul>), [text](url)
    links, **bold**, and blank-line-separated paragraphs. Everything else is
    HTML-escaped first, so no raw markup from the source can leak through.
    """
    if not section_markdown.strip():
        return ""

    lines = section_markdown.splitlines()
    out: list[str] = []
    list_open = False

    def close_list():
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph:
            out.append("<p>" + " ".join(_inline_html(p) for p in paragraph) + "</p>")
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            close_list()
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            close_list()
            out.append(f"<h3>{_inline_html(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            flush_paragraph()
            close_list()
            out.append(f"<h2>{_inline_html(stripped[3:])}</h2>")
        elif stripped.startswith("- "):
            flush_paragraph()
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{_inline_html(stripped[2:])}</li>")
        else:
            close_list()
            paragraph.append(stripped)

    flush_paragraph()
    close_list()
    return "\n".join(out)


def enrich_report_in_place(report, section_markdown: str, footer_markdown: str) -> None:
    """Append the market section + cost footer to both report.markdown and report.html.

    Mutates `report` so every downstream consumer (file write, email/Telegram
    delivery) sees the enriched content, rather than only the local variable
    the caller happens to write to disk.
    """
    report.markdown = report.markdown + section_markdown + footer_markdown

    fragment = render_section_html(section_markdown) + render_section_html(footer_markdown)
    if not fragment:
        return
    existing_html = report.html or ""
    if "</body>" in existing_html:
        report.html = existing_html.replace("</body>", fragment + "</body>", 1)
    else:
        report.html = existing_html + fragment
