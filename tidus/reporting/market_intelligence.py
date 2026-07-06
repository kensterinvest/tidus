"""Claude-authored 'Market Intelligence' magazine section + AI-cost footer.

Both entry points are fail-open: a missing client or a failed API call
falls back to a minimal factual section rather than breaking the magazine
build. `render_market_intelligence` does NOT construct its own Anthropic
client — the client is always injected by the caller (see Task 9 wiring).
"""
from __future__ import annotations

import json

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
