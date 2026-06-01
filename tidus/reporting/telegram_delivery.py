"""Telegram delivery for the pricing-sync magazine.

Additive, env-gated, fail-open delivery channel that posts each magazine issue to
a dedicated Telegram bot/chat:

  1. ``sendMessage``  — a concise summary (new models + top price moves) in-chat.
  2. ``sendDocument`` — the full styled HTML report attached, so the giant price
     table doesn't have to fit in a 4096-char message.

Enabled only when both ``TIDUS_TELEGRAM_BOT_TOKEN`` and ``TIDUS_TELEGRAM_CHAT_ID``
are set. Any failure is logged and swallowed so the rest of the pipeline (GitHub
push, landing-page update) still completes — exactly like the email path.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import structlog
from dotenv import load_dotenv

# Load .env so credentials work without exporting system environment variables.
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

log = structlog.get_logger(__name__)

_API_BASE = "https://api.telegram.org"
_TELEGRAM_MSG_LIMIT = 4096
_MAX_MOVES = 12  # most-significant price moves to list in the chat summary
_TIMEOUT_S = 30.0


def build_summary(report) -> str:
    """Build a Telegram-friendly text summary from a PricingReport.

    Uses the report's structured fields (never the markdown table) so the chat
    message stays short and readable. Hard-capped at the Telegram message limit.
    """
    lines: list[str] = [
        f"\U0001F4CA Tidus Pricing Update — {report.report_date}",
        f"{report.total_models} models tracked",
        "",
    ]

    if report.new_models:
        lines.append(f"\U0001F195 New models ({len(report.new_models)}):")
        for m in report.new_models:
            lines.append(
                f"  • {m.model_id} ({m.vendor}) — "
                f"${m.input_usd_per_1m:g}/${m.output_usd_per_1m:g} per 1M"
            )
        lines.append("")

    if report.price_changes:
        lines.append(f"\U0001F4B8 Price changes ({len(report.price_changes)}):")
        for c in report.price_changes[:_MAX_MOVES]:
            arrow = "↓" if c.delta_pct < 0 else "↑"
            lines.append(
                f"  {arrow} {c.model_id} ({c.vendor}) {c.field} "
                f"${c.old_usd_per_1m:g}→${c.new_usd_per_1m:g} "
                f"({c.delta_pct:+.1f}%)"
            )
        if len(report.price_changes) > _MAX_MOVES:
            lines.append(f"  …and {len(report.price_changes) - _MAX_MOVES} more")
        lines.append("")
    else:
        lines.append("No price changes this issue.")
        lines.append("")

    lines.append("Full report attached ⬇️")

    summary = "\n".join(lines)
    if len(summary) > _TELEGRAM_MSG_LIMIT:
        cut = "\n…(truncated — see attached report)"
        summary = summary[: _TELEGRAM_MSG_LIMIT - len(cut)] + cut
    return summary


class TelegramDelivery:
    """Posts the magazine to a dedicated Telegram bot/chat. Env-gated, fail-open."""

    def __init__(self) -> None:
        self._token = os.getenv("TIDUS_TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.getenv("TIDUS_TELEGRAM_CHAT_ID", "")

    @property
    def enabled(self) -> bool:
        return bool(self._token) and bool(self._chat_id)

    def deliver(self, report, html_path: Path) -> bool:
        """Send the summary message + attach the full HTML report.

        Returns True on success. Never raises: a missing config skips quietly and
        any transport error is logged so the pipeline still finishes.
        """
        if not self.enabled:
            log.info("telegram_delivery_skipped", reason="not configured")
            return False

        try:
            base = f"{_API_BASE}/bot{self._token}"
            summary = build_summary(report)

            httpx.post(
                f"{base}/sendMessage",
                data={"chat_id": self._chat_id, "text": summary},
                timeout=_TIMEOUT_S,
            ).raise_for_status()

            html_path = Path(html_path)
            with html_path.open("rb") as fh:
                httpx.post(
                    f"{base}/sendDocument",
                    data={
                        "chat_id": self._chat_id,
                        "caption": f"Tidus magazine — {report.report_date}",
                    },
                    files={"document": (html_path.name, fh, "text/html")},
                    timeout=_TIMEOUT_S,
                ).raise_for_status()

            log.info("telegram_delivered", report_date=str(report.report_date))
            return True
        except Exception as exc:  # noqa: BLE001 — fail-open by design
            log.error("telegram_delivery_failed", error=str(exc))
            return False
