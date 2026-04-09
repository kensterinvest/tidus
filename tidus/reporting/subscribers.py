"""Pricing report subscriber management.

Subscribers receive the Tidus AI Model Latest Pricing Report after each weekly sync.
Delivery is currently via email (SMTP) or stdout (dev mode when SMTP not configured).

Subscriber list is stored in config/subscribers.yaml:
    subscribers:
      - email: user@example.com
        name: Kenny Wong
        active: true
        subscribed_at: 2026-04-09
"""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import structlog

from tidus.utils.yaml_loader import load_yaml

log = structlog.get_logger(__name__)

_SUBSCRIBERS_FILE = Path("config/subscribers.yaml")


@dataclass
class Subscriber:
    email: str
    name: str
    active: bool = True
    subscribed_at: date | None = None


def load_subscribers() -> list[Subscriber]:
    """Load active subscribers from config/subscribers.yaml."""
    if not _SUBSCRIBERS_FILE.exists():
        return []
    raw = load_yaml(str(_SUBSCRIBERS_FILE))
    result = []
    for entry in raw.get("subscribers", []):
        if entry.get("active", True):
            result.append(Subscriber(
                email=entry["email"],
                name=entry.get("name", entry["email"]),
                active=entry.get("active", True),
                subscribed_at=entry.get("subscribed_at"),
            ))
    return result


def add_subscriber(email: str, name: str = "") -> None:
    """Add a subscriber to config/subscribers.yaml (idempotent)."""
    existing = []
    if _SUBSCRIBERS_FILE.exists():
        raw = load_yaml(str(_SUBSCRIBERS_FILE))
        existing = raw.get("subscribers", [])

    # Check if already subscribed
    for entry in existing:
        if entry.get("email") == email:
            if not entry.get("active", True):
                entry["active"] = True
                log.info("subscriber_reactivated", email=email)
            else:
                log.info("subscriber_already_exists", email=email)
            _write_subscribers(existing)
            return

    existing.append({
        "email": email,
        "name": name or email.split("@")[0],
        "active": True,
        "subscribed_at": str(date.today()),
    })
    _write_subscribers(existing)
    log.info("subscriber_added", email=email)


def _write_subscribers(entries: list[dict]) -> None:
    _SUBSCRIBERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    import yaml  # type: ignore[import-untyped]
    with _SUBSCRIBERS_FILE.open("w", encoding="utf-8") as f:
        yaml.dump({"subscribers": entries}, f, default_flow_style=False, allow_unicode=True)


class ReportDelivery:
    """Delivers pricing reports to subscribers via email or stdout."""

    def __init__(self) -> None:
        import os
        self._smtp_host = os.getenv("TIDUS_SMTP_HOST", "")
        self._smtp_port = int(os.getenv("TIDUS_SMTP_PORT", "587"))
        self._smtp_user = os.getenv("TIDUS_SMTP_USER", "")
        self._smtp_pass = os.getenv("TIDUS_SMTP_PASS", "")
        self._from_addr = os.getenv("TIDUS_SMTP_FROM", "tidus-reports@noreply.com")

    def deliver(self, report_markdown: str, subject: str, subscribers: list[Subscriber]) -> int:
        """Deliver report to all active subscribers. Returns count delivered."""
        if not subscribers:
            log.info("report_delivery_skipped", reason="no active subscribers")
            return 0

        if not self._smtp_host:
            # Dev mode: log to stdout
            log.info(
                "report_delivery_dev_mode",
                recipients=[s.email for s in subscribers],
                subject=subject,
            )
            safe_body = report_markdown[:2000].encode("ascii", "replace").decode("ascii")
            truncated = "...[truncated]" if len(report_markdown) > 2000 else ""
            print(f"\n{'='*72}")
            print(f"[DEV MODE] Would send to: {', '.join(s.email for s in subscribers)}")
            print(f"Subject: {subject}")
            print(f"{'='*72}")
            print(safe_body + truncated)
            print(f"{'='*72}\n")
            return len(subscribers)

        delivered = 0
        for subscriber in subscribers:
            try:
                self._send_email(subscriber, subject, report_markdown)
                delivered += 1
                log.info("report_delivered", email=subscriber.email)
            except Exception as exc:
                log.error("report_delivery_failed", email=subscriber.email, error=str(exc))

        return delivered

    def _send_email(
        self, subscriber: Subscriber, subject: str, markdown_body: str
    ) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from_addr
        msg["To"] = subscriber.email

        # Plain text version
        text_part = MIMEText(markdown_body, "plain", "utf-8")
        # Simple HTML wrapper for markdown
        html_body = self._markdown_to_html(markdown_body, subscriber.name)
        html_part = MIMEText(html_body, "html", "utf-8")

        msg.attach(text_part)
        msg.attach(html_part)

        context = ssl.create_default_context()
        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(self._smtp_user, self._smtp_pass)
            server.sendmail(self._from_addr, subscriber.email, msg.as_string())

    def _markdown_to_html(self, markdown: str, recipient_name: str) -> str:
        """Minimal markdown → HTML for email. Uses pre-formatted block as fallback."""
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; }}
  h1 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 10px; }}
  h2 {{ color: #0f3460; margin-top: 30px; }}
  h3 {{ color: #533483; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th {{ background: #0f3460; color: white; padding: 8px 12px; text-align: left; }}
  td {{ border: 1px solid #ddd; padding: 6px 12px; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
  blockquote {{ border-left: 4px solid #0f3460; margin: 10px 0; padding: 8px 16px;
               background: #f0f4ff; color: #555; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;
             font-size: 12px; color: #888; }}
</style>
</head>
<body>
<p>Hi {recipient_name},</p>
<p>Your weekly AI pricing report from Tidus is ready.</p>
<pre style="background:#f8f9fa;padding:20px;border-radius:6px;
            font-size:12px;overflow-x:auto;white-space:pre-wrap;">{markdown}</pre>
<div class="footer">
  <p>You're receiving this because you subscribed to Tidus AI pricing reports.<br>
  To unsubscribe, remove your entry from <code>config/subscribers.yaml</code>.</p>
</div>
</body>
</html>"""
