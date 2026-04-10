"""Pricing report subscriber management.

Delivery priority:
  1. Resend API  — if RESEND_API_KEY is set (recommended, no password needed)
  2. SMTP        — if TIDUS_SMTP_HOST is set (fallback)
  3. Dev mode    — prints to stdout (no credentials configured)

Subscriber list stored in config/subscribers.yaml.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import structlog
from dotenv import load_dotenv

from tidus.utils.yaml_loader import load_yaml

# Load .env so credentials work without setting system environment variables.
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

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
    return [
        Subscriber(
            email=e["email"],
            name=e.get("name", e["email"]),
            active=e.get("active", True),
            subscribed_at=e.get("subscribed_at"),
        )
        for e in raw.get("subscribers", [])
        if e.get("active", True)
    ]


def add_subscriber(email: str, name: str = "") -> None:
    """Add a subscriber to config/subscribers.yaml (idempotent)."""
    existing: list[dict] = []
    if _SUBSCRIBERS_FILE.exists():
        existing = load_yaml(str(_SUBSCRIBERS_FILE)).get("subscribers", [])

    for entry in existing:
        if entry.get("email") == email:
            if not entry.get("active", True):
                entry["active"] = True
                log.info("subscriber_reactivated", email=email)
                _write_subscribers(existing)
            else:
                log.info("subscriber_already_exists", email=email)
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
    import yaml  # type: ignore[import-untyped]
    _SUBSCRIBERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _SUBSCRIBERS_FILE.open("w", encoding="utf-8") as f:
        yaml.dump({"subscribers": entries}, f, default_flow_style=False, allow_unicode=True)


class ReportDelivery:
    """Delivers pricing reports via Resend API, SMTP, or stdout (dev mode).

    Provider selection (in order):
      1. Resend  — RESEND_API_KEY is set
      2. SMTP    — TIDUS_SMTP_HOST is set
      3. Dev     — neither configured; prints to stdout
    """

    def __init__(self) -> None:
        self._resend_key  = os.getenv("RESEND_API_KEY", "")
        self._smtp_host   = os.getenv("TIDUS_SMTP_HOST", "")
        self._smtp_port   = int(os.getenv("TIDUS_SMTP_PORT", "587"))
        self._smtp_user   = os.getenv("TIDUS_SMTP_USER", "")
        self._smtp_pass   = os.getenv("TIDUS_SMTP_PASS", "")
        self._from_addr   = os.getenv("TIDUS_SMTP_FROM", "Tidus Reports <onboarding@resend.dev>")

    @property
    def provider(self) -> str:
        if self._resend_key:
            return "resend"
        if self._smtp_host:
            return "smtp"
        return "dev"

    def deliver(
        self,
        report_markdown: str,
        subject: str,
        subscribers: list[Subscriber],
        report_html: str = "",
    ) -> int:
        """Deliver report to all active subscribers. Returns count delivered."""
        if not subscribers:
            log.info("report_delivery_skipped", reason="no active subscribers")
            return 0

        log.info("report_delivery_start", provider=self.provider,
                 recipients=len(subscribers))

        if self.provider == "dev":
            return self._deliver_dev(report_markdown, subject, subscribers)

        delivered = 0
        for subscriber in subscribers:
            try:
                if self.provider == "resend":
                    self._send_resend(subscriber, subject, report_markdown, report_html)
                else:
                    self._send_smtp(subscriber, subject, report_markdown, report_html)
                delivered += 1
                log.info("report_delivered", provider=self.provider,
                         email=subscriber.email)
            except Exception as exc:
                log.error("report_delivery_failed", provider=self.provider,
                          email=subscriber.email, error=str(exc))
        return delivered

    # ── Resend ────────────────────────────────────────────────────────────────

    def _send_resend(
        self, subscriber: Subscriber, subject: str, markdown_body: str, html_body: str = ""
    ) -> None:
        import resend  # type: ignore[import-untyped]
        resend.api_key = self._resend_key

        resend.Emails.send({
            "from": self._from_addr,
            "to": [subscriber.email],
            "subject": subject,
            "text": markdown_body,
            "html": html_body or self._to_html(markdown_body, subscriber.name),
        })

    # ── SMTP fallback ─────────────────────────────────────────────────────────

    def _send_smtp(
        self, subscriber: Subscriber, subject: str, markdown_body: str, html_body: str = ""
    ) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self._from_addr
        msg["To"]      = subscriber.email
        msg.attach(MIMEText(markdown_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body or self._to_html(markdown_body, subscriber.name), "html", "utf-8"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(self._smtp_user, self._smtp_pass)
            server.sendmail(self._from_addr, subscriber.email, msg.as_string())

    # ── Dev mode ──────────────────────────────────────────────────────────────

    def _deliver_dev(
        self, report_markdown: str, subject: str, subscribers: list[Subscriber]
    ) -> int:
        safe = report_markdown[:3000].encode("ascii", "replace").decode("ascii")
        truncated = "\n...[truncated — see reports/ directory for full report]" \
                    if len(report_markdown) > 3000 else ""
        print(f"\n{'='*72}")
        print(f"[DEV MODE — no email provider configured]")
        print(f"To:      {', '.join(s.email for s in subscribers)}")
        print(f"Subject: {subject}")
        print(f"{'='*72}")
        print(safe + truncated)
        print(f"{'='*72}")
        print(f"\nTo send real emails, add RESEND_API_KEY to .env")
        print(f"Get a free key at: https://resend.com (3,000 emails/month free)\n")
        return len(subscribers)

    # ── HTML renderer ─────────────────────────────────────────────────────────

    def _to_html(self, markdown: str, recipient_name: str) -> str:
        # Escape HTML special chars in the markdown body
        escaped = (markdown
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 820px; margin: 40px auto; padding: 0 24px; color: #1a1a2e;
    line-height: 1.6;
  }}
  .header {{
    background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
    color: white; padding: 24px 28px; border-radius: 8px; margin-bottom: 28px;
  }}
  .header h1 {{ margin: 0; font-size: 22px; }}
  .header p  {{ margin: 6px 0 0; font-size: 13px; opacity: 0.85; }}
  pre {{
    background: #f8f9fa; padding: 20px; border-radius: 8px;
    font-size: 12.5px; overflow-x: auto; white-space: pre-wrap;
    border: 1px solid #e9ecef; line-height: 1.5;
  }}
  .footer {{
    margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6;
    font-size: 12px; color: #868e96;
  }}
</style>
</head>
<body>
  <div class="header">
    <h1>&#x1F4CA; Tidus AI Model Latest Pricing Report</h1>
    <p>Your weekly snapshot of AI model pricing across all vendors.</p>
  </div>
  <p>Hi {recipient_name},</p>
  <pre>{escaped}</pre>
  <div class="footer">
    You&rsquo;re receiving this because you subscribed to Tidus AI pricing reports.<br>
    To unsubscribe, set <code>active: false</code> next to your entry in
    <code>config/subscribers.yaml</code>.
  </div>
</body>
</html>"""
