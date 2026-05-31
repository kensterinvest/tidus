"""Unit tests for Telegram magazine delivery.

The pricing-sync pipeline delivers each issue to a dedicated Telegram bot: a
concise summary in-chat plus the full styled report attached as a document.
Delivery is env-gated and fail-open, mirroring the email ReportDelivery path.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from tidus.reporting import telegram_delivery
from tidus.reporting.telegram_delivery import TelegramDelivery, build_summary

_TOKEN = "123456:ABC-DEF1234567890"
_CHAT = "987654321"


def _new_model(model_id: str, vendor: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        vendor=vendor,
        display_name=f"[auto-promoted] {vendor}: {model_id}",
        tier=3,
        input_usd_per_1m=1.0,
        output_usd_per_1m=2.0,
        max_context_k=128,
        capabilities=["chat"],
    )


def _change(model_id: str, vendor: str, delta_pct: float) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        vendor=vendor,
        display_name=model_id,
        field="input_price",
        old_usd_per_1m=10.0,
        new_usd_per_1m=10.0 * (1 + delta_pct / 100),
        delta_pct=delta_pct,
    )


def _report(new_models=None, price_changes=None) -> SimpleNamespace:
    return SimpleNamespace(
        report_date=date(2026, 5, 31),
        total_models=217,
        new_models=new_models or [],
        price_changes=price_changes or [],
        markdown="# magazine\n## 📋 Full Current Price Table\n| ... |",
        html="<html>magazine</html>",
    )


# ── enabled gating ──────────────────────────────────────────────────────────


class TestEnabled:
    def test_disabled_when_token_missing(self, monkeypatch):
        monkeypatch.delenv("TIDUS_TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("TIDUS_TELEGRAM_CHAT_ID", _CHAT)
        assert TelegramDelivery().enabled is False

    def test_disabled_when_chat_missing(self, monkeypatch):
        monkeypatch.setenv("TIDUS_TELEGRAM_BOT_TOKEN", _TOKEN)
        monkeypatch.delenv("TIDUS_TELEGRAM_CHAT_ID", raising=False)
        assert TelegramDelivery().enabled is False

    def test_enabled_when_both_present(self, monkeypatch):
        monkeypatch.setenv("TIDUS_TELEGRAM_BOT_TOKEN", _TOKEN)
        monkeypatch.setenv("TIDUS_TELEGRAM_CHAT_ID", _CHAT)
        assert TelegramDelivery().enabled is True


# ── summary builder ─────────────────────────────────────────────────────────


class TestBuildSummary:
    def test_includes_new_models_and_price_moves(self):
        report = _report(
            new_models=[_new_model("gpt-5.5", "openai")],
            price_changes=[_change("qwen3.7-max", "qwen", -50.0)],
        )
        summary = build_summary(report)
        assert "gpt-5.5" in summary
        assert "qwen3.7-max" in summary
        assert "50" in summary  # the percentage move surfaces

    def test_omits_full_price_table(self):
        report = _report(price_changes=[_change("deepseek-v3", "deepseek", -28.5)])
        summary = build_summary(report)
        assert "Full Current Price Table" not in summary

    def test_capped_at_telegram_limit(self):
        # 500 price changes must still fit a single Telegram message.
        changes = [_change(f"model-{i}", "openai", -float(i % 90 + 1)) for i in range(500)]
        summary = build_summary(_report(price_changes=changes))
        assert len(summary) <= 4096

    def test_handles_quiet_week_with_no_changes(self):
        summary = build_summary(_report())
        assert summary  # non-empty
        assert "217" in summary  # still reports total models tracked


# ── delivery ────────────────────────────────────────────────────────────────


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None


class TestDeliver:
    def test_sends_message_then_document(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TIDUS_TELEGRAM_BOT_TOKEN", _TOKEN)
        monkeypatch.setenv("TIDUS_TELEGRAM_CHAT_ID", _CHAT)
        html = tmp_path / "pricing-2026-05-31.html"
        html.write_text("<html>magazine</html>", encoding="utf-8")

        calls: list[tuple[str, dict]] = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs))
            return _FakeResponse()

        monkeypatch.setattr(telegram_delivery.httpx, "post", fake_post)

        report = _report(new_models=[_new_model("gpt-5.5", "openai")])
        ok = TelegramDelivery().deliver(report=report, html_path=html)

        assert ok is True
        assert len(calls) == 2
        assert calls[0][0].endswith("/sendMessage")
        assert calls[0][1]["data"]["chat_id"] == _CHAT
        assert "text" in calls[0][1]["data"]
        assert calls[1][0].endswith("/sendDocument")
        assert calls[1][1]["data"]["chat_id"] == _CHAT
        assert "files" in calls[1][1]

    def test_fail_open_on_http_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TIDUS_TELEGRAM_BOT_TOKEN", _TOKEN)
        monkeypatch.setenv("TIDUS_TELEGRAM_CHAT_ID", _CHAT)
        html = tmp_path / "pricing-2026-05-31.html"
        html.write_text("<html>magazine</html>", encoding="utf-8")

        def boom(url, **kwargs):
            raise RuntimeError("telegram unreachable")

        monkeypatch.setattr(telegram_delivery.httpx, "post", boom)

        # Must NOT raise — fail-open so the pipeline still finishes.
        ok = TelegramDelivery().deliver(report=_report(), html_path=html)
        assert ok is False

    def test_skips_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TIDUS_TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TIDUS_TELEGRAM_CHAT_ID", raising=False)
        html = tmp_path / "pricing-2026-05-31.html"
        html.write_text("<html>magazine</html>", encoding="utf-8")

        called = False

        def fake_post(url, **kwargs):
            nonlocal called
            called = True
            return _FakeResponse()

        monkeypatch.setattr(telegram_delivery.httpx, "post", fake_post)

        ok = TelegramDelivery().deliver(report=_report(), html_path=html)
        assert ok is False
        assert called is False
