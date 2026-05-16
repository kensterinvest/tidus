"""Unit tests for the AI-verifier section in the pricing report."""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as _date

from tidus.reporting.pricing_report import (
    PricingReport,
    PricingReportGenerator,
)


def _empty_report(ai_rejected: list[dict] | None = None) -> PricingReport:
    """Build a PricingReport with everything else empty; lets us assert
    purely on the ai_rejected rendering path."""
    return PricingReport(
        generated_at=datetime.now(UTC),
        report_date=_date.today(),
        current_revision_id="abcdef12-0000",
        base_revision_id=None,
        new_models=[],
        price_changes=[],
        stale_models=[],
        total_models=0,
        ai_rejected=list(ai_rejected or []),
    )


def _render(report: PricingReport) -> str:
    """Call the markdown renderer directly. PricingReportGenerator doesn't
    need a session for this code path — _render_markdown only reads its args."""
    gen = PricingReportGenerator.__new__(PricingReportGenerator)
    return gen._render_markdown(report, specs={})


class TestAIRejectedSection:
    def test_section_omitted_when_no_rejections(self):
        md = _render(_empty_report(ai_rejected=[]))
        assert "AI Verifier" not in md
        assert "🤖" not in md

    def test_section_renders_each_rejection_as_a_row(self):
        rejected = [
            {
                "model_id":  "gpt-4o",
                "field":     "input_price",
                "delta_pct": -92.5,
                "reasoning": "OpenAI hasn't announced a price cut of this magnitude.",
            },
            {
                "model_id":  "claude-opus-4-7",
                "field":     "output_price",
                "delta_pct": 200.0,
                "reasoning": "A 200% jump on a flagship model would have been newsworthy; no source confirms it.",
            },
        ]
        md = _render(_empty_report(ai_rejected=rejected))

        assert "## 🤖 AI Verifier" in md
        assert "Claude rejected **2** anomalous" in md
        assert "`gpt-4o`" in md
        assert "input_price" in md
        assert "-92.5%" in md
        assert "OpenAI hasn't announced" in md
        assert "`claude-opus-4-7`" in md
        assert "+200.0%" in md

    def test_pipe_in_reasoning_is_escaped_for_markdown_table(self):
        """A naive renderer would break the table when reasoning contains '|'."""
        rejected = [
            {
                "model_id":  "test-model",
                "field":     "input_price",
                "delta_pct": -75.0,
                "reasoning": "Possible confusion with mistral-large | Together-hosted variant.",
            },
        ]
        md = _render(_empty_report(ai_rejected=rejected))
        # The literal `|` inside the reasoning must be escaped so it doesn't
        # close the markdown table cell early.
        assert "mistral-large \\| Together-hosted" in md

    def test_section_position_above_stale_models(self):
        rejected = [{
            "model_id":  "x",
            "field":     "input_price",
            "delta_pct": -60.0,
            "reasoning": "test",
        }]
        report = _empty_report(ai_rejected=rejected)
        report.stale_models = ["some-stale-model"]
        md = _render(report)
        ai_idx = md.find("AI Verifier")
        stale_idx = md.find("Stale Pricing")
        assert ai_idx != -1 and stale_idx != -1
        assert ai_idx < stale_idx, "AI Verifier section must come before Stale Pricing"


class TestPipelineResultPlumbing:
    def test_pricing_report_default_is_empty_list_not_none(self):
        """ai_rejected must default to a list so renderers can call len() safely."""
        report = _empty_report()
        assert report.ai_rejected == []
        assert isinstance(report.ai_rejected, list)
