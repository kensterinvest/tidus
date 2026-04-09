"""Pricing report generator — produces Tidus AI Model Latest Pricing Report.

Generates a markdown report comparing two registry revisions, highlighting:
  - New models added to the catalog
  - Price increases and decreases with context
  - Models with stale or missing prices
  - Market narrative summary

Usage:
    generator = PricingReportGenerator(session_factory)
    report = await generator.generate(revision_id=current, base_revision_id=previous)
    print(report.markdown)
    print(report.github_release_body)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import structlog

from tidus.models.model_registry import ModelSpec

log = structlog.get_logger(__name__)

# Thresholds for classifying price moves
_BIG_INCREASE_PCT   = 50.0   # ≥50%  increase → major move
_INCREASE_PCT       = 5.0    # ≥5%   increase → notable
_BIG_DECREASE_PCT   = 50.0   # ≥50%  decrease → major move
_DECREASE_PCT       = 5.0    # ≥5%   decrease → notable

# Vendor display names for narrative
_VENDOR_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai":    "OpenAI",
    "google":    "Google",
    "deepseek":  "DeepSeek",
    "xai":       "xAI",
    "alibaba":   "Alibaba (Qwen)",
    "mistral":   "Mistral AI",
    "perplexity": "Perplexity",
    "ollama":    "Local (Ollama)",
    "moonshot":  "Moonshot (Kimi)",
}


@dataclass
class PriceChange:
    model_id: str
    vendor: str
    display_name: str
    field: str            # input_price | output_price
    old_usd_per_1m: float
    new_usd_per_1m: float
    delta_pct: float      # positive = increase, negative = decrease

    @property
    def direction(self) -> str:
        return "UP" if self.delta_pct > 0 else "DOWN"

    @property
    def abs_pct(self) -> float:
        return abs(self.delta_pct)

    @property
    def emoji(self) -> str:
        if self.delta_pct > 0:
            return "📈" if self.abs_pct >= _BIG_INCREASE_PCT else "↑"
        return "📉" if self.abs_pct >= _BIG_DECREASE_PCT else "↓"


@dataclass
class NewModel:
    model_id: str
    vendor: str
    display_name: str
    tier: int
    input_usd_per_1m: float
    output_usd_per_1m: float
    max_context_k: int
    capabilities: list[str]


@dataclass
class PricingReport:
    generated_at: datetime
    report_date: date
    current_revision_id: str
    base_revision_id: str | None
    new_models: list[NewModel]
    price_changes: list[PriceChange]    # sorted: biggest moves first
    stale_models: list[str]             # model_ids with no hardcoded source entry
    total_models: int
    markdown: str = field(default="", repr=False)
    github_release_body: str = field(default="", repr=False)


class PricingReportGenerator:
    """Generates Tidus AI Model Latest Pricing Report from registry revisions."""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    async def generate(
        self,
        revision_id: str | None = None,    # defaults to current ACTIVE
        base_revision_id: str | None = None,  # defaults to previous (most recent SUPERSEDED)
    ) -> PricingReport:
        """Generate a pricing report comparing two revisions."""
        from tidus.db.repositories.registry_repo import (
            get_active_revision,
            get_entries_for_revision,
        )
        from sqlalchemy import text

        async with self._sf() as session:
            # Resolve current revision
            if revision_id is None:
                rev = await get_active_revision(self._sf)
                if rev is None:
                    raise ValueError("No active revision found")
                revision_id = rev.revision_id

            # Resolve base revision (most recent SUPERSEDED)
            if base_revision_id is None:
                result = await session.execute(text("""
                    SELECT revision_id FROM model_catalog_revisions
                    WHERE status = 'superseded'
                    ORDER BY activated_at DESC
                    LIMIT 1
                """))
                row = result.fetchone()
                base_revision_id = row[0] if row else None

            # Load current entries
            current_entries = await get_entries_for_revision(self._sf, revision_id)
            current_specs: dict[str, ModelSpec] = {}
            for e in current_entries:
                try:
                    current_specs[e.model_id] = ModelSpec.model_validate(e.spec_json)
                except Exception:
                    continue

            # Load base entries
            base_specs: dict[str, ModelSpec] = {}
            if base_revision_id:
                base_entries = await get_entries_for_revision(self._sf, base_revision_id)
                for e in base_entries:
                    try:
                        base_specs[e.model_id] = ModelSpec.model_validate(e.spec_json)
                    except Exception:
                        continue

        # Find new models
        new_models = self._find_new_models(current_specs, base_specs)

        # Find price changes
        price_changes = self._find_price_changes(current_specs, base_specs)

        # Find stale models (no price check in last 14 days)
        cutoff = date.today()
        stale = [
            m for m, s in current_specs.items()
            if not s.is_local and s.last_price_check
            and (cutoff - s.last_price_check).days > 14
        ]

        report = PricingReport(
            generated_at=datetime.now(UTC),
            report_date=date.today(),
            current_revision_id=revision_id,
            base_revision_id=base_revision_id,
            new_models=new_models,
            price_changes=price_changes,
            stale_models=stale,
            total_models=len(current_specs),
        )
        report.markdown = self._render_markdown(report, current_specs)
        report.github_release_body = self._render_github_release(report)
        return report

    # ── Diff helpers ──────────────────────────────────────────────────────────

    def _find_new_models(
        self,
        current: dict[str, ModelSpec],
        base: dict[str, ModelSpec],
    ) -> list[NewModel]:
        new_ids = set(current) - set(base)
        result = []
        for model_id in sorted(new_ids):
            s = current[model_id]
            if s.is_local:
                continue   # local model additions are routine infra; skip in report
            result.append(NewModel(
                model_id=model_id,
                vendor=s.vendor,
                display_name=s.display_name or model_id,
                tier=s.tier,
                input_usd_per_1m=round(s.input_price * 1000, 4),
                output_usd_per_1m=round(s.output_price * 1000, 4),
                max_context_k=s.max_context // 1000,
                capabilities=[c.value for c in s.capabilities],
            ))
        return result

    def _find_price_changes(
        self,
        current: dict[str, ModelSpec],
        base: dict[str, ModelSpec],
    ) -> list[PriceChange]:
        changes: list[PriceChange] = []
        for model_id, new_spec in current.items():
            if new_spec.is_local:
                continue
            old_spec = base.get(model_id)
            if old_spec is None:
                continue
            for field_name, new_val, old_val in [
                ("input_price",  new_spec.input_price,  old_spec.input_price),
                ("output_price", new_spec.output_price, old_spec.output_price),
            ]:
                if old_val == 0 and new_val == 0:
                    continue
                ref = old_val if old_val != 0 else new_val
                delta_pct = (new_val - old_val) / ref * 100
                if abs(delta_pct) < _DECREASE_PCT:
                    continue
                changes.append(PriceChange(
                    model_id=model_id,
                    vendor=new_spec.vendor,
                    display_name=new_spec.display_name or model_id,
                    field=field_name,
                    old_usd_per_1m=round(old_val * 1000, 4),
                    new_usd_per_1m=round(new_val * 1000, 4),
                    delta_pct=round(delta_pct, 2),
                ))
        # Sort: biggest absolute move first
        return sorted(changes, key=lambda c: -c.abs_pct)

    # ── Renderers ─────────────────────────────────────────────────────────────

    def _render_markdown(
        self, report: PricingReport, specs: dict[str, ModelSpec]
    ) -> str:
        lines: list[str] = []

        # Header
        lines += [
            f"# Tidus AI Model Latest Pricing Report",
            f"",
            f"**Report Date:** {report.report_date}  ",
            f"**Active Revision:** `{report.current_revision_id[:8]}…`  ",
            f"**Models Tracked:** {report.total_models}  ",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
            f"---",
            f"",
        ]

        # Executive Summary
        increases = [c for c in report.price_changes if c.delta_pct > 0]
        decreases = [c for c in report.price_changes if c.delta_pct < 0]
        big_moves = [c for c in report.price_changes if c.abs_pct >= _BIG_INCREASE_PCT]

        lines += [
            f"## Executive Summary",
            f"",
            f"This week in AI pricing: **{len(report.price_changes)} price changes** across "
            f"**{len({c.model_id for c in report.price_changes})} models**.",
        ]

        if report.new_models:
            lines.append(
                f"**{len(report.new_models)} new model(s)** added to the catalog."
            )

        if increases:
            vendors_up = ", ".join(sorted({_VENDOR_NAMES.get(c.vendor, c.vendor) for c in increases}))
            lines.append(f"- **Increases ({len(increases)} changes):** {vendors_up}")
        if decreases:
            vendors_down = ", ".join(sorted({_VENDOR_NAMES.get(c.vendor, c.vendor) for c in decreases}))
            lines.append(f"- **Decreases ({len(decreases)} changes):** {vendors_down}")
        if big_moves:
            lines.append(
                f"- **Major moves (≥50%):** {', '.join(sorted({c.model_id for c in big_moves}))}"
            )

        lines += ["", "---", ""]

        # New Models section
        if report.new_models:
            lines += ["## 🆕 New Models Added", ""]
            lines += [
                f"| Model | Vendor | Tier | Input $/1M | Output $/1M | Context | Capabilities |",
                f"|---|---|---|---|---|---|---|",
            ]
            for m in report.new_models:
                caps = ", ".join(m.capabilities)
                lines.append(
                    f"| `{m.model_id}` | {_VENDOR_NAMES.get(m.vendor, m.vendor)} | "
                    f"Tier {m.tier} | ${m.input_usd_per_1m:.3f} | ${m.output_usd_per_1m:.3f} | "
                    f"{m.max_context_k}K | {caps} |"
                )
            lines += [""]

        # Price changes by vendor group
        if report.price_changes:
            lines += ["## 📊 Price Changes", ""]
            vendors_changed = sorted({c.vendor for c in report.price_changes})
            for vendor in vendors_changed:
                vendor_changes = [c for c in report.price_changes if c.vendor == vendor]
                vendor_name = _VENDOR_NAMES.get(vendor, vendor)
                # Group by model to generate narrative
                models_changed = sorted({c.model_id for c in vendor_changes})
                lines += [f"### {vendor_name}", ""]
                lines += [
                    f"| Model | Field | Old $/1M | New $/1M | Change |",
                    f"|---|---|---|---|---|",
                ]
                for change in vendor_changes:
                    field_label = "Input" if change.field == "input_price" else "Output"
                    arrow = change.emoji
                    sign = "+" if change.delta_pct > 0 else ""
                    lines.append(
                        f"| `{change.model_id}` | {field_label} | "
                        f"${change.old_usd_per_1m:.3f} | ${change.new_usd_per_1m:.3f} | "
                        f"{arrow} {sign}{change.delta_pct:.1f}% |"
                    )
                lines += [""]

                # Narrative for this vendor's changes
                narrative = self._generate_vendor_narrative(vendor, vendor_changes, specs)
                if narrative:
                    lines += [f"> {narrative}", ""]

        # Current full price table
        lines += [
            "## 📋 Full Current Price Table",
            "",
            f"*Prices as of {report.report_date} — $/1M tokens*",
            "",
        ]
        # Group by vendor
        by_vendor: dict[str, list[ModelSpec]] = {}
        for spec in sorted(specs.values(), key=lambda s: (s.vendor, s.model_id)):
            if spec.is_local:
                continue
            by_vendor.setdefault(spec.vendor, []).append(spec)

        for vendor, vendor_specs in sorted(by_vendor.items()):
            vendor_name = _VENDOR_NAMES.get(vendor, vendor)
            lines += [f"### {vendor_name}", ""]
            lines += [
                "| Model | Tier | Input $/1M | Output $/1M | Cache Read $/1M | Max Context |",
                "|---|---|---|---|---|---|",
            ]
            for s in vendor_specs:
                cache = f"${s.cache_read_price * 1000:.3f}" if s.cache_read_price > 0 else "—"
                ctx = f"{s.max_context // 1000}K"
                lines.append(
                    f"| `{s.model_id}` | Tier {s.tier} | "
                    f"${s.input_price * 1000:.3f} | ${s.output_price * 1000:.3f} | "
                    f"{cache} | {ctx} |"
                )
            lines += [""]

        # Stale models warning
        if report.stale_models:
            lines += [
                "## ⚠️ Models With Stale Pricing",
                "",
                "The following models have not had prices verified in the last 14 days. "
                "Prices shown are from the last known source.",
                "",
            ]
            for m in sorted(report.stale_models):
                s = specs.get(m)
                last_check = s.last_price_check if s else "unknown"
                lines.append(f"- `{m}` — last verified: {last_check}")
            lines += [""]

        # Footer
        lines += [
            "---",
            "",
            f"*Report generated by [Tidus](https://github.com/kensterinvest/tidus) "
            f"v1.1.0 — Multi-Source Self-Healing Registry.*  ",
            f"*Source data: HardcodedSource (confidence 0.7) + "
            f"TidusPricingFeedSource (confidence 0.85, if configured).*  ",
            f"*Prices sourced from official vendor pricing pages. Not affiliated with "
            f"OpenRouter or any aggregator.*",
        ]

        return "\n".join(lines)

    def _render_github_release(self, report: PricingReport) -> str:
        """Short release body for GitHub Release notes."""
        lines: list[str] = []
        n_changes = len({c.model_id for c in report.price_changes})
        n_new = len(report.new_models)

        summary_parts = []
        if n_new:
            summary_parts.append(f"{n_new} new model(s)")
        if n_changes:
            summary_parts.append(f"prices updated for {n_changes} model(s)")
        summary = ", ".join(summary_parts) or "no pricing changes"

        lines += [
            f"## Tidus Pricing Update — {report.report_date}",
            f"",
            f"Weekly AI pricing sync: **{summary}**.",
            f"",
        ]

        if report.new_models:
            lines += ["**New models:**"]
            for m in report.new_models:
                lines.append(
                    f"- `{m.model_id}` ({_VENDOR_NAMES.get(m.vendor, m.vendor)}) — "
                    f"${m.input_usd_per_1m:.3f}/${m.output_usd_per_1m:.3f} per 1M tokens"
                )
            lines += [""]

        big_moves = [c for c in report.price_changes if c.abs_pct >= _BIG_DECREASE_PCT]
        if big_moves:
            lines += ["**Major price moves (≥50%):**"]
            seen = set()
            for c in big_moves:
                if c.model_id in seen:
                    continue
                seen.add(c.model_id)
                sign = "+" if c.delta_pct > 0 else ""
                lines.append(
                    f"- `{c.model_id}` {c.emoji} {sign}{c.delta_pct:.0f}%"
                )
            lines += [""]

        lines += [
            f"Full report: see `reports/pricing-{report.report_date}.md` in this release.",
            f"",
            f"[View full price table](https://github.com/kensterinvest/tidus/blob/main/docs/pricing-model.md)",
        ]
        return "\n".join(lines)

    def _generate_vendor_narrative(
        self,
        vendor: str,
        changes: list[PriceChange],
        specs: dict[str, ModelSpec],
    ) -> str:
        """Generate a one-sentence market narrative for a vendor's price changes."""
        increases = [c for c in changes if c.delta_pct > 0]
        decreases = [c for c in changes if c.delta_pct < 0]
        vendor_name = _VENDOR_NAMES.get(vendor, vendor)

        if decreases and not increases:
            avg_drop = sum(abs(c.delta_pct) for c in decreases) / len(decreases)
            models = ", ".join(sorted({f"`{c.model_id}`" for c in decreases[:3]}))
            if avg_drop >= _BIG_DECREASE_PCT:
                return (
                    f"{vendor_name} cut prices significantly (avg {avg_drop:.0f}% down) on "
                    f"{models} — likely a competitive response to market pressure."
                )
            return (
                f"{vendor_name} reduced prices on {models} by an average of {avg_drop:.0f}%, "
                f"making these models more cost-effective for high-volume workloads."
            )

        if increases and not decreases:
            avg_up = sum(c.delta_pct for c in increases) / len(increases)
            models = ", ".join(sorted({f"`{c.model_id}`" for c in increases[:3]}))
            return (
                f"{vendor_name} raised prices on {models} by an average of {avg_up:.0f}%, "
                f"reflecting updated positioning for these models."
            )

        if increases and decreases:
            return (
                f"{vendor_name} adjusted its pricing mix: {len(decreases)} reductions "
                f"and {len(increases)} increases, suggesting a tier realignment."
            )
        return ""
