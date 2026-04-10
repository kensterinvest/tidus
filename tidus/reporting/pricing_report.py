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

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

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

# ── Magazine HTML email CSS ────────────────────────────────────────────────────
# Defined as a plain string (not f-string) so CSS {} braces need no escaping.
_HTML_CSS = """\
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#f0f2f5;color:#1a1a2e;line-height:1.6;}
.wrapper{max-width:680px;margin:0 auto;background:#ffffff;}

.header{background:linear-gradient(135deg,#0f3460 0%,#533483 60%,#e94560 100%);
        padding:40px 40px 32px;text-align:center;}
.header-logo{font-size:30px;font-weight:800;color:white;letter-spacing:-0.5px;}
.header-logo .dot{color:#a78bfa;}
.header-tagline{font-size:11px;text-transform:uppercase;letter-spacing:2px;
                color:rgba(255,255,255,0.6);margin-top:4px;}
.header-title{font-size:22px;font-weight:700;color:white;margin-top:20px;line-height:1.3;}
.header-date{font-size:13px;color:rgba(255,255,255,0.75);margin-top:8px;}

.stats-bar{background:#1a1a2e;padding:20px 40px;display:flex;
           justify-content:space-around;align-items:center;}
.stat{text-align:center;flex:1;}
.stat-value{font-size:28px;font-weight:800;color:#a78bfa;line-height:1;}
.stat-drops{color:#4ade80 !important;}
.stat-label{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
            color:rgba(255,255,255,0.5);margin-top:4px;}
.stat-sep{width:1px;height:40px;background:rgba(255,255,255,0.1);}

.content{padding:32px 40px;}
.exec-summary{font-size:14px;color:#555;margin-bottom:8px;}

.section-header{display:flex;align-items:center;gap:10px;margin:32px 0 16px;
                padding-bottom:10px;border-bottom:2px solid #e9ecef;}
.section-icon{font-size:22px;}
.section-title{font-size:18px;font-weight:700;color:#1a1a2e;}

table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px;}
th{background:#f8f9fa;text-align:left;padding:8px 12px;font-size:11px;
   text-transform:uppercase;letter-spacing:0.8px;color:#6c757d;border-bottom:2px solid #dee2e6;}
td{padding:10px 12px;border-bottom:1px solid #f0f0f0;vertical-align:middle;}
tr:last-child td{border-bottom:none;}
.col-price{color:#888;}
.col-price-main{font-weight:600;}
.col-field{color:#888;font-size:12px;}
.col-vendor{color:#888;font-size:11px;white-space:nowrap;}
.col-ctx{color:#888;font-size:11px;white-space:nowrap;}
.table-note{font-size:12px;color:#888;margin-bottom:12px;}
.no-changes{padding:20px;text-align:center;color:#888;font-style:italic;margin-bottom:16px;}

.model-id{font-family:'SF Mono','Fira Code',monospace;font-size:12px;
          background:#f0f0f0;padding:2px 6px;border-radius:4px;color:#333;}
.change-up{color:#dc3545;font-weight:700;}
.change-down{color:#198754;font-weight:700;}
.badge-up{background:#fff0f0;color:#dc3545;padding:2px 8px;
          border-radius:12px;font-size:11px;font-weight:700;white-space:nowrap;}
.badge-down{background:#f0fff4;color:#198754;padding:2px 8px;
            border-radius:12px;font-size:11px;font-weight:700;white-space:nowrap;}

.vendor-header{font-size:13px;font-weight:700;color:#533483;text-transform:uppercase;
               letter-spacing:1px;padding:12px 0 6px;margin-top:8px;}
.narrative{background:linear-gradient(135deg,#f8f4ff,#fff);border-left:3px solid #a78bfa;
           padding:10px 14px;font-size:13px;color:#4a4a6a;font-style:italic;
           margin-bottom:20px;border-radius:0 6px 6px 0;}

.model-card{border:1px solid #e9ecef;border-radius:10px;padding:18px 20px;
            margin-bottom:16px;background:#fafbfc;}
.model-card-header{display:flex;align-items:flex-start;
                   justify-content:space-between;gap:12px;margin-bottom:10px;}
.model-card-left{flex:1;}
.model-card-vendor{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                   color:#a78bfa;font-weight:700;margin-bottom:4px;}
.model-card-id{font-family:monospace;font-size:14px;font-weight:700;color:#0f3460;
               background:#e8edf7;padding:3px 8px;border-radius:5px;
               display:inline-block;margin-bottom:4px;}
.model-card-tagline{font-size:13px;color:#555;line-height:1.4;}
.model-card-pricing{text-align:right;flex-shrink:0;}
.price-input{font-size:16px;font-weight:800;color:#0f3460;}
.price-output{font-size:13px;color:#888;margin-top:2px;}
.price-unit{font-size:10px;color:#aaa;margin-top:1px;}
.model-strengths{font-size:12px;color:#666;margin-top:8px;padding-left:16px;}
.model-strengths li{margin-bottom:3px;}
.model-context{font-size:12px;color:#888;font-style:italic;margin-top:10px;
               padding-top:8px;border-top:1px solid #eee;}
.model-meta{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;}
.meta-tag{background:#f0f2f5;color:#555;font-size:11px;padding:2px 8px;border-radius:10px;}
.tier-tag{background:#e8edf7;color:#0f3460;font-size:11px;
          font-weight:600;padding:2px 8px;border-radius:10px;}

.cta-box{background:linear-gradient(135deg,#0f3460,#533483);border-radius:12px;
         padding:28px 32px;margin:32px 0;text-align:center;}
.cta-title{color:white;font-size:18px;font-weight:700;margin-bottom:8px;}
.cta-sub{color:rgba(255,255,255,0.75);font-size:13px;margin-bottom:20px;}
.cta-btn{display:inline-block;background:#a78bfa;color:white !important;
         text-decoration:none;padding:12px 28px;border-radius:8px;
         font-weight:700;font-size:14px;letter-spacing:0.3px;}

.footer{background:#f8f9fa;padding:24px 40px;border-top:1px solid #e9ecef;
        font-size:11px;color:#adb5bd;text-align:center;line-height:1.7;}
.footer a{color:#533483;text-decoration:none;}
code{background:#f0f0f0;padding:1px 5px;border-radius:3px;
     font-family:monospace;font-size:11px;color:#555;}

@media(max-width:600px){
  .content,.header,.footer,.stats-bar{padding:20px;}
  .model-card-header{flex-direction:column;}
  .stats-bar{flex-wrap:wrap;gap:12px;}
  .stat-sep{display:none;}
}
"""


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
    html: str = field(default="", repr=False)


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
        from sqlalchemy import text

        from tidus.db.repositories.registry_repo import (
            get_active_revision,
            get_entries_for_revision,
        )

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

        # Load model descriptions for new-model intro cards
        descriptions = self._load_descriptions()

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
        report.html = self._render_html(report, current_specs, descriptions)
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
            "# Tidus AI Model Latest Pricing Report",
            "",
            f"**Report Date:** {report.report_date}  ",
            f"**Active Revision:** `{report.current_revision_id[:8]}…`  ",
            f"**Models Tracked:** {report.total_models}  ",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
        ]

        # Executive Summary
        increases = [c for c in report.price_changes if c.delta_pct > 0]
        decreases = [c for c in report.price_changes if c.delta_pct < 0]
        big_moves = [c for c in report.price_changes if c.abs_pct >= _BIG_INCREASE_PCT]

        lines += [
            "## Executive Summary",
            "",
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
                "| Model | Vendor | Tier | Input $/1M | Output $/1M | Context | Capabilities |",
                "|---|---|---|---|---|---|---|",
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
                # Group by model to generate narrative (reserved for future narrative expansion)
                lines += [f"### {vendor_name}", ""]
                lines += [
                    "| Model | Field | Old $/1M | New $/1M | Change |",
                    "|---|---|---|---|---|",
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
            "*Report generated by [Tidus](https://github.com/kensterinvest/tidus) "
            "v1.1.0 — Multi-Source Self-Healing Registry.*  ",
            "*Source data: HardcodedSource (confidence 0.7) + "
            "TidusPricingFeedSource (confidence 0.85, if configured).*  ",
            "*Prices sourced from official vendor pricing pages. Not affiliated with "
            "OpenRouter or any aggregator.*",
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
            "",
            f"Weekly AI pricing sync: **{summary}**.",
            "",
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
            "",
            "[View full price table](https://github.com/kensterinvest/tidus/blob/main/docs/pricing-model.md)",
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

    # ── Model descriptions ────────────────────────────────────────────────────

    @staticmethod
    def _load_descriptions() -> dict[str, dict]:
        """Load model descriptions from config/model_descriptions.yaml."""
        from pathlib import Path

        from tidus.utils.yaml_loader import load_yaml
        path = Path("config/model_descriptions.yaml")
        if not path.exists():
            return {}
        try:
            raw = load_yaml(str(path))
            return raw.get("models", {})
        except Exception as exc:
            log.warning("model_descriptions_load_failed", error=str(exc))
            return {}

    # ── Magazine HTML renderer ────────────────────────────────────────────────

    def _render_html(
        self,
        report: PricingReport,
        specs: dict[str, ModelSpec],
        descriptions: dict[str, dict],
    ) -> str:
        """Generate a magazine-style HTML email newsletter."""
        n_changes  = len(report.price_changes)
        n_new      = len(report.new_models)
        n_drops    = sum(1 for c in report.price_changes if c.delta_pct < 0)
        increases  = [c for c in report.price_changes if c.delta_pct > 0]
        decreases  = [c for c in report.price_changes if c.delta_pct < 0]

        # Executive summary line
        if n_changes == 0 and n_new == 0:
            exec_summary = "No pricing changes this week — all models stable."
        else:
            parts: list[str] = []
            if decreases:
                parts.append(f"<strong>{len(decreases)} price drop{'s' if len(decreases)>1 else ''}</strong>")
            if increases:
                parts.append(f"<strong>{len(increases)} price hike{'s' if len(increases)>1 else ''}</strong>")
            if n_new:
                parts.append(f"<strong>{n_new} new model{'s' if n_new>1 else ''}</strong>")
            n_v = len({c.vendor for c in report.price_changes}) if report.price_changes else 0
            v_str = f" across {n_v} vendor{'s' if n_v != 1 else ''}" if n_v else ""
            exec_summary = f"This week: {', '.join(parts)}{v_str}."

        # ── New models section ────────────────────────────────────────────────
        new_models_html = ""
        if report.new_models:
            cards = []
            for m in report.new_models:
                desc = descriptions.get(m.model_id, {})
                tagline = desc.get(
                    "tagline",
                    f"New {_VENDOR_NAMES.get(m.vendor, m.vendor)} model in the Tidus catalog.",
                )
                strengths_items = "".join(
                    f"<li>{s}</li>" for s in desc.get("strengths", [])[:4]
                )
                strengths_block = (
                    f'<ul class="model-strengths">{strengths_items}</ul>'
                    if strengths_items else ""
                )
                context = desc.get("context", "")
                context_block = (
                    f'<div class="model-context">{context}</div>' if context else ""
                )
                caps_str = ", ".join(m.capabilities[:3]) if m.capabilities else ""
                meta = (
                    f'<span class="tier-tag">Tier {m.tier}</span>'
                    f'<span class="meta-tag">{m.max_context_k}K ctx</span>'
                )
                if caps_str:
                    meta += f'<span class="meta-tag">{caps_str}</span>'
                vname = _VENDOR_NAMES.get(m.vendor, m.vendor)
                cards.append(
                    f'<div class="model-card">'
                    f'<div class="model-card-header">'
                    f'<div class="model-card-left">'
                    f'<div class="model-card-vendor">{vname}</div>'
                    f'<div class="model-card-id">{m.model_id}</div>'
                    f'<div class="model-card-tagline">{tagline}</div>'
                    f'</div>'
                    f'<div class="model-card-pricing">'
                    f'<div class="price-input">${m.input_usd_per_1m:.3f}</div>'
                    f'<div class="price-output">${m.output_usd_per_1m:.3f} out</div>'
                    f'<div class="price-unit">per 1M tokens</div>'
                    f'</div></div>'
                    f'{strengths_block}'
                    f'<div class="model-meta">{meta}</div>'
                    f'{context_block}'
                    f'</div>'
                )
            new_models_html = (
                '<div class="section-header">'
                '<span class="section-icon">&#x1F195;</span>'
                '<span class="section-title">New Models This Week</span>'
                '</div>'
                + "".join(cards)
            )

        # ── Price changes section ─────────────────────────────────────────────
        if report.price_changes:
            vendor_blocks: list[str] = []
            for vendor in sorted({c.vendor for c in report.price_changes}):
                vc = [c for c in report.price_changes if c.vendor == vendor]
                vname = _VENDOR_NAMES.get(vendor, vendor)
                narrative = self._generate_vendor_narrative(vendor, vc, specs)
                rows = []
                for ch in vc:
                    fl = "Input" if ch.field == "input_price" else "Output"
                    sign = "+" if ch.delta_pct > 0 else ""
                    bc = "badge-up" if ch.delta_pct > 0 else "badge-down"
                    pc = "change-up" if ch.delta_pct > 0 else "change-down"
                    rows.append(
                        f"<tr>"
                        f'<td><span class="model-id">{ch.model_id}</span></td>'
                        f'<td class="col-field">{fl}</td>'
                        f'<td class="col-price">${ch.old_usd_per_1m:.3f}</td>'
                        f'<td class="col-price {pc}">${ch.new_usd_per_1m:.3f}</td>'
                        f'<td><span class="{bc}">{sign}{ch.delta_pct:.1f}%</span></td>'
                        f"</tr>"
                    )
                narr_block = (
                    f'<div class="narrative">{narrative}</div>' if narrative else ""
                )
                vendor_blocks.append(
                    f'<div class="vendor-header">{vname}</div>'
                    f"<table><thead><tr>"
                    f"<th>Model</th><th>Field</th><th>Old $/1M</th>"
                    f"<th>New $/1M</th><th>Change</th>"
                    f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
                    f"{narr_block}"
                )
            price_changes_html = (
                '<div class="section-header">'
                '<span class="section-icon">&#x1F4CA;</span>'
                '<span class="section-title">Price Changes</span>'
                '</div>'
                + "".join(vendor_blocks)
            )
        else:
            price_changes_html = (
                '<div class="no-changes">'
                "No price changes this week &#x2014; all models stable."
                "</div>"
            )

        # ── Full price table ──────────────────────────────────────────────────
        by_vendor: dict[str, list[ModelSpec]] = {}
        for spec in sorted(specs.values(), key=lambda s: (s.vendor, s.model_id)):
            if not spec.is_local:
                by_vendor.setdefault(spec.vendor, []).append(spec)
        table_rows: list[str] = []
        for vendor, vspecs in sorted(by_vendor.items()):
            vname = _VENDOR_NAMES.get(vendor, vendor)
            for s in vspecs:
                ctx_k = s.max_context // 1000
                table_rows.append(
                    f"<tr>"
                    f'<td class="col-vendor">{vname}</td>'
                    f'<td><span class="model-id">{s.model_id}</span></td>'
                    f'<td class="col-price-main">${s.input_price * 1000:.3f}</td>'
                    f'<td class="col-price">${s.output_price * 1000:.3f}</td>'
                    f'<td class="col-ctx">{ctx_k}K</td>'
                    f"</tr>"
                )
        price_table_html = (
            '<div class="section-header">'
            '<span class="section-icon">&#x1F4CB;</span>'
            '<span class="section-title">Full Price Table</span>'
            "</div>"
            f'<p class="table-note">All prices USD/1M tokens &middot; Updated {report.report_date}</p>'
            "<table><thead><tr>"
            "<th>Vendor</th><th>Model</th><th>Input $/1M</th><th>Output $/1M</th><th>Context</th>"
            f"</tr></thead><tbody>{''.join(table_rows)}</tbody></table>"
        )

        # ── Assemble ──────────────────────────────────────────────────────────
        parts_html: list[str] = [
            f"<!DOCTYPE html>\n<html>\n<head>\n"
            f'<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>Tidus AI Pricing Report &mdash; {report.report_date}</title>\n"
            f"<style>\n",
            _HTML_CSS,
            f"</style>\n</head>\n<body>\n<div class=\"wrapper\">\n\n"

            f'<div class="header">\n'
            f'  <div class="header-logo">tidus<span class="dot">.</span>ai</div>\n'
            f'  <div class="header-tagline">WEEKLY AI INTELLIGENCE REPORT</div>\n'
            f'  <div class="header-title">AI Model Pricing Report</div>\n'
            f'  <div class="header-date">{report.report_date}</div>\n'
            f"</div>\n\n"

            f'<div class="stats-bar">\n'
            f'  <div class="stat"><div class="stat-value">{report.total_models}</div>'
            f'<div class="stat-label">Models tracked</div></div>\n'
            f'  <div class="stat-sep"></div>\n'
            f'  <div class="stat"><div class="stat-value">{n_changes}</div>'
            f'<div class="stat-label">Price changes</div></div>\n'
            f'  <div class="stat-sep"></div>\n'
            f'  <div class="stat"><div class="stat-value">{n_new}</div>'
            f'<div class="stat-label">New models</div></div>\n'
            f'  <div class="stat-sep"></div>\n'
            f'  <div class="stat"><div class="stat-value stat-drops">{n_drops}</div>'
            f'<div class="stat-label">Price drops</div></div>\n'
            f"</div>\n\n"

            f'<div class="content">\n'
            f'<p class="exec-summary">{exec_summary}</p>\n'
            f"{new_models_html}\n"
            f"{price_changes_html}\n"
            f"{price_table_html}\n"

            f'<div class="cta-box">\n'
            f'  <div class="cta-title">&#x1F4EC; Know someone who tracks AI costs?</div>\n'
            f"  <div class=\"cta-sub\">Forward this report or share the link &mdash; "
            f"it&rsquo;s free, weekly, and open source.</div>\n"
            f'  <a href="https://github.com/kensterinvest/tidus#subscribe" class="cta-btn">'
            f"Subscribe to weekly AI pricing updates</a>\n"
            f"</div>\n\n"
            f"</div>\n\n"

            f'<div class="footer">\n'
            f"  You&rsquo;re receiving this because you subscribed to Tidus AI weekly pricing reports.<br>\n"
            f"  To unsubscribe, set <code>active: false</code> next to your entry in "
            f"<code>config/subscribers.yaml</code>.<br>\n"
            f'  <a href="https://github.com/kensterinvest/tidus">Tidus v1.1.0</a> &middot; '
            f'<a href="https://github.com/kensterinvest/tidus/blob/main/docs/pricing-model.md">'
            f"Pricing docs</a> &middot; Apache 2.0\n"
            f"</div>\n\n"
            f"</div>\n</body>\n</html>",
        ]
        return "".join(parts_html)
