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

.stats-bar{background:#1a1a2e;padding:20px 32px;display:flex;
           justify-content:space-around;align-items:center;}
.stat{text-align:center;flex:1;}
.stat-value{font-size:26px;font-weight:800;color:#a78bfa;line-height:1;}
.stat-drops{color:#4ade80 !important;}
.stat-rises{color:#f87171 !important;}
.stat-label{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
            color:rgba(255,255,255,0.5);margin-top:4px;}
.stat-sep{width:1px;height:36px;background:rgba(255,255,255,0.1);}

.content{padding:32px 40px;}
.exec-summary{font-size:14px;color:#555;line-height:1.7;margin-bottom:6px;}

.section-header{display:flex;align-items:center;gap:10px;margin:32px 0 16px;
                padding-bottom:10px;border-bottom:2px solid #e9ecef;}
.section-icon{font-size:22px;}
.section-title{font-size:18px;font-weight:700;color:#1a1a2e;}

/* Price change cards */
.changes-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
              gap:14px;margin-bottom:16px;}
.cc-card{border-radius:10px;padding:16px;border:1.5px solid;}
.cc-card.drop{background:#f0fff4;border-color:#86efac;}
.cc-card.rise{background:#fff0f0;border-color:#fca5a5;}
.cc-card.new{background:#f0f4ff;border-color:#93c5fd;}
.cc-model{font-family:'SF Mono','Fira Code',monospace;font-size:13px;font-weight:700;
          color:#0f3460;margin-bottom:2px;}
.cc-vendor{font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:#888;
           font-weight:600;margin-bottom:10px;}
.cc-row{display:flex;align-items:center;gap:5px;font-size:12px;margin-bottom:4px;flex-wrap:wrap;}
.cc-lbl{color:#888;font-size:11px;min-width:38px;}
.cc-old{text-decoration:line-through;color:#aaa;}
.cc-nv-drop{color:#16a34a;font-weight:700;}
.cc-nv-rise{color:#dc2626;font-weight:700;}
.cc-nv-new{color:#2563eb;font-weight:700;}
.d-drop{background:#dcfce7;color:#16a34a;padding:1px 6px;border-radius:8px;
        font-size:11px;font-weight:700;}
.d-rise{background:#fee2e2;color:#dc2626;padding:1px 6px;border-radius:8px;
        font-size:11px;font-weight:700;}
.d-new{background:#dbeafe;color:#2563eb;padding:1px 6px;border-radius:8px;
       font-size:11px;font-weight:700;}

/* Vendor narrative */
.narrative{background:linear-gradient(135deg,#f8f4ff,#fff);border-left:3px solid #a78bfa;
           padding:10px 14px;font-size:13px;color:#4a4a6a;font-style:italic;
           margin:4px 0 20px;border-radius:0 6px 6px 0;}

/* Optimisation tips */
.tips-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px;}
.tip-card{padding:14px;background:#f8f9ff;border-radius:10px;border:1px solid #e9ecef;}
.tip-title{font-size:12px;font-weight:700;color:#0f3460;margin-bottom:6px;}
.tip-desc{font-size:12px;color:#555;line-height:1.6;}
.tip-savings{color:#16a34a;font-weight:700;}

/* New model cards */
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

/* Price table */
table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px;}
th{background:#f8f9fa;text-align:left;padding:8px 12px;font-size:11px;
   text-transform:uppercase;letter-spacing:0.8px;color:#6c757d;border-bottom:2px solid #dee2e6;}
td{padding:10px 12px;border-bottom:1px solid #f0f0f0;vertical-align:middle;}
tr:last-child td{border-bottom:none;}
.col-price{color:#888;font-size:12px;}
.col-price-main{font-weight:700;color:#0f3460;}
.col-vendor{color:#888;font-size:11px;white-space:nowrap;}
.col-ctx{color:#888;font-size:11px;white-space:nowrap;}
.col-rank{color:#bbb;font-size:11px;text-align:right;padding-right:4px;width:28px;}
.table-note{font-size:12px;color:#888;margin-bottom:12px;}
.no-changes{padding:20px;text-align:center;color:#888;font-style:italic;margin-bottom:16px;}
.model-id{font-family:'SF Mono','Fira Code',monospace;font-size:12px;
          background:#f0f0f0;padding:2px 6px;border-radius:4px;color:#333;}

/* Legend */
.price-legend{background:#f8f6ff;border:1px solid #e0d9f7;border-radius:10px;
              padding:16px 20px;margin-bottom:20px;}
.legend-title{font-size:13px;font-weight:700;color:#533483;margin-bottom:12px;}
.legend-grid{display:flex;flex-direction:column;gap:10px;}
.legend-item{display:grid;grid-template-columns:130px 1fr;gap:8px;align-items:start;}
.legend-label{font-size:12px;font-weight:700;color:#0f3460;padding-top:2px;white-space:nowrap;}
.legend-desc{font-size:12px;color:#555;line-height:1.5;}

/* CTA and footer */
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
  .tips-grid{grid-template-columns:1fr;}
  .changes-grid{grid-template-columns:1fr;}
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
    # Discovered models (from vendor /v1/models) that are NOT in the active
    # registry. These are surface-only candidates — they don't route until
    # a maintainer adds them to models.yaml + hardcoded_source.py.
    discovery_report: object | None = None  # tidus.sync.discovery.DiscoveryReport
    markdown: str = field(default="", repr=False)
    github_release_body: str = field(default="", repr=False)
    html: str = field(default="", repr=False)


def _rank_key(spec) -> tuple[float, int, str]:
    """Sort key for ranking paid models in the report.

    Returns ``(blended_cost, released_ord, model_id)``. Used with
    ``sorted(..., reverse=True)`` so:
      1. Higher blended cost ranks first.
      2. On cost tie, the more recently released model wins (so opus-4-7
         outranks opus-4-6 at the same $15/M).
      3. On date tie, higher model_id string wins — deterministic tiebreak.

    Requires ``released_at`` to be populated on specs (applied via the YAML
    overlay in :meth:`PricingReportGenerator.generate` before sorting).
    """
    blended = (spec.input_price + spec.output_price) / 2
    released_ord = spec.released_at.toordinal() if spec.released_at else 0
    return (blended, released_ord, spec.model_id)


class PricingReportGenerator:
    """Generates Tidus AI Model Latest Pricing Report from registry revisions."""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    async def generate(
        self,
        revision_id: str | None = None,    # defaults to current ACTIVE
        base_revision_id: str | None = None,  # defaults to previous (most recent SUPERSEDED)
        discovery_report=None,             # optional tidus.sync.discovery.DiscoveryReport
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

        # Overlay released_at from YAML — DB spec_json may predate this field,
        # and we rely on released_at as the tie-breaker for ranking (Opus 4.7
        # above Opus 4.6 when they share the same $15/M blended cost).
        try:
            from tidus.router.registry import ModelRegistry
            from tidus.settings import get_settings
            _yaml_reg = ModelRegistry.load(get_settings().models_config_path)
            for mid, spec in list(current_specs.items()):
                y = _yaml_reg.get(mid)
                if y and y.released_at and spec.released_at is None:
                    current_specs[mid] = spec.model_copy(update={"released_at": y.released_at})
        except Exception:
            pass  # best-effort; ranking falls back to model_id lex

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
            discovery_report=discovery_report,
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
                # m.tier may be a ModelTier enum or already an int — normalise so the
                # generated Markdown always shows "Tier 2" not "Tier ModelTier.mid".
                tier_int = int(m.tier.value) if hasattr(m.tier, "value") else int(m.tier)
                lines.append(
                    f"| `{m.model_id}` | {_VENDOR_NAMES.get(m.vendor, m.vendor)} | "
                    f"Tier {tier_int} | ${m.input_usd_per_1m:.3f} | ${m.output_usd_per_1m:.3f} | "
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
            f"*Prices as of {report.report_date} — USD/1M tokens · Ranked by blended cost*",
            "*Blended = (input + output) / 2 — equal input:output token split*",
            "",
            "| # | Vendor | Model | Blended $/1M | Input $/1M | Output $/1M | Context |",
            "|---|---|---|---|---|---|---|",
        ]
        paid_specs_md = sorted(
            (s for s in specs.values() if not s.is_local),
            key=_rank_key,
            reverse=True,
        )
        for rank, s in enumerate(paid_specs_md, start=1):
            blended = (s.input_price + s.output_price) / 2 * 1000
            vendor_name = _VENDOR_NAMES.get(s.vendor, s.vendor)
            ctx = f"{s.max_context // 1000}K"
            lines.append(
                f"| {rank} | {vendor_name} | `{s.model_id}` | "
                f"${blended:.2f} | ${s.input_price * 1000:.3f} | "
                f"${s.output_price * 1000:.3f} | {ctx} |"
            )
        lines += [""]

        # Vendor-discovered candidates (surface-only — not yet routable)
        dr = report.discovery_report
        if dr is not None and (dr.new_this_run or dr.pending_review or dr.removed_from_vendor):
            lines += [
                "## 🔎 Vendor-Discovered Models (Pending Review)",
                "",
                "Models surfaced from vendor `/v1/models` endpoints that are NOT in "
                "the active routing catalog. Pricing is intentionally not shown — "
                "promotion to routable status requires a maintainer to verify "
                "pricing and add the model to `config/models.yaml` + "
                "`tidus/sync/pricing/hardcoded_source.py`.",
                "",
                f"*Sources polled: {', '.join(dr.sources_run) or 'none'}"
                + (f" · skipped (no API key): {', '.join(dr.sources_skipped)}" if dr.sources_skipped else "")
                + "*",
                "",
            ]
            if dr.new_this_run:
                lines += [
                    "### 🆕 First-seen this run",
                    "",
                    "| Vendor | Model ID | Vendor ID | Display Name |",
                    "|---|---|---|---|",
                ]
                for m in dr.new_this_run:
                    name = m.display_name or "—"
                    lines.append(
                        f"| {m.vendor} | `{m.model_id}` | `{m.vendor_id}` | {name} |"
                    )
                lines += [""]
            if dr.pending_review:
                lines += [
                    "### ⏳ Backlog (previously seen, not yet promoted)",
                    "",
                    "| Vendor | Model ID | Vendor ID |",
                    "|---|---|---|",
                ]
                for m in dr.pending_review:
                    lines.append(f"| {m.vendor} | `{m.model_id}` | `{m.vendor_id}` |")
                lines += [""]
            if dr.removed_from_vendor:
                lines += [
                    "### 🚫 Absent this run (possibly deprecated upstream)",
                    "",
                ]
                for mid in dr.removed_from_vendor:
                    lines.append(f"- `{mid}`")
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
            "v1.2.0 — Comprehensive Review Hardening.*  ",
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
        """Generate a magazine-style HTML email newsletter matching index.html layout."""
        from collections import defaultdict

        increases  = [c for c in report.price_changes if c.delta_pct > 0]
        decreases  = [c for c in report.price_changes if c.delta_pct < 0]
        n_drops    = len({c.model_id for c in decreases})
        n_rises    = len({c.model_id for c in increases})
        n_new      = len(report.new_models)
        n_updated  = len({c.model_id for c in report.price_changes})

        def _blended(s: ModelSpec) -> float:
            return (s.input_price + s.output_price) / 2 * 1000  # $/1M

        paid = [s for s in specs.values() if not s.is_local and s.input_price > 0]
        cheapest = min(paid, key=lambda s: s.input_price, default=None)
        priciest = max(paid, key=_blended, default=None)
        biggest_drop = (
            min(decreases, key=lambda c: c.delta_pct) if decreases else None
        )

        # ── Rich executive summary ─────────────────────────────────────────────
        paras: list[str] = []
        if biggest_drop:
            vname = _VENDOR_NAMES.get(biggest_drop.vendor, biggest_drop.vendor)
            fl = "input" if biggest_drop.field == "input_price" else "output"
            paras.append(
                f"<strong>{vname}</strong> leads this week's activity — "
                f"<strong>{biggest_drop.model_id}</strong> {fl} dropped "
                f"<strong>{abs(biggest_drop.delta_pct):.1f}%</strong> to "
                f"<strong>${biggest_drop.new_usd_per_1m:.3f}/1M</strong>."
            )
        if increases:
            vnames = sorted({_VENDOR_NAMES.get(c.vendor, c.vendor) for c in increases})
            paras.append(
                f"{', '.join(vnames)} raised prices on "
                f"{len({c.model_id for c in increases})} model"
                f"{'s' if len({c.model_id for c in increases}) > 1 else ''}."
            )
        if n_new:
            new_names = ", ".join(
                f"<strong>{m.model_id}</strong>" for m in report.new_models[:3]
            )
            paras.append(
                f"{n_new} new model{'s' if n_new > 1 else ''} added to the catalog: {new_names}."
            )
        if cheapest and priciest:
            paras.append(
                f"The catalog spans <strong>{cheapest.model_id}</strong> at "
                f"<strong>${cheapest.input_price * 1000:.3f}/1M</strong> input "
                f"(cheapest) to <strong>{priciest.model_id}</strong> at "
                f"<strong>${_blended(priciest):.2f}/1M</strong> blended (most expensive)."
            )
        if not paras:
            paras = [
                "All models stable this week — no pricing changes detected across the "
                f"{report.total_models}-model catalog."
            ]
        exec_summary_html = "  ".join(paras)

        # ── New models section ─────────────────────────────────────────────────
        new_models_html = ""
        if report.new_models:
            cards = []
            for m in report.new_models:
                desc = descriptions.get(m.model_id, {})
                tagline = desc.get(
                    "tagline",
                    f"New {_VENDOR_NAMES.get(m.vendor, m.vendor)} model added to the catalog.",
                )
                strengths_items = "".join(
                    f"<li>{s}</li>" for s in desc.get("strengths", [])[:4]
                )
                strengths_block = (
                    f'<ul class="model-strengths">{strengths_items}</ul>'
                    if strengths_items else ""
                )
                context_block = (
                    f'<div class="model-context">{desc["context"]}</div>'
                    if desc.get("context") else ""
                )
                caps_str = ", ".join(m.capabilities[:3]) if m.capabilities else ""
                tier_int = int(m.tier.value) if hasattr(m.tier, "value") else int(m.tier)
                meta = (
                    f'<span class="tier-tag">Tier {tier_int}</span>'
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

        # ── Price changes — card grid ──────────────────────────────────────────
        if report.price_changes:
            # Group per-field changes back to per-model cards
            by_model: dict[str, list[PriceChange]] = defaultdict(list)
            for c in report.price_changes:
                by_model[c.model_id].append(c)

            cards_html: list[str] = []
            for model_id in sorted(by_model):
                mchanges = by_model[model_id]
                drops_m  = [c for c in mchanges if c.delta_pct < 0]
                card_type = "drop" if len(drops_m) >= len(mchanges) - len(drops_m) else "rise"
                vname = _VENDOR_NAMES.get(mchanges[0].vendor, mchanges[0].vendor)
                rows_html = ""
                for ch in mchanges:
                    fl  = "Input" if ch.field == "input_price" else "Output"
                    sign = "−" if ch.delta_pct < 0 else "+"
                    nv_cls    = "cc-nv-drop" if ch.delta_pct < 0 else "cc-nv-rise"
                    badge_cls = "d-drop"     if ch.delta_pct < 0 else "d-rise"
                    rows_html += (
                        f'<div class="cc-row">'
                        f'<span class="cc-lbl">{fl}</span>'
                        f'<span class="cc-old">${ch.old_usd_per_1m:.3f}</span>'
                        f'<span style="color:#ccc">&#8594;</span>'
                        f'<span class="{nv_cls}">${ch.new_usd_per_1m:.3f}</span>'
                        f'<span class="{badge_cls}">{sign}{abs(ch.delta_pct):.1f}%</span>'
                        f'</div>'
                    )
                cards_html.append(
                    f'<div class="cc-card {card_type}">'
                    f'<div class="cc-model">{model_id}</div>'
                    f'<div class="cc-vendor">{vname}</div>'
                    f'{rows_html}'
                    f'</div>'
                )

            # Vendor narratives below the grid
            narratives_html = ""
            for vendor in sorted({c.vendor for c in report.price_changes}):
                vc = [c for c in report.price_changes if c.vendor == vendor]
                narr = self._generate_vendor_narrative(vendor, vc, specs)
                if narr:
                    narratives_html += f'<div class="narrative">{narr}</div>'

            price_changes_html = (
                '<div class="section-header">'
                '<span class="section-icon">&#x1F4C9;</span>'
                '<span class="section-title">Price Changes This Week</span>'
                '</div>'
                f'<div class="changes-grid">{"".join(cards_html)}</div>'
                + narratives_html
            )
        else:
            price_changes_html = (
                '<div class="no-changes">'
                "&#x2714; No price changes this week &mdash; all models stable."
                "</div>"
            )

        # ── Cost optimisation tips ─────────────────────────────────────────────
        tips: list[str] = []

        # Tip 1: biggest input drop this week
        input_drops = [c for c in decreases if c.field == "input_price"]
        if input_drops:
            bd = min(input_drops, key=lambda c: c.delta_pct)
            vname = _VENDOR_NAMES.get(bd.vendor, bd.vendor)
            tips.append(
                f'<div class="tip-card">'
                f'<div class="tip-title">&#x1F4C9; {bd.model_id} dropped {abs(bd.delta_pct):.0f}%</div>'
                f'<div class="tip-desc">'
                f'<strong>{vname}</strong>&rsquo;s <strong>{bd.model_id}</strong> input is now '
                f'<strong>${bd.new_usd_per_1m:.3f}/1M</strong> (was ${bd.old_usd_per_1m:.3f}/1M). '
                f'If you route similar-complexity tasks to a pricier model, this week is the right '
                f'time to benchmark an alternative.'
                f'</div></div>'
            )

        # Tip 2: cheapest economy vs most expensive — ratio insight
        economy = sorted(
            [s for s in paid if int(s.tier) == 3],
            key=_blended,
        )
        premium = sorted([s for s in paid if int(s.tier) == 1], key=_rank_key, reverse=True)
        if economy and premium:
            eco, prem = economy[0], premium[0]
            ratio = _blended(prem) / _blended(eco) if _blended(eco) > 0 else 0
            tips.append(
                f'<div class="tip-card">'
                f'<div class="tip-title">&#x1F4B0; Economy pick: {eco.model_id}</div>'
                f'<div class="tip-desc">'
                f'At <strong>${_blended(eco):.3f}/1M</strong> blended, '
                f'<strong>{eco.model_id}</strong> is '
                f'<strong class="tip-savings">{ratio:.0f}&times; cheaper</strong> than '
                f'{prem.model_id} (${_blended(prem):.2f}/1M). '
                f'Ideal for classification, summarisation, and simple generation tasks.'
                f'</div></div>'
            )

        tips_html = ""
        if tips:
            tips_html = (
                '<div class="section-header">'
                '<span class="section-icon">&#x1F4A1;</span>'
                '<span class="section-title">Cost Optimisation Opportunities</span>'
                '</div>'
                f'<div class="tips-grid">{"".join(tips)}</div>'
            )

        # ── Full price table ───────────────────────────────────────────────────
        paid_specs = sorted(paid, key=_rank_key, reverse=True)
        table_rows: list[str] = []
        for rank, s in enumerate(paid_specs, start=1):
            ctx_k  = s.max_context // 1000
            vname  = _VENDOR_NAMES.get(s.vendor, s.vendor)
            blended = _blended(s)
            table_rows.append(
                f"<tr>"
                f'<td class="col-rank">{rank}</td>'
                f'<td class="col-vendor">{vname}</td>'
                f'<td><span class="model-id">{s.model_id}</span></td>'
                f'<td class="col-price-main">${blended:.2f}</td>'
                f'<td class="col-price">${s.input_price * 1000:.3f}</td>'
                f'<td class="col-price">${s.output_price * 1000:.3f}</td>'
                f'<td class="col-ctx">{ctx_k}K</td>'
                f"</tr>"
            )
        price_legend_html = (
            '<div class="price-legend">'
            '<div class="legend-title">&#x2139;&#xFE0F; How to read these prices</div>'
            '<div class="legend-grid">'
            '<div class="legend-item">'
            '<span class="legend-label">Blended $/1M</span>'
            '<span class="legend-desc">Average of input and output price per 1M tokens — '
            'the fairest single-number comparison. Assumes equal input:output token volume, '
            'which is neutral across use cases.</span>'
            '</div>'
            '<div class="legend-item">'
            '<span class="legend-label">Input $/1M</span>'
            '<span class="legend-desc">Cost per 1M tokens you <strong>send</strong> — '
            'your prompt, system instruction, and conversation history.</span>'
            '</div>'
            '<div class="legend-item">'
            '<span class="legend-label">Output $/1M</span>'
            '<span class="legend-desc">Cost per 1M tokens the model <strong>generates</strong>. '
            'Output is typically 3–5&times; more expensive than input.</span>'
            '</div>'
            '</div>'
            '</div>'
        )
        price_table_html = (
            '<div class="section-header">'
            '<span class="section-icon">&#x1F4CB;</span>'
            '<span class="section-title">Full Model Catalog &mdash; '
            f'{report.report_date.strftime("%B %Y")}</span>'
            '</div>'
            f'<p class="table-note">Ranked by blended cost &mdash; highest first &middot; '
            f'Blended&nbsp;=&nbsp;(input&nbsp;+&nbsp;output)&nbsp;/&nbsp;2 &middot; '
            f'All prices USD/1M tokens &middot; Updated {report.report_date}</p>'
            + price_legend_html
            + "<table><thead><tr>"
            "<th>#</th><th>Vendor</th><th>Model</th>"
            "<th>Blended $/1M</th><th>Input $/1M</th><th>Output $/1M</th><th>Context</th>"
            f"</tr></thead><tbody>{''.join(table_rows)}</tbody></table>"
        )

        # ── Assemble ───────────────────────────────────────────────────────────
        parts_html: list[str] = [
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>Tidus AI Pricing Report &mdash; {report.report_date}</title>\n"
            "<style>\n",
            _HTML_CSS,
            f'</style>\n</head>\n<body>\n<div class="wrapper">\n\n'

            # Header
            f'<div class="header">\n'
            f'  <div class="header-logo">tidus<span class="dot">.</span>magazine</div>\n'
            f'  <div class="header-tagline">AI MODEL MARKET INTELLIGENCE &middot; WEEKLY EDITION</div>\n'
            f'  <div class="header-title">AI Model Pricing Report &mdash; '
            f'{report.report_date.strftime("%B %Y")}</div>\n'
            f'  <div class="header-date">Week of {report.report_date} &middot; '
            f'Powered by Tidus v1.2.0</div>\n'
            f'</div>\n\n'

            # Stats bar (5 stats matching index.html magazine)
            f'<div class="stats-bar">\n'
            f'  <div class="stat"><div class="stat-value">{report.total_models}</div>'
            f'<div class="stat-label">Models Tracked</div></div>\n'
            f'  <div class="stat-sep"></div>\n'
            f'  <div class="stat"><div class="stat-value stat-drops">{n_drops}</div>'
            f'<div class="stat-label">Price Drops</div></div>\n'
            f'  <div class="stat-sep"></div>\n'
            f'  <div class="stat"><div class="stat-value stat-rises">{n_rises}</div>'
            f'<div class="stat-label">Price Rises</div></div>\n'
            f'  <div class="stat-sep"></div>\n'
            f'  <div class="stat"><div class="stat-value">{n_updated}</div>'
            f'<div class="stat-label">Models Updated</div></div>\n'
            f'  <div class="stat-sep"></div>\n'
            f'  <div class="stat"><div class="stat-value">{n_new}</div>'
            f'<div class="stat-label">New Models</div></div>\n'
            f'</div>\n\n'

            # Content
            f'<div class="content">\n'
            f'<div style="background:#f8f9ff;border-left:4px solid #533483;'
            f'border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:28px;">\n'
            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1.5px;color:#533483;margin-bottom:6px;">Executive Summary</div>\n'
            f'<p class="exec-summary" style="margin:0">{exec_summary_html}</p>\n'
            f'</div>\n'
            f"{new_models_html}\n"
            f"{price_changes_html}\n"
            f"{tips_html}\n"
            f"{price_table_html}\n"

            # CTA
            f'<div class="cta-box">\n'
            f'  <div class="cta-title">&#x1F4EC; Know someone who tracks AI costs?</div>\n'
            f'  <div class="cta-sub">Forward this report &mdash; '
            f"it&rsquo;s free, weekly, and open source.</div>\n"
            f'  <a href="https://z-tidus.com#subscribe" class="cta-btn">'
            f'Subscribe at z-tidus.com &rarr;</a>\n'
            f'</div>\n\n'
            f'</div>\n\n'

            # Footer
            f'<div class="footer">\n'
            f"  You&rsquo;re receiving this because you subscribed to Tidus AI weekly pricing reports.<br>\n"
            f'  <a href="https://z-tidus.com#subscribe">Unsubscribe</a> &middot; '
            f'  <a href="https://github.com/kensterinvest/tidus">Tidus on GitHub</a> &middot; '
            f'  Apache 2.0\n'
            f'</div>\n\n'
            f'</div>\n</body>\n</html>',
        ]
        return "".join(parts_html)
