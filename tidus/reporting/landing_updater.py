"""LandingPageUpdater — regenerates index.html's magazine section and MODELS array from the
active registry revision and pushes to both kensterinvest/tidus and
kensterinvest/kensterinvest.github.io.

Called every Sunday after the weekly price sync so the landing page always
reflects the same prices as the weekly email report.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Minimum absolute % move to count as a price change (matches pricing_report.py)
_CHANGE_THRESHOLD_PCT = 5.0

# Display names for the landing page table
_VENDOR_NAMES: dict[str, str] = {
    "anthropic":   "Anthropic",
    "openai":      "OpenAI",
    "google":      "Google",
    "deepseek":    "DeepSeek",
    "xai":         "xAI",
    "alibaba":     "Alibaba",
    "qwen":        "Alibaba",
    "mistral":     "Mistral",
    "perplexity":  "Perplexity",
    "moonshot":    "Moonshot",
    "cohere":      "Cohere",
    "together":    "Together AI",
    "groq":        "Groq",
    "ollama":      "Local",
}

# Static pricing legend — does not change between runs
_PRICING_LEGEND_HTML = (
    '<div class="pricing-legend" style="margin-bottom:14px">'
    '<div class="leg"><div class="leg-lbl">Input Price</div>'
    '<div class="leg-desc">Cost per 1M tokens you <strong>send</strong> to the model &mdash; '
    'your prompt, system instruction, conversation history, and any context you attach. '
    'Longer prompts or large document uploads drive this cost up.</div></div>'
    '<div class="leg"><div class="leg-lbl">Output Price</div>'
    '<div class="leg-desc">Cost per 1M tokens the model <strong>generates</strong> &mdash; '
    'every word in its response. Verbose answers, long code completions, and streaming '
    'replies all accrue output cost. Output is typically 3&ndash;5&times; more expensive '
    'than input.</div></div>'
    '<div class="leg"><div class="leg-lbl">$/1M tokens</div>'
    '<div class="leg-desc">Industry-standard unit. To estimate a task: '
    '<em>(prompt tokens &divide; 1,000,000) &times; input price</em> + '
    '<em>(response tokens &divide; 1,000,000) &times; output price</em>. '
    'A typical 500-word prompt (&asymp;700 tokens) + 500-word reply costs under $0.01 '
    'on most models.</div></div>'
    '</div>'
)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _js_num(val: float) -> str:
    """Format a $/1M float for a JS numeric literal — no unnecessary trailing zeros."""
    if val >= 1:
        return f"{val:.2f}"
    if val >= 0.1:
        return f"{val:.3f}"
    return f"{val:.4g}"


def _date_label(d: date) -> str:
    """Return 'April 13, 2026' style label (cross-platform, no leading zero)."""
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    return f"{months[d.month - 1]} {d.day}, {d.year}"


def _blended_1m(input_price: float, output_price: float) -> float:
    """Blended cost in $/1M = (input + output) / 2 * 1000."""
    return (input_price + output_price) / 2 * 1000


def _next_sync_label(today: date) -> str:
    """Return 'Month D, YYYY 02:00 UTC' for next Sunday."""
    # today.weekday(): 0=Mon … 6=Sun
    days_until_sunday = (6 - today.weekday()) % 7
    if days_until_sunday == 0:
        days_until_sunday = 7  # next Sunday, not today
    next_sunday = today + timedelta(days=days_until_sunday)
    return f"{_date_label(next_sunday)} 02:00 UTC"


# ── Main class ─────────────────────────────────────────────────────────────────

class LandingPageUpdater:
    """Reads the active revision from DB, regenerates the full magazine section and
    MODELS array in index.html, then pushes to GitHub."""

    def __init__(self, index_html_path: str = "index.html") -> None:
        self._index_path = Path(index_html_path)

    async def update(self, session_factory) -> bool:
        """Update index.html and push to both repos. Returns True on success."""
        from sqlalchemy import text

        from tidus.db.repositories.registry_repo import (
            get_active_revision,
            get_entries_for_revision,
        )
        from tidus.models.model_registry import ModelSpec

        active = await get_active_revision(session_factory)
        if active is None:
            log.warning("landing_update_skipped", reason="no_active_revision")
            return False

        # Load current revision specs
        entries = await get_entries_for_revision(session_factory, active.revision_id)
        all_specs: dict[str, ModelSpec] = {}
        for entry in entries:
            try:
                all_specs[entry.model_id] = ModelSpec.model_validate(entry.spec_json)
            except Exception:
                continue

        paid_specs = [s for s in all_specs.values() if not s.is_local and s.input_price > 0]
        if not paid_specs:
            log.warning("landing_update_skipped", reason="no_paid_specs")
            return False

        # Load previous (superseded) revision for diff + get issue number
        base_specs: dict[str, ModelSpec] = {}
        async with session_factory() as session:
            row = (await session.execute(text("""
                SELECT revision_id FROM model_catalog_revisions
                WHERE status = 'superseded'
                ORDER BY activated_at DESC
                LIMIT 1
            """))).fetchone()
            prev_revision_id = row[0] if row else None

            issue_number = (await session.execute(text(
                "SELECT COUNT(*) FROM model_catalog_revisions WHERE status != 'failed'"
            ))).scalar() or 1

        if prev_revision_id:
            prev_entries = await get_entries_for_revision(session_factory, prev_revision_id)
            for e in prev_entries:
                try:
                    base_specs[e.model_id] = ModelSpec.model_validate(e.spec_json)
                except Exception:
                    continue

        # Compute diffs
        changes   = self._compute_changes(all_specs, base_specs)
        new_models = self._compute_new_models(all_specs, base_specs)

        today = date.today()

        # Sort paid specs by blended cost descending for MODELS array + table
        paid_specs.sort(key=lambda s: (s.input_price + s.output_price) / 2, reverse=True)

        # Build the complete magazine HTML block
        magazine_html = self._build_magazine_html(
            paid_specs, all_specs, changes, new_models, today, issue_number
        )

        # Build MODELS JS array
        models_js = self._build_models_js(paid_specs)

        # Apply all replacements and write
        content = self._index_path.read_text(encoding="utf-8")
        content = self._replace_magazine_section(content, magazine_html)
        content = self._replace_models_array(content, models_js)
        content = self._update_dates(content, today)
        self._index_path.write_text(content, encoding="utf-8")

        log.info(
            "landing_magazine_updated",
            models=len(paid_specs),
            changes=len(changes),
            new_models=len(new_models),
            issue=issue_number,
            date=today,
        )

        self._git_push(today)
        return True

    # ── Price diff helpers ─────────────────────────────────────────────────────

    def _compute_changes(
        self,
        current: dict[str, Any],
        base: dict[str, Any],
    ) -> list[dict]:
        """Return list of price change dicts, sorted by absolute % move descending."""
        changes: list[dict] = []
        for model_id, spec in current.items():
            if spec.is_local:
                continue
            old_spec = base.get(model_id)
            if old_spec is None:
                continue
            for field, new_val, old_val in [
                ("input",  spec.input_price,  old_spec.input_price),
                ("output", spec.output_price, old_spec.output_price),
            ]:
                if old_val == 0 and new_val == 0:
                    continue
                ref = old_val if old_val != 0 else new_val
                delta_pct = (new_val - old_val) / ref * 100
                if abs(delta_pct) < _CHANGE_THRESHOLD_PCT:
                    continue
                changes.append({
                    "model_id": model_id,
                    "vendor":   spec.vendor,
                    "field":    field,        # "input" | "output"
                    "old_1m":   round(old_val * 1000, 4),
                    "new_1m":   round(new_val * 1000, 4),
                    "delta_pct": round(delta_pct, 2),
                })
        return sorted(changes, key=lambda c: -abs(c["delta_pct"]))

    def _compute_new_models(
        self,
        current: dict[str, Any],
        base: dict[str, Any],
    ) -> list[Any]:
        """Return ModelSpec objects for models in current but not in base."""
        new_ids = set(current) - set(base)
        return [
            current[mid] for mid in sorted(new_ids)
            if not current[mid].is_local
        ]

    # ── Magazine HTML builder ──────────────────────────────────────────────────

    def _build_magazine_html(
        self,
        paid_specs: list,
        all_specs: dict,
        changes: list[dict],
        new_models: list,
        today: date,
        issue: int,
    ) -> str:
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        month_year  = f"{months[today.month - 1]} {today.year}"
        date_label  = _date_label(today)
        next_sync   = _next_sync_label(today)

        drops     = [c for c in changes if c["delta_pct"] < 0]
        rises     = [c for c in changes if c["delta_pct"] > 0]
        n_drops   = len({c["model_id"] for c in drops})
        n_rises   = len({c["model_id"] for c in rises})
        n_updated = len({c["model_id"] for c in changes})
        n_new     = len(new_models)
        n_models  = len(paid_specs)

        headline       = self._build_headline(changes, new_models, today)
        exec_sum       = self._build_exec_summary(paid_specs, changes, new_models, today)
        changes_html   = self._build_changes_html(changes, new_models)
        tips_html      = self._build_tips_html(paid_specs, changes)

        return (
            '<div class="mag-card">\n'
            '      <div class="mag-header">\n'
            '        <div class="mag-logo">tidus<span>.</span>magazine</div>\n'
            '        <div class="mag-sub-tag">AI Model Market Intelligence &middot; Weekly Edition</div>\n'
            f'        <div class="mag-title">{headline}</div>\n'
            f'        <div class="mag-date">Week of {date_label} &middot; Issue #{issue}</div>\n'
            '      </div>\n'
            '      <div class="mag-stats-bar">\n'
            f'        <div class="ms"><div class="ms-val">{n_models}</div><div class="ms-lbl">Models Tracked</div></div>\n'
            f'        <div class="ms"><div class="ms-val g">{n_drops}</div><div class="ms-lbl">Price Drops</div></div>\n'
            f'        <div class="ms"><div class="ms-val r">{n_rises}</div><div class="ms-lbl">Price Rises</div></div>\n'
            f'        <div class="ms"><div class="ms-val">{n_updated}</div><div class="ms-lbl">Models Updated</div></div>\n'
            f'        <div class="ms"><div class="ms-val">{n_new}</div><div class="ms-lbl">New Models</div></div>\n'
            '      </div>\n'
            '      <div class="mag-body">\n'
            f'        <div class="exec-sum"><strong>Executive Summary:</strong> {exec_sum}</div>\n\n'
            f'        {changes_html}\n\n'
            f'        {tips_html}\n\n'
            f'        <div class="mag-sec-title" style="margin-top:28px">&#x1F4CB; Full Model Catalog &mdash; {month_year}</div>\n'
            f'        <p style="font-size:12px;color:#888;margin-bottom:14px">Ranked by blended cost &mdash; highest first &middot; All prices USD/1M tokens &middot; Updated {date_label}</p>\n'
            f'        {_PRICING_LEGEND_HTML}\n'
            '        <div class="table-wrap">\n'
            '          <table id="model-table">\n'
            '            <thead>\n'
            '              <tr>\n'
            '                <th class="col-rank">#</th>\n'
            '                <th>Vendor</th>\n'
            '                <th>Model</th>\n'
            '                <th>Blended $/1M</th>\n'
            '                <th>Input $/1M</th>\n'
            '                <th>Output $/1M</th>\n'
            '                <th>Context</th>\n'
            '              </tr>\n'
            '            </thead>\n'
            '            <tbody id="model-tbody"></tbody>\n'
            '          </table>\n'
            '        </div>\n'
            f'        <p class="table-note">Prices from official vendor pages via multi-source consensus &middot; Ranked by blended cost &middot; Updated {date_label}</p>\n'
            '      </div>\n'
            '      <div class="mag-footer">\n'
            f'        <p>Prices verified via multi-source consensus &middot; Next sync: {next_sync}</p>\n'
            '        <a href="#subscribe">Subscribe to weekly reports &rarr;</a>\n'
            '      </div>\n'
            '    </div>'
        )

    def _build_headline(
        self,
        changes: list[dict],
        new_models: list,
        today: date,
    ) -> str:
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        month = months[today.month - 1]
        year  = today.year

        bullets: list[str] = []
        seen: set[str] = set()
        for ch in changes:  # already sorted by abs delta
            mid = ch["model_id"]
            if mid in seen:
                continue
            seen.add(mid)
            vname = _VENDOR_NAMES.get(ch["vendor"], ch["vendor"].title())
            sign  = "&darr;" if ch["delta_pct"] < 0 else "&uarr;"
            bullets.append(f"{vname} {sign}{abs(ch['delta_pct']):.0f}%")
            if len(bullets) == 2:
                break

        if new_models and len(bullets) < 3:
            count = len(new_models)
            bullets.append(f"{count} New Model{'s' if count > 1 else ''}")

        if not bullets:
            return f"{month} {year} Edition &middot; All Prices Stable"

        return f"{month} {year} Update: {' &middot; '.join(bullets)}"

    def _build_exec_summary(
        self,
        paid_specs: list,
        changes: list[dict],
        new_models: list,
        today: date,
    ) -> str:
        drops = [c for c in changes if c["delta_pct"] < 0]
        rises = [c for c in changes if c["delta_pct"] > 0]
        paras: list[str] = []

        biggest_drop = min(drops, key=lambda c: c["delta_pct"]) if drops else None
        if biggest_drop:
            vname = _VENDOR_NAMES.get(biggest_drop["vendor"], biggest_drop["vendor"].title())
            fl = biggest_drop["field"]
            paras.append(
                f"<strong>{vname}</strong> leads this week &mdash; "
                f"<strong>{biggest_drop['model_id']}</strong> {fl} price dropped "
                f"<strong>{abs(biggest_drop['delta_pct']):.1f}%</strong> to "
                f"<strong>${biggest_drop['new_1m']:.3f}/1M</strong>."
            )

        if rises:
            vnames = sorted({_VENDOR_NAMES.get(c["vendor"], c["vendor"].title()) for c in rises})
            n_rise_models = len({c["model_id"] for c in rises})
            paras.append(
                f"{', '.join(vnames)} raised prices on "
                f"{n_rise_models} model{'s' if n_rise_models > 1 else ''}."
            )

        if new_models:
            names = ", ".join(f"<strong>{m.model_id}</strong>" for m in new_models[:3])
            paras.append(
                f"{len(new_models)} new model{'s' if len(new_models) > 1 else ''} "
                f"added to the catalog: {names}."
            )

        cheapest = min(paid_specs, key=lambda s: s.input_price, default=None)
        priciest = max(paid_specs, key=lambda s: _blended_1m(s.input_price, s.output_price), default=None)
        if cheapest and priciest:
            paras.append(
                f"The catalog spans <strong>{cheapest.model_id}</strong> at "
                f"<strong>${cheapest.input_price * 1000:.3f}/1M</strong> input "
                f"(cheapest) to <strong>{priciest.model_id}</strong> at "
                f"<strong>${_blended_1m(priciest.input_price, priciest.output_price):.2f}/1M</strong> "
                f"blended (most expensive)."
            )

        if not paras:
            return (
                f"All {len(paid_specs)} models stable this week &mdash; no pricing changes detected. "
                f"Routing teams can rely on current cost estimates for budget planning."
            )

        return "  ".join(paras)

    def _build_changes_html(self, changes: list[dict], new_models: list) -> str:
        if not changes and not new_models:
            return (
                '<div class="mag-sec-title">&#x1F4C9; Price Changes This Week</div>'
                '<p style="color:#888;font-size:13px;font-style:italic;margin-bottom:20px">'
                '&#x2714; No price changes this week &mdash; all models stable.</p>'
            )

        # Group per-field changes back to per-model cards
        by_model: dict[str, list[dict]] = defaultdict(list)
        for c in changes:
            by_model[c["model_id"]].append(c)

        cards: list[str] = []

        for model_id in sorted(by_model):
            mchanges = by_model[model_id]
            drops_m  = [c for c in mchanges if c["delta_pct"] < 0]
            card_type = "drop" if len(drops_m) >= (len(mchanges) - len(drops_m)) else "rise"
            vname = _VENDOR_NAMES.get(mchanges[0]["vendor"], mchanges[0]["vendor"].title())
            rows = ""
            for ch in mchanges:
                fl        = "Input" if ch["field"] == "input" else "Output"
                sign      = "&minus;" if ch["delta_pct"] < 0 else "+"
                val_cls   = "drop" if ch["delta_pct"] < 0 else "rise"
                badge_cls = "d-drop" if ch["delta_pct"] < 0 else "d-rise"
                rows += (
                    f'<div class="cc-row">'
                    f'<span class="cc-lbl">{fl}</span>'
                    f'<span class="cc-old">${ch["old_1m"]:.3f}</span>'
                    f'<span>&rarr;</span>'
                    f'<span class="cc-new {val_cls}">${ch["new_1m"]:.3f}</span>'
                    f'<span class="cc-delta {badge_cls}">{sign}{abs(ch["delta_pct"]):.1f}%</span>'
                    f'</div>'
                )
            cards.append(
                f'<div class="cc-card {card_type}">'
                f'<div class="cc-model">{model_id}</div>'
                f'<div class="cc-vendor">{vname}</div>'
                f'{rows}'
                f'</div>'
            )

        # New model cards
        for spec in new_models:
            vname = _VENDOR_NAMES.get(spec.vendor, spec.vendor.title())
            rows = (
                f'<div class="cc-row">'
                f'<span class="cc-lbl">Input</span>'
                f'<span class="cc-new new">${spec.input_price * 1000:.3f}</span>'
                f'<span class="cc-delta d-new">NEW</span>'
                f'</div>'
                f'<div class="cc-row">'
                f'<span class="cc-lbl">Output</span>'
                f'<span class="cc-new new">${spec.output_price * 1000:.3f}</span>'
                f'</div>'
            )
            cards.append(
                f'<div class="cc-card new">'
                f'<div class="cc-model">{spec.model_id}</div>'
                f'<div class="cc-vendor">{vname} &middot; New Catalog Entry</div>'
                f'{rows}'
                f'</div>'
            )

        return (
            '<div class="mag-sec-title">&#x1F4C9; Price Changes This Week</div>'
            f'<div class="changes-grid">{"".join(cards)}</div>'
        )

    def _build_tips_html(self, paid_specs: list, changes: list[dict]) -> str:
        input_drops = [c for c in changes if c["delta_pct"] < 0 and c["field"] == "input"]
        economy = sorted(
            [s for s in paid_specs if int(s.tier) == 3],
            key=lambda s: _blended_1m(s.input_price, s.output_price),
        )
        premium = sorted(
            [s for s in paid_specs if int(s.tier) == 1],
            key=lambda s: _blended_1m(s.input_price, s.output_price),
            reverse=True,
        )

        tips: list[str] = []

        if input_drops:
            bd    = min(input_drops, key=lambda c: c["delta_pct"])
            vname = _VENDOR_NAMES.get(bd["vendor"], bd["vendor"].title())
            tips.append(
                f'<div style="padding:14px;background:#f8f9ff;border-radius:10px;border:1px solid #e9ecef">'
                f'<div style="font-size:12px;font-weight:700;color:#0f3460;margin-bottom:6px">'
                f'&#x1F4C9; {bd["model_id"]} dropped {abs(bd["delta_pct"]):.0f}%</div>'
                f'<div style="font-size:12px;color:#555;line-height:1.6">'
                f'<strong>{vname}</strong>&rsquo;s <strong>{bd["model_id"]}</strong> input is now '
                f'<strong>${bd["new_1m"]:.3f}/1M</strong> (was ${bd["old_1m"]:.3f}/1M). '
                f'Consider benchmarking it against your current routing choice this week.</div></div>'
            )

        if economy and premium:
            eco, prem = economy[0], premium[0]
            eco_b  = _blended_1m(eco.input_price, eco.output_price)
            prem_b = _blended_1m(prem.input_price, prem.output_price)
            ratio  = prem_b / eco_b if eco_b > 0 else 0
            tips.append(
                f'<div style="padding:14px;background:#f8f9ff;border-radius:10px;border:1px solid #e9ecef">'
                f'<div style="font-size:12px;font-weight:700;color:#0f3460;margin-bottom:6px">'
                f'&#x1F4B0; Economy pick: {eco.model_id}</div>'
                f'<div style="font-size:12px;color:#555;line-height:1.6">'
                f'At <strong>${eco_b:.3f}/1M</strong> blended, <strong>{eco.model_id}</strong> is '
                f'<strong style="color:#16a34a">{ratio:.0f}&times; cheaper</strong> than '
                f'{prem.model_id} (${prem_b:.2f}/1M). '
                f'Ideal for classification, summarisation, and simple generation tasks.</div></div>'
            )

        if not tips:
            return ""

        return (
            '<div class="mag-sec-title">&#x1F4A1; Cost Optimisation Opportunities</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">{"".join(tips)}</div>'
        )

    # ── Replacement helpers ────────────────────────────────────────────────────

    def _replace_magazine_section(self, content: str, magazine_html: str) -> str:
        """Replace everything between TIDUS:MAG-CARD comment markers."""
        pattern = r'<!-- TIDUS:MAG-CARD:START -->.*?<!-- TIDUS:MAG-CARD:END -->'
        replacement = (
            f'<!-- TIDUS:MAG-CARD:START -->\n'
            f'      {magazine_html}\n'
            f'      <!-- TIDUS:MAG-CARD:END -->'
        )
        replaced, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
        if n == 0:
            log.warning("landing_magazine_markers_not_found")
        return replaced

    def _build_models_js(self, specs: list) -> str:
        lines = ["const MODELS = ["]
        for spec in specs:
            v   = _VENDOR_NAMES.get(spec.vendor, spec.vendor.title())
            b   = _js_num((spec.input_price + spec.output_price) / 2 * 1000)
            i   = _js_num(spec.input_price * 1000)
            o   = _js_num(spec.output_price * 1000)
            ctx = spec.max_context // 1000
            lines.append(
                f'  {{id:"{spec.model_id}", v:"{v}", t:{int(spec.tier)},'
                f' b:{b}, i:{i}, o:{o}, ctx:{ctx}}},'
            )
        lines.append("];")
        return "\n".join(lines)

    def _replace_models_array(self, content: str, new_js: str) -> str:
        """Replace the entire MODELS = [...]; block."""
        pattern  = r"const MODELS = \[.*?\];"
        replaced, n = re.subn(pattern, new_js, content, flags=re.DOTALL)
        if n == 0:
            log.warning("landing_models_array_not_found")
        return replaced

    def _update_dates(self, content: str, today: date) -> str:
        """Update 'Updated Month D, YYYY' occurrences to today."""
        label   = _date_label(today)
        pattern = r"Updated [A-Z][a-z]+ \d{1,2}, \d{4}"
        return re.sub(pattern, f"Updated {label}", content)

    def _git_push(self, today: date) -> None:
        """Commit index.html and push to tidus repo + github.io repo."""
        repo = self._index_path.parent
        msg  = f"magazine: update pricing table to {today}"

        # ── Push to tidus repo ────────────────────────────────────────────────
        try:
            subprocess.run(["git", "-C", str(repo), "add", "index.html"], check=True)
            result = subprocess.run(
                ["git", "-C", str(repo), "diff", "--cached", "--quiet"],
                capture_output=True,
            )
            if result.returncode != 0:  # there are staged changes
                subprocess.run(
                    ["git", "-C", str(repo), "commit", "-m", msg],
                    check=True,
                )
                subprocess.run(["git", "-C", str(repo), "push"], check=True)
                log.info("landing_pushed_tidus", date=today)
            else:
                log.info("landing_no_changes_to_push")
                return
        except subprocess.CalledProcessError as exc:
            log.error("landing_push_tidus_failed", error=str(exc))
            return

        # ── Push to github.io repo ────────────────────────────────────────────
        try:
            remote_result = subprocess.run(
                ["git", "-C", str(repo), "remote", "get-url", "origin"],
                capture_output=True, text=True, check=True,
            )
            tidus_remote      = remote_result.stdout.strip()
            github_io_remote  = re.sub(r"/tidus(\.git)?$", "/kensterinvest.github.io", tidus_remote)

            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["git", "clone", github_io_remote, tmp], check=True)
                import shutil
                shutil.copy(str(self._index_path), str(Path(tmp) / "index.html"))
                subprocess.run(["git", "-C", tmp, "add", "index.html"], check=True)
                subprocess.run(["git", "-C", tmp, "commit", "-m", msg], check=True)
                subprocess.run(["git", "-C", tmp, "push"], check=True)
            log.info("landing_pushed_github_io", date=today)
        except subprocess.CalledProcessError as exc:
            log.error("landing_push_github_io_failed", error=str(exc))
