"""AutoPromoter — converts discovered+priced models into routable entries.

The weekly sync used to surface new vendor models as "pending review" rows
in the magazine and stop there. Promotion required a human edit to
`config/models.yaml` + `hardcoded_source.py` before routing would consider
the new model. That manual step accumulated months of backlog.

This module closes the loop: when discovery surfaces a model with usable
pricing from OpenRouter, the promoter writes a conservative ModelSpec to
`config/models.auto.yaml`. ModelRegistry.load() merges that file in
alongside the hand-curated catalog (auto entries lose conflicts), so the
next pipeline run picks the model up and starts routing to it.

Guardrails (aggressive promotion, but bounded):
  * Vendor must be in the known allow-list. Promotes Google, Anthropic,
    OpenAI, DeepSeek, xAI, Mistral, Moonshot, Cohere, Qwen — same set
    Tidus already has adapters for. A new "Foobar AI" model gets surfaced
    in the discovery report but never auto-promoted.
  * Both prompt AND completion pricing must be > 0. Free-tier or
    unpriced entries are ignored.
  * Models matching obvious junk patterns (`:free`, `:beta`, `:nightly`,
    `-preview`, `-experimental`, `-test`) are skipped. Pre-GA gates
    promotion behind an explicit yaml edit.
  * Defaults are deliberately conservative: tier=3 (economy), capabilities
    [chat] only, max_complexity=moderate, no fallbacks, enabled=true.
    These don't preempt premium routing for high-stakes traffic but do
    make the model selectable for simple/moderate tasks.

Kill-switch:
  Setting `auto_promote_enabled=False` turns the whole pass into a no-op
  while leaving the file in place. The next run rebuilds from a clean
  slate when re-enabled.

File format:
  Same schema as `config/models.yaml`. Header comment marks it as
  generated — operators should NOT hand-edit it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import structlog
import yaml

from tidus.models.model_registry import ModelSpec
from tidus.sync.ai_verifier import (
    ClaudeDiscoveryVerifier,
    DiscoveryCandidate,
)
from tidus.sync.discovery.base import DiscoveredModel

log = structlog.get_logger(__name__)


# Vendors we have adapters for. Promotion to a vendor outside this set
# would create a routable entry that no adapter can serve — silently
# 500-ing in production. Keep this list aligned with tidus/adapters/.
_KNOWN_VENDORS: frozenset[str] = frozenset({
    "openai", "anthropic", "google", "mistral", "deepseek", "xai",
    "moonshot", "cohere", "qwen", "alibaba", "meta",
})

# Tokenizer guess by vendor — best-effort. Wrong guess just means worse
# token-count estimates; routing still works.
_TOKENIZER_BY_VENDOR: dict[str, str] = {
    "openai":    "tiktoken_o200k",
    "anthropic": "anthropic",
    "google":    "google",
    "mistral":   "sentencepiece",
    "deepseek":  "tiktoken_cl100k",
    "xai":       "tiktoken_cl100k",
    "moonshot":  "tiktoken_cl100k",
    "cohere":    "tiktoken_cl100k",
    "qwen":      "tiktoken_cl100k",
    "alibaba":   "tiktoken_cl100k",
    "meta":      "tiktoken_cl100k",
}

# Patterns that mark a model as pre-GA / variant / not-yet-stable. We
# trust the magazine's pending-review surface for these and require a
# hand edit before they route.
_SKIP_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r":(free|nitro|beta|nightly|experimental|preview)"),
    re.compile(r"-(preview|experimental|test|alpha|beta|nightly)(-|$)"),
    re.compile(r"^(test-|stub-)"),
)


@dataclass
class AutoPromoteResult:
    """Returned by AutoPromoter.run() for the weekly_full_sync log line."""

    promoted: list[ModelSpec]            # entries written this run
    skipped_known: int                   # already in the hand-curated yaml
    skipped_unknown_vendor: int          # vendor not in adapter set
    skipped_no_price: int                # missing or zero pricing
    skipped_variant: int                 # `:free`, `-preview`, etc.
    ai_rejected: int = 0                 # passed rule-based filters but Claude said no

    @property
    def total_evaluated(self) -> int:
        return (
            len(self.promoted)
            + self.skipped_known
            + self.skipped_unknown_vendor
            + self.skipped_no_price
            + self.skipped_variant
            + self.ai_rejected
        )


def _is_skip_variant(or_id: str) -> bool:
    return any(p.search(or_id) for p in _SKIP_PATTERNS)


def _parse_pricing(meta: dict) -> tuple[float, float, float, float] | None:
    """Return (input_per_1k, output_per_1k, cache_read_per_1k, cache_write_per_1k)
    or None if the model lacks usable pricing.

    OpenRouter stores per-token prices as decimal strings; Tidus stores
    per-1K floats in ModelSpec. Multiplying by 1000 converts.
    """
    pricing = meta.get("pricing") or {}
    try:
        prompt_per_token = float(pricing.get("prompt") or 0)
        completion_per_token = float(pricing.get("completion") or 0)
    except (TypeError, ValueError):
        return None
    if prompt_per_token <= 0 or completion_per_token <= 0:
        return None
    try:
        cache_r = float(pricing.get("input_cache_read") or 0)
        cache_w = float(pricing.get("input_cache_write") or 0)
    except (TypeError, ValueError):
        cache_r, cache_w = 0.0, 0.0
    return (
        prompt_per_token * 1000,
        completion_per_token * 1000,
        cache_r * 1000,
        cache_w * 1000,
    )


def _build_spec(model: DiscoveredModel, prices: tuple[float, float, float, float]) -> ModelSpec:
    """Build the conservative ModelSpec for an auto-promoted discovery."""
    input_p, output_p, cache_r, cache_w = prices
    raw_meta = model.raw_metadata or {}
    context_length = int(raw_meta.get("context_length") or 8000)
    if context_length <= 0:
        context_length = 8000

    tokenizer = _TOKENIZER_BY_VENDOR.get(model.vendor, "tiktoken_cl100k")
    display = f"[auto-promoted] {model.display_name or model.model_id}"

    return ModelSpec.model_validate({
        "model_id":         model.model_id,
        "display_name":     display,
        "vendor":           model.vendor,
        "tier":             3,                       # economy default
        "max_context":      context_length,
        "input_price":      input_p,
        "output_price":     output_p,
        "cache_read_price": cache_r,
        "cache_write_price": cache_w,
        "tokenizer":        tokenizer,
        "latency_p50_ms":   1500,                    # placeholder; health probe will refine
        "capabilities":     ["chat"],
        "min_complexity":   "simple",
        "max_complexity":   "moderate",              # never auto-claims critical
        "is_local":         False,
        "enabled":          True,
        "deprecated":       False,
        "fallbacks":        [],                      # operator adds these on review
        "last_price_check": date.today().isoformat(),
    })


class AutoPromoter:
    """Promotes priced, vendor-known discovered models into models.auto.yaml.

    Designed to be called from `scripts/weekly_full_sync.py` AFTER
    discovery returns its report and BEFORE the price-sync pipeline runs.
    Writing happens before pipeline.run_price_sync_cycle() so the
    pipeline picks the new entries up in `yaml_by_id` on the same cycle.
    """

    def __init__(
        self,
        *,
        auto_yaml_path: str | Path = "config/models.auto.yaml",
        enabled: bool = True,
        ai_verifier: ClaudeDiscoveryVerifier | None = None,
    ) -> None:
        self._path = Path(auto_yaml_path)
        self._enabled = enabled
        self._ai_verifier = ai_verifier

    async def run(
        self,
        *,
        discovered: Iterable[DiscoveredModel],
        hand_curated_ids: set[str],
    ) -> AutoPromoteResult:
        """Evaluate discovered models and rewrite config/models.auto.yaml.

        The auto-yaml is fully rewritten each run — entries that no longer
        show up in discovery, or that an operator has since promoted to
        hand-curated, disappear automatically. First-seen timestamps live
        in discovered_models.json, not here.
        """
        promoted: list[ModelSpec] = []
        skipped_known = 0
        skipped_unknown_vendor = 0
        skipped_no_price = 0
        skipped_variant = 0
        seen_canonicals: set[str] = set()

        if not self._enabled:
            log.info("auto_promote_disabled")
            return AutoPromoteResult(
                promoted=[],
                skipped_known=0,
                skipped_unknown_vendor=0,
                skipped_no_price=0,
                skipped_variant=0,
            )

        # Materialize once — `discovered` may be a generator, and the AI
        # verifier branch below reads it a second time.
        discovered = list(discovered)

        for model in discovered:
            canonical = model.model_id
            if not canonical or canonical in seen_canonicals:
                continue
            seen_canonicals.add(canonical)

            if canonical in hand_curated_ids:
                # Operator has explicitly vetted this id — leave it alone.
                skipped_known += 1
                continue

            if model.vendor not in _KNOWN_VENDORS:
                skipped_unknown_vendor += 1
                continue

            if _is_skip_variant(model.vendor_id or canonical):
                skipped_variant += 1
                continue

            prices = _parse_pricing(model.raw_metadata or {})
            if prices is None:
                skipped_no_price += 1
                continue

            try:
                spec = _build_spec(model, prices)
            except Exception as exc:
                log.warning(
                    "auto_promote_spec_build_failed",
                    model_id=canonical,
                    error=str(exc),
                )
                continue

            promoted.append(spec)
            log.info(
                "auto_promoted",
                model_id=canonical,
                vendor=model.vendor,
                input_price=spec.input_price,
                output_price=spec.output_price,
            )

        # AI confirmation pass — the rule-based filters above are the floor;
        # Claude is the ceiling. Closes the gap where a plausible-looking
        # name + plausible-looking price slip past structural rules.
        ai_rejected_count = 0
        if self._ai_verifier and self._ai_verifier.is_available and promoted:
            spec_by_id = {s.model_id: s for s in promoted}
            # Build DiscoveryCandidate inputs from the matching DiscoveredModel
            # objects (re-read raw_metadata for the prices the model showed
            # before promotion).
            discovered_by_id = {d.model_id: d for d in discovered}
            candidates: list[DiscoveryCandidate] = []
            for spec in promoted:
                disc = discovered_by_id.get(spec.model_id)
                candidates.append(
                    DiscoveryCandidate(
                        model_id=spec.model_id,
                        vendor=spec.vendor,
                        openrouter_id=(disc.vendor_id if disc else spec.model_id),
                        display_name=(disc.display_name if disc else None),
                        input_price_per_1m=spec.input_price * 1000,
                        output_price_per_1m=spec.output_price * 1000,
                    )
                )
            verdict = await self._ai_verifier.verify(candidates)
            log.info(
                "auto_promote_ai_verify_complete",
                candidates=len(candidates),
                accepted=len(verdict.accepted),
                rejected=len(verdict.rejected),
                skipped=verdict.skipped,
            )
            if verdict.rejected:
                rejected_ids = {r.candidate.model_id for r in verdict.rejected}
                promoted = [s for s in promoted if s.model_id not in rejected_ids]
                ai_rejected_count = len(verdict.rejected)
                # Re-sync spec_by_id (used nowhere downstream, but keeps the
                # invariant in case anything is appended later).
                _ = spec_by_id

        self._write_file(promoted)
        return AutoPromoteResult(
            promoted=promoted,
            skipped_known=skipped_known,
            skipped_unknown_vendor=skipped_unknown_vendor,
            skipped_no_price=skipped_no_price,
            skipped_variant=skipped_variant,
            ai_rejected=ai_rejected_count,
        )

    def _write_file(self, specs: list[ModelSpec]) -> None:
        """Atomically rewrite the auto.yaml file. Empty list → file with just
        the header (so an operator can still see the alarm: "auto promoter
        ran but nothing qualified")."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            "# Tidus auto-promoted model catalog — DO NOT EDIT BY HAND.\n"
            "# Generated by tidus/sync/auto_promote.py on each weekly sync.\n"
            "# Entries are vendor-discovered models with live pricing that\n"
            "# have NOT been hand-vetted. To promote an entry to vetted status,\n"
            "# copy it to config/models.yaml and remove it from here (the next\n"
            "# sync would otherwise rewrite it back).\n"
            "#\n"
            "# Safety: ModelRegistry.load() drops auto entries whose model_id\n"
            "# also appears in models.yaml — hand-curated always wins.\n"
        )

        payload = {
            "models": [
                spec.model_dump(mode="json", exclude_none=False)
                for spec in specs
            ]
        }

        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(
                payload,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        tmp.replace(self._path)
