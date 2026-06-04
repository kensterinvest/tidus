"""PriceConsensus — MAD-based outlier detection across multiple pricing sources.

Algorithm (Modified Z-Score using Median Absolute Deviation):

  For each model with N quotes:
    median_price = median(all quotes' input_price)
    MAD          = median(|price_i - median_price| for each quote i)
    z_score(i)   = 0.6745 × |price_i − median_price| / MAD

    Reject quote if z_score > outlier_z_threshold (default 3.5)
    If MAD == 0 (all sources agree exactly): no rejection
    If only one source: accept but lower effective confidence by 0.2
    If all sources rejected: raise ConsensusError (systemic data issue)

After outlier removal, the highest-confidence non-outlier quote wins.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import structlog

from tidus.sync.pricing.base import PriceQuote

log = structlog.get_logger(__name__)

_DEFAULT_Z_THRESHOLD = 3.5
_SINGLE_SOURCE_CONFIDENCE_PENALTY = 0.2
# MAD screening is inert with exactly two sources (z is always 0.6745). For that
# regime we fall back to a relative-spread alarm: flag when max/min exceeds this
# ratio, and dock the winner's confidence so the distrust propagates downstream.
_DEFAULT_TWO_SOURCE_RATIO_THRESHOLD = 2.0
_TWO_SOURCE_DISAGREEMENT_PENALTY = 0.3


class ConsensusError(Exception):
    """Raised when all quotes for a model are rejected as outliers."""


@dataclass
class ConsensusResult:
    """Output of PriceConsensus.resolve()."""

    quotes: dict[str, PriceQuote]         # model_id → winning quote
    single_source_models: list[str]        # models with only one source
    rejection_summary: dict[str, list[str]] = field(default_factory=dict)
    # rejection_summary: {model_id: [rejected_source_name, ...]}
    screening_bypassed: list[str] = field(default_factory=list)
    # models with exactly two sources, where MAD outlier screening is inert
    flagged_disagreements: dict[str, float] = field(default_factory=dict)
    # {model_id: price_ratio} for two-source spreads beyond the ratio threshold


class PriceConsensus:
    """Resolves a list of quotes from multiple sources into one quote per model."""

    def __init__(
        self,
        outlier_z_threshold: float = _DEFAULT_Z_THRESHOLD,
        two_source_ratio_threshold: float = _DEFAULT_TWO_SOURCE_RATIO_THRESHOLD,
    ) -> None:
        self._z_threshold = outlier_z_threshold
        self._two_source_ratio_threshold = two_source_ratio_threshold

    @staticmethod
    def _penalise_confidence(q: PriceQuote, penalty: float) -> PriceQuote:
        """Return a copy of ``q`` with ``source_confidence`` reduced (floored 0)."""
        return PriceQuote(
            model_id=q.model_id,
            input_price=q.input_price,
            output_price=q.output_price,
            cache_read_price=q.cache_read_price,
            cache_write_price=q.cache_write_price,
            currency=q.currency,
            effective_date=q.effective_date,
            retrieved_at=q.retrieved_at,
            source_name=q.source_name,
            source_confidence=max(0.0, q.source_confidence - penalty),
            evidence_url=q.evidence_url,
        )

    def resolve(self, all_quotes: list[PriceQuote]) -> ConsensusResult:
        """Apply MAD outlier detection and return the winning quote per model.

        Args:
            all_quotes: Combined list of quotes from all sources.

        Returns:
            ConsensusResult with winning quotes + single-source model list.

        Raises:
            ConsensusError: If ALL quotes for any model are statistical outliers.
        """
        from collections import defaultdict

        by_model: dict[str, list[PriceQuote]] = defaultdict(list)
        for q in all_quotes:
            by_model[q.model_id].append(q)

        winners: dict[str, PriceQuote] = {}
        single_source_models: list[str] = []
        rejection_summary: dict[str, list[str]] = {}
        screening_bypassed: list[str] = []
        flagged_disagreements: dict[str, float] = {}

        for model_id, quotes in by_model.items():
            if len(quotes) == 1:
                # Single source — accept but reduce confidence
                q = quotes[0]
                penalised = PriceQuote(
                    model_id=q.model_id,
                    input_price=q.input_price,
                    output_price=q.output_price,
                    cache_read_price=q.cache_read_price,
                    cache_write_price=q.cache_write_price,
                    currency=q.currency,
                    effective_date=q.effective_date,
                    retrieved_at=q.retrieved_at,
                    source_name=q.source_name,
                    source_confidence=max(0.0, q.source_confidence - _SINGLE_SOURCE_CONFIDENCE_PENALTY),
                    evidence_url=q.evidence_url,
                )
                winners[model_id] = penalised
                single_source_models.append(model_id)
                log.debug("consensus_single_source", model_id=model_id, source=q.source_name)
                continue

            # Multi-source: apply MAD outlier detection on input_price
            prices = [q.input_price for q in quotes]
            median_price = statistics.median(prices)
            mad = statistics.median([abs(p - median_price) for p in prices])

            # MAD is statistically inert at exactly two sources (z is always
            # 0.6745, below any sane threshold). Record that screening did not
            # really run, and fall back to a relative-spread alarm so a gross
            # 2-source disagreement is not silently accepted.
            if len(quotes) == 2:
                screening_bypassed.append(model_id)
                lo, hi = min(prices), max(prices)
                ratio = (hi / lo) if lo > 0 else float("inf")
                if ratio > self._two_source_ratio_threshold:
                    flagged_disagreements[model_id] = round(ratio, 3)
                    log.warning(
                        "consensus_two_source_disagreement",
                        model_id=model_id,
                        prices=prices,
                        ratio=round(ratio, 3),
                        sources=[q.source_name for q in quotes],
                    )

            non_outliers: list[PriceQuote] = []
            rejected: list[str] = []

            for q in quotes:
                if mad == 0:
                    # All sources agree exactly — no rejection possible
                    non_outliers.append(q)
                else:
                    z = 0.6745 * abs(q.input_price - median_price) / mad
                    if z <= self._z_threshold:
                        non_outliers.append(q)
                    else:
                        rejected.append(q.source_name)
                        log.warning(
                            "consensus_quote_rejected",
                            model_id=model_id,
                            source=q.source_name,
                            z_score=round(z, 3),
                            threshold=self._z_threshold,
                        )

            if rejected:
                rejection_summary[model_id] = rejected

            if not non_outliers:
                raise ConsensusError(
                    f"All {len(quotes)} quotes for model {model_id!r} rejected as outliers. "
                    f"Sources: {[q.source_name for q in quotes]}. "
                    "This indicates a systemic data quality problem — manual investigation required."
                )

            # Pick highest-confidence non-outlier.
            # Tie-breaker (Fix 8): prefer the more recent effective_date, then
            # the more recent retrieved_at. Older feeds with stale prices lose
            # to fresher feeds even when confidence scores are identical.
            winner = max(
                non_outliers,
                key=lambda q: (q.source_confidence, q.effective_date, q.retrieved_at),
            )
            if model_id in flagged_disagreements:
                winner = self._penalise_confidence(winner, _TWO_SOURCE_DISAGREEMENT_PENALTY)
            winners[model_id] = winner
            log.debug(
                "consensus_winner",
                model_id=model_id,
                source=winner.source_name,
                input_price=winner.input_price,
                rejected_count=len(rejected),
            )

        return ConsensusResult(
            quotes=winners,
            single_source_models=single_source_models,
            rejection_summary=rejection_summary,
            screening_bypassed=screening_bypassed,
            flagged_disagreements=flagged_disagreements,
        )
