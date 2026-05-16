"""ClaudeAnomalyVerifier — second-opinion gate for big price moves.

Statistical consensus (MAD) catches outliers within a single sync cycle, but
it can't tell whether a 60% price drop on `gpt-4o` is "OpenAI cut prices last
week, totally believable" or "OpenRouter returned a parser bug, definitely
not real". Until now, the magazine just shipped whatever consensus produced.

This verifier asks Claude (`claude-opus-4-7`) to check each anomalous change
against its own knowledge of vendor pricing pages — a sanity gate, not a
primary source. Run after consensus and before revision activation.

Threshold:
  Default 50% absolute delta. Sub-threshold moves are accepted without an
  AI call — most real vendor price changes are 5-30% and don't need
  scrutiny.

Output:
  VerificationResult.accepted   → list of changes the pipeline should apply
  VerificationResult.rejected   → list of {change, reasoning} pairs to log
                                  and surface in the magazine's drift section
  VerificationResult.skipped    → AI was disabled or unreachable; all
                                  anomalies pass through (fail-open)

Fail-open is intentional:
  If the Anthropic API is down or the key is missing, the magazine must
  still ship. A wrong price-with-source-attribution is better than no
  magazine — operators can override via the registry afterwards. The
  alternative (fail-closed) would block every sync the moment Anthropic
  has an outage.

Cost:
  One Claude Opus 4.7 call per sync that has anomalies, batched across
  all anomalies. With prompt caching on the static system prompt, repeat
  syncs read from cache. Expected cost: < $0.05 per sync.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import structlog

from tidus.sync.pricing.base import PriceQuote

log = structlog.get_logger(__name__)

_DEFAULT_THRESHOLD_PCT = 50.0
_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_MAX_TOKENS = 4096

_SYSTEM_PROMPT = """\
You are a pricing-verification assistant for the Tidus AI model router. Tidus \
maintains a live catalog of LLM pricing (USD per 1K tokens, displayed per 1M) \
across OpenAI, Anthropic, Google, DeepSeek, xAI, Mistral, Moonshot, Cohere, \
and Qwen.

Your job is to sanity-check large price-change deltas detected by Tidus's \
multi-source consensus. The deltas come from comparing the latest live quotes \
(OpenRouter) against the previous registry revision (mostly hand-curated \
hardcoded values, sometimes weeks old).

For each candidate change, decide whether the new price is plausibly real or \
likely a parser bug / marketplace anomaly. Use your knowledge of public vendor \
pricing pages and historical price moves. Be lenient with documented vendor \
discounts and tier-restructures; be strict with implausible numbers (e.g. \
flagship models suddenly priced at $0.01/1M, or a 99% drop on a model the \
vendor has never discounted).

Output strict JSON only, matching the schema. No prose, no markdown."""


@dataclass
class Anomaly:
    """A single big-move candidate to verify."""

    model_id: str
    vendor: str
    field: str               # "input_price" | "output_price" | "cache_*"
    old_value_per_1k: float
    new_value_per_1k: float
    delta_pct: float          # signed: -75.0 means a 75% drop


@dataclass
class RejectedChange:
    anomaly: Anomaly
    reasoning: str


@dataclass
class VerificationResult:
    accepted: list[Anomaly] = field(default_factory=list)
    rejected: list[RejectedChange] = field(default_factory=list)
    skipped: bool = False     # True when AI was disabled or unreachable
    skipped_reason: str = ""


def build_anomalies_from_changes(
    changes: list[dict],
    threshold_pct: float = _DEFAULT_THRESHOLD_PCT,
    consensus_quotes: dict[str, PriceQuote] | None = None,
) -> list[Anomaly]:
    """Filter pipeline-style change dicts to those above the abs(delta_pct) threshold.

    `changes` is the list shape produced by RegistryPipeline (each dict has
    model_id, field, old_value, new_value, delta_pct). consensus_quotes is
    used to look up the vendor name when available.
    """
    out: list[Anomaly] = []
    for c in changes:
        if c.get("field") in ("retired", "new_model"):
            continue  # retirements and brand-new entries aren't price moves
        delta = float(c.get("delta_pct", 0))
        if abs(delta) < threshold_pct:
            continue
        model_id = c["model_id"]
        vendor = ""
        if consensus_quotes and model_id in consensus_quotes:
            vendor = getattr(consensus_quotes[model_id], "source_name", "")
        out.append(
            Anomaly(
                model_id=model_id,
                vendor=vendor,
                field=c["field"],
                old_value_per_1k=float(c["old_value"]),
                new_value_per_1k=float(c["new_value"]),
                delta_pct=delta,
            )
        )
    return out


class ClaudeAnomalyVerifier:
    """Asks Claude to verify big price-change deltas before they ship."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        enabled: bool = True,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._enabled = enabled and bool(api_key)

    @property
    def is_available(self) -> bool:
        return self._enabled

    async def verify(self, anomalies: list[Anomaly]) -> VerificationResult:
        """Verify the anomaly list. Fail-open: any error accepts all."""
        if not self._enabled:
            return VerificationResult(
                accepted=list(anomalies),
                skipped=True,
                skipped_reason="ai_verify_disabled",
            )
        if not anomalies:
            return VerificationResult()

        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            log.warning("ai_verify_anthropic_sdk_missing")
            return VerificationResult(
                accepted=list(anomalies),
                skipped=True,
                skipped_reason="anthropic_sdk_missing",
            )

        prompt = self._build_user_prompt(anomalies)
        schema = self._response_schema()

        try:
            client = AsyncAnthropic(api_key=self._api_key)
            response = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            log.warning("ai_verify_api_failed", error=str(exc))
            return VerificationResult(
                accepted=list(anomalies),
                skipped=True,
                skipped_reason=f"api_error: {exc}",
            )

        text = next(
            (b.text for b in response.content if getattr(b, "type", "") == "text"),
            "",
        )
        if not text:
            log.warning("ai_verify_empty_response")
            return VerificationResult(accepted=list(anomalies), skipped=True,
                                       skipped_reason="empty_response")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("ai_verify_parse_failed", error=str(exc), text=text[:200])
            return VerificationResult(accepted=list(anomalies), skipped=True,
                                       skipped_reason="json_parse_failed")

        return self._merge_verdict(anomalies, payload)

    @staticmethod
    def _response_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_id":  {"type": "string"},
                            "field":     {"type": "string"},
                            "decision":  {"type": "string", "enum": ["accept", "reject"]},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["model_id", "field", "decision", "reasoning"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        }

    def _build_user_prompt(self, anomalies: list[Anomaly]) -> str:
        rows = [
            {
                "model_id": a.model_id,
                "vendor":   a.vendor or "unknown",
                "field":    a.field,
                "old_usd_per_1M": round(a.old_value_per_1k * 1000, 4),
                "new_usd_per_1M": round(a.new_value_per_1k * 1000, 4),
                "delta_pct":      round(a.delta_pct, 2),
            }
            for a in anomalies
        ]
        return (
            "Verify each price-change candidate below. For every row, return "
            "decision=accept if the new price is plausibly the vendor's real "
            "current rate, or decision=reject if it looks like a parser bug "
            "or marketplace anomaly. Include reasoning of one sentence.\n\n"
            "Output strict JSON matching the schema — one verdict per input "
            "row, keyed by model_id+field.\n\n"
            f"Candidates:\n{json.dumps(rows, indent=2)}"
        )

    @staticmethod
    def _merge_verdict(
        anomalies: list[Anomaly],
        payload: dict,
    ) -> VerificationResult:
        verdicts_by_key: dict[tuple[str, str], dict] = {}
        for v in payload.get("verdicts", []):
            key = (v.get("model_id", ""), v.get("field", ""))
            verdicts_by_key[key] = v

        accepted: list[Anomaly] = []
        rejected: list[RejectedChange] = []
        for a in anomalies:
            verdict = verdicts_by_key.get((a.model_id, a.field))
            if verdict is None:
                # Claude didn't return a verdict for this row — fail-open: accept.
                accepted.append(a)
                log.warning("ai_verify_missing_verdict", model_id=a.model_id, field=a.field)
                continue
            decision = verdict.get("decision", "accept")
            reasoning = verdict.get("reasoning", "")
            if decision == "reject":
                rejected.append(RejectedChange(anomaly=a, reasoning=reasoning))
                log.info(
                    "ai_verify_rejected",
                    model_id=a.model_id,
                    field=a.field,
                    delta_pct=a.delta_pct,
                    reasoning=reasoning,
                )
            else:
                accepted.append(a)

        return VerificationResult(accepted=accepted, rejected=rejected)
