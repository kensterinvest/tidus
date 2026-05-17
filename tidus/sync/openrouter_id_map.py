"""Shared OpenRouter → Tidus canonical id mapping.

Originally lived only in `tidus/sync/pricing/openrouter_source.py`, but the
discovery source (`tidus/sync/discovery/openrouter.py`) did its own
slash-strip canonicalization and produced a different answer for the same
OpenRouter id. That meant:

  pricing source:    "anthropic/claude-opus-4.6" → "claude-opus-4-6"  (dash)
  discovery source:  "anthropic/claude-opus-4.6" → "claude-opus-4.6"  (dot)

Same model, different model_id → ModelRegistry treated them as separate
entries, the dot version got auto-promoted into models.auto.yaml, and any
routing decision that selected it would 404 against api.anthropic.com
(which only knows the dash form).

This module is the single source of truth. Both sources call
`canonical_from_openrouter()` and get the same answer for the same input.
"""

from __future__ import annotations

# OpenRouter id → Tidus canonical id. Only listed when slash-strip + suffix
# stripping doesn't already produce the right answer. Keep this list short
# and well-justified; the slash-strip path covers most cases.
#
# How to extend: when OpenRouter ships a model Tidus knows under a different
# canonical id, add the exact OpenRouter id (left) → Tidus id (right). Verify
# both ids actually exist in their respective catalogs first.
OPENROUTER_TO_TIDUS: dict[str, str] = {
    # Anthropic — OpenRouter uses dot-versioned names (claude-opus-4.6);
    # Tidus's canonical form uses dashes (claude-opus-4-6) matching the
    # Anthropic API's model_id format. Mapping is REQUIRED for routing —
    # without it, the discovery source produces dot-versioned duplicates
    # of hand-curated entries that 404 if selected.
    "anthropic/claude-opus-4.7":     "claude-opus-4-7",
    "anthropic/claude-opus-4.6":     "claude-opus-4-6",
    "anthropic/claude-sonnet-4.6":   "claude-sonnet-4-6",
    "anthropic/claude-haiku-4.5":    "claude-haiku-4-5",
    # OpenAI — codex variants
    "openai/gpt-5-codex":            "gpt-5-codex",
    "openai/codex-mini":             "codex-mini-latest",
    "openai/gpt-oss-120b":           "gpt-oss-120b",
    # DeepSeek — Tidus uses short ids; OpenRouter often appends -chat / -base
    "deepseek/deepseek-r1":          "deepseek-r1",
    "deepseek/deepseek-chat":        "deepseek-v3",
    "deepseek/deepseek-v3":          "deepseek-v3",
    "deepseek/deepseek-v4":          "deepseek-v4",
    # xAI
    "x-ai/grok-4":                   "grok-4",
    "x-ai/grok-3":                   "grok-3",
    "x-ai/grok-3-fast":              "grok-3-fast",
    # Moonshot
    "moonshotai/kimi-k2.5":          "kimi-k2.5",
    # Mistral
    "mistralai/mistral-large":       "mistral-large-3",
    "mistralai/mistral-medium":      "mistral-medium",
    "mistralai/mistral-small":       "mistral-small",
    "mistralai/mistral-nemo":        "mistral-nemo",
    "mistralai/codestral":           "codestral",
    "mistralai/devstral":            "devstral",
    "mistralai/devstral-small":      "devstral-small",
}


def strip_variant(or_id: str) -> str:
    """OpenRouter appends `:free` / `:nitro` / `:beta` to some models.
    Strip the suffix so canonical lookup doesn't care which variant
    OpenRouter is brokering on a given day."""
    return or_id.split(":", 1)[0]


def canonical_from_openrouter(or_id: str) -> str | None:
    """Return the Tidus canonical id for an OpenRouter id, or None if
    the model can't be canonicalized (no slash, empty).

    Resolution order:
      1. Strip OpenRouter variant suffix (`:free`, `:nitro`, ...).
      2. Exact match in OPENROUTER_TO_TIDUS → use mapped value.
      3. Drop the `vendor/` prefix and use what's left.

    Returning None — rather than guessing — is intentional. Adding a
    new model to the routing catalog needs `config/models.yaml` + an
    entry in OPENROUTER_TO_TIDUS if naming diverges from slash-strip.
    """
    base = strip_variant(or_id)
    if base in OPENROUTER_TO_TIDUS:
        return OPENROUTER_TO_TIDUS[base]
    if "/" not in base:
        return None
    _, suffix = base.split("/", 1)
    return suffix or None
