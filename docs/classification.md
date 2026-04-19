# Multi-Axis Request Classification — Domain, Complexity, Privacy

**Author:** Kenny Wong (<lapkei01@gmail.com>)
**Published:** 2026-04-20
**Version:** 1.0 — v1.3.0 auto-classification layer
**Status:** Research complete. Shipping preparation.
**Current baseline:** 89.2% confidential recall (CI [80.7%, 94.2%]) on cross-family-validated ground truth.

Tidus classifies every incoming AI request across three dimensions before routing:

| Axis | Values | Purpose in routing |
|---|---|---|
| **domain** | chat, code, reasoning, extraction, classification, summarization, creative | Selects models with matching domain capabilities (hard constraint at Stage 1) |
| **complexity** | simple, moderate, complex, critical | Sets tier ceiling (Stage 3) — `critical` routes to tier 1 only; `simple` permits all tiers |
| **privacy** | public, internal, confidential | Under `strict` enforcement, `confidential` forces local-only routing |

Classification runs entirely within the deployment boundary via a five-tier cascade: caller override → heuristic regex + keywords → trained sentence-embedding encoder → Presidio NER → language-model fallback. No raw prompt ever leaves the deployment except via the explicit routing decision that follows classification (and that routing respects the classification).

## Full technical specification

See the **Technical Specification** section at the bottom of the [Tidus landing page](https://kensterinvest.github.io/#classification-workflow) (also available in-repo at `index.html`, section id `classification-workflow`). That document covers:

1. Abstract and field of application
2. Problem statement and prior art gap
3. Five-tier architecture with per-tier latency budgets
4. Asymmetric-safety OR-rule for combining tier outputs
5. Two-valued `privacy_enforcement` config (`strict` / `disabled`) and its distinction from classifier location
6. Labeled corpus (2,669 WildChat prompts + 25 manual overrides + 14 IRR-adjudicated overrides)
7. Three empirical validation studies:
   - Cross-family inter-rater reliability (Claude + GPT + Gemini, weighted κ 0.68–0.78)
   - Ensemble rule evaluation on adjudicated ground truth (E1 = 89.2%, E2 = 83.1%)
   - Entity/topic bifurcation analysis (50/50 split, motivating the Tier 2b/Tier 5 separation)
   - Credential re-leak observation (longitudinal user-scoped leak pattern)
8. Current shipping baseline and self-improvement trajectory
9. Per-lever accuracy improvement roadmap (→ 95–97% within 12 months)
10. Enterprise deployment guide
11. Summary of claims-adjacent novel aspects (patent-supporting)
12. References (reference-only style, 17 citations)

## Configuration

See `config/policies.yaml` for:
- `privacy_enforcement` — `strict` (default) or `disabled`
- `classification.presidio_rule` — `E1` (default, 89.2% recall, 49% flag rate) or `E2` (83.1% recall, 19% flag rate)

See `plan.md` for the full shipping sequence (Stages A through D) and the four-lever accuracy improvement roadmap.
See `findings.md` for the research evidence document with full tables and methodology.

## Reproduction artifacts

All studies are reproducible from the repository. Relevant scripts:

- `scripts/label_wildchat.py` — rubric (`SYSTEM_PROMPT`) + labeling pipeline
- `scripts/ensemble_presidio.py` — Tier 2b rule evaluation (E0 through E7)
- `scripts/irr_build_external_pack.py` — cross-family IRR blind labeling pack builder
- `scripts/irr_score.py` — Cohen's κ, Fleiss κ, weighted κ, confusion matrices, adjudication
- `scripts/irr_score_all_axes.py` — three-axis IRR scoring (domain + complexity + privacy)
- `scripts/irr_flip_analysis.py` — entity vs. topic classification of adjudicated flips

Labeled data: `tests/classification/chunks/labels_*.jsonl` (2,669 rows) + `tests/classification/label_overrides.jsonl` (25 manual) + `tests/classification/label_overrides_irr.jsonl` (14 cross-family adjudicated).

IRR report: `tests/classification/irr/irr_report.md`.

## FAQ

**Why 89.2% and not 95%?** The 95.8% number initially observed under single-labeler ground truth was upwardly biased. Cross-family inter-rater reliability adjudication surfaced 14 additional confidentials that a single labeler missed, dropping recall to 89.2%. The new number is honest; the old number overstated real-world coverage. See §7.2 of the Technical Specification.

**Will it reach 95% at deployment?** Yes, on real enterprise traffic, via four compounding feedback mechanisms over approximately 12 months. Active learning on tier-disagreement + pattern library expansion + encoder upgrades + per-tenant fine-tuning together target 95–97% at the 12-month mark. See §9 of the Technical Specification.

**Does the classifier ever send my prompt outside my deployment?** No — not when `privacy_enforcement=strict` (the default). The classifier runs entirely in-process or on localhost. Under `privacy_enforcement=disabled` (opt-in), the classifier still runs locally; what changes is only whether a `confidential` classification *forces* local-only routing of the underlying request. See §5.

**What if I can't reach 95%?** The ceiling is approximately 97–98%, set by rubric ambiguity rather than model capability. Beyond that, improving accuracy means refining the rubric (clarifying the internal/confidential boundary with better borderline examples) rather than improving the model.
