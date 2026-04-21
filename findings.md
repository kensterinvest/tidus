# Tidus — Research Findings on Multi-Axis Request Classification

**Author:** Kenny Wong (<lapkei01@gmail.com>)
**Published:** 2026-04-20
**Version:** 1.0 — v1.3.0 auto-classification layer
**Project:** [kensterinvest/tidus](https://github.com/kensterinvest/tidus)
**License (this document):** CC-BY 4.0 where not superseded by patent filing
**Related:**
- Landing-page technical specification: <https://kensterinvest.github.io/#classification-workflow>
- Implementation plan: [`plan.md`](plan.md)
- Concise reference: [`docs/classification.md`](docs/classification.md)
- **Dataset (separate distribution):** <https://huggingface.co/datasets/kensterinvest/tidus-wildchat-classification> — 2,669 labels + IRR study artifacts + overrides. Distributed on Hugging Face Datasets rather than committed to the main git repo because a small number of prompts contain real credential patterns (Telegram bot tokens, FB/Instagram access tokens, OpenAI key templates) inherited from upstream WildChat-1M; these are the evidentiary basis of Finding #4 (credential re-leak) and are preserved in the dataset.

---

> Load-bearing empirical claims, methodology, and open gaps for the v1.3.0
> auto-classification layer. Intended both for enterprise evaluation and as
> prior-art documentation supporting patent filing. Reproducible end-to-end
> from scripts under `scripts/` and data under `tests/classification/`.

---

## Executive summary

Tidus's 5-tier privacy classifier achieves **89.2% recall [80.7, 94.2]** on
cross-family-validated confidential ground truth (n=83, WildChat corpus),
with the 10.8% miss rate falling entirely into a named, topic-based class
that motivates the Tier-5 LLM escalation layer. Inter-rater reliability
across Claude (Anthropic), GPT (OpenAI), and Gemini (Google) produces
weighted Cohen's κ = 0.68–0.78 (all "substantial") on a stratified n=149
sample, closing the label-trust gap that single-labeler methodology leaves
open.

---

## Finding 1 — Ground truth is cross-family-validated

**Claim.** The privacy labels used in this study are trustworthy under the
field-standard rubric for inter-rater reliability; they survive independent
replication by raters from two other model families.

**Methodology.** A stratified sample of n=149 prompts (all 69 unique
confidentials + 40 random internal + 40 random public) was labeled
independently by three raters: Claude (Anthropic, via the original labeling
pipeline), GPT (OpenAI, via Microsoft Copilot Think-Deeper), and Gemini
(Google, via Gemini 2.5 Pro deep-thinking). All raters saw the same frozen
rubric derived from `scripts/label_wildchat.py::SYSTEM_PROMPT`; none saw the
others' labels or rationale. Each labeled in a fresh session to avoid
context-carryover between raters.

**Results.**

| Metric (n=149) | Value | Interpretation |
|---|---|---|
| Weighted Cohen's κ — Claude vs GPT | **0.709** | substantial |
| Weighted Cohen's κ — Claude vs Gemini | **0.783** | substantial |
| Weighted Cohen's κ — GPT vs Gemini | **0.677** | substantial |
| Fleiss' κ (3 raters, unweighted) | 0.577 | moderate |
| Observed agreement — all three raters | 63.1% | — |
| Observed agreement — GPT/Gemini | 77.2% | — |
| Claude vs. majority-vote | 78.5% | — |

**Interpretation.** Classes are ordinal (public < internal < confidential),
so quadratic-weighted κ is the appropriate metric — it penalizes distant
disagreements (public↔confidential) more than adjacent ones
(public↔internal). All three pairwise weighted κ values cross the field
threshold for "substantial" agreement. The unweighted Fleiss' κ of 0.577 is
reported for transparency; the gap between it and the weighted values
reflects that the internal/confidential boundary is where most disagreement
lives — expected from the rubric's explicit asymmetric-safety rule.

**Reproduction.** Scripts:
- `scripts/irr_build_external_pack.py` — builds blind labeling packs.
- `scripts/irr_score.py` — computes κ, Fleiss', weighted κ, confusion
  matrices, disagreement lists, asymmetric-safety adjudication.
- Outputs: `tests/classification/irr/pack_01..10.md`,
  `tests/classification/irr/responses/`, `tests/classification/irr/irr_report.md`,
  `tests/classification/label_overrides_irr.jsonl`.

---

## Finding 2 — The tiered architecture is empirically validated

**Claim.** The 5-tier classifier design (cheap entity-based detection
followed by LLM-topic escalation) is not a convenience — it maps to a
real bifurcation in the data. Confidentials divide cleanly into
*entity-bearing* (detectable by Presidio+regex) and *topic-bearing*
(requiring semantic reasoning).

**Methodology.** The IRR study produces 14 "asymmetric-safety flips" — rows
where Claude labeled non-confidential but at least one of GPT or Gemini
labeled confidential. Under the deployed asymmetric-safety rule (any rater
says confidential → confidential), these 14 are adjudicated to confidential
in the published ground truth. `scripts/irr_flip_analysis.py` then checks
whether Presidio's PERSON detector catches each flip.

**Results (n=12 flips in ensemble's joinable pool).**

| Sub-class | Count | Examples of miss content |
|---|---|---|
| **E1 catches** (PERSON entity detected) | 6/12 | Real names in contact info, driver license numbers, email+named-sender combinations |
| **E1 misses** (no entity detected) | 6/12 | "I'm from Slovakia, no money" (financial distress), "generate valid openai api keys" (credential request), "human resources complaint" (employment-legal), `/Users/surabhi/Documents/...` (filesystem path with real username), SSH audit log with IP+host, "apartment cost 100/mo how do I survive" (financial hardship) |

**Interpretation.** The split is exactly 50/50. E1's 6 misses all share one
structural property: no identifying entity, but topic-sensitive subject
matter (financial hardship, credential-harvesting intent, employment
disputes, infrastructure-identifying paths). These are the cases that
motivate Tier-5 LLM topic review. The 3-case structural-miss audit from the
original single-labeler pass (Vue/SCSS flip, Canadian work permit, Russian
mental health) was labeler intuition; the IRR flip analysis confirms the
pattern holds across independent raters and a larger sample.

**Corroborating evidence.** The original 3-case audit received **unanimous
3/3 agreement** in the IRR study:

| Case | Claude | GPT | Gemini |
|---|---|---|---|
| Case 1 — Vue/SCSS Chinese placeholder | public | public | public |
| Case 2 — Canadian work permit | confidential | confidential | confidential |
| Case 3 — Russian mental health | confidential | confidential | confidential |

Case 1's flip (originally contested) is now validated by two additional
independent model families. Cases 2 and 3's classification as
LLM-tier-territory is independently confirmed.

---

## Finding 3 — Cheap-tier recall under validated ground truth

**Claim.** The E1 rule (Presidio PERSON entity alone triggers confidential)
achieves 89.2% [80.7, 94.2] recall on cross-family-adjudicated ground
truth. This is the defensible number for production deployment.

**Methodology.** `scripts/ensemble_presidio.py` re-run with combined
overrides (original 25 manual + 14 IRR-adjudicated, merged by
`train_encoder._load_overrides`). Ground-truth confidentials: gt_conf=83
post-adjudication, up from 71 pre-IRR. Encoder is 5-fold OOF
`all-MiniLM-L6-v2` + sklearn LR; Presidio is default en_core_web_sm.

**Results.**

| Metric | Pre-IRR (Claude-only) | Post-IRR (cross-family) | Delta |
|---|---|---|---|
| gt_conf | 71 | 83 | +12 |
| E0 POC+Encoder only | 57/71 = 80.3% [69.6, 87.9] | 59/83 = 71.1% [60.6, 79.7] | −9.2 pp |
| E1 PERSON alone | 68/71 = 95.8% [88.3, 98.6] | **74/83 = 89.2% [80.7, 94.2]** | **−6.6 pp** |
| E2 PERSON + Enc-non-pub | 64/71 = 90.1% [81.0, 95.1] | 69/83 = 83.1% [73.7, 89.7] | −7.0 pp |
| High-trust Presidio only | 61/71 = 85.9% [76.0, 92.2] | 63/83 = 75.9% [65.7, 83.8] | −10.0 pp |

**Interpretation.** All signals drop under validated ground truth; the drop
is larger for weaker signals. E1's 6.6-point drop is the smallest — the
PERSON-alone rule is the most robust to the labeling bias IRR surfaces.
The dropped points are exactly the topic-based confidentials Finding 2
identifies; they were always LLM-tier cases.

**Why this is a better number than 95.8%.** The pre-IRR 95.8% was a ceiling
defined by single-labeler ground truth. The post-IRR 89.2% incorporates
rows where two independent model families flagged confidentials that
Claude originally did not — a stricter definition of what "confidential"
means. The 6.6-point difference is the quantified label bias; reporting
only the pre-IRR number would overstate real-world recall.

**Production deployment.** E1 at 89.2% with Tier-5 LLM escalation for
topic-based residue is the defensible architecture. Per-request latency:
p95 = 59 ms at Tier 4 (CPU). Tier 5 latency depends on deployment SKU —
CPU-only: disqualified (measured 34.7 s p95 for Phi-3.5-mini Q4 on
Intel i7-9700KF, see `tests/classification/t5_bench_results.md`); GPU-required
Enterprise SKU: estimated 150-500 ms p95 on 8+ GB VRAM GPU (verify on
deployment hardware). See `docs/hardware-requirements.md` for the full
hardware spec.

---

## Finding 4 — Credential re-leak is the deployment-relevant problem

**Claim.** Across the 2669-row labeled WildChat corpus, **4 identified
cases** of the same real credential appear in multiple user sessions from
the same user. This is evidence that credential persistence — not
one-shot detection — is the operational problem enterprise routers must
solve.

**Observed cases.**

| # | Credential type | Re-leak pattern |
|---|---|---|
| 1 | Telegram bot token `5828712341:AAG5HJa37u32...` | Same token, chunks 055 #10 and 059 #31 (same user, different session) |
| 2 | VK bot token `vk1.a.KdAkdaN3v-HPcxH8...` | Same token, chunks 048 #2 and 061 #9 |
| 3 | Instagram/Instaloader credentials + FB access_token | Same credentials, chunks 048 and 062 #31 |
| 4 | Discord webhook tokens (piranhaPeche, sardineInsolente) + Steam Web API key | Multiple credentials exposed in chunk 060 (Discord #8, Steam #29) |

**Interpretation.** Stateless request-time classification (the standard
model of privacy-enforcing systems) implicitly assumes each prompt is
independent. The re-leak pattern shows it isn't: users who leak once tend
to leak again with the same credential in subsequent sessions. A router
that caches "this user previously leaked token X" can detect re-leaks even
when a single-prompt classifier doesn't. This is a direction Tidus's
architecture supports (the Audit layer already logs prompt fingerprints)
but the classifier itself does not exploit. Publishable as a deployment
insight: **enterprise routers should treat credential-leak as a
longitudinal property of a user/session, not a per-request property**.

---

## Known open gaps (honest limitations)

1. **Single-dataset (WildChat).** All evidence comes from one corpus of
   English-dominant web-chat prompts. Generalization to enterprise support
   tickets, internal Slack, email, or voice transcripts is unvalidated.
   *Proposed experiment:* run E1 on 100-200 labels from a second source
   (Enron email dump, public support-ticket corpora) and report recall.

2. **Rubric ambiguity at internal/confidential boundary.** Unweighted κ
   (0.577) is materially lower than weighted κ (0.68-0.78), indicating
   disagreements cluster at the internal/confidential boundary rather than
   public/confidential. A rubric revision with clearer borderline examples
   would tighten future labeling.

3. **Tier-5 empirical bound not yet established at target precision.**
   The IRR study provides promising directional evidence: the 6 topic-based
   IRR flips (Finding 2) were caught by at least one non-Claude LLM rater,
   and the 3 audit cases (Case 1 public×3, Cases 2 & 3 confidential×3) were
   unanimous. Combined: **n=9 Tier-5-proxy-positives with 0 losses** in the
   evidence base. The direction is strongly supported; precision-bound
   reporting would want n≥30-50 for a published rate. A follow-up study
   could evaluate Tier-5 LLM classification on a broader sample of
   topic-based content and establish a recall estimate with CI.

4. **Per-language coverage.** The rubric handles English-only (per
   language filter in `label_wildchat.py`). Privacy signals in Chinese,
   Russian, Arabic prompts are not separately evaluated. Case 1
   (张三/13845257654) and Case 3 (Russian depression disclosure) both
   required human understanding of language-specific placeholder conventions
   and semantic disclosure patterns; automated detection at production
   scale would need per-language entity recognizers.

---

## What's settled, what's still open

| Research question | Status |
|---|---|
| Is the ground truth trustworthy? | **SETTLED** — weighted κ substantial across 3 families (Finding 1) |
| Is the tiered architecture justified? | **SETTLED** — 50/50 entity/topic split confirmed (Finding 2) |
| What's the honest cheap-tier recall? | **SETTLED** — E1 = 89.2% [80.7, 94.2] on adjudicated ground truth (Finding 3) |
| Does E1 generalize beyond WildChat? | **OPEN** — single-dataset study |
| Is the rubric reproducibly applied at the internal boundary? | **OPEN** — unweighted κ reveals boundary ambiguity |
| What's the Tier-5 LLM's bound recall? | **OPEN** — audit-case 3/3 unanimous but n=3 is too small |

---

## Scripts / artifacts

- `scripts/label_wildchat.py` — original rubric (`SYSTEM_PROMPT`) + labeling pipeline
- `scripts/train_encoder.py` — 5-fold OOF encoder training; `_load_overrides()` merges manual + IRR overrides
- `scripts/ensemble_presidio.py` — 10-rule E-series recall sweep against gt_conf
- `scripts/irr_build_external_pack.py` — blind 3-rater pack builder
- `scripts/irr_score.py` — Cohen/Fleiss/weighted κ + confusion + adjudication
- `scripts/irr_flip_analysis.py` — entity/topic classification of IRR-adjudicated flips

- `tests/classification/chunks/labels_001..065.jsonl` — 2669 labeled prompts
- `tests/classification/label_overrides.jsonl` — 25 manual audit overrides
- `tests/classification/label_overrides_irr.jsonl` — 14 cross-family-adjudicated overrides
- `tests/classification/irr/irr_report.md` — full IRR report with confusion matrices + disagreement list
- `audit_all_missed.txt` — 3-case structural-miss audit from pre-IRR single-labeler pass
