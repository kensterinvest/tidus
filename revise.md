# Project: Tidus
# Version: 1.3.0
# Plan Date: 2026-04-18
# Source: Claude Code implementation plan (research-validated)
# Description: Auto-Classification Layer — privacy-first cascade (heuristics → trained
#              encoder || Presidio NER → local-LLM fallback) that auto-detects complexity,
#              domain, privacy, and token count so callers no longer need to supply metadata.
#              All classification happens in-process or on localhost — customer messages
#              never leave the deployment boundary for classification purposes.

---

# Tidus v1.3.0 — Auto-Classification Layer

## Deliverables

| Artifact | Path | Purpose |
|---|---|---|
| Claude plan file | `C:\Users\OWNER\.claude\plans\cozy-marinating-flame.md` | Internal planning reference |
| **Project plan** | **`D:\dev\tidus\plan.md`** | **Detailed implementation plan with per-tier specs** |
| **External plan** | **`D:\dev\tidus\revise.md`** | **External-facing plan for handoff to other AI agents, CI systems, or documentation tools** |

---

## Context

Today Tidus's 5-stage router requires callers to supply `complexity`, `domain`, `privacy`, and `estimated_input_tokens` with every request. v1.3.0 makes these optional — callers send raw messages, Tidus classifies internally, then routes to the cheapest capable model. Explicit fields still override everything (backward-compatible).

**The hard constraint that drives the design:** Enterprise HIPAA / SOC 2 / GDPR deployments cannot transmit customer messages to external services for classification. Every tier of the classification cascade runs either in-process (in the FastAPI Python worker) or on localhost (Ollama subprocess). Confidential data never leaves the deployment boundary.

**Prior work (preserved):** POC validated 2026-04-17 at 94.5% domain / 86.9% complexity / 99.6% privacy on 1889 synthetic cases. Research round (2026-04-18) validated architecture against vLLM Semantic Router (closest public precedent) and verified privacy stack via Presidio + detect-secrets documentation.

**Version note:** v1.2.0 was used for the comprehensive review hardening release (2026-04). Auto-classification ships as v1.3.0.

---

## Architecture: Four-Tier Cascade

```
Incoming message
      │
      ▼
[Tier 0]  Caller override                         (< 1μs, 0 cost)
      │   explicit fields → skip all tiers
      ▼
[Tier 1]  Heuristic fast-path                     (~5-10ms, in-process, 0 cost)
      │   POC regex + detect-secrets + keyword lists + structural signals
      │   ~30-40% of traffic short-circuits here
      ▼
[Tier 2 ∥ Tier 2b]  Parallel (asyncio.gather)     (~15-50ms, in-process, 0 cost)
      │   Tier 2:   DeBERTa-v3-xsmall (44M) multi-head   — domain / complexity / privacy
      │   Tier 2b:  Presidio NER-minimal                 — PII entity detection
      │   Merged via authoritative privacy rule (below)
      ▼
[Tier 3]  Local LLM fallback                      (~200-500ms, localhost Ollama)
      │   Phi-3.5-mini-instruct (MIT), structured JSON output
      │   < 2% of traffic after encoder is trained
      ▼
Authoritative classification → existing 5-stage router
```

---

## Architectural Decisions (Research-Validated)

| # | Decision | Locked Choice | Rationale |
|---|---|---|---|
| 1 | External LLM tier | **Dropped entirely** (was "off by default" in v1.2 draft) | Hard enterprise privacy constraint — no customer messages leave deployment boundary |
| 2 | Encoder backbone | **DeBERTa-v3-xsmall (44M)** | +11 GLUE over DistilBERT at 1/3 the params; CPU-friendly; MIT license; mmBERT-32K (307M, vLLM SR base) is CPU-catastrophic |
| 3 | Encoder training recipe | **Two candidates — pick winner on Phase 1 eval:** Recipe A = LoRA port of vLLM SR's `ft_linear_lora.py`; Recipe B = frozen sentence-transformer + class-weighted logistic heads | Recipe A is production-proven at 50k scale; unproven at 3k. Recipe B is safer at 3k and near-calibrated out-of-box. Cost of training both: one afternoon. |
| 4 | Primary fallback | **Trained encoder, NOT local LLM** | Latency: encoder 3-15ms CPU vs LLM 200-500ms (50× gap). Accuracy: encoder 88-94% vs small-LLM 70-85% on 7-way classification |
| 5 | PII stack | **POC regex + detect-secrets + Presidio NER-minimal + custom keywords** | No single tool covers all of SSN/CC/AWS-keys/GitHub-tokens/Slack/OpenAI/IBAN/crypto/medical-license/country-IDs; combination is the industry pattern |
| 6 | Presidio NLP engine | **`en_core_web_sm` + `SpacyRecognizer` removed** | `trf` 600ms+ catastrophic; `lg` ~50ms borderline; `sm` with NER-recognizer-stripped expected <30ms but **unverified — Phase 0.5 benchmark gates placement** |
| 7 | Tier 3 LLM model | **Phi-3.5-mini-instruct (MIT)** | Top MMLU (69.0) and BBH (69.0) in 3B class; MIT license is cleanest; Llama-3.2-3B as fallback if Phi-3.5 underfits Tidus taxonomy |
| 8 | Tier 3 deployment form | **Ollama (localhost subprocess)** NOT llama-cpp-python in-process | Process isolation, crash containment, independent OOM, audit trail; 2ms HTTP overhead negligible on 500ms budget |
| 9 | Phase 0 labeling data | **WildChat-1M (Allen AI, ODC-BY)** | Real ChatGPT/Claude prompts with real PII; stress-tests privacy detection harder than synthetic |
| 10 | Privacy merge rule | **Asymmetric overclassification to `confidential`** on ANY signal (regex, secrets, keywords, Presidio PII, encoder privacy=`confidential`). Never emit `public`. | False-negative on confidential = compliance incident; false-positive = routing optimization loss. Asymmetric cost. |

---

## Tier-by-Tier Components

### Tier 0 — Caller Override
Unchanged from v1.1. Explicit fields → skip classification entirely.

### Tier 1 — Heuristic Fast-Path (< 10ms)
- **POC regex** (preserved): SSN + valid-prefix exclusion, credit card + Luhn, AWS keys, GitHub tokens, generic secrets (`api_key=…`)
- **detect-secrets in-memory** (Yelp, Apache 2.0): AWS, Azure, GCP, GitHub, GitLab, Slack, OpenAI, Stripe, JWT, private keys. KeywordDetector and BasicAuthDetector DISABLED (high FP)
- **Custom keyword layer** (Aho-Corasick): medical (MeSH-seeded), legal (homebrew), financial (homebrew + PCI DSS glossary)
- **Structural domain signals**: code fences → code@0.90, shebang → code@0.95, operator density > 0.08 → code@0.75
- **Token count estimate**: `max(1, len(text) // 4.5)` — stays here, not in encoder or LLM

Short-circuit on high-confidence all-fields match (~30-40% of traffic).

### Tier 2 — Trained Encoder (3-15ms CPU, the workhorse)
- DeBERTa-v3-xsmall (44M, MIT) via ONNX int8 quantized, loaded at FastAPI startup
- **Input policy (all tiers):** classify the **last user message only**, truncated to first 2000 chars. Multi-turn conversations classified by most recent user turn, not full history. Rationale: classifying the current request's nature, not the conversation.
- Shared encoder forward pass → three classification heads:
  - Domain (7-way): chat / code / reasoning / extraction / classification / summarization / creative
  - Complexity (4-way): simple / moderate / complex / critical
  - Privacy (3-way): public / internal / confidential (document-level)
- Per-head temperature-scaled softmax for confidence gating
- Escalate to Tier 3 if any head's confidence < threshold

### Tier 2b — Presidio NER (PARALLEL to Tier 2 via `asyncio.gather`)
- `AnalyzerEngine(en_core_web_sm)` with `SpacyRecognizer` removed from registry
- Pattern-based recognizers only (structural PII)
- Covers: CC+Luhn recheck, IBAN, phone, email, URL, crypto wallets, medical_license, country-specific IDs (US SSN/ITIN/passport/driver's license + UK/ES/IT/PL/SG/AU/IN/FI/KR/NG/TH)
- Gaps (covered by Tier 1): AWS/GCP/Azure keys, GitHub/Slack tokens
- **Latency risk: unverified.** Phase 0.5 benchmark gates placement:
  - p95 ≤ 30ms → keep as parallel Tier 2b
  - p95 > 30ms → demote to conditional Tier 3. Trigger: `encoder.privacy_confidence < classify_privacy_threshold AND not tier1.any_hit` (Tier 1 hit already forces `confidential`)

### Tier 3 — Local LLM Fallback (200-500ms, Ollama localhost)
- Phi-3.5-mini-instruct (MIT — **verify current HF repo license before shipping**), Q4_K_M GGUF (~2.4 GB)
- Structured JSON output via Ollama grammar constraints
- SHA-256 cached (TTL 1h, LRU max 10K)
- Rate-limited per worker-minute to prevent Ollama saturation on traffic spikes
- Expected volume: < 2% of traffic

---

## Concurrency Pattern (Authoritative)

Tier 2 encoder and Tier 2b Presidio run in parallel via `asyncio.gather` — total latency is `max(encoder_ms, presidio_ms)`, not their sum.

```python
async def classify_tier_2(text: str) -> tuple[EncoderResult, PresidioResult]:
    encoder_task = asyncio.to_thread(run_encoder, text)
    presidio_task = asyncio.to_thread(run_presidio, text)
    return await asyncio.gather(encoder_task, presidio_task)
```

ONNX Runtime inference and Presidio `AnalyzerEngine.analyze()` are both thread-safe; `asyncio.to_thread` dispatches without blocking the FastAPI event loop.

---

## Privacy Merge Rule (Authoritative)

```python
def merge_privacy(
    tier1: Tier1Signals,
    encoder_privacy: Privacy | None,
    presidio_pii_found: bool,
) -> Privacy:
    if (tier1.any_regex_hit
        or tier1.any_keyword_hit
        or presidio_pii_found
        or encoder_privacy == Privacy.confidential):
        return Privacy.confidential

    # Never emit public — safety default
    return Privacy.internal
```

**Invariants:**
- Never emits `public`. Ever.
- Any PII signal from any tier forces `confidential`.
- Encoder's `public` prediction is silently upgraded to `internal`.

---

## Implementation Phases

```
Step 1:  Label 1000 WildChat-1M prompts via Sonnet           [Phase 0, ~$15]
Step 2:  Backtest POC heuristics on labeled set              [Phase 0 — GATE CHECK]
         → Gate uses 95% CI lower bound (±2.7% sampling error at n=1000)
         → if CI lower bound ≥ 82% domain + ≥ 93% privacy: training may be skipped
         → else proceed to Step 4
         → POC is FROZEN for this comparison (no new heuristics until after gate)
Step 3:  Benchmark Presidio CPU latency                      [Phase 0.5, parallel to 1-2]
         → if p95 > 30ms: demote Presidio to conditional Tier 3
Step 4:  Train encoder — BOTH recipes                        [Phase 1]
         Recipe A: LoRA-on-DeBERTa-v3-xsmall (port vLLM SR recipe)
         Recipe B: frozen sentence-transformer + logistic heads
Step 5:  Eval both on CLEAN eval tier (not Sonnet-labeled held-out)  [Phase 1]
         → Clean eval tier = 100-150 prompts, Sonnet-labeled AND human-verified
           (~2 hours one-time human verification during Phase 0)
         → Rationale: both recipes trained on Sonnet labels; Sonnet-only held-out
           measures noise memorization, not generalization
         → Recipe A > Recipe B by ≥ 2pp macro-F1 on clean eval → A ; else → B
Step 6:  Integration + Tier 3 confidence calibration         [Phase 2]
Step 7:  Adversarial eval harness before shipping            [Phase 3]
```

Steps 1-3 run concurrently. Step 2 is the kill-switch — skip training if heuristics are already sufficient on real-world prompts.

---

## New Endpoint

`POST /api/v1/classify` — returns `domain`, `complexity`, `privacy`, `estimated_input_tokens`, `classification_tier` (override / heuristic / encoder / presidio / llm_fallback), per-field confidence scores, and per-tier debug info.

## API Contract (Backward Compatible)

```python
# Before (v1.0 / v1.1) — required
complexity: Complexity
domain: Domain
privacy: Privacy
estimated_input_tokens: int

# After (v1.3) — optional; auto-classified when None
complexity: Complexity | None = None
domain: Domain | None = None
privacy: Privacy | None = None
estimated_input_tokens: int | None = None
```

## New Settings (12 total)

```python
auto_classify_enabled: bool = True
classify_encoder_path: str = "tidus/classification/weights/encoder_v1.onnx"
classify_llm_model_id: str = "phi3.5:mini-instruct"
classify_llm_endpoint: str = "http://localhost:11434"
classify_privacy_threshold: float = 0.75
classify_domain_threshold: float = 0.70
classify_complexity_threshold: float = 0.65
classify_presidio_enabled: bool = True
classify_presidio_parallel: bool = True
classify_cache_ttl: int = 3600
classify_cache_max_entries: int = 10_000
classify_llm_rate_limit_per_minute: int = 60
```

## New Dependencies

```toml
dependencies = [
    "presidio-analyzer >= 2.2.362",              # MIT
    "detect-secrets @ git+https://github.com/Yelp/detect-secrets@<PIN_SHA>",  # Apache 2.0
    "pyahocorasick >= 2.0.0",                    # BSD-3
    "onnxruntime >= 1.20.0",                     # MIT
    "spacy >= 3.7.0",                            # MIT
    "sentence-transformers >= 3.0.0",            # Apache 2.0 (already installed)
]
# Post-install:  python -m spacy download en_core_web_sm
```

---

## Tests

| File | Covers |
|---|---|
| `tests/unit/classification/test_heuristics.py` | Regex, Luhn, structural signals, token estimation |
| `tests/unit/classification/test_secrets.py` | In-memory scanning, no temp file, plugin subset |
| `tests/unit/classification/test_keywords.py` | Aho-Corasick, MeSH loader, case-insensitive |
| `tests/unit/classification/test_encoder.py` | ONNX loading, multi-head inference, temperature scaling, confidence gating |
| `tests/unit/classification/test_presidio_wrapper.py` | SpacyRecognizer removal, async wrapping, empty input |
| `tests/unit/classification/test_llm_classifier.py` | Ollama JSON parsing, malformed fallback, cache, rate limit |
| `tests/unit/classification/test_merge_rule.py` | Full 16+ truth table of privacy merge |
| `tests/unit/classification/test_classifier.py` | Tier cascade, short-circuit, caller override, concurrency |
| `tests/integration/test_classify_endpoint.py` | `/classify`, `/complete` without metadata, SSN → confidential routing |
| `tests/integration/test_backward_compat.py` | v1.1 requests (all fields provided) still work identically |

## Verification Checklist

1. `POST /classify` with code message → `domain=code, complexity=moderate, tier=heuristic`
2. Same message repeated → `tier=heuristic` OR `tier=cached`, latency unchanged
3. `POST /complete` with only `team_id + messages` → 200, `chosen_model_id` present
4. SSN in message → `privacy=confidential`, only `is_local=True` models
5. `"diagnose my symptoms"` → `complexity=critical`
6. `auto_classify_enabled=false` + fields omitted → 422 validation error
7. `GET /metrics` → `tidus_classify_tier_total{tier="heuristic|encoder|llm"}` increments
8. Presidio disabled → no import penalty, no latency hit
9. Concurrent load 200 req/s → encoder || Presidio confirmed parallel (max, not sum)
10. Ollama unavailable → encoder output returned with `confidence_warning` flag

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Presidio latency > 30ms on CPU | Medium | Medium | Phase 0.5 benchmark-first; demote to conditional Tier 3 |
| Recipe A (LoRA) underfits at 3k scale | Medium | Low | Train Recipe B in parallel; pick winner on **clean eval tier** (not Sonnet-only) |
| WildChat distribution ≠ real Tidus traffic | High | Medium | Regenerate eval set post-deployment on audit log |
| Adversarial PII breaks 99.6% claim | High | High | Phase 3 adversarial eval gate; Privacy SLO ≤ 1% FN; compliance sign-off |
| **Sonnet label noise propagates to encoder (3-8% est)** | **Medium** | **Medium** | **Weak supervision: require Sonnet + POC agreement for high-confidence labels. Human-audit 50 disagreement prompts. Clean eval tier selects recipe winner.** |
| **Non-English input breaks English-only tiers** | **Medium** | **Medium** | **langdetect (<1ms); non-English → `internal`, skip Tier 2, route via Tier 0/1 only. Multi-language deferred to v1.4.** |
| Ollama unavailable at request time | Medium | Low | Graceful degradation: encoder output + confidence warning |

---

## Privacy SLO (Enterprise Compliance Commitment)

**False-negative rate on the adversarial eval set (Phase 3) must be ≤ 1%.**

- Re-verified quarterly post-ship against maintained adversarial harness
- If breached: auto-rollback to Tier 1-only (regex + Presidio + keywords, no encoder, no LLM) until fixed
- The customer-facing commitment. "Asymmetric overclassification" is engineering philosophy; ≤ 1% FN is the signable guarantee
- Applies to `privacy == confidential` predictions specifically; domain/complexity FN are routing losses, not compliance incidents

## What NOT to Do

- Never transmit customer messages externally for classification
- Never write confidential data to disk (LLM Guard's temp-file pattern is banned)
- Never default privacy to `public`
- Never skip adversarial eval before shipping privacy tier
- Never use `en_core_web_lg` or `en_core_web_trf` for Presidio (budget-breakers)
- Never count tokens via encoder or LLM (unreliable)
- Never reuse mmBERT-based classifiers from vLLM SR on CPU-only Tidus deployments (307M params is CPU-catastrophic)
- Never select recipe A vs B on full Sonnet-labeled held-out (noise memorization confound — use clean eval tier only)
- Never update POC heuristics during Phase 0 gate check (avoid moving the target)

---

## Research Sources

- **vLLM Semantic Router** — `github.com/vllm-project/semantic-router` (closest public precedent; training recipe port; PII/jailbreak weights deferred to future GPU deployment)
- **RouteLLM** — arXiv:2406.18665 (trained routers beat heuristics)
- **FrugalGPT** — arXiv:2305.05176 (cascade + confidence-gate)
- **RouterBench** — arXiv:2403.12031 (simple trained routers outperform random)
- **Presidio** — `github.com/microsoft/presidio` v2.2.362 (2026-03-18), MIT
- **detect-secrets** — `github.com/Yelp/detect-secrets` (master active through 2026-04)
- **DeBERTa-v3** — Microsoft, MIT, +11 GLUE over DistilBERT
- **Phi-3.5-mini-instruct** — Microsoft, MIT
- **WildChat-1M** — Allen AI, ODC-BY
- **MeSH** — NLM (verify commercial ToS before redistribution)

---

## POC Script (preserved from 2026-04-17)

`scripts/poc_classifier.py` — 94.5% domain / 86.9% complexity / 99.6% privacy on 1889 synthetic. Standalone, no Tidus imports. Used for Phase 0 backtest comparison:

```bash
uv run python scripts/poc_classifier.py --no-embedding     # Tier 1 only
uv run python scripts/poc_classifier.py                     # Tier 1 + Tier 2 embedding
uv run python scripts/poc_classifier.py --verbose           # per-case diagnostics
```

---

**Status:** Ready for implementation. Phase 0 (Steps 1-3) can begin immediately.
