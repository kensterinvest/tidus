# Tidus v1.3.0 — Auto-Classification Layer: Implementation Plan

## Status: DESIGN APPROVED (2026-04-18)

**Prior state:** POC validated 2026-04-17 on 1889 synthetic cases. Research rounds 2026-04-18 (vLLM Semantic Router deep dive + privacy stack verification) validated the architecture. All 10 design decisions locked.

---

## Context

Tidus's 5-stage router works, but callers must supply `complexity`, `domain`, `privacy`, and `estimated_input_tokens` with every request. v1.3.0 makes these optional — Tidus classifies raw messages internally and routes to the cheapest capable model without caller-side bookkeeping. Explicit fields still override everything (backward compatible).

**Hard constraint (new, overrides earlier "external LLM as Tier 3, off by default"):** Classification must happen in-process or on localhost. Customer messages never leave the deployment boundary for classification purposes — ever. This is the enterprise HIPAA/SOC 2 air-gap requirement and is non-negotiable.

---

## Budget

| Dimension | Limit |
|---|---|
| Fast-path latency (Tier 1 short-circuit) | p95 < 10ms |
| Extended-path latency (Tier 1 + 2 + 2b) | p95 < 50ms |
| Fallback-path latency (Tier 3) | p95 < 500ms |
| Worker memory budget | < 500 MB additional RAM |
| CPU assumption | 4-8 vCPUs x86, no GPU |
| Dependencies | Open-source only (MIT / Apache / BSD) |

---

## Architecture

```
Incoming message
      │
      ▼
[Tier 0]  Caller override                            (< 1μs)
      │   explicit fields in request → skip all tiers
      ▼
[Tier 1]  Heuristic fast-path                        (~5-10ms total)
      │   a. POC regex (SSN / CC+Luhn / AWS / GH / generic secrets)
      │   b. detect-secrets in-memory subset         (2-5ms, UNVERIFIED)
      │   c. Custom keyword layer (Aho-Corasick)     (< 1ms)
      │      medical (MeSH-seeded), legal (homebrew), financial (homebrew + PCI DSS)
      │   d. Structural domain signals               (< 1ms)
      │      code fence, shebang, operator density
      │   e. Token count estimate                    (< 1μs)
      │      estimate_tokens(text) = max(1, len(text) // 4.5)
      │
      │   High-confidence hits → short-circuit       (~30-40% of traffic)
      │   Privacy: ANY hit → confidential (asymmetric, overclassify safe)
      ▼
┌─────────────────────────────┬───────────────────────────────┐
│ [Tier 2] Trained encoder    │ [Tier 2b] Presidio NER        │  — PARALLEL via asyncio.gather()
│  (3-15ms CPU, BENCHMARK)    │  (benchmark-gated, UNVERIFIED)│
│                             │                               │
│  DeBERTa-v3-xsmall (44M)    │  AnalyzerEngine(              │
│  + LoRA multi-head:         │    en_core_web_sm,            │
│   - Domain (7-way)          │    SpacyRecognizer removed)   │
│   - Complexity (4-way)      │                               │
│   - Privacy (3-way)         │  Detects: IBAN, phone, email, │
│                             │  URL, crypto, medical_license,│
│  ONNX int8, in-process      │  country-specific IDs         │
└─────────────────────────────┴───────────────────────────────┘
              │
              ▼   merge via privacy rule (below)
              │
              │   if any head encoder max_softmax < threshold
              ▼
[Tier 3]  Local LLM fallback                         (~200-500ms via Ollama)
      │   Phi-3.5-mini-instruct (MIT, MMLU 69.0, BBH 69.0)
      │   Ollama localhost, structured JSON via grammar
      │   <2% of traffic after encoder is trained
      ▼
Authoritative classification → 5-stage router (existing behavior)
```

---

## Concurrency Pattern (Authoritative)

Tier 2 encoder and Tier 2b Presidio execute in parallel — NOT in series. Total latency is `max(encoder_ms, presidio_ms)`, not their sum.

```python
async def classify_tier_2(text: str) -> tuple[EncoderResult, PresidioResult]:
    encoder_task = asyncio.to_thread(run_encoder, text)
    presidio_task = asyncio.to_thread(run_presidio, text)
    return await asyncio.gather(encoder_task, presidio_task)
```

Both block the event loop if called synchronously, so `asyncio.to_thread` (or `run_in_executor` with a sized threadpool) is required. ONNX Runtime session inference is thread-safe; Presidio's `AnalyzerEngine.analyze()` is thread-safe.

---

## Privacy Merge Rule (Authoritative Truth Table)

After Tier 2 runs, merge encoder privacy + Presidio + Tier 1 signals via this rule:

```python
def merge_privacy(
    tier1: Tier1Signals,
    encoder_privacy: Privacy | None,     # None if encoder not called
    presidio_pii_found: bool,
) -> Privacy:
    # Overclassify to confidential on any signal (asymmetric cost)
    if (tier1.any_regex_hit
        or tier1.any_keyword_hit
        or presidio_pii_found
        or encoder_privacy == Privacy.confidential):
        return Privacy.confidential

    # Otherwise trust encoder but never emit public
    if encoder_privacy in (Privacy.internal, Privacy.public):
        return Privacy.internal  # safety default: never public

    # Fallthrough (encoder skipped + no Tier 1 signals)
    return Privacy.internal
```

**Invariants:**
- Never returns `public`. Ever. (The existing POC rule — preserved.)
- Any PII detection from any tier forces `confidential`.
- Encoder's `public` prediction is silently upgraded to `internal`.

---

## Phase Sequencing

```
Step 1: Label 1000 WildChat-1M prompts via Sonnet         [Phase 0, ~$15]
Step 2: Backtest POC heuristics on labeled set            [Phase 0 GATE]
        → Gate uses 95% CI lower bound, not point estimate (±2.7% sampling error at n=1000)
        → if 95% CI lower bound ≥ 82% domain + ≥ 93% privacy: training may be skipped
        → else proceed to Step 4
        → POC is FROZEN for this comparison — do not add new heuristics until after
          gate check (avoid moving the target)
Step 3: Benchmark Presidio CPU latency                    [Phase 0.5, parallel to 1-2]
        → if p95 > 30ms: demote Presidio to conditional Tier 3 for privacy-only
Step 4: Train encoder — BOTH recipes                      [Phase 1]
        Recipe A: LoRA-on-DeBERTa-v3-xsmall (port vLLM SR ft_linear_lora.py)
        Recipe B: frozen sentence-transformer + 3 class-weighted logistic heads
Step 5: Eval both encoders on CLEAN eval tier             [Phase 1]
        → Clean eval tier = 100-150 prompts, Sonnet-labeled AND human-verified
          (human verification during Phase 0, ~2 hours one-time effort by Kenny)
        → Rationale: both recipes are trained on Sonnet labels; measuring on Sonnet-only
          held-out measures recipe ability to memorize Sonnet noise, not generalization
        → Pick winner by macro-F1 on clean eval (Recipe A > Recipe B by ≥ 2pp → A; else B)
Step 6: Integration + Tier 3 confidence calibration       [Phase 2]
Step 7: Adversarial eval harness before shipping          [Phase 3]
```

Steps 1-3 are independent and run concurrently. Step 2 is the kill-switch — skip all training if Phase 0 shows heuristics are already sufficient on real-world prompts.

**Training decision rule (Step 5):**
- Recipe A beats Recipe B by ≥2pp macro-F1 on held-out → pick Recipe A (LoRA adapters)
- Otherwise → pick Recipe B (frozen + linear heads: simpler, near-calibrated out of box, retrains in seconds)

---

## Components by Tier

### Tier 0: Caller Override (unchanged from v1.1)

If caller provides `complexity`, `domain`, `privacy`, `estimated_input_tokens` → use them as-is, skip all tiers. Maintains full backward compatibility.

### Tier 1: Heuristic Fast-Path

**(a) Existing POC regex** (preserved from 2026-04-17 POC):
- SSN with valid-prefix exclusion
- Credit card with Luhn checksum
- AWS keys (AKIA / AGPA / AIDA / AROA + 16 chars)
- GitHub tokens (ghp_ / gho_ / ghs_ / ghr_ / ghu_ + 36 chars)
- Generic secrets (`api_key = ...`, `password = ...`, `token = ...`)

**(b) detect-secrets (Yelp) in-memory subset:**
- Enabled plugins: `AWSKeyDetector`, `AzureStorageKeyDetector`, `GitHubTokenDetector`, `GitLabTokenDetector`, `SlackDetector`, `OpenAIDetector`, `StripeDetector`, `TwilioKeyDetector`, `JwtTokenDetector`, `PrivateKeyDetector`, `Base64HighEntropyString` (limit=4.5)
- **Disabled plugins:** `KeywordDetector` (redundant with Tidus's keyword layer), `BasicAuthDetector` (high FP rate)
- **In-memory only** — no temp file writes (confidential-data safety). Use lower-level plugin API, not `scan_file`.
- Pin to git SHA (last tagged release is May 2024; master is active through April 2026)

**(c) Custom keyword layer (Aho-Corasick via pyahocorasick):**
- Medical: MeSH slice (disease, diagnosis, procedure, medication terms) + HIPAA-identifier phrases
- Legal: homebrew (privilege, attorney-client, work-product, litigation-hold, deposition, trade-secret, NDA, non-disclosure)
- Financial: homebrew + PCI DSS glossary (wire-transfer, SWIFT, ACH, routing-number, tax-return, W-2, 1099, K-1, PCI, cardholder-data, earnings-call)

**(d) Structural domain signals:**
- Backtick code fences → domain=code, conf=0.90
- Shebang line → domain=code, conf=0.95
- Operator density > 0.08 → domain=code, conf=0.75
- Sigmoid on keyword coverage for non-code domains

**(e) Token count estimate:**
```python
estimate_tokens(text) = max(1, len(text) // 4.5)
```
Stays in Tier 1 — NOT in encoder, NOT in LLM. Classification models don't count tokens reliably.

**Short-circuit rule:**
- High-confidence Tier 1 hits on all three fields (privacy + domain + complexity confidence > threshold) → skip Tier 2, return Tier 1 result directly (~30-40% of traffic)
- Any PII hit (regex, detect-secrets, keyword) → privacy = confidential, still run Tier 2 for domain/complexity

### Tier 2: Trained Encoder (Workhorse)

**Backbone:** DeBERTa-v3-xsmall — 44M params, MIT license
**Deployment:** ONNX int8 quantized via `onnxruntime`, loaded at FastAPI worker startup, reused across requests
**Memory footprint:** ~90 MB weights + small session overhead

**Input policy (applies to all classification tiers):**
- Input to classify = **last user message only**, truncated to first 2000 chars
- For multi-turn conversations, only the most recent user turn is classified
- Rationale: classifying the current request's nature, not conversation history
- DeBERTa-v3-xsmall max seq len is 512 tokens (~2000 chars); beyond that output is truncation-dependent
- vLLM SR's TextRank+TF-IDF compression technique is deferred to v1.4 if Phase 3 adversarial eval shows degradation on long prompts

**Multi-head architecture:**
- Shared encoder forward pass (one CPU inference)
- Three classification heads on `[CLS]` embedding:
  - Domain head: 7-way (chat / code / reasoning / extraction / classification / summarization / creative)
  - Complexity head: 4-way (simple / moderate / complex / critical)
  - Privacy head: 3-way (public / internal / confidential) — document-level sensitivity, complements Tier 1 regex and Tier 2b NER

**Training recipes — both candidates, pick winner on Phase 1 eval:**

**Recipe A — LoRA-on-DeBERTa (port vLLM SR ft_linear_lora.py):**
- LoRA rank 16-32, alpha = 2×rank, target modules `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, `mlp.Wo`
- Loss: CrossEntropy summed across heads, privacy head weighted 2× (asymmetric cost)
- Optimizer: AdamW, lr 2e-5 to 3e-5, weight decay 0.1, cosine schedule, warmup 0.06, grad accum 2, grad clip 1.0
- Epochs: 3-5, batch 8-32, `load_best_model_at_end=True`
- Training data: 1889 synthetic POC + 1000 WildChat Sonnet-labeled
- Calibration: temperature-scale per-head logits on held-out slice
- Proven at 50k scale in vLLM SR; UNTESTED at 3k scale for Tidus

**Recipe B — Frozen sentence-transformer + linear heads:**
- Backbone: `all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5` (frozen, no gradient)
- Heads: three class-weighted logistic regression heads on frozen `[CLS]` embedding
- Near-calibrated out of box (no temperature scaling needed)
- Retrains in seconds; iteration cost is trivial
- Safer at 3k scale; fallback if Recipe A underfits

**Selection criterion:**
- Macro-F1 on **clean eval tier** (100-150 prompts, Sonnet-labeled AND human-verified)
- Full WildChat held-out slice (Sonnet-only labels) measures noise memorization, not generalization — NOT used for final selection
- Recipe A > Recipe B by ≥ 2pp on clean eval → pick A
- Else → pick B (simpler, calibration-free, faster iteration)

**Confidence gating (Tier 3 escalation trigger):**
```python
confidence = {head: softmax(logits)[predicted_class] for head in heads}
escalate = any(confidence[head] < threshold[head] for head in heads)
```
Per-head thresholds from `settings.classify_*_threshold`.

### Tier 2b: Presidio (PARALLEL to Tier 2, benchmark-gated)

**Configuration:**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
analyzer.registry.remove_recognizer("SpacyRecognizer")  # disable NER-driven recognizer
```

**Covers:** CC+Luhn (redundant recheck), IBAN, phone, email, URL, crypto wallets, medical_license, country-specific IDs (SSN/ITIN/passport/driver's license variants across US/UK/ES/IT/PL/SG/AU/IN/FI/KR/NG/TH)

**Gaps filled by Tier 1:** AWS/GCP/Azure keys, GitHub/GitLab/Slack tokens, OpenAI/Anthropic keys (all in `detect-secrets` — not in Presidio's built-in recognizers)

**Latency risk:** UNVERIFIED. Microsoft's maintainer says "we don't have any formal results, and it's somewhat intentional." Phase 0.5 benchmark decides Tier 2b's placement:
- p95 ≤ 30ms → keep as parallel Tier 2b
- p95 > 30ms → demote to conditional Tier 3. Trigger: `encoder.privacy_confidence < classify_privacy_threshold AND not tier1.any_hit` (Tier 1 hit already forces `confidential`; Presidio recheck adds no value in that path)

**Output:** `presidio.detected_any_pii: bool` → feeds privacy merge rule

### Tier 3: Local LLM Fallback

**Model:** Phi-3.5-mini-instruct
- Params: 3.8B, MIT license (**verify current HF repo license before shipping** — Microsoft has shifted Phi licensing terms historically)
- MMLU 69.0, BBH 69.0 — top accuracy-per-dollar in 3B class
- Quantization: Q4_K_M GGUF (~2.4 GB), Q5_K_M (~2.8 GB)

**Alternative:** Llama-3.2-3B-Instruct (Llama 3.2 Community License, <700M MAU OK, "Built with Llama" attribution required) — switch if Phi-3.5-mini underfits Tidus's taxonomy

**Deployment:** Ollama on localhost
- Process isolation, independent OOM domain, crash containment, audit trail
- HTTP overhead ~2-5ms — negligible on 500ms budget
- Rationale: llama-cpp-python in-process offers no benefit for Tidus's Tier 3 (no KV-cache sharing need, and in-process serializes under FastAPI concurrency)

**Prompt:** ~130-token system instruction + user input truncated to 512 tokens
**Output:** structured JSON via Ollama grammar constraints:
```json
{"domain": "code", "complexity": "moderate", "privacy": "internal"}
```

**Caching:** SHA-256 fingerprint of input → TTL 1h, max 10K entries (LRU eviction)

**Rate limiting:** Per-worker-minute budget prevents Ollama from pinning at 100% on traffic spikes

**Trigger:** `any(encoder.softmax[head] < threshold[head] for head in heads) and not tier1.any_hit and not presidio.detected_any_pii`

**Expected volume:** <2% of traffic after encoder is trained

---

## Files to Create

| File | Purpose |
|---|---|
| `tidus/classification/__init__.py` | Package marker |
| `tidus/classification/models.py` | `ClassificationResult`, `Tier1Signals`, `EncoderResult`, `PresidioResult` dataclasses |
| `tidus/classification/heuristics.py` | Regex + keyword detection, structural signals, token estimation |
| `tidus/classification/secrets.py` | detect-secrets in-memory wrapper (no temp files) |
| `tidus/classification/keywords.py` | Aho-Corasick keyword trie, MeSH slice loader |
| `tidus/classification/encoder.py` | ONNX session + multi-head classification + temperature scaling |
| `tidus/classification/presidio_wrapper.py` | `AnalyzerEngine` + `SpacyRecognizer` removal + async wrapping |
| `tidus/classification/llm_classifier.py` | Ollama client + grammar + cache |
| `tidus/classification/classifier.py` | `TaskClassifier` — orchestrates Tier 0→1→(2‖2b)→3 + merge rule |
| `tidus/classification/weights/encoder_v1.onnx` | Trained encoder (git-lfs) |
| `tidus/classification/keywords/medical.txt` | MeSH-derived medical keyword list |
| `tidus/classification/keywords/legal.txt` | Homebrew legal keywords |
| `tidus/classification/keywords/financial.txt` | Homebrew financial keywords + PCI DSS glossary |
| `scripts/label_wildchat.py` | Phase 0 Sonnet batch-labeler |
| `scripts/benchmark_presidio.py` | Phase 0.5 latency benchmark |
| `scripts/train_encoder.py` | Phase 1 training (Recipe A + Recipe B) |
| `scripts/adversarial_eval.py` | Phase 3 adversarial test suite |
| `tests/classification/real_traffic_eval.jsonl` | Phase 0 labeled eval set (1000 prompts) — permanent eval harness |

## Files to Modify

| File | Change |
|---|---|
| `tidus/settings.py` | Add 9 new settings (see below) |
| `tidus/api/deps.py` | Add `get_classifier()` singleton; init in `build_singletons()` |
| `tidus/api/v1/complete.py` | Make `complexity`, `domain`, `privacy`, `estimated_input_tokens` → `Optional[...]`; call classifier on None |
| `tidus/api/v1/route.py` | Same Optional fields + classifier call |
| `tidus/api/v1/classify.py` | NEW — `POST /api/v1/classify` endpoint |
| `tidus/main.py` | Call `await TaskClassifier.startup()` in lifespan (load encoder, warm Presidio, ping Ollama) |
| `pyproject.toml` | Add dependencies (see below) |
| `alembic/versions/` | Migration for classifier decision log (optional — audit trail of which tier fired) |

## New Settings (tidus/settings.py)

```python
# Auto-classification layer (v1.3.0)
auto_classify_enabled: bool = True
classify_encoder_path: str = "tidus/classification/weights/encoder_v1.onnx"
classify_llm_model_id: str = "phi3.5:mini-instruct"    # Ollama model name
classify_llm_endpoint: str = "http://localhost:11434"  # Ollama URL
classify_privacy_threshold: float = 0.75               # per-head confidence gate
classify_domain_threshold: float = 0.70
classify_complexity_threshold: float = 0.65
classify_presidio_enabled: bool = True                 # set False to disable entirely
classify_presidio_parallel: bool = True                # False = demote to conditional Tier 3
classify_cache_ttl: int = 3600                         # Tier 3 LLM cache TTL (seconds)
classify_cache_max_entries: int = 10_000
classify_llm_rate_limit_per_minute: int = 60           # per-worker rate cap
```

## New Dependencies (pyproject.toml)

```toml
[project]
dependencies = [
    # ... existing ...
    "presidio-analyzer >= 2.2.362",   # MIT — PII NER (pattern-based + context)
    "detect-secrets @ git+https://github.com/Yelp/detect-secrets@<PIN_SHA>",  # Apache 2.0 — secrets
    "pyahocorasick >= 2.0.0",         # BSD-3 — O(n) keyword matching
    "onnxruntime >= 1.20.0",          # MIT — encoder inference
    "spacy >= 3.7.0",                 # MIT — Presidio tokenization
    "sentence-transformers >= 3.0.0", # Apache 2.0 — Recipe B fallback, already used by cache
    # spaCy model installed post-install:  python -m spacy download en_core_web_sm
]
```

## New Endpoint

`POST /api/v1/classify`

**Request:**
```json
{
  "messages": [{"role": "user", "content": "diagnose my symptoms"}],
  "team_id": "team-a"
}
```

**Response:**
```json
{
  "domain": "reasoning",
  "complexity": "critical",
  "privacy": "internal",
  "estimated_input_tokens": 5,
  "classification_tier": "heuristic",
  "confidence": {
    "domain": 0.92,
    "complexity": 1.0,
    "privacy": 0.85
  },
  "debug": {
    "tier1_hits": {"regex": [], "secrets": [], "keywords": ["diagnose"]},
    "tier2_called": false,
    "tier2b_presidio_hits": [],
    "tier3_called": false,
    "latency_ms": 2.3
  }
}
```

---

## Tests

| File | Covers |
|---|---|
| `tests/unit/classification/test_heuristics.py` | Regex, Luhn, keyword lookups, structural signals, token estimation |
| `tests/unit/classification/test_secrets.py` | In-memory scanning (no temp file), plugin subset, FP rates on safe text |
| `tests/unit/classification/test_keywords.py` | Aho-Corasick correctness, MeSH loader, case-insensitive matching |
| `tests/unit/classification/test_encoder.py` | ONNX loading, multi-head forward pass, temperature scaling, confidence gating |
| `tests/unit/classification/test_presidio_wrapper.py` | SpacyRecognizer removal, parallel execution, NPE on empty input |
| `tests/unit/classification/test_llm_classifier.py` | Ollama JSON parsing, malformed-JSON fallback, cache hit/miss, rate limit |
| `tests/unit/classification/test_merge_rule.py` | Full truth table of privacy merge (16+ cases) |
| `tests/unit/classification/test_classifier.py` | Tier cascade, short-circuit logic, caller override, concurrency |
| `tests/integration/test_classify_endpoint.py` | `POST /classify`, `POST /complete` without metadata, SSN → confidential routing |
| `tests/integration/test_backward_compat.py` | v1.1 requests (all fields provided) still work identically |

## Verification Checklist

1. `POST /api/v1/classify` with code message → `domain=code, complexity=moderate, tier=heuristic`
2. Same message repeated → `tier=heuristic` (heuristics hit every time) OR `tier=cached` (if LLM escalation happened and cache hit)
3. `POST /api/v1/complete` with only `team_id + messages` → 200 OK, `chosen_model_id` present
4. SSN in message → `privacy=confidential`, `chosen_model_id` is `is_local=True`
5. `"diagnose my symptoms"` → `complexity=critical` (medical keyword veto)
6. `auto_classify_enabled=false` + fields omitted → 422 validation error
7. `GET /metrics` → `tidus_classify_tier_total{tier="heuristic|encoder|llm"}` increments
8. Presidio disabled via `classify_presidio_enabled=false` → no import of presidio_analyzer, no latency penalty
9. Concurrent request load (200 req/s) → encoder + Presidio run in parallel (total latency = max(), not sum)
10. Tier 3 rate limit reached → encoder output used directly with `confidence_warning: true` flag

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Presidio latency exceeds 30ms on CPU | Medium | Medium | Phase 0.5 benchmark-first; demote to conditional Tier 3 if needed |
| Recipe A (LoRA) underfits at 3k scale | Medium | Low | Train Recipe B in parallel; pick winner on **clean eval tier** (not Sonnet-labeled held-out) |
| WildChat-1M distribution ≠ real Tidus traffic | High | Medium | Re-run Phase 0 on actual audit log post-deployment; regenerate eval set |
| Adversarial PII examples break 99.6% claim | High | High | Adversarial eval harness (Phase 3); Privacy SLO gate (≤1% FN); compliance sign-off |
| **Sonnet label noise propagates to encoder (3-8% estimated)** | **Medium** | **Medium** | **Weak supervision: require Sonnet + POC heuristic agreement for high-confidence labels. Human-audit 50 disagreement prompts (signal-rich). Clean eval tier (100-150 human-verified) used for recipe selection, NOT Sonnet-only held-out.** |
| **Non-English input breaks English-only regex/keywords/spaCy** | **Medium** | **Medium** | **Detect language (langdetect, <1ms); non-English → default `internal`, skip Tier 2, route via Tier 0/1 only. Multi-language support deferred to v1.4.** |
| mmBERT-based PII detector unusable without GPU | Low | Low | Already deferred; Tidus uses Presidio + detect-secrets + custom |
| Ollama Tier 3 unavailable at request time | Medium | Low | Graceful degradation: use encoder output with `confidence_warning: true` flag |
| detect-secrets stale release breaks on Python 3.13 | Low | Low | Pinned to active master SHA; test matrix covers Python 3.12/3.13 |

---

## Privacy SLO (Enterprise Compliance Commitment)

**False-negative rate on the adversarial eval set (Phase 3) must be ≤ 1%.**

- Re-verified quarterly post-ship against the maintained adversarial eval harness
- If breached: auto-rollback classifier to Tier 1-only (regex + Presidio + keywords, no encoder, no LLM) until fixed
- This is the customer-facing compliance handshake. "Asymmetric overclassification" is engineering philosophy; ≤1% FN rate is the signable commitment
- Applies to `privacy == confidential` predictions specifically; domain/complexity FN are routing-optimization losses, not compliance incidents

## What NOT to Do

- Never transmit messages to external services for classification (hard constraint)
- Never write confidential data to disk (LLM Guard's temp-file pattern is banned)
- Never default privacy to `public` (always `internal` or `confidential`)
- Never skip the adversarial eval before shipping the privacy tier
- Never ship with `KeywordDetector` or `BasicAuthDetector` enabled in detect-secrets (redundant + high FP)
- Never use `en_core_web_trf` or `en_core_web_lg` for Presidio's NLP engine — `en_core_web_sm` only, NER recognizer removed
- Never count tokens via the encoder or LLM — stays in Tier 1 heuristic
- Never select recipe A vs B on full Sonnet-labeled held-out (noise memorization confound — use clean eval tier only)
- Never update the POC heuristics during Phase 0 gate check (avoid moving the target)

---

## Sources / Research Validation

- **vLLM Semantic Router** — `https://github.com/vllm-project/semantic-router` (architecture precedent, training recipe port, PII/jailbreak weights available MIT/Apache 2.0)
- **RouteLLM** — arXiv:2406.18665 (trained routers beat heuristics at routing-scale 50k examples)
- **FrugalGPT** — arXiv:2305.05176 (cascade-with-confidence-gate pattern validated)
- **RouterBench** — arXiv:2403.12031 (simple trained routers outperform heuristics)
- **Presidio** — `https://github.com/microsoft/presidio` v2.2.362 (2026-03-18), MIT, active
- **detect-secrets** — `https://github.com/Yelp/detect-secrets` (master active through 2026-04, tagged v1.5.0 May 2024)
- **DeBERTa-v3** — Microsoft, MIT, 22M/44M backbone variants, +11 GLUE over DistilBERT
- **Phi-3.5-mini-instruct** — Microsoft, MIT, MMLU 69.0 / BBH 69.0
- **WildChat-1M** — Allen AI, ODC-BY, real ChatGPT/Claude conversations with real PII
- **MeSH** — NLM, free programmatic access (verify ToS before commercial redistribution)

---

## POC Script (unchanged)

`scripts/poc_classifier.py` — validation script from 2026-04-17, 94.5% domain / 86.9% complexity / 99.6% privacy on 1889 synthetic cases. Runs standalone, no Tidus imports. Kept for Phase 0 backtest comparison.

```bash
uv run python scripts/poc_classifier.py --no-embedding     # Tier 1 heuristics only
uv run python scripts/poc_classifier.py                     # Tier 1 + Tier 2 embedding
uv run python scripts/poc_classifier.py --verbose           # per-case diagnostics
```
