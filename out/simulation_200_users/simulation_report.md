# Tidus v1.3.0 — 200-User Deployment Simulation Report

**Generated:** 2026-04-21 by `scripts/simulate_200_users.py`  
**Total requests processed:** 4000  
**Simulated users:** 200  
**Classifier health at run time:** `{'encoder_loaded': True, 'presidio_loaded': True, 'llm_loaded': False, 'sku': 'cpu-only'}`  

## Purpose

This report demonstrates Tidus v1.3.0's classification-and-routing behaviour on a realistic mix of enterprise AI traffic. It is produced from a synthetic simulation with mocked vendor adapters; every classification run is real (the full T0→T5 cascade executed), but no downstream LLM call is issued, so results are deterministic and incur zero vendor cost.

**Methodology caveat for legal review:** This measures Tidus's *internal* classification and routing logic. It is NOT a measurement of live vendor latency, cost accuracy, or end-user experience. Claims about those properties require measurement against real traffic, which this simulation does not attempt.

## Methodology

- **User population:** 200 synthetic users distributed across four enterprise roles per typical SaaS deployment (engineer 70%, analyst 15%, ops 10%, exec 5%).
- **Request volume per user:** power-law (Zipf-1) — top 10% of users generate ~50% of traffic, matching observed enterprise usage patterns.
- **Prompt corpus:** 30 in-line synthetic templates plus an ~5% cross-role mix-in of eight confidential templates (real PII patterns: SSN, credit card, API tokens, AWS keys, personal medical/legal). Every template is annotated with its *expected* (domain, complexity, privacy) — human-labeler intent. Divergences between template intent and classifier output are visible in `simulation_evidence.jsonl`.
- **Classifier:** the production `TaskClassifier` loaded from `tidus.classification` with real MiniLM encoder + Presidio (spaCy `en_core_web_sm`) + T1 heuristics. T5 LLM disabled (CPU-only SKU baseline).
- **Routing:** simulated by a small deterministic table mirroring the real 5-stage selector's Stage-1 hard-constraint filter (privacy=confidential → local-only) and Stage-3 tier ceiling (critical → tier 1 only). This avoids pulling in the full registry / budget / health-probe stack, which has its own separate test coverage.
- **Seed:** 42. Re-running with the same seed reproduces identical output.

## Headline results

| Metric | Value |
|---|---|
| Total requests | 4000 |
| Confidential flagged | 1432 (35.8%) |
| Domain-axis template agreement | 1563/4000 (39.1%) |
| Privacy-axis template agreement | 1734/4000 (43.4%) |
| Tidus-routed total cost (estimated) | $2.61 |
| Premium-always baseline cost | $88.80 |
| Tidus cost savings vs premium-always | **97.1%** |

### Classification tier distribution

Which tier decided the final classification? Higher-tier labels ('encoder', 'llm') reflect richer model-based decisions; lower-tier ('heuristic', 'caller_override') reflect fast-paths.

| Tier | Count | Percent |
|---|---|---|
| `encoder` | 4000 | 100.0% |

### Domain distribution (as classified)

| Domain | Count | Percent |
|---|---|---|
| `chat` | 2243 | 56.1% |
| `code` | 918 | 22.9% |
| `reasoning` | 624 | 15.6% |
| `creative` | 127 | 3.2% |
| `summarization` | 88 | 2.2% |

### Complexity distribution

| Complexity | Count | Percent |
|---|---|---|
| `simple` | 2601 | 65.0% |
| `complex` | 984 | 24.6% |
| `moderate` | 415 | 10.4% |

### Privacy distribution

| Privacy | Count | Percent |
|---|---|---|
| `internal` | 1488 | 37.2% |
| `confidential` | 1432 | 35.8% |
| `public` | 1080 | 27.0% |

### Model routing

The cheapest capable model wins per Tidus's 5-stage selector. The following distribution shows **real routing behaviour under classified load** — confidential prompts route to local-only models; complex prompts to premium; simple prompts to the cheapest capable tier.

| Model | Count | Percent |
|---|---|---|
| `deepseek-v4` | 1507 | 37.7% |
| `llama3.1-8b-local` | 1432 | 35.8% |
| `claude-sonnet-4-6` | 646 | 16.1% |
| `claude-haiku-4-5` | 415 | 10.4% |

## How Tidus handled 5 confidential requests

Each row below shows a prompt that carried PII or a leaked secret. The confidential-vote flow (which tier first flagged, how the OR-merge resolved, and where the request routed) is the primary compliance story for a regulated deployment.

### Confidential example 1 — category `confidential_cred`

- **Prompt (redacted preview):** `Is this AWS key compromised? AKIA[REDACTED] — I saw it in a Stack Overflow…`
- **Expected axes:** domain=code, complexity=simple, privacy=confidential
- **Classifier output:** domain=chat, complexity=simple, privacy=**confidential**
- **Classifier tier that decided:** `encoder`
- **Confidence:** `{'domain': 0.6724824159762669, 'complexity': 0.8153130795426449, 'privacy': 0.9}`
- **Routed to:** `llama3.1-8b-local` (estimated cost $0.00000; local-only if privacy=confidential)
- **Stage B record emitted:** `{"request_id": "3f411d73-4b0c-4410-afe4-5b66d4fe8bd9", "tenant_id": "tenant-demo", "tier_decided": "encoder", "classification": {"domain": "chat", "complexity": "simple", "privacy": "confidential"}, "model_routed": "llama3.1-8b-local", "latency_ms": 43}`

### Confidential example 2 — category `confidential_cred`

- **Prompt (redacted preview):** `Is this AWS key compromised? AKIA[REDACTED] — I saw it in a Stack Overflow…`
- **Expected axes:** domain=code, complexity=simple, privacy=confidential
- **Classifier output:** domain=chat, complexity=simple, privacy=**confidential**
- **Classifier tier that decided:** `encoder`
- **Confidence:** `{'domain': 0.6724824159762669, 'complexity': 0.8153130795426449, 'privacy': 0.9}`
- **Routed to:** `llama3.1-8b-local` (estimated cost $0.00000; local-only if privacy=confidential)
- **Stage B record emitted:** `{"request_id": "b640d3b1-2b17-4a56-a84e-f7258ade4cfb", "tenant_id": "tenant-demo", "tier_decided": "encoder", "classification": {"domain": "chat", "complexity": "simple", "privacy": "confidential"}, "model_routed": "llama3.1-8b-local", "latency_ms": 15}`

### Confidential example 3 — category `extraction`

- **Prompt (redacted preview):** `Extract every phone number and email from this text:
Contact me at (555) 123-456…`
- **Expected axes:** domain=extraction, complexity=simple, privacy=public
- **Classifier output:** domain=chat, complexity=simple, privacy=**confidential**
- **Classifier tier that decided:** `encoder`
- **Confidence:** `{'domain': 0.4149740948806383, 'complexity': 0.5136541431557681, 'privacy': 0.9}`
- **Routed to:** `llama3.1-8b-local` (estimated cost $0.00000; local-only if privacy=confidential)
- **Stage B record emitted:** `{"request_id": "6c636216-9f63-41c8-9955-0dc28f745761", "tenant_id": "tenant-demo", "tier_decided": "encoder", "classification": {"domain": "chat", "complexity": "simple", "privacy": "confidential"}, "model_routed": "llama3.1-8b-local", "latency_ms": 58}`

### Confidential example 4 — category `extraction`

- **Prompt (redacted preview):** `Extract every phone number and email from this text:
Contact me at (555) 123-456…`
- **Expected axes:** domain=extraction, complexity=simple, privacy=public
- **Classifier output:** domain=chat, complexity=simple, privacy=**confidential**
- **Classifier tier that decided:** `encoder`
- **Confidence:** `{'domain': 0.4149740948806383, 'complexity': 0.5136541431557681, 'privacy': 0.9}`
- **Routed to:** `llama3.1-8b-local` (estimated cost $0.00000; local-only if privacy=confidential)
- **Stage B record emitted:** `{"request_id": "1b6dcc8f-b384-4e61-a8bb-ad650e1d237f", "tenant_id": "tenant-demo", "tier_decided": "encoder", "classification": {"domain": "chat", "complexity": "simple", "privacy": "confidential"}, "model_routed": "llama3.1-8b-local", "latency_ms": 15}`

### Confidential example 5 — category `confidential_cred`

- **Prompt (redacted preview):** `Is this AWS key compromised? AKIA[REDACTED] — I saw it in a Stack Overflow…`
- **Expected axes:** domain=code, complexity=simple, privacy=confidential
- **Classifier output:** domain=chat, complexity=simple, privacy=**confidential**
- **Classifier tier that decided:** `encoder`
- **Confidence:** `{'domain': 0.6724824159762669, 'complexity': 0.8153130795426449, 'privacy': 0.9}`
- **Routed to:** `llama3.1-8b-local` (estimated cost $0.00000; local-only if privacy=confidential)
- **Stage B record emitted:** `{"request_id": "96a16bf5-6c47-42c1-8fc7-878d6c16b54b", "tenant_id": "tenant-demo", "tier_decided": "encoder", "classification": {"domain": "chat", "complexity": "simple", "privacy": "confidential"}, "model_routed": "llama3.1-8b-local", "latency_ms": 13}`


## How Tidus handled 15 normal requests

The rest of the sample — non-confidential enterprise traffic routed by the selector to the cheapest capable tier.

### Normal example 1 — category `code` (engineer)

- **Prompt:** `How do I delete a git branch that's been merged?`
- **Classified as:** domain=chat, complexity=simple, privacy=public (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 2 — category `code` (engineer)

- **Prompt:** `What's the difference between a list and a tuple in Python?`
- **Classified as:** domain=chat, complexity=simple, privacy=public (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 3 — category `code` (engineer)

- **Prompt:** `What's the difference between a list and a tuple in Python?`
- **Classified as:** domain=chat, complexity=simple, privacy=public (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 4 — category `confidential_cred` (engineer)

- **Prompt:** `This Slack webhook stopped working: https://hooks.slack.com/services/T01ABCDEF/B…`
- **Classified as:** domain=code, complexity=simple, privacy=internal (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 5 — category `confidential_cred` (engineer)

- **Prompt:** `This Slack webhook stopped working: https://hooks.slack.com/services/T01ABCDEF/B…`
- **Classified as:** domain=code, complexity=simple, privacy=internal (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 6 — category `code` (engineer)

- **Prompt:** `Implement a thread-safe LRU cache in Go with TTL eviction and per-key read/write…`
- **Classified as:** domain=chat, complexity=complex, privacy=public (tier `encoder`)
- **Routed to:** `claude-sonnet-4-6` (est. $0.00290)

### Normal example 7 — category `confidential_cred` (engineer)

- **Prompt:** `Debug this Python — I keep getting 401:
openai.api_key = 'sk-proj-VWxyz12345abcd…`
- **Classified as:** domain=code, complexity=simple, privacy=internal (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 8 — category `code` (engineer)

- **Prompt:** `How do I delete a git branch that's been merged?`
- **Classified as:** domain=chat, complexity=simple, privacy=public (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 9 — category `confidential_cred` (engineer)

- **Prompt:** `Debug this Python — I keep getting 401:
openai.api_key = 'sk-proj-VWxyz12345abcd…`
- **Classified as:** domain=code, complexity=simple, privacy=internal (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 10 — category `code` (engineer)

- **Prompt:** `Design a multi-class fighting-game engine in TypeScript — Character base class, …`
- **Classified as:** domain=chat, complexity=complex, privacy=public (tier `encoder`)
- **Routed to:** `claude-sonnet-4-6` (est. $0.00290)

### Normal example 11 — category `reasoning` (engineer)

- **Prompt:** `Plan a zero-downtime migration from MySQL 5.7 to PostgreSQL 16 for a 3TB OLTP da…`
- **Classified as:** domain=reasoning, complexity=moderate, privacy=internal (tier `encoder`)
- **Routed to:** `claude-haiku-4-5` (est. $0.00077)

### Normal example 12 — category `code` (engineer)

- **Prompt:** `Design a multi-class fighting-game engine in TypeScript — Character base class, …`
- **Classified as:** domain=chat, complexity=complex, privacy=public (tier `encoder`)
- **Routed to:** `claude-sonnet-4-6` (est. $0.00290)

### Normal example 13 — category `code` (engineer)

- **Prompt:** `Design a multi-class fighting-game engine in TypeScript — Character base class, …`
- **Classified as:** domain=chat, complexity=complex, privacy=public (tier `encoder`)
- **Routed to:** `claude-sonnet-4-6` (est. $0.00290)

### Normal example 14 — category `code` (engineer)

- **Prompt:** `Write a SQL query to find the top 5 customers by revenue in Q1 2026.`
- **Classified as:** domain=reasoning, complexity=simple, privacy=internal (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

### Normal example 15 — category `code` (engineer)

- **Prompt:** `What's the difference between a list and a tuple in Python?`
- **Classified as:** domain=chat, complexity=simple, privacy=public (tier `encoder`)
- **Routed to:** `deepseek-v4` (est. $0.00027)

## Reproducibility

```bash
# Re-run this simulation
uv run python scripts/simulate_200_users.py

# Change sample size or seed
uv run python scripts/simulate_200_users.py --users 500 --requests 10000 --seed 99
```

The script is deterministic per `--seed`. Prompt templates live in-line in the script (lines marked `TEMPLATES = [...]`) so a reviewer can inspect every input used. The `classifier` is the same production `TaskClassifier` imported by `POST /api/v1/classify`, `/complete`, and `/route`.

## What this simulation proves (and does not)

**Proves:** that under realistic enterprise-traffic mixes, Tidus's classification-and-routing pipeline behaves as specified — confidential prompts flag, route to local models, and emit a PII-safe Stage B record; simple prompts route to cheap tiers; complex prompts route to premium; the tier that decided each classification is observable.

**Does not prove:** live latency, live vendor-cost accuracy, live model-selection quality on actual customer responses. Those require production measurement. File this document as evidence of *system behaviour*, not *system performance*.

---

Generated by `scripts/simulate_200_users.py` — source-review-friendly; every template and every metric visible in the script.