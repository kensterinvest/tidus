# Configuration Reference

Tidus uses three YAML configuration files in the `config/` directory.

| File | Purpose |
|------|---------|
| `config/models.yaml` | Model registry: all models, pricing, capabilities, tiers |
| `config/budgets.yaml` | Per-team and per-workflow spending limits |
| `config/policies.yaml` | Guardrail limits and routing scoring weights |

---

## config/models.yaml

Each entry defines one model. Fields are validated against `ModelSpec` (Pydantic v2) at startup.

### Schema

```yaml
- model_id: "claude-haiku-4-5"    # unique string ID used in API calls
  vendor: "anthropic"              # matches adapter class vendor constant
  tier: 3                          # 1=premium, 2=mid, 3=economy, 4=local/free

  max_context: 200000              # maximum context window in tokens
  input_price: 0.001               # USD per 1,000 input tokens  (0.0 for local)
  output_price: 0.005              # USD per 1,000 output tokens (0.0 for local)

  tokenizer: "anthropic"           # see Tokenizer Types below
  latency_p50_ms: 300              # observed median latency; updated by health probe

  capabilities:                    # list — see Capability Values below
    - chat
    - extraction
    - classification
    - summarization

  min_complexity: "simple"         # lowest complexity this model is designed for
  max_complexity: "moderate"       # highest complexity this model is designed for

  is_local: false                  # true = no API key, on-prem/Ollama
  enabled: true                    # false = excluded from all routing
  deprecated: false                # true = EOL, set enabled=false automatically

  fallbacks:                       # ordered fallback chain if this model fails
    - "gpt-4o-mini"
    - "mistral-small"
    - "llama4-scout-ollama"

  last_price_check: "2026-03-26"   # ISO 8601 date; updated by weekly sync job
  last_health_check: null          # ISO 8601 datetime; updated by 5-min health probe
```

### Tier Values

| Tier | Value | Use For | Examples |
|------|-------|---------|---------|
| `premium` | 1 | Planning, reasoning, compliance, critical decisions | o3, claude-opus-4-6, gpt-5-codex, deepseek-r1 |
| `mid` | 2 | Summarisation, extraction, complex code, routine tasks | claude-sonnet-4-6, deepseek-v3, mistral-large-3 |
| `economy` | 3 | Classification, filtering, simple code, pre-processing | claude-haiku-4-5, gpt-4o-mini, mistral-small |
| `local` | 4 | On-prem, zero API cost, privacy-sensitive | llama4-maverick-ollama, gemini-nano, phi-4-ollama |

### Complexity Values

| Value | Maps To | Model Design Range |
|-------|---------|-------------------|
| `simple` | order 0 | Filtering, routing, simple Q&A |
| `moderate` | order 1 | Summarisation, extraction, standard code |
| `complex` | order 2 | Multi-step reasoning, complex code, analysis |
| `critical` | order 3 | Compliance, planning, high-stakes decisions |

The `complexity_mismatch` rejection (stage 1) fires when a task's complexity falls outside a model's `[min_complexity, max_complexity]` range. This is a hard constraint — not scoring preference.

### Tokenizer Types

| Value | Used By | Method |
|-------|---------|--------|
| `tiktoken_o200k` | OpenAI GPT-4o, GPT-4.1, o3, Codex | tiktoken local encoding |
| `tiktoken_cl100k` | DeepSeek, xAI, Kimi | tiktoken local encoding |
| `anthropic` | Claude family | Anthropic count_tokens API |
| `sentencepiece` | Mistral family | sentencepiece local |
| `google` | Gemini family | google-generativeai SDK |
| `ollama` | Local models | Ollama tokenize endpoint |

### Capability Values

| Value | Meaning |
|-------|---------|
| `chat` | General conversational tasks |
| `code` | Code generation, debugging, review |
| `reasoning` | Multi-step logical reasoning, planning |
| `extraction` | Structured data extraction from text |
| `classification` | Labelling, categorisation, intent detection |
| `summarization` | Document and conversation summarisation |
| `creative` | Creative writing, brainstorming |
| `multimodal` | Image + text inputs |
| `long_context` | Optimised for very long documents (>100K tokens) |
| `agents` | Tool use, function calling, multi-step agentic tasks |

### Complexity Tier Ceiling (Stage 3)

The tier ceiling enforces that higher-complexity tasks use higher-quality models:

| Task Complexity | Max Tier Allowed | Excluded |
|----------------|-----------------|---------|
| `simple` | 4 (any) | Nothing excluded |
| `moderate` | 3 (economy) | Tier 1–2 models still eligible; tier ceiling only blocks upward here in reverse — actually this limits to tiers 1-3 |
| `complex` | 2 (mid) | Tier 3–4 models excluded |
| `critical` | 1 (premium) | Only tier 1 models |

*Note: the ceiling prevents routing critical tasks to economy models, but cheaper tier-1 models (e.g. deepseek-r1 at $0.00055/1K) still win via the cost-weighted scoring formula.*

---

## config/budgets.yaml

```yaml
budgets:
  - policy_id: "team-eng-monthly"
    scope: team                      # team | workflow
    scope_id: "team-engineering"     # team_id or workflow_id
    period: monthly                  # daily | weekly | monthly | rolling_30d
    limit_usd: 500.00
    warn_at_pct: 0.80                # alert at 80% utilisation
    hard_stop: true                  # true = reject; false = warn only
```

See [budgets-and-guardrails.md](budgets-and-guardrails.md) for full details.

---

## config/policies.yaml

```yaml
guardrails:
  max_agent_depth: 5          # Stage 2 hard constraint
  max_tokens_per_step: 8000   # Stage 2 hard constraint
  max_retries_per_task: 3     # enforced in AgentGuard

  # Roadmap:
  # max_concurrent_agents: 10
  # max_reflection_loops: 3
  # max_total_tokens_session: 50000

routing:
  cost_weight: 0.70           # Stage 5 scoring weights
  tier_weight: 0.20
  latency_weight: 0.10
  # must sum to 1.0

  complexity_tier_ceiling:    # Stage 3 — maximum tier per complexity level
    simple: 4
    moderate: 3
    complex: 2
    critical: 1

cost:
  estimate_buffer_pct: 0.15   # Safety buffer on cost estimates (15%)
```

### Routing Weights

The scoring formula at Stage 5 is:
```
score = cost_normalised × cost_weight
      + tier_normalised × tier_weight
      + latency_normalised × latency_weight
```

All three components are min-max normalised across eligible candidates before weighting, so the score always falls in [0, 1]. Lower score = better. This means:
- A model that costs 1/10th of another gets a 70% cost-component advantage
- A tier-2 model beats a tier-1 model of equal cost due to the 20% tier weight
- Latency only contributes 10% — correctness and cost matter more than speed

To prioritise latency over cost (e.g. for real-time interactive tasks), increase `latency_weight` and decrease `cost_weight`. Weights must sum to 1.0.

---

## Environment Variables (`.env`)

```env
# Vendor API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
XAI_API_KEY=...
MOONSHOT_API_KEY=...

# Local inference
OLLAMA_BASE_URL=http://localhost:11434

# Application
DATABASE_URL=sqlite+aiosqlite:///./tidus.db
LOG_LEVEL=INFO
ENVIRONMENT=development

# Caching (optional — defaults shown)
CACHE_EXACT_TTL_SECONDS=3600
CACHE_SEMANTIC_TTL_SECONDS=900
SEMANTIC_CACHE_THRESHOLD=0.95

# Redis backend (optional — defaults to in-memory; roadmap)
# REDIS_URL=redis://localhost:6379/0
```

Only the keys for vendors you intend to use are required. Tidus routes only to models whose vendor adapter has a valid API key (or is local).
