# Tidus — Enterprise AI Router

> **The AI system that governs all AIs.**

Tidus is a vendor-agnostic, enterprise-grade, cost-aware AI model router and governance system. Every AI request in your organisation passes through Tidus before reaching any model or AI service. Tidus selects the **cheapest model capable of doing the job**, enforces team and workflow budgets, prevents runaway multi-agent loops, and gives you full visibility into your AI spend.

```
┌───────────────────────────────────────────────────────────────┐
│                Enterprise AI Router (Tidus)                    │
├───────────────────────────────────────────────────────────────┤
│  Tiered Model Strategy   │  Router Agent Intelligence          │
│  Response Caching        │  Agent Autonomy Limits              │
│  Vendor-Agnostic Design  │  Budget + Policy Enforcement        │
├───────────────────────────────────────────────────────────────┤
│  OpenAI │ Anthropic │ Google │ Mistral │ DeepSeek              │
│  xAI    │ Kimi      │ Ollama (local/on-prem)                   │
└───────────────────────────────────────────────────────────────┘
```

## Why Tidus?

Most AI systems default every request to the most powerful (most expensive) model — even for simple tasks like chat replies, classification, or summarisation. In enterprise multi-agent workflows, this cost multiplies with every agent hop, retry, and reflection loop.

**Tidus implements five cost-control pillars** that together eliminate 70–90% of enterprise AI spend:

| Pillar | What It Does | Status |
|--------|-------------|--------|
| **1. Tiered Model Strategy** | Routes to the cheapest capable tier — local models for filtering/routing, mid-tier for summarisation/extraction, premium only for reasoning/compliance | ✅ Live |
| **2. Router Agent Intelligence** | Decides model, local vs. cloud, multi-agent need, and budget feasibility before any compute runs | ✅ Live |
| **3. Response Caching** | Exact-match and semantic caching — identical or near-identical queries pay once; 30–50% additional reduction | Planned Phase 4 |
| **4. Agent Autonomy Limits** | Max depth, agents, tokens-per-step, retries, and reflection loops — prevents runaway compute | Partial (depth + tokens live; reflection loops in Phase 4) |
| **5. Vendor-Agnostic Architecture** | Pluggable adapters + MCP + A2A protocol — swap vendors without rewriting your system | Partial (adapters in Phase 4, MCP/A2A in Phase 6) |

### Cost Impact — 500 users × 200 requests/day

| Scenario | Monthly AI Cost | Tidus Fee | Net Cost | Saving |
|----------|----------------|-----------|---------|--------|
| Without Tidus (always Claude Opus 4.6) | $63,000 | — | $63,000 | — |
| Tidus Pro — Pillar 1+2 tiered routing | $8,400 | $99 | $8,499 | **87%** |
| Tidus Business — + local models | $4,200 | $499 | $4,699 | **93%** |
| Tidus Business — + caching (Phase 4) | ~$2,100 | $499 | $2,599 | **96%** |

*Based on real 2026 pricing. DeepSeek R1 wins critical reasoning at $0.00055/1K input vs. Claude Opus at $0.005/1K — a 9× cost difference for tier-1 quality.*

See [docs/roi-calculator.md](docs/roi-calculator.md) for your custom calculation.

## The 5-Stage Selection Algorithm (Pillars 1 & 2)

Every request passes through five sequential stages. The first failure rejects with a typed reason:

```
Stage 1 — Hard Constraints
  enabled, context window fits, domain capability supported,
  confidential tasks → local-only models,
  task complexity within model's min/max designed range

Stage 2 — Guardrails (Pillar 4 — partially live)
  agent_depth < max_agent_depth (default: 5)
  estimated_input_tokens < max_tokens_per_step (default: 8,000)

Stage 3 — Complexity Tier Ceiling
  simple   → any tier  (prefer cheapest)
  moderate → tier ≤ 3
  complex  → tier ≤ 2
  critical → tier 1 only

Stage 4 — Budget Filter
  reject if estimated cost exceeds team/workflow budget

Stage 5 — Score & Select
  score = cost×0.70 + tier_penalty×0.20 + latency×0.10
  lowest score wins
```

## Features

| Category | Feature |
|----------|---------|
| **Routing** | 5-stage model selection: hard constraints → guardrails → tier ceiling → budget → scoring |
| **Complexity** | Per-model `min_complexity`/`max_complexity` — prevents routing simple tasks to premium models |
| **Cost** | Token accounting with 15% safety buffer, per-call cost logging, weekly price sync |
| **Budget** | Per-team and per-workflow limits with hard-stop or warn-only behaviour |
| **Guardrails** | Max agent depth (5), max tokens per step (8K), max retries (3), session tracking |
| **Caching** | *(Phase 4)* Exact + semantic response caching — 30–50% additional cost reduction |
| **Vendors** | OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, xAI, Kimi, Ollama (local) |
| **Integration** | *(Phase 6)* MCP server for Claude Desktop / Cursor; A2A agent interoperability |
| **Visibility** | *(Phase 5)* Dashboard: cost by model, budget utilisation, active sessions, cache hit rate |
| **Resilience** | Automatic fallback chains, health probes, model deprecation handling |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/lapkei01/tidus.git
cd tidus
uv sync

# 2. Configure
cp .env.example .env
# Edit .env — add your API keys

# 3. Run
uvicorn tidus.main:app --reload

# 4. Route a request (selects best model — no execution)
curl -X POST http://localhost:8000/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-engineering",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 200,
    "messages": [{"role": "user", "content": "Summarise this ticket in one sentence."}]
  }'
```

Response:
```json
{
  "task_id": "a1b2c3...",
  "accepted": true,
  "chosen_model_id": "deepseek-v3",
  "estimated_cost_usd": 0.000056,
  "score": 0.09
}
```

See [docs/quickstart.md](docs/quickstart.md) for the full guide including `/api/v1/complete` (route + execute).

## Supported Models (28 total, prices verified 2026-03-26)

| Family | Models | Tier | Notes |
|--------|--------|------|-------|
| OpenAI GPT | o3, GPT-4.1, GPT-4o mini, GPT-OSS 120B | 1–3 | o3 replaces GPT-5 |
| OpenAI GPT | o4-mini | 1 | Compact reasoning |
| OpenAI Codex | gpt-5-codex, codex-mini-latest | 1, 3 | Specialised code models |
| Anthropic Claude | Opus 4.6, Sonnet 4.6, Haiku 4.5 | 1–3 | 1M context on Opus/Sonnet |
| Google Gemini | 3.1 Pro, 3.1 Flash, Nano | 1, 3, 4 | Nano is local/free |
| Mistral | Large 3, Medium, Small, Codestral, Devstral | 2–3 | Codestral/Devstral for code |
| DeepSeek | R1 (tier 1), V3 (tier 2) | 1–2 | R1: cheapest tier-1 reasoning |
| xAI | Grok 3 | 1 | |
| Kimi | K2.5 | 2 | 1M context |
| Local (Ollama) | Llama 4 Maverick, Llama 4 Scout, Mistral Small, Phi-4, Gemma 3 | 4 | Free, on-prem |

## Pricing

| Tier | Price | Requests/month | Key Features |
|------|-------|---------------|-------------|
| Community | Free | 10K | Model routing only, no budget enforcement |
| Pro | $99/month | 100K | Budget enforcement, dashboard, guardrails |
| Business | $499/month | 1M | MCP integration, caching, audit logs, Docker |
| Enterprise | Custom | Unlimited | On-prem/VPC, SSO/OIDC, RBAC, SLA, data residency |

See [docs/pricing.md](docs/pricing.md) for full tier details.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Configuration Reference](docs/configuration.md)
- [API Reference](docs/api-reference.md)
- [Adapters & Vendors](docs/adapters.md)
- [Budgets & Guardrails](docs/budgets-and-guardrails.md)
- [Caching](docs/caching.md) *(Phase 4)*
- [MCP Integration](docs/mcp-integration.md)
- [Dashboard](docs/dashboard.md)
- [Deployment](docs/deployment.md)
- [Pricing](docs/pricing.md)
- [ROI Calculator](docs/roi-calculator.md)

## Build Status

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 — Foundation | Scaffolding, models, DB, config | ✅ Complete |
| Phase 2 — Core Logic | Selector, cost engine, budget enforcer, guardrails | ✅ Complete |
| Phase 3 — API Layer | REST endpoints, integration tests, pricing corrections | ✅ Complete |
| Phase 4 — Adapters + Cache | All 8 vendor adapters, `/complete`, semantic caching | In progress |
| Phase 5 — Dashboard | Cost SPA, cache hit-rate panel, agent autonomy metrics | Planned |
| Phase 6 — MCP + Docker | MCP server, A2A protocol, Docker Compose | Planned |
| Phase 7 — Release | Full docs, v0.1.0 tag, public launch | Planned |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

**Kenny Wong**
lapkei01@gmail.com
Creator of Tidus — the AI system that governs all AIs.
