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
│  OpenAI │ Anthropic │ Google  │ Mistral  │ DeepSeek            │
│  xAI    │ Kimi      │ Ollama (local/on-prem)                   │
│  Cohere │ Groq      │ Qwen    │ Perplexity │ Together AI        │
└───────────────────────────────────────────────────────────────┘
```

## Why Tidus?

Most AI systems default every request to the most powerful (most expensive) model — even for simple tasks like chat replies, classification, or summarisation. In enterprise multi-agent workflows, this cost multiplies with every agent hop, retry, and reflection loop.

**Tidus implements five cost-control pillars** that together eliminate 70–90% of enterprise AI spend:

| Pillar | What It Does | Status |
|--------|-------------|--------|
| **1. Tiered Model Strategy** | Routes to the cheapest capable tier — local models for filtering/routing, mid-tier for summarisation/extraction, premium only for reasoning/compliance | ✅ Live |
| **2. Router Agent Intelligence** | Decides model, local vs. cloud, multi-agent need, and budget feasibility before any compute runs | ✅ Live |
| **3. Response Caching** | Exact-match and semantic caching — identical or near-identical queries pay once; 30–50% additional reduction | ✅ Live (Phase 4) |
| **4. Agent Autonomy Limits** | Max depth, agents, tokens-per-step, retries, and reflection loops — prevents runaway compute | ✅ Live (depth, tokens, retries) |
| **5. Vendor-Agnostic Architecture** | Pluggable adapters + MCP server — swap vendors without rewriting your system | ✅ Live (8 adapters + MCP) |

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
| **Classification** | Local three-axis classifier — **domain** (task type), **complexity** (cognitive load), **privacy** (content sensitivity) — with 89.2% confidential recall on cross-family-validated ground truth. Self-improving to 95–97% over 12 months via disagreement-capture active learning. Full spec at [docs/classification.md](docs/classification.md) or the Technical Specification section at the bottom of the [landing page](https://kensterinvest.github.io/). |
| **Complexity** | Per-model `min_complexity`/`max_complexity` — prevents routing simple tasks to premium models |
| **Cost** | Token accounting with 15% safety buffer, per-call cost logging, weekly price sync |
| **Budget** | Per-team and per-workflow limits with hard-stop or warn-only behaviour |
| **Guardrails** | Max agent depth (5), max tokens per step (8K), max retries (3), session tracking |
| **Caching** | Exact + semantic response caching — 30–50% additional cost reduction |
| **Vendors** | OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, xAI, Kimi, Ollama (local) + Cohere, Groq, Qwen, Perplexity, Together AI (adapters in progress) |
| **Integration** | MCP server (`tidus-mcp`) for Claude Desktop / Cursor — 4 tools: route, complete, budget, models |
| **Visibility** | Dashboard at `/dashboard/`: cost by model, budget utilisation, active sessions, registry health |
| **Resilience** | Automatic fallback chains, health probes, model deprecation handling |

## Install

```bash
# pip (recommended)
pip install tidus

# Docker
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  tidus/tidus:latest

# From source (uv)
git clone https://github.com/kensterinvest/tidus.git && cd tidus && uv sync
```

## Quick Start

```bash
# 1. Install
pip install tidus

# 2. Configure — create .env with your API keys
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
EOF

# 3. Run
tidus

# or with uvicorn directly:
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

## Supported Models (55 total, 45 enabled across 13 vendors — prices verified 2026-04-17)

**8 production adapters live. 5 adapters in progress (models registered, adapters coming).**

| Family | Models | Tier | Adapter |
|--------|--------|------|---------|
| OpenAI GPT | o3, o4-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-oss-120b | 1–3 | ✅ Live |
| OpenAI Codex | gpt-5-codex, codex-mini-latest | 1, 3 | ✅ Live |
| Anthropic Claude | Opus 4.6, Sonnet 4.6, Haiku 4.5 | 1–3 | ✅ Live |
| Google Gemini | 3.1 Pro, 3.1 Flash, 2.5 Pro, 2.5 Flash, 2.0 Flash | 1–3 | ✅ Live |
| Mistral | Large 3, Medium, Small, Nemo, Codestral, Devstral, Devstral Small | 2–3 | ✅ Live |
| DeepSeek | R1, V3, V4 | 1–2 | ✅ Live |
| xAI | Grok 3, Grok 3 Fast | 1 | ✅ Live |
| Kimi | K2.5 | 2 | ✅ Live |
| Local (Ollama) | Llama 4 Maverick, Llama 4 Scout, Mistral Small, Phi-4, Gemma 3, DeepSeek Coder V2, Qwen 2.5, Qwen 2.5 Coder, Falcon 11B | 4 | ✅ Live |
| Cohere | Command R, Command R+ | 2–3 | 🔧 In progress |
| Groq | Llama 4 Maverick, DeepSeek R1 | 3–1 | 🔧 In progress |
| Qwen / Alibaba | Qwen Max, Qwen Plus, Qwen Flash | 1–3 | 🔧 In progress |
| Perplexity | Sonar Pro, Sonar | 1–3 | 🔧 In progress |
| Together AI | Llama 4 Maverick | 3 | 🔧 In progress |

## Pricing

Tidus is priced to pay for itself — the AI cost savings Tidus delivers typically exceed the subscription fee within the first day of use.

| Tier | Price | Requests/month | Key Features |
|------|-------|---------------|-------------|
| **Community** | Free | 10,000 | 5-stage routing, single team, no budget enforcement |
| **Pro** | $99/month | 100,000 | Budget enforcement + guardrails, dashboard, multi-team |
| **Business** | $499/month | 1,000,000 | MCP server, response caching (exact + semantic), audit logs, Docker |
| **Enterprise** | from $2,000/month | Unlimited | SSO/OIDC + RBAC (live), on-prem / data residency (roadmap), SLA, dedicated support |

See [docs/pricing.md](docs/pricing.md) for full tier details and ROI calculations, or the [How to Use guide](docs/how-to-use.md#15-enterprise-tier) for enterprise onboarding steps.

## Get Started

**New to Tidus?** → [How to Use Tidus — Step-by-Step Guide for New Users](docs/how-to-use.md)

Free for individuals and small organisations (< 1,000 AI users). No credit card needed.

## Documentation

- [**How to Use Tidus (New User Guide)**](docs/how-to-use.md) ← Start here
- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Configuration Reference](docs/configuration.md)
- [API Reference](docs/api-reference.md)
- [Adapters & Vendors](docs/adapters.md)
- [Budgets & Guardrails](docs/budgets-and-guardrails.md)
- [Caching](docs/caching.md)
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
| Phase 4 — Adapters + Cache | All 8 vendor adapters, `/complete`, semantic caching, sync scheduler | ✅ Complete |
| Phase 5 — Dashboard | Cost SPA, budget bars, active sessions, registry health | ✅ Complete |
| Phase 6 — MCP + Docker | MCP server (`tidus-mcp`), Docker Compose, Ollama profile | ✅ Complete |
| Phase 7 — Release | Full docs, v0.1.0 tag, 115 tests passing | ✅ Complete |
| Phase 8 — SSO/OIDC + RBAC | Enterprise auth, role-based access control | ✅ Complete |
| Phase 9 — Audit Logs | Structured audit trail, queryable log records | ✅ Complete |
| Phase 10 — PostgreSQL/Redis | Production database + cache config | ✅ Complete |
| Phase 11 — Kubernetes + Helm | K8s manifests, Helm chart, Prometheus/Grafana, HPA | ✅ Complete |
| v1.0.0-community — Public Release | PyPI publish, Docker Hub images, community open-source launch | ✅ Complete |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

**Kenny Wong**
lapkei01@gmail.com
Creator of Tidus — the AI system that governs all AIs.
