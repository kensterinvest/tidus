# Tidus — Enterprise AI Router

> **The AI system that governs all AIs.**

Tidus is a vendor-agnostic, enterprise-grade, cost-aware AI model router and governance system. Every AI request in your organisation passes through Tidus before reaching any model or AI service. Tidus selects the **cheapest model capable of doing the job**, enforces team and workflow budgets, prevents runaway multi-agent loops, and gives you full visibility into your AI spend.

```
┌───────────────────────────────────────────────────────┐
│           Enterprise AI Router (Tidus)                 │
├───────────────────────────────────────────────────────┤
│  Model Selection Engine  │  Cost Optimizer             │
│  Vendor Abstraction      │  Token Governance           │
│  Policy Enforcement      │  Multi-Agent Guardrails     │
├───────────────────────────────────────────────────────┤
│  OpenAI │ Anthropic │ Google │ Mistral │ DeepSeek      │
│  xAI    │ Kimi      │ Ollama (local/on-prem)           │
└───────────────────────────────────────────────────────┘
```

## Why Tidus?

Most AI systems default every request to the most powerful (most expensive) model — even for simple tasks like chat replies, classification, or summarisation. In enterprise multi-agent workflows, this multiplies quickly.

**Without Tidus:** 500 users × 200 requests/day × GPT-4o pricing = ~$15,000/month

**With Tidus:** Routing 70% of requests to Claude Haiku / Gemini Flash, 20% to mid-tier models, 10% to premium = ~$2,100/month + $99 Tidus fee = **$2,199/month — an 85% reduction**.

## Features

| Category | Feature |
|----------|---------|
| **Routing** | Automatic model selection based on task complexity, domain, and privacy |
| **Cost** | Token accounting with safety buffer, per-call cost logging, weekly price sync |
| **Budget** | Per-team and per-workflow limits with hard-stop or warn-only behaviour |
| **Guardrails** | Max agent depth, max tokens per step, max retries, session tracking |
| **Vendors** | OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, xAI, Kimi, Ollama (local) |
| **Integration** | MCP server (Claude Desktop / Cursor compatible), REST API, Python library |
| **Visibility** | Dashboard: cost by model, budget utilisation, active sessions |
| **Resilience** | Automatic fallback chains, health probes, model deprecation handling |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/kensterinvest/tidus.git
cd tidus
uv sync

# 2. Configure
cp .env.example .env
# Edit .env — add your API keys

# 3. Run
uvicorn tidus.main:app --reload

# 4. Try it
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

See [docs/quickstart.md](docs/quickstart.md) for the full guide.

## ROI Calculator

| Scenario | Monthly AI Cost | Tidus Fee | Net Cost | Saving |
|----------|----------------|-----------|---------|--------|
| Without Tidus (all GPT-4o) | $15,000 | — | $15,000 | — |
| Tidus Pro — tiered routing | $2,100 | $99 | $2,199 | **85%** |
| Tidus Business — + local models | $1,400 | $499 | $1,899 | **87%** |

*Based on 500 users × 200 requests/day × ~1,000 tokens average.*

See [docs/roi-calculator.md](docs/roi-calculator.md) for your custom calculation.

## Pricing

| Tier | Price | Requests/month |
|------|-------|---------------|
| Community | Free | 10K |
| Pro | $99/month | 100K |
| Business | $499/month | 1M |
| Enterprise | Custom | Unlimited |

See [docs/pricing.md](docs/pricing.md) for full tier details.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Configuration Reference](docs/configuration.md)
- [API Reference](docs/api-reference.md)
- [Adapters & Vendors](docs/adapters.md)
- [MCP Integration](docs/mcp-integration.md)
- [Budgets & Guardrails](docs/budgets-and-guardrails.md)
- [Dashboard](docs/dashboard.md)
- [Deployment](docs/deployment.md)
- [Pricing](docs/pricing.md)
- [ROI Calculator](docs/roi-calculator.md)

## Supported Models

| Family | Models |
|--------|--------|
| OpenAI GPT | GPT-5, GPT-5.2, GPT-4.1, GPT-4o mini, GPT-OSS 120B |
| Anthropic Claude | Opus 4.6, Sonnet 4.6, Haiku 4.5 |
| Google Gemini | 3.1 Pro, 3.1 Flash, Nano |
| Mistral | Large 3, Medium, Small, Codestral, Devstral |
| DeepSeek | R1, V3 |
| xAI | Grok 3 |
| Kimi | K2.5 |
| Local (Ollama) | Llama 4 Maverick/Scout, Mistral Small, Phi-4, Gemma 3, Falcon |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

**Kenny Wong**
lapkei01@gmail.com
Creator of Tidus — the AI system that governs all AIs.
