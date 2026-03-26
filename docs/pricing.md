# Pricing

Tidus is priced to pay for itself. At any meaningful request volume, the AI cost savings Tidus delivers exceed the subscription fee within the first day of use.

---

## Subscription Tiers

| Tier | Price | Requests/month | Key Features |
|------|-------|---------------|-------------|
| **Community** | Free | 10,000 | Model routing (Pillars 1+2), no budget enforcement, single team |
| **Pro** | $99/month | 100,000 | Budget enforcement, guardrails, dashboard, multi-team |
| **Business** | $499/month | 1,000,000 | MCP integration, response caching (Pillar 3), audit logs, Docker |
| **Enterprise** | Custom | Unlimited | On-prem/VPC, SSO/OIDC, RBAC, SLA, dedicated support, data residency |

---

## What Each Tier Unlocks

### Community (Free)
- `POST /api/v1/route` — full 5-stage model selection
- 28-model registry with real pricing
- REST API with Swagger UI
- Single team, no budget enforcement
- Best for: developers evaluating Tidus, single-project use

### Pro ($99/month)
Everything in Community, plus:
- **Budget enforcement** — hard-stop or warn-only per-team limits
- **Agent guardrails** — max depth, tokens per step, retries
- **Usage dashboard** — cost by model, budget utilisation
- **Multi-team support** — separate budgets per team
- **Usage API** — `GET /api/v1/usage/summary`
- Best for: teams routing AI requests in production

### Business ($499/month)
Everything in Pro, plus:
- **Response caching** *(Phase 4)* — exact + semantic cache, 30–50% additional savings
- **MCP server** *(Phase 6)* — plug Tidus into Claude Desktop, Cursor, and any MCP client
- **Audit logs** — full routing decision history with rejection reasons
- **Docker Compose** — one-command production deployment
- **A2A protocol** *(Phase 6)* — agent interoperability across vendor boundaries
- Best for: enterprises replacing direct vendor calls with a governed proxy

### Enterprise (Custom, from $2,000/month)
Everything in Business, plus:
- **On-prem / VPC deployment** — your infrastructure, your data
- **SSO/OIDC** — integrate with your identity provider
- **RBAC** — per-role model and budget access controls
- **SLA** — guaranteed uptime and response time
- **Dedicated support** — named support contact, priority response
- **Data residency** — choose where routing logs and cost data are stored
- **Custom adapter development** — proprietary or internal model integration

---

## Add-Ons

| Add-On | Price |
|--------|-------|
| Overage requests | $0.05 per 1,000 requests above plan limit |
| Additional vendor adapter (managed hosting) | $50/adapter/month |
| Professional onboarding & integration | $5,000 one-time |

---

## ROI at Each Tier

Based on 500 users × 200 requests/day × realistic task mix (2026 pricing):

| Tier | Subscription | Monthly AI Saving | Net Monthly Benefit | Payback Period |
|------|-------------|------------------|--------------------|-|
| Community | $0 | ~$48,000 | $48,000 | Immediate |
| Pro | $99 | ~$50,000 | $49,901 | < 1 day |
| Business | $499 | ~$51,000 | $50,501 | < 1 day |
| Enterprise | $2,000 | ~$51,000 | $49,000 | < 1 day |

*AI saving calculated vs. always-Claude-Opus baseline. See [roi-calculator.md](roi-calculator.md) for methodology.*

---

## Charging Unlock by Phase

| Phase | Tier Unlocked | What Becomes Available |
|-------|--------------|----------------------|
| Phase 2 — Core Logic | Community | Python library: `selector.select_model(task)` |
| Phase 3 — API Layer | Community fully live | `POST /api/v1/route` as a service |
| Phase 4 — Adapters + Cache | **Pro** | Full AI proxy via `/api/v1/complete`; Pillar 3 caching |
| Phase 5 — Dashboard | **Business** | Savings chart, budget bars — the sales demo |
| Phase 6 — MCP + Docker | Business fully live | MCP tools for any AI agent; A2A interoperability |
| Phase 7 — Release | **Enterprise** | Public launch, docs site, enterprise sales conversations |
