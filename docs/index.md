# Tidus Documentation

> The AI system that governs all AIs.

## What is Tidus?

Tidus is an enterprise AI governance and routing system. Every AI request in your organisation passes through Tidus, which selects the cheapest capable model, enforces budgets, and prevents runaway agent costs — implementing five cost-control pillars that together reduce AI spend by 70–96%.

## Five Cost-Control Pillars

| Pillar | Description | Status |
|--------|-------------|--------|
| 1. Tiered Model Strategy | Route tasks to the cheapest capable tier | ✅ Live |
| 2. Router Agent Intelligence | 5-stage selector decides model before compute runs | ✅ Live |
| 3. Cache Everything | Exact + semantic caching prevents paying twice | ✅ Live |
| 4. Agent Autonomy Limits | Depth, tokens, retries, reflection loops | ✅ Live |
| 5. Vendor-Agnostic Design | 8 production adapters + 5 in progress + MCP server — swap vendors without rewriting your system | ✅ Live |

## Table of Contents

| Document | Description | Status |
|----------|-------------|--------|
| [Quickstart](quickstart.md) | Up and running in 5 minutes | ✅ Live |
| [First 15 Minutes](first-15-minutes.md) | Outcome-oriented onboarding: see savings in minutes | ✅ Live |
| [Architecture](architecture.md) | System design, five pillars, data flow | ✅ Live |
| [Configuration](configuration.md) | models.yaml, budgets.yaml, policies.yaml reference | ✅ Live |
| [API Reference](api-reference.md) | All REST endpoints with request/response schemas | ✅ Live |
| [Adapters](adapters.md) | Supported vendors and adding new ones | ✅ Live |
| [Budgets & Guardrails](budgets-and-guardrails.md) | Spending limits and agent autonomy controls | ✅ Live |
| [Caching](caching.md) | Exact + semantic response caching — Pillar 3 | ✅ Live |
| [MCP Integration](mcp-integration.md) | Connect Claude Desktop, Cursor, and other agents | ✅ Live |
| [Dashboard](dashboard.md) | Cost visibility UI at `/dashboard/` | ✅ Live |
| [Deployment](deployment.md) | Docker, production setup, PostgreSQL migration | ✅ Live |
| [Pricing](pricing.md) | Tidus subscription tiers | Live |
| [ROI Calculator](roi-calculator.md) | Calculate your enterprise savings | Live |
| [Savings Report](savings-report.md) | Monthly AI savings report API — local, no external dependency | ✅ Live |
| [Troubleshooting](troubleshooting.md) | Top 10 first-run issues with exact fixes | ✅ Live |
| [Enterprise: RBAC](enterprise/rbac.md) | Role-based access control | Roadmap |
| [Enterprise: SSO](enterprise/sso.md) | SSO/OIDC integration | Roadmap |
| [Enterprise: Data Residency](enterprise/data-residency.md) | On-prem/VPC deployment | Roadmap |
