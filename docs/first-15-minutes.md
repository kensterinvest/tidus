# First 15 Minutes with Tidus

This guide gets you from zero to seeing real AI cost savings in 15 minutes.
No manual YAML editing required.

---

## What you'll need

- Python 3.12+ **or** Docker
- At least one API key (OpenAI, Anthropic, Google, etc.) **or** [Ollama](https://ollama.ai) for free local models

---

## Minutes 0–2: Install

**Option A — Docker (recommended, zero Python setup):**
```bash
git clone https://github.com/kensterinvest/tidus.git
cd tidus
cp .env.example .env
# Add at least one API key to .env (see next step)
docker compose up -d
```

**Option B — Local Python:**
```bash
git clone https://github.com/kensterinvest/tidus.git
cd tidus
uv sync
cp .env.example .env
```

---

## Minutes 2–5: Add an API key and run setup

Open `.env` and uncomment at least one key:
```env
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
# or for free inference with no API key:
OLLAMA_BASE_URL=http://localhost:11434
```

Then run the setup wizard — it detects your keys, creates a team budget, and tests connectivity:
```bash
uv run tidus-setup
# Or non-interactively (Docker, CI):
uv run tidus-setup --defaults
```

You'll see output like:
```
[Step 1/4] Detecting API keys…
  ✓ ANTHROPIC_API_KEY found — Claude (Opus / Sonnet / Haiku)
  ✓ OPENAI_API_KEY found — GPT-4o / GPT-4o-mini
  ○ OLLAMA — not running (skip)

[Step 2/4] Setting up your first team budget…
  Team name [team-default]:
  Monthly spend limit in USD [500]:
  ✓ Written to config/budgets.yaml

[Step 3/4] Testing vendor connectivity…
  ✓ Claude (Opus / Sonnet / Haiku): healthy (312 ms)
  ✓ GPT-4o / GPT-4o-mini: healthy (287 ms)

[Step 4/4] Ready!
```

Start the server:
```bash
uvicorn tidus.main:app --reload
```

---

## Minutes 5–8: Send your first requests

Tidus routes each request to the cheapest model capable of handling it.
Run a mix to see smart routing in action:

```bash
# Simple chat → routes to cheapest tier-2 model (e.g. deepseek-v3 at $0.000014/1K)
curl -s -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-default",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 500,
    "messages": [{"role": "user", "content": "Summarise this in one sentence: Tidus routes AI requests to the cheapest capable model."}]
  }' | python -m json.tool

# Critical reasoning → routes to tier-1 (e.g. deepseek-r1 or claude-opus)
curl -s -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-default",
    "complexity": "critical",
    "domain": "reasoning",
    "estimated_input_tokens": 2000,
    "messages": [{"role": "user", "content": "What are the second-order effects of reducing interest rates during deflation?"}]
  }' | python -m json.tool

# Confidential → routes to local Ollama (no data leaves your server)
curl -s -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-default",
    "complexity": "simple",
    "domain": "chat",
    "privacy": "confidential",
    "estimated_input_tokens": 200,
    "messages": [{"role": "user", "content": "Summarise this confidential document: [your text]"}]
  }' | python -m json.tool
```

Each response includes `cost_usd` and `chosen_model_id` so you can see exactly what was selected and why.

---

## Minutes 8–12: Open the dashboard and see your savings

Navigate to **http://localhost:8000/dashboard/**

You'll see the **Saved vs Baseline** KPI light up in green — this is the estimated saving
compared to routing every request to Claude Opus 4.6.

For a realistic enterprise workload of 500 users × 200 requests/day, Tidus typically
saves **$48,000–$51,000/month** (92–98% reduction).

Use the `[7d] [30d] [90d]` toggle to change the time window.

---

## Minutes 12–15: Monthly savings report and budget

Generate your first monthly report:
```bash
curl "http://localhost:8000/api/v1/reports/monthly" | python -m json.tool
```

This shows:
- Total actual spend
- Estimated savings vs premium baseline
- Per-day trend
- Top models by traffic

Check the budget panel:
```bash
curl http://localhost:8000/api/v1/budgets/status/team/team-default | python -m json.tool
```

---

## What's next?

| Goal | Guide |
|---|---|
| Add more teams or adjust budget limits | [budgets-and-guardrails.md](budgets-and-guardrails.md) |
| Connect Claude Desktop or Cursor via MCP | [mcp-integration.md](mcp-integration.md) |
| Set up SSO/OIDC authentication | [enterprise/sso.md](enterprise/sso.md) |
| Deploy to Kubernetes | [deployment.md](deployment.md) |
| Understand the 5-stage routing algorithm | [architecture.md](architecture.md) |
| Estimate savings for your workload | [roi-calculator.md](roi-calculator.md) |
| Common setup problems | [troubleshooting.md](troubleshooting.md) |
