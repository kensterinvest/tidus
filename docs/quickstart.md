# Quickstart

Get Tidus routing AI requests in under 5 minutes.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- At least one API key (or [Ollama](https://ollama.ai) running locally for free testing)

## 1. Install

```bash
git clone https://github.com/kensterinvest/tidus.git
cd tidus
uv sync
```

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — add at least one vendor key:

```env
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
# or for free local testing (no API key needed):
OLLAMA_BASE_URL=http://localhost:11434
```

## 3. Start the server

Run the automated setup wizard — it detects your keys, creates a team budget, and tests vendor connectivity:

```bash
uv run tidus-setup
# Non-interactive (Docker / CI):
uv run tidus-setup --defaults
```

Then start the server:
```bash
uvicorn tidus.main:app --reload
```

You should see:
```
INFO  tidus_starting environment=development tier=community
INFO  database_ready
INFO  Application startup complete.
```

## 4. Your first routing request

```bash
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

Tidus returns the selected model, estimated cost, and a normalised score (lower = better):

```json
{
  "task_id": "a1b2c3d4-...",
  "accepted": true,
  "chosen_model_id": "deepseek-v3",
  "estimated_cost_usd": 0.000056,
  "score": 0.096
}
```

`deepseek-v3` wins simple/chat because it is the cheapest tier-2 model with `chat` capability in the `simple–complex` complexity range. The `score` is computed as `cost×0.70 + tier×0.20 + latency×0.10`, normalised to [0, 1].

## 5. Try with different complexity levels

Tidus routes differently based on task complexity:

```bash
# critical reasoning → tier-1 model only (e.g. deepseek-r1 or claude-opus)
curl -X POST http://localhost:8000/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-engineering",
    "complexity": "critical",
    "domain": "reasoning",
    "estimated_input_tokens": 2000,
    "messages": [{"role": "user", "content": "Analyse the regulatory implications of this contract."}]
  }'

# confidential task → local model only (no data leaves your infrastructure)
curl -X POST http://localhost:8000/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-engineering",
    "complexity": "simple",
    "domain": "chat",
    "privacy": "confidential",
    "estimated_input_tokens": 200,
    "messages": [{"role": "user", "content": "Summarise this internal document."}]
  }'
```

## 6. Execute the request

`/api/v1/complete` routes **and** executes in one call — Tidus selects the model, calls the vendor API, logs the cost, and returns the response:

```bash
curl -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-engineering",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 200,
    "messages": [{"role": "user", "content": "Hello! What can you help me with?"}]
  }'
```

Response:
```json
{
  "task_id": "b2c3d4e5-...",
  "chosen_model_id": "deepseek-v3",
  "content": "I can help you with a wide range of tasks...",
  "input_tokens": 18,
  "output_tokens": 42,
  "cost_usd": 0.0000084,
  "latency_ms": 612.3
}
```

## 7. Open the dashboard and see your savings

Navigate to **http://localhost:8000/dashboard/** to see:
- **Saved vs Baseline** — estimated savings vs routing everything to Claude Opus (typically 92–98%)
- AI spend by model — toggle between 7d / 30d / 90d views
- Budget utilisation per team
- Active agent sessions
- Model registry health

After your first few requests, the green **Saved vs Baseline** KPI shows your real-time savings.

## 8. Generate your first monthly savings report

```bash
curl "http://localhost:8000/api/v1/reports/monthly" | python -m json.tool
```

This shows total spend, estimated savings vs the premium baseline, per-day breakdown,
and top models by traffic. All data stays on your server — nothing is sent externally.

See [docs/savings-report.md](savings-report.md) for the full report reference.

## 8. Connect via MCP (Claude Desktop / Cursor)

Start the MCP server to expose Tidus as native tools for any AI agent:

```bash
tidus-mcp
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tidus": {
      "command": "tidus-mcp",
      "args": [],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

See [MCP Integration](mcp-integration.md) for full setup details.

## Next Steps

- [First 15 Minutes guide](first-15-minutes.md) — outcome-oriented walkthrough
- [Configure your model registry](configuration.md)
- [Set up team budgets](budgets-and-guardrails.md)
- [Monthly savings report](savings-report.md)
- [Understand the selection algorithm](architecture.md)
- [View all API endpoints](api-reference.md)
- [Connect via MCP](mcp-integration.md)
- [Deploy with Docker](deployment.md)
- [Troubleshooting](troubleshooting.md)
