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
# or for free local testing:
OLLAMA_BASE_URL=http://localhost:11434
```

## 3. Start the server

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

Tidus will return a `RoutingDecision` showing which model was selected and why:

```json
{
  "decision_id": "...",
  "task_id": "...",
  "selected_model_id": "claude-haiku-4-5",
  "selected_vendor": "anthropic",
  "explanation": "Selected claude-haiku-4-5 (tier 3): cheapest model capable of simple/chat. Estimated cost: $0.00016",
  "chosen_estimate": {
    "model_id": "claude-haiku-4-5",
    "estimated_cost_usd": 0.00016,
    "buffer_pct": 0.15
  }
}
```

## 5. Execute the request

Use `/api/v1/complete` to route **and** execute in one call:

```bash
curl -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-engineering",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 200,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Next Steps

- [Configure your model registry](configuration.md)
- [Set up team budgets](budgets-and-guardrails.md)
- [View the dashboard](dashboard.md)
- [Connect via MCP](mcp-integration.md)
