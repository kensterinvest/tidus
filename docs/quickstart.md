# Quickstart

Get Tidus routing AI requests in under 5 minutes.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- At least one API key (or [Ollama](https://ollama.ai) running locally for free testing)

## 1. Install

```bash
git clone https://github.com/lapkei01/tidus.git
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

`deepseek-v3` wins simple/chat because it is the cheapest tier-2 model with `chat` capability and a complexity range that includes `simple`. The `score` is computed as `cost×0.70 + tier×0.20 + latency×0.10`, normalised to [0, 1].

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

# confidential task → local model only (privacy enforcement)
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

## 6. Execute the request (Phase 4)

`/api/v1/complete` routes **and** executes in one call — available once vendor adapters are built in Phase 4:

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
- [Understand the selection algorithm](architecture.md)
- [View all API endpoints](api-reference.md)
- [Connect via MCP](mcp-integration.md) *(Phase 6)*
