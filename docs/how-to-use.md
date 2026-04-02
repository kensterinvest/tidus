# How to Use Tidus — Step-by-Step Guide for Home Users

> Get Tidus running at home for **free** in under 10 minutes. No credit card needed.

Tidus is an AI router — it sits between your apps/scripts and AI models, automatically picking the **cheapest capable model** for every request. This guide gets you from zero to routing real AI requests using either free local models or a single low-cost API key.

---

## Table of Contents

1. [What You'll Need](#1-what-youll-need)
2. [Install Tidus](#2-install-tidus)
3. [Track A — Fully Free (Ollama + Local Models)](#3-track-a--fully-free-ollama--local-models)
4. [Track B — Cloud API Keys (Cheapest Option)](#4-track-b--cloud-api-keys-cheapest-option)
5. [Start the Server](#5-start-the-server)
6. [Your First Request](#6-your-first-request)
7. [Route vs Complete — What's the Difference?](#7-route-vs-complete--whats-the-difference)
8. [Working Examples](#8-working-examples)
9. [The Dashboard](#9-the-dashboard)
10. [Connect to Claude Desktop via MCP](#10-connect-to-claude-desktop-via-mcp)
11. [Set a Budget (Avoid Surprise Bills)](#11-set-a-budget-avoid-surprise-bills)
12. [Understanding the Routing Logic](#12-understanding-the-routing-logic)
13. [Troubleshooting](#13-troubleshooting)
14. [What's Next](#14-whats-next)

---

## 1. What You'll Need

| Requirement | Notes |
|---|---|
| **Python 3.12+** | Check with `python --version` |
| **[uv](https://docs.astral.sh/uv/)** | Fast Python package manager (free) |
| **Git** | To clone the repo |
| **API key OR Ollama** | Pick Track A (free) or Track B (cheap) below |

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. Install Tidus

```bash
git clone https://github.com/lapkei01/tidus.git
cd tidus
uv sync
```

This installs all dependencies into an isolated virtual environment. No system-wide changes.

Verify it worked:

```bash
uv run python -c "import tidus; print('Tidus ready!')"
```

---

## 3. Track A — Fully Free (Ollama + Local Models)

**Best for:** Home use, privacy-sensitive work, no API keys, offline operation.

### Step 3.1 — Install Ollama

Download from [ollama.ai](https://ollama.ai) and install for your platform.

Start the Ollama service:

```bash
ollama serve
```

### Step 3.2 — Pull a Free Model

Tidus supports these local models out of the box:

| Model | Size | Best For |
|---|---|---|
| `llama4-scout` | ~6 GB | General chat, fast |
| `llama4-maverick` | ~18 GB | Better quality |
| `phi4` | ~9 GB | Code, reasoning |
| `gemma3` | ~5 GB | Lightweight general |
| `mistral-small` | ~12 GB | Instruction following |

Pull a model (start with Llama 4 Scout for speed):

```bash
ollama pull llama4:scout
```

Or pull the smaller, faster option:

```bash
ollama pull phi4
```

### Step 3.3 — Configure Tidus for Local-Only

```bash
cp .env.example .env
```

Edit `.env` to add just one line:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

No API keys needed. Tidus will route all requests to your local Ollama instance for free.

---

## 4. Track B — Cloud API Keys (Cheapest Option)

**Best for:** Better quality responses, no GPU required, still very cheap.

The cheapest cloud option is **DeepSeek** — roughly $0.00028 per 1,000 tokens (input), about 18× cheaper than GPT-4o.

```bash
cp .env.example .env
```

Edit `.env` and add your key(s):

```env
# Pick at least one — DeepSeek is cheapest
DEEPSEEK_API_KEY=sk-...

# Or Anthropic (Claude Haiku 4.5 is very cheap)
ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI
OPENAI_API_KEY=sk-...
```

Get free API credits:
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com) — often has free trial credits
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com) — $5 free credit on signup
- **OpenAI**: [platform.openai.com](https://platform.openai.com) — $5 free credit on signup

You can combine both tracks — Tidus will use local models when possible and fall back to cloud only when the task complexity requires it.

---

## 5. Start the Server

```bash
uvicorn tidus.main:app --reload
```

You should see:

```
INFO  tidus_starting environment=development tier=community
INFO  database_ready
INFO  Application startup complete.
```

Tidus is now running at `http://localhost:8000`.

Verify it's alive:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## 6. Your First Request

Open a new terminal tab and send your first routing request:

```bash
curl -X POST http://localhost:8000/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "home",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 50,
    "messages": [{"role": "user", "content": "Summarise this sentence in three words."}]
  }'
```

Example response:

```json
{
  "task_id": "a1b2c3d4-...",
  "accepted": true,
  "chosen_model_id": "llama4-scout",
  "estimated_cost_usd": 0.0,
  "score": 0.02
}
```

Tidus chose `llama4-scout` (local, free) because the task is simple. For a `critical` complexity reasoning task, it would select a tier-1 model like DeepSeek R1 or Claude Opus instead.

---

## 7. Route vs Complete — What's the Difference?

| Endpoint | What it does |
|---|---|
| `POST /api/v1/route` | Picks the best model and returns the recommendation — **does not execute** |
| `POST /api/v1/complete` | Picks the best model **and executes the request** — returns the AI response |

Use `/route` when you want to inspect what Tidus would pick before committing.

Use `/complete` for end-to-end AI calls in your app.

---

## 8. Working Examples

### Example 1 — Simple Chat (Free, Local)

```bash
curl -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "home",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 100,
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Expected: Tidus routes to a local model (free). Response includes `content`, `cost_usd: 0.0`, and `latency_ms`.

---

### Example 2 — Code Generation (Tier 2–3)

```bash
curl -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "home",
    "complexity": "moderate",
    "domain": "code",
    "estimated_input_tokens": 300,
    "messages": [
      {"role": "user", "content": "Write a Python function that checks if a string is a palindrome."}
    ]
  }'
```

Expected: Tidus selects a mid-tier model like DeepSeek V3 or Mistral Medium — capable for code, but not burning tier-1 budget.

---

### Example 3 — Critical Reasoning (Tier 1 Only)

```bash
curl -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "home",
    "complexity": "critical",
    "domain": "reasoning",
    "estimated_input_tokens": 1500,
    "messages": [
      {"role": "user", "content": "What are the main trade-offs between microservices and monolithic architectures for a 3-person startup team?"}
    ]
  }'
```

Expected: Tidus uses a tier-1 model (DeepSeek R1 is cheapest at $0.00055/1K input — 9× cheaper than Claude Opus). The routing algorithm only escalates to tier 1 when `complexity: "critical"`.

---

### Example 4 — Confidential Data (Local Only, No Cloud)

```bash
curl -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "home",
    "complexity": "simple",
    "domain": "chat",
    "privacy": "confidential",
    "estimated_input_tokens": 200,
    "messages": [
      {"role": "user", "content": "Summarise this internal document: [your private text here]"}
    ]
  }'
```

Expected: Tidus **forces** a local Ollama model regardless of quality or complexity. No data leaves your machine. This is how you handle sensitive or proprietary content.

---

### Example 5 — Multi-Turn Conversation

```bash
curl -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "home",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 200,
    "messages": [
      {"role": "user", "content": "My name is Alex."},
      {"role": "assistant", "content": "Nice to meet you, Alex!"},
      {"role": "user", "content": "What's my name?"}
    ]
  }'
```

Tidus passes the full conversation to the selected model. The router picks based on the **total complexity** of the task, not per-message.

---

### Example 6 — Check Available Models

```bash
curl http://localhost:8000/api/v1/models
```

Returns all registered models with their tier, pricing, and capability tags. Use this to understand what Tidus has available on your setup.

---

### Example 7 — Check Your Budget Usage

```bash
curl "http://localhost:8000/api/v1/budget?team_id=home"
```

Returns current spend, budget limit, and utilisation percentage for your team.

---

## 9. The Dashboard

Open your browser and go to:

```
http://localhost:8000/dashboard/
```

The dashboard shows:

- **AI spend by model** — see exactly where your money is going
- **Budget utilisation** — per-team spend bars vs. limits
- **Active sessions** — ongoing agent loops and their depth
- **Model registry health** — which adapters are connected and responsive

No login needed in local development mode.

---

## 10. Connect to Claude Desktop via MCP

If you use Claude Desktop, you can expose Tidus as native tools — your Claude sessions will automatically route through Tidus for cheaper completions.

### Step 10.1 — Start the MCP Server

```bash
# In a new terminal (keep uvicorn running in the other tab)
tidus-mcp
```

### Step 10.2 — Configure Claude Desktop

Find your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the Tidus MCP server:

```json
{
  "mcpServers": {
    "tidus": {
      "command": "tidus-mcp",
      "args": [],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

Restart Claude Desktop. You'll now have four Tidus tools available in every Claude conversation:

| Tool | What it does |
|---|---|
| `tidus_route` | Ask Tidus which model to use for a task |
| `tidus_complete` | Route + execute through Tidus |
| `tidus_budget` | Check team budget status |
| `tidus_models` | List available models and tiers |

---

## 11. Set a Budget (Avoid Surprise Bills)

By default, there is no spend limit. Add a team budget to cap API costs:

```bash
curl -X POST http://localhost:8000/api/v1/teams \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "home",
    "monthly_budget_usd": 5.00,
    "budget_behaviour": "hard_stop"
  }'
```

`hard_stop` = reject requests when the budget is hit. Use `warn` to allow overage with a logged warning instead.

Check budget at any time:

```bash
curl "http://localhost:8000/api/v1/budget?team_id=home"
```

---

## 12. Understanding the Routing Logic

Tidus runs five stages for every request. Understanding this helps you write better requests:

```
Stage 1 — Hard Constraints
  Is the model enabled? Does the context window fit?
  Is this a confidential task? → local-only
  Does the task complexity match the model's range?

Stage 2 — Guardrails
  Agent depth < 5 (prevents infinite loops)
  Input tokens < 8,000 per step

Stage 3 — Complexity Tier Ceiling
  simple   → any tier (cheapest wins)
  moderate → tier 3 or lower
  complex  → tier 2 or lower
  critical → tier 1 only

Stage 4 — Budget Filter
  Reject if estimated cost exceeds team/workflow budget

Stage 5 — Score and Select
  score = cost×0.70 + tier_penalty×0.20 + latency×0.10
  Lowest score wins
```

**Key insight:** Set `complexity: "simple"` for chat/summarisation tasks and you'll almost always get a free local model or the cheapest cloud tier. Escalate to `"critical"` only when you genuinely need the best reasoning.

### Complexity Reference

| Task type | Recommended complexity |
|---|---|
| Simple chat, FAQ answers | `simple` |
| Summarisation, extraction, classification | `simple` or `moderate` |
| Code generation, data analysis | `moderate` |
| Multi-step reasoning, architecture decisions | `complex` |
| Legal/compliance review, financial analysis | `critical` |

---

## 13. Troubleshooting

### "No models available" or empty response

- Check Ollama is running: `ollama list`
- Check `.env` has at least one key set
- Restart the server: `uvicorn tidus.main:app --reload`

### 500 error on `/complete`

- The selected model's vendor API is unreachable or the API key is invalid
- Check the server logs — they show which adapter failed
- Try `/route` first to confirm model selection, then test the vendor separately

### Ollama model not being selected

- Verify Ollama is running at the URL in your `.env` (`OLLAMA_BASE_URL`)
- Pull the model first: `ollama pull llama4:scout`
- Check `GET /api/v1/models` — local models show `adapter: "ollama"`

### MCP tools not appearing in Claude Desktop

- Ensure `tidus-mcp` is on your PATH: `which tidus-mcp`
- Check Claude Desktop was fully restarted after config change
- Verify the config file path is correct for your OS

### Budget errors

- Check current spend: `curl "http://localhost:8000/api/v1/budget?team_id=home"`
- Use a higher budget or switch `budget_behaviour` to `warn`

---

## 14. What's Next

Now that Tidus is running at home, explore these features:

| Feature | Doc |
|---|---|
| Full API reference (all endpoints) | [api-reference.md](api-reference.md) |
| Configure custom model registry | [configuration.md](configuration.md) |
| Caching to reduce repeat costs | [caching.md](caching.md) |
| Budget and guardrail options | [budgets-and-guardrails.md](budgets-and-guardrails.md) |
| All 8 vendor adapters | [adapters.md](adapters.md) |
| Deployment with Docker | [deployment.md](deployment.md) |
| MCP deep dive | [mcp-integration.md](mcp-integration.md) |
| Architecture overview | [architecture.md](architecture.md) |

### Estimated Monthly Cost

| Usage level | Track A (Ollama free) | Track B (DeepSeek cloud) |
|---|---|---|
| Light (100 req/day) | $0 | ~$0.03 |
| Medium (500 req/day) | $0 | ~$0.15 |
| Heavy (2,000 req/day) | $0 | ~$0.60 |

*Track A requires a GPU with ≥8 GB VRAM for best performance. Track B works on any machine.*

---

**Questions?** Open an issue at [github.com/lapkei01/tidus](https://github.com/lapkei01/tidus) or check the [full documentation index](index.md).
