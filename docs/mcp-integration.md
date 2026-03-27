# MCP Integration

Tidus runs as an MCP (Model Context Protocol) server, exposing its routing and execution capabilities as tools that any MCP-compatible AI agent can call directly.

## What Is MCP?

MCP is an open protocol for connecting AI agents to tools and data sources. When Tidus is configured as an MCP server, AI agents (Claude Desktop, Cursor, custom agents) can call `tidus_route_task` and `tidus_complete_task` as natural tool calls — no HTTP client code required.

## Quick Start

### 1. Install Tidus

```bash
pip install tidus
# or from source:
uv sync
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set at least one vendor API key.

### 3. Run the MCP Server

```bash
tidus-mcp
```

The server starts on **stdio** (standard input/output) — the standard MCP transport for local tools.

## Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tidus": {
      "command": "tidus-mcp",
      "args": [],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-...",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

Restart Claude Desktop. The Tidus tools appear in the tool picker.

## Configure Cursor

In Cursor settings → MCP → Add server:

```json
{
  "name": "tidus",
  "command": "tidus-mcp"
}
```

## Available Tools

### `tidus_route_task`

Select the cheapest capable model without executing. Returns routing decision.

**Input:**
```json
{
  "team_id": "team-engineering",
  "complexity": "simple",
  "domain": "code",
  "estimated_input_tokens": 500,
  "messages": [{"role": "user", "content": "Fix the null check in utils.py"}],
  "privacy": "internal",
  "max_cost_usd": 0.01
}
```

**Output:**
```json
{
  "chosen_model_id": "deepseek-v3",
  "vendor": "deepseek",
  "estimated_cost_usd": 0.00008,
  "tier": 2
}
```

### `tidus_complete_task`

Route and execute. Calls the selected adapter and returns the LLM response.

**Input:** Same as `tidus_route_task` plus optional `estimated_output_tokens` and `agent_depth`.

**Output:**
```json
{
  "content": "Here's the fixed null check: ...",
  "model_id": "deepseek-v3",
  "vendor": "deepseek",
  "input_tokens": 487,
  "output_tokens": 312,
  "cost_usd": 0.000095,
  "latency_ms": 820.4
}
```

### `tidus_get_budget_status`

Check team spend vs. limit.

**Input:** `{"team_id": "team-engineering"}`

**Output:**
```json
{
  "team_id": "team-engineering",
  "spent_usd": 12.45,
  "limit_usd": 500.00,
  "utilisation_pct": 2.49,
  "is_hard_stopped": false
}
```

### `tidus_list_models`

List all models in the registry.

**Input:** `{"enabled_only": true}`

**Output:** Array of model objects with `model_id`, `vendor`, `tier`, `enabled`, `is_local`, `input_price`, `output_price`, `max_context`.

## Privacy and Confidential Data

Set `"privacy": "confidential"` to force routing to `is_local=True` models only. No data leaves your infrastructure.

```json
{
  "team_id": "legal",
  "complexity": "moderate",
  "domain": "summarization",
  "privacy": "confidential",
  "messages": [{"role": "user", "content": "Summarize this M&A document..."}]
}
```

Tidus selects Llama 4, Mistral Small, or another locally-hosted model automatically.

## Agent Loop Example

```python
# In a multi-step agent loop, pass agent_depth to enforce guardrails
for step in range(10):
    result = await tidus_complete_task({
        "team_id": "agent-team",
        "complexity": "moderate",
        "domain": "reasoning",
        "agent_depth": step,  # Tidus rejects at depth >= max_agent_depth (default: 5)
        "messages": messages,
    })
    if "error" in result:
        break  # Guardrail triggered — stop the loop
```
