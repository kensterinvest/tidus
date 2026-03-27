# Dashboard

The Tidus dashboard is a single-page application (SPA) served at `/dashboard`. It provides real-time visibility into AI cost, budget utilization, active agent sessions, and model health — the management layer that turns cost data into a boardroom story.

## Accessing the Dashboard

After starting Tidus:

```
http://localhost:8000/dashboard/
# or shortcut:
http://localhost:8000/dash
```

No login required in v0.1 (authentication is a Phase 8 feature).

## Panels

### 1. Cost Overview (KPI Cards)

Six cards showing:

| Card | Description |
|---|---|
| **7-Day Cost** | Total USD spent across all vendors in the last 7 days |
| **Total Requests** | Request count in the last 7 days |
| **Requests Today** | Today's request count (UTC day) |
| **Avg Cost / Request** | Mean cost across all routed requests |
| **Most Used Model** | Model with highest request count (by router selection) |
| **Cheapest Model Used** | Model with lowest average cost per request |

### 2. Cost by Model (Bar Chart)

A 7-day bar chart showing USD spend per model, sorted highest to lowest. Colour-coded by tier:

- **Red** — Premium (Tier 1): o3, Claude Opus, Grok 3
- **Yellow** — Mid (Tier 2): DeepSeek V3, Kimi K2.5, GPT-4.1
- **Blue** — Economy (Tier 3): GPT-4o-mini, Gemini Flash, Mistral Small
- **Green** — Local/Free (Tier 4): Llama 4, Phi-4, Gemma (via Ollama)

The chart shows the routing algorithm working: most bars should be blue and green. Red bars indicate complex/critical tasks that genuinely required premium models.

### 3. Budget Utilization

Progress bars for each team with an active budget policy:

- **Green** — Under warning threshold (default 80%)
- **Yellow/WARN** — Above warning threshold, approaching limit
- **Red/STOPPED** — Hard-stop triggered, team requests are being rejected

### 4. Active Agent Sessions

A table of currently active multi-agent sessions showing:
- Session ID, team, current depth, step count, total tokens, start time
- Depth badges: Green (0-1), Yellow (2-3), Red (4+) — approaching the guardrail limit

### 5. Registry Health

Badges showing the enabled/disabled status of every model in the registry. Red badges indicate models auto-disabled by the health probe after 3 consecutive failures.

## Auto-Refresh

The dashboard refreshes every **30 seconds** automatically. All data comes from a single API call to `GET /api/v1/dashboard/summary`.

## Dashboard API

The dashboard data is also available as JSON for custom integrations:

```bash
curl http://localhost:8000/api/v1/dashboard/summary
```

Response fields: `cost`, `cost_by_model`, `budgets`, `sessions`, `registry_health`, `generated_at`.

## Using the Dashboard as a Sales Demo

The cost-by-model chart is the most powerful tool in the Tidus sales story. To demonstrate ROI:

1. Run Tidus for a few days with real traffic
2. Open the dashboard
3. The chart shows how tasks were distributed across tiers
4. Compare `7-Day Cost` against what the same requests would have cost at GPT-4o prices
5. The difference is the monthly saving — multiply by 4 for monthly ROI

See [roi-calculator.md](roi-calculator.md) for the full calculation methodology.
