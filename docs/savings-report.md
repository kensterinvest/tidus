# Monthly Savings Report

Tidus generates a self-contained monthly savings report from your local database.
**No data is sent to any external service** — everything is computed on your own infrastructure.

---

## What the Report Shows

The report answers one question: *How much did Tidus save vs always routing to a premium AI model?*

| Field | Description |
|---|---|
| `total_cost_usd` | What Tidus actually spent (via smart routing) |
| `baseline_cost_usd` | What the same requests would have cost via the baseline model (default: Claude Opus 4.6) |
| `estimated_savings_usd` | `baseline_cost_usd − total_cost_usd` |
| `savings_pct` | `savings / baseline × 100` |
| `top_models` | Models that handled the most traffic (up to 10) |
| `daily_breakdown` | Per-day cost and savings for trend analysis |

---

## Generating a Report

```bash
# Current month, all teams
curl http://localhost:8000/api/v1/reports/monthly | python -m json.tool

# Specific month
curl "http://localhost:8000/api/v1/reports/monthly?year=2026&month=3"

# Specific team
curl "http://localhost:8000/api/v1/reports/monthly?team_id=team-engineering"

# With a different baseline (compare vs GPT-4o instead of Claude Opus)
curl "http://localhost:8000/api/v1/reports/monthly?baseline_model_id=gpt-4o"
```

### Example Response

```json
{
  "period": "2026-04",
  "team_id": "all",
  "total_requests": 84000,
  "total_cost_usd": 23.40,
  "baseline_cost_usd": 1449.00,
  "estimated_savings_usd": 1425.60,
  "savings_pct": 98.38,
  "avg_cost_per_request_usd": 0.000279,
  "baseline_model_id": "claude-opus-4-6",
  "top_models": [
    {
      "model_id": "deepseek-v3",
      "vendor": "deepseek",
      "requests": 50400,
      "cost_usd": 14.11,
      "pct_of_traffic": 60.0
    },
    {
      "model_id": "gpt-4o-mini",
      "vendor": "openai",
      "requests": 16800,
      "cost_usd": 3.36,
      "pct_of_traffic": 20.0
    }
  ],
  "daily_breakdown": [
    { "date": "2026-04-01", "requests": 2800, "cost_usd": 0.78, "savings_usd": 47.46 },
    { "date": "2026-04-02", "requests": 2800, "cost_usd": 0.78, "savings_usd": 47.46 }
  ],
  "generated_at": "2026-04-03T10:00:00+00:00",
  "note": "All data is computed from your local Tidus database. Nothing is sent to any external service."
}
```

---

## Query Parameters

| Parameter | Default | Description |
|---|---|---|
| `year` | Current year | Report year (2024–2099) |
| `month` | Current month | Report month (1–12) |
| `team_id` | `"all"` | Team to report on, or `"all"` for all teams |
| `baseline_model_id` | `claude-opus-4-6` | Model used for "what-if premium" comparison |

---

## Access Control

| Role | What they see |
|---|---|
| `admin`, `team_manager` | All teams |
| `developer`, `service_account`, `read_only` | Own team only |

---

## Exporting and Sharing

Since the report is a plain JSON endpoint, you can export it with standard tools:

```bash
# Save to file
curl "http://localhost:8000/api/v1/reports/monthly?year=2026&month=4" \
  > tidus-savings-april-2026.json

# Pretty-print to PDF (requires pandoc + wkhtmltopdf)
curl "http://localhost:8000/api/v1/reports/monthly" | python -m json.tool > report.json
pandoc report.json -o report.pdf

# Monthly totals only (using jq)
curl "http://localhost:8000/api/v1/reports/monthly" | jq '{
  period,
  total_requests,
  actual_cost: .total_cost_usd,
  savings: .estimated_savings_usd,
  savings_pct
}'
```

To share your savings with the Tidus project or community:
1. Run the report
2. Extract the top-level summary fields
3. Share via email or GitHub Discussions — no automation, full control

---

## Dashboard Integration

The dashboard's **Saved vs Baseline** KPI card (top-right of the overview grid) shows
a rolling calculation from the same data, updated every 30 seconds. Use the `7d / 30d / 90d`
toggle to change the window.

For the full monthly breakdown with per-day trends, use the API endpoint above.

---

## How the Savings Are Calculated

```
baseline_cost = (total_input_tokens / 1000 × baseline_input_price)
              + (total_output_tokens / 1000 × baseline_output_price)

actual_cost   = sum of cost_usd from all CostRecord rows in the period

savings_usd   = max(0, baseline_cost − actual_cost)
savings_pct   = savings_usd / baseline_cost × 100
```

Both token counts and prices come from your local registry — no external API calls.

**Why `claude-opus-4-6` as the default baseline?**
It is the most expensive widely-used model, so it produces the largest (most impressive)
savings number. For a more conservative estimate, use `gpt-4o` or `deepseek-r1` as
the baseline — these are premium but not the most expensive.
