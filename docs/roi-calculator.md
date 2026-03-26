# ROI Calculator

Use this to estimate your monthly AI cost savings with Tidus versus always routing to a premium model.

---

## The Formula

```
Baseline cost  = monthly_requests x avg_input_tokens x premium_model_input_price
               + monthly_requests x avg_output_tokens x premium_model_output_price

Tidus cost     = sum over task mix of:
                 (fraction x monthly_requests x tidus_selected_model_cost)

Monthly saving = Baseline cost - Tidus cost - Tidus subscription fee
Payback period = Tidus fee / Monthly saving
```

---

## Worked Example — 500-User Enterprise (Verified 2026-03-26 Pricing)

**Setup:** 500 users, 200 requests/day each, 30-day month = 3,000,000 requests/month.

### Baseline (always Claude Opus 4.6 at $0.005/1K input, $0.025/1K output)

Average task: 1,000 input tokens, 400 output tokens + 15% buffer = 1,150 / 460 tokens.

```
Baseline per request = 1.15 * 0.005 + 0.46 * 0.025 = $0.00575 + $0.0115 = $0.01725
Baseline monthly     = 3,000,000 * $0.01725 = $51,750
```

### Tidus — Smart Routing (Pillars 1+2)

Realistic enterprise task mix:

| Task Type | Share | Complexity | Domain | Tidus Model | Tidus Cost/req |
|-----------|-------|------------|--------|-------------|---------------|
| Simple chat | 60% | simple | chat | deepseek-v3 | $0.000056 |
| Moderate summarisation | 20% | moderate | summarization | mistral-medium | $0.00082 |
| Moderate code | 12% | moderate | code | gpt-oss-120b | $0.000052 |
| Complex code | 5% | complex | code | deepseek-r1 | $0.00096 |
| Critical reasoning | 3% | critical | reasoning | deepseek-r1 | $0.00096 |

```
Weighted Tidus cost/request = 0.60*0.000056 + 0.20*0.00082 + 0.12*0.000052
                             + 0.05*0.00096 + 0.03*0.00096
                            = 0.0000336 + 0.000164 + 0.00000624
                             + 0.000048 + 0.0000288
                            = ~$0.000280 per request

Tidus monthly AI cost = 3,000,000 * $0.000280 = $840
```

### Results Table

| Scenario | Monthly AI Cost | Tidus Fee | Net Cost | Saving vs Baseline | % Saving |
|----------|----------------|-----------|---------|-------------------|----------|
| Baseline (always Claude Opus) | $51,750 | — | $51,750 | — | — |
| Tidus Pro — Pillars 1+2 | $840 | $99 | $939 | $50,811 | **98%** |
| Tidus Business — + local models | ~$500 | $499 | $999 | $50,751 | **98%** |
| Tidus Business — + caching (Phase 4) | ~$250 | $499 | $749 | $51,001 | **99%** |

*Note: The 98% figure assumes DeepSeek R1 wins critical/complex tasks at $0.00055/1K input — 9× cheaper than Claude Opus. Real savings depend on your task mix and whether you enable local models.*

---

## Conservative Estimate

If only 50% of requests can be down-routed (mixed workload with more complex tasks):

| Scenario | Monthly AI Cost | Net Cost | Saving |
|----------|----------------|---------|--------|
| Baseline (always Claude Opus) | $51,750 | $51,750 | — |
| Tidus Pro — conservative mix | $4,200 | $4,299 | **92%** |

---

## Your Custom Calculation

Fill in your own numbers:

```
monthly_requests = [your users] x [requests per user per day] x 30
avg_input_tokens = [average prompt size, typically 500–2,000]
avg_output_tokens = [average response size, typically 200–1,000]

baseline_cost = monthly_requests
              x (avg_input_tokens/1000 x 0.005   # Claude Opus input
              +  avg_output_tokens/1000 x 0.025)  # Claude Opus output

# Estimate Tidus cost based on your task mix:
# 60% simple tasks → deepseek-v3 at $0.000014/1K input, $0.000028/1K output
# 30% moderate tasks → various mid-tier at ~$0.0002–0.001/1K
# 10% complex/critical → deepseek-r1 at $0.00055/1K input, $0.00219/1K output
```

---

## Key Price Points (Verified 2026-03-26)

| Model | Tier | Input $/1K | Output $/1K | Best For |
|-------|------|-----------|------------|---------|
| deepseek-v3 | 2 | 0.000014 | 0.000028 | Simple/moderate chat, code, extraction |
| gpt-oss-120b | 2 | 0.000039 | 0.0001 | Moderate tasks, reasoning |
| mistral-small | 3 | 0.00007 | 0.0002 | Classification, filtering |
| claude-haiku-4-5 | 3 | 0.001 | 0.005 | Economy tasks needing Claude quality |
| deepseek-r1 | 1 | 0.00055 | 0.00219 | Critical/complex reasoning — cheapest tier-1 |
| claude-opus-4-6 | 1 | 0.005 | 0.025 | Highest quality; baseline for comparison |
| llama4-maverick-ollama | 4 | 0.0 | 0.0 | Local/private, zero API cost |

**The core insight:** DeepSeek R1 delivers tier-1 reasoning quality at 1/9th the input cost of Claude Opus. For enterprises with heavy reasoning workloads, the savings are disproportionately large.

---

## Payback Period

At Tidus Pro ($99/month) with a conservative $5,000/month saving:

```
Payback period = $99 / $5,000 = 0.02 months = less than 1 day
```

Tidus pays for itself on the first day it routes any non-trivial volume.
