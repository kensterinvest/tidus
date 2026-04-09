# How Tidus Selects an AI Model — A Complete Guide

> **Audience:** Engineers integrating Tidus, operators tuning routing policies, anyone who wants to
> understand exactly why Tidus picked the model it did for a given request.

---

## The Core Idea

Every AI request carries a `TaskDescriptor` — a structured description of what the caller needs.
Tidus runs that descriptor through **five sequential stages**, each of which eliminates models that
cannot or should not handle the task. The model that survives all five stages with the lowest
composite score is selected.

```
All 53 models in registry
        │
   ┌────▼──────────────────────────────────┐
   │  Stage 1 — Hard Constraints           │  binary pass/fail per model
   │  (enabled, context, domain, privacy,  │
   │   complexity range)                   │
   └────┬──────────────────────────────────┘
        │ eligible set (may be much smaller)
   ┌────▼──────────────────────────────────┐
   │  Stage 2 — Guardrails                 │  operator-defined safety limits
   │  (agent depth, tokens-per-step)       │
   └────┬──────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────┐
   │  Stage 3 — Complexity Tier Ceiling    │  prevents over-engineering cheap tasks
   │  (simple→tier 4, critical→tier 1)    │
   └────┬──────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────┐
   │  Stage 4 — Budget Filter              │  enforces per-request and team budgets
   │  (per-request cap + team budget)      │
   └────┬──────────────────────────────────┘
        │ scored candidates (≥1 required)
   ┌────▼──────────────────────────────────┐
   │  Stage 5 — Score & Select             │  picks cheapest viable model
   │  cost×0.70 + tier×0.20 + lat×0.10    │
   └────┬──────────────────────────────────┘
        │
   RoutingDecision (chosen model + cost estimate)
```

If **any stage reduces the eligible set to zero**, Tidus raises a `ModelSelectionError` that
names the stage and lists every rejection reason. This makes debugging straightforward: you know
exactly which constraint killed the last candidate.

---

## The TaskDescriptor

Before walking the stages, here is what a `TaskDescriptor` looks like:

```python
class TaskDescriptor(BaseModel):
    task_id:              str           # UUID, for tracing
    team_id:              str           # Budget scoping
    workflow_id:          str | None    # Sub-budget scoping
    domain:               Domain        # chat | code | reasoning | extraction | …
    complexity:           Complexity    # simple | moderate | complex | critical
    privacy:              Privacy       # public | internal | confidential
    estimated_input_tokens: int         # Caller's estimate of prompt size
    agent_depth:          int           # How deep in an agent chain (0 = top-level)
    max_cost_usd:         float | None  # Optional per-request cost cap
    preferred_model_id:   str | None    # Optional caller preference (still enforced)
```

Every field affects at least one stage. The caller sets these; Tidus enforces them.

---

## Stage 1 — Hard Constraints

**Code:** `tidus/router/capability_matcher.py` → `_check_hard_constraints()`

This stage asks: "Is this model **physically and contractually capable** of handling this task?"
Each check is binary — pass or fail. The first failure eliminates the model.

### Check 1.1 — Model Enabled

```python
if not spec.enabled:
    return RejectionReason.model_disabled
```

`enabled=false` models are completely invisible to routing. This is set by the `DriftEngine` when
a model goes critical (auto `hard_disable_model` override) or by an admin override.

> **Note on deprecated models:** Deprecated models are intentionally **not** rejected here. A
> model marked `deprecated=true` is in its wind-down window — still capable, just being phased
> out. It reaches Stage 5 where it receives a score penalty of +0.15. This gives operators time
> to migrate workloads without a hard cutoff.

### Check 1.2 — Context Window Fit

```python
if task.estimated_input_tokens > spec.max_context:
    return RejectionReason.context_too_large
```

If the caller estimates the prompt is larger than the model's context window, the model is
eliminated. The caller's `estimated_input_tokens` must be accurate — Tidus trusts this value.

**Example:** A 90,000-token document summary sent to `gpt-4o-mini` (128K context) passes.
The same task sent to `mistral-small-ollama` (32K context) fails at Stage 1.

### Check 1.3 — Domain Capability

```python
required_capability = _DOMAIN_TO_CAPABILITY.get(task.domain)
if required_capability and required_capability not in spec.capabilities:
    return RejectionReason.domain_not_supported
```

Each model declares a `capabilities` set (e.g., `{chat, code, summarization}`). Each task
declares a `domain`. A code-generation task sent to a model without `Capability.code` fails here.

**Domain → Capability mapping:**

| Domain | Required Capability |
|---|---|
| `chat` | `chat` |
| `code` | `code` |
| `reasoning` | `reasoning` |
| `extraction` | `extraction` |
| `classification` | `classification` |
| `summarization` | `summarization` |
| `creative` | `creative` |

### Check 1.4 — Privacy

```python
if task.privacy == Privacy.confidential and not spec.is_local:
    return RejectionReason.privacy_violation
```

`Privacy.confidential` means the data must not leave the customer's infrastructure. Only models
with `is_local=true` (Ollama-served, on-prem) pass this check. All cloud models are eliminated.

This is an enterprise-critical safety rail. A HIPAA-regulated healthcare workflow tagging its
tasks as `confidential` will never accidentally route through OpenAI or Anthropic.

### Check 1.5 — Complexity Range

```python
task_order = _COMPLEXITY_ORDER[task.complexity]
model_min = _COMPLEXITY_ORDER.get(Complexity(spec.min_complexity), 0)
model_max = _COMPLEXITY_ORDER.get(Complexity(spec.max_complexity), 3)
if task_order < model_min or task_order > model_max:
    return RejectionReason.complexity_mismatch
```

Each model has a declared `min_complexity` and `max_complexity`. This prevents:
- A `simple` task (e.g., "What is 2+2?") from being sent to a model designed for `complex`
  reasoning — wasteful and potentially slower.
- A `critical` task from being sent to a model rated only up to `simple`.

**Complexity ordering:** `simple(0) < moderate(1) < complex(2) < critical(3)`

---

## Stage 2 — Guardrail Constraints

**Code:** `tidus/router/capability_matcher.py` → `_check_guardrails()`  
**Config:** `config/policies.yaml` → `guardrails:`

Guardrails are **operator-defined global limits** that apply to all models equally. They protect
against runaway agent behavior and token abuse.

### Check 2.1 — Agent Depth

```python
if task.agent_depth > self._guardrails.max_agent_depth:
    return RejectionReason.agent_depth_exceeded
```

When an AI agent calls another AI agent, the `agent_depth` increments. This prevents infinite
recursion loops — a common failure mode in agentic workflows. Default limit: **5 levels deep**.

### Check 2.2 — Tokens Per Step

```python
if task.estimated_input_tokens > self._guardrails.max_tokens_per_step:
    return RejectionReason.token_limit_exceeded
```

Even if a model technically supports a 200K context window, an operator may set a per-step token
limit of 8,000 to control costs across all models uniformly. This limit is global — it does not
vary per model. Default: **8,000 tokens per step**.

> **Why are Stage 1 and Stage 2 separate?**  
> Stage 1 checks model-specific properties (context window varies per model). Stage 2 checks
> task-level properties against global policies. A task that violates guardrails is rejected for
> all models simultaneously — there is no point checking each model individually.

---

## Stage 3 — Complexity Tier Ceiling

**Code:** `tidus/router/selector.py` → `select()` lines 91–112  
**Config:** `config/policies.yaml` → `routing.complexity_tier_ceiling`

This stage enforces the principle that **expensive, powerful models should only be used when
complexity demands it**. Tidus uses a 4-tier model hierarchy:

| Tier | Description | Examples |
|---|---|---|
| 1 | Premium cloud — frontier models | claude-opus-4-6, gpt-4o, o3 |
| 2 | Standard cloud — strong generalist | claude-sonnet-4-6, gpt-4o |
| 3 | Economy cloud — fast and cheap | gpt-4.1-mini, claude-haiku-4-5 |
| 4 | Local / free — on-prem Ollama | llama4-scout-ollama, phi-4-ollama |

The **ceiling is determined by task complexity**:

| Task Complexity | Max Tier Allowed | Effect |
|---|---|---|
| `simple` | 4 | Any model is eligible |
| `moderate` | 3 | Tier 4 local models are blocked |
| `complex` | 2 | Only Tier 1–2 cloud models |
| `critical` | 1 | Only Tier 1 frontier models |

```python
_COMPLEXITY_TIER_CEILING = {
    Complexity.simple:   4,
    Complexity.moderate: 3,
    Complexity.complex:  2,
    Complexity.critical: 1,
}
tier_ceiling = _COMPLEXITY_TIER_CEILING[task.complexity]
after_tier = [s for s in eligible if s.tier <= tier_ceiling]
```

**Example:** A `simple` chat request can route to a free local Ollama model. A `critical`
reasoning task requiring high accuracy is restricted to Tier 1 only (claude-opus-4-6, o3, etc.).

---

## Stage 4 — Budget Filter

**Code:** `tidus/router/selector.py` → `select()` lines 114–157  
**Components:** `CostEngine`, `BudgetEnforcer`

This stage eliminates models that would bust the caller's budget. It applies **two independent
checks** per model:

### Check 4.1 — Per-Request Cost Cap

```python
estimate = await self._cost_engine.estimate(spec, task)
cost_usd = estimate.estimated_cost_usd
if task.max_cost_usd is not None and cost_usd > task.max_cost_usd:
    # rejected: budget_exceeded
```

The caller may set `max_cost_usd` on the task. This is a hard ceiling — the router will not
exceed it even if it means routing to a worse model. Estimate formula:

```
estimated_cost_usd = (input_tokens × input_price + output_tokens × output_price)
                     × (1 + estimate_buffer_pct)    ← default 15% safety buffer
```

### Check 4.2 — Team Budget

```python
can_spend = await self._enforcer.can_spend(
    team_id=task.team_id,
    workflow_id=task.workflow_id,
    amount_usd=cost_usd,
)
```

Even if a single request is cheap, the team may have exhausted its monthly budget. The
`BudgetEnforcer` checks cumulative spending. Budget periods reset on the 1st of each month.

> **Cost estimate accuracy:** The 15% buffer (`estimate_buffer_pct`) accounts for tokenization
> variance (models tokenize differently from estimations) and output length uncertainty (the
> model may produce more tokens than expected). Actual cost is logged in `cost_records` and
> compared against estimates in billing reconciliation.

---

## Stage 5 — Score & Select

**Code:** `tidus/router/selector.py` → `_score_and_pick()`

The models that survived Stages 1–4 are all **capable of handling the task within budget**.
Stage 5 picks the **best** one using a weighted composite score. Lower score = better.

### The Scoring Formula

```
score = cost_norm × 0.70
      + tier_norm × 0.20
      + latency_norm × 0.10
      [+ 0.15 deprecated penalty if applicable]
```

Each dimension is **min-max normalized** across the surviving candidates:

```python
def _normalize(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)   # all equal → all score 0 on this dimension
    return [(v - lo) / (hi - lo) for v in values]
```

This normalization is critical: it means scores are **relative to the competition**, not
absolute. A $15/1M model that is the cheapest in the pool gets `cost_norm = 0.0` (best possible
on cost). The same model in a pool where everything else costs $0.10/1M would get `cost_norm = 1.0`.

### Scoring Dimensions

**Cost (70% weight)** — the primary driver. Tidus is a cost-minimizing router. If two models are
equally capable, the cheaper one wins. The cost used here is the Stage 4 estimate.

**Tier (20% weight)** — a tiebreaker that prefers lower-overhead models. Between two models with
identical cost estimates, the economy-tier model beats the premium-tier model. This prevents
accidentally routing to a slower, heavier model when a lighter one would do.

**Latency (10% weight)** — uses `spec.latency_p50_ms` from the model catalog (updated by health
probes). Faster models score slightly better. This weight is intentionally small — latency is a
secondary concern after cost.

### Deprecated Model Penalty

```python
if spec.deprecated:
    score += 0.15   # _DEPRECATED_SCORE_PENALTY
```

Deprecated models receive a flat +0.15 penalty after normalization. This ensures they lose to
non-deprecated models with equivalent pricing. However:

- If a deprecated model is **significantly cheaper** (enough to overcome the 0.15 penalty), it
  can still win. The intention is a soft preference, not a hard block.
- When a deprecated model is selected, Tidus logs `routing_deprecated_model` with the model ID
  and score for observability.

### Preferred Model Shortcut

```python
if task.preferred_model_id:
    for spec, cost in costed:
        if spec.model_id == task.preferred_model_id:
            return RoutingDecision(..., score=0.0)  # bypass scoring
```

If the caller pinned a preferred model and it survived Stages 1–4, it is returned immediately
without going through Stage 5 scoring. Budget enforcement still applies.

### Worked Example

**Task:** `domain=code, complexity=moderate, estimated_input_tokens=2000, team_id="eng"`

**After Stage 1–3,** three candidates survive:

| Model | Tier | Input $/1M | Output $/1M | latency_p50_ms |
|---|---|---|---|---|
| claude-sonnet-4-6 | 2 | $3.00 | $15.00 | 800ms |
| gpt-4.1-mini | 3 | $0.40 | $1.60 | 400ms |
| gemini-2.5-flash | 3 | $0.30 | $2.50 | 450ms |

**Stage 4 cost estimates** (2000 input + 500 output tokens, 15% buffer):

| Model | Estimate |
|---|---|
| claude-sonnet-4-6 | (2000×0.003 + 500×0.015) / 1000 × 1.15 = $0.0155 |
| gpt-4.1-mini | (2000×0.0004 + 500×0.0016) / 1000 × 1.15 = $0.00184 |
| gemini-2.5-flash | (2000×0.0003 + 500×0.0025) / 1000 × 1.15 = $0.00212 |

**Stage 5 normalization:**

| Model | cost_norm | tier_norm | lat_norm | score |
|---|---|---|---|---|
| claude-sonnet-4-6 | 1.0 | 0.0 | 1.0 | 1.0×0.70 + 0.0×0.20 + 1.0×0.10 = **0.80** |
| gpt-4.1-mini | 0.0 | 1.0 | 0.0 | 0.0×0.70 + 1.0×0.20 + 0.0×0.10 = **0.20** |
| gemini-2.5-flash | 0.21 | 1.0 | 0.10 | 0.21×0.70 + 1.0×0.20 + 0.10×0.10 = **0.36** |

**Winner: `gpt-4.1-mini`** with score 0.20. It is the cheapest option with lowest latency in
this candidate set. Despite Gemini-2.5-flash having a slightly lower token price, gpt-4.1-mini
wins on latency.

---

## What Happens When Everything Fails

If **every model is eliminated** before Stage 5, Tidus raises:

```python
raise ModelSelectionError(
    message="All models exceed budget for task ...",
    stage=4,
    rejections=[...]   # every rejection reason from all stages
)
```

The API returns HTTP 422 with a structured body listing every model and why it was rejected. This
is intentional: silent fallback to a wrong model is worse than a clear error.

**Common causes by stage:**

| Stage | Common cause | Fix |
|---|---|---|
| 1 | All models disabled | Check drift events, active overrides |
| 1 | Context too large | Chunk the input, or use a 200K+ context model |
| 1 | No model with required capability | Check YAML capabilities for target domain |
| 2 | Agent depth exceeded | Reduce recursion depth or raise `max_agent_depth` |
| 3 | No model in tier ≤ ceiling | Check that `complex`/`critical` tasks have Tier 1 models enabled |
| 4 | Budget exhausted | Contact team admin to increase budget or wait for monthly reset |
| 4 | Per-request cap too low | Raise `max_cost_usd` or chunk the request |

---

## Configuration Reference

All tunable parameters in `config/policies.yaml`:

```yaml
guardrails:
  max_agent_depth: 5          # Stage 2 depth limit
  max_tokens_per_step: 8000   # Stage 2 token limit

routing:
  cost_weight:    0.70         # Stage 5 scoring
  tier_weight:    0.20
  latency_weight: 0.10
  complexity_tier_ceiling:
    simple:   4               # Stage 3 ceilings
    moderate: 3
    complex:  2
    critical: 1

cost:
  estimate_buffer_pct: 0.15   # Stage 4 estimate padding
```

---

## Summary

| Stage | Question Asked | Failure Mode |
|---|---|---|
| 1 | Can this model do the job at all? | Hard rejection — wrong tool |
| 2 | Is this request within safe operating limits? | Hard rejection — guardrail breach |
| 3 | Is this model the right power level for this task? | Ceiling rejection — over/under powered |
| 4 | Can the team afford this model? | Budget rejection |
| 5 | Among viable options, which is cheapest overall? | No failure (winner is always returned) |

Tidus never routes to a wrong model silently. Every rejection is logged with a reason code, and
every selection is logged with the score, cost estimate, and number of candidates considered.
