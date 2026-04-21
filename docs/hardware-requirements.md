# Hardware Requirements

Tidus is a **software-only** deployment for most use cases. The cascade is
designed to short-circuit at cheap tiers (T0-T4), leaving only ~2% of traffic
for T5 LLM escalation. T5 is the only component that benefits from a GPU.

This document captures the two supported deployment tiers and what each costs.

## At a glance

| Component | **CPU-only deployment** | **Enterprise deployment (with T5 GPU)** |
|---|---|---|
| Compute | 4-8 vCPU, 8+ GB RAM | 4-8 vCPU, 8+ GB RAM |
| **GPU** | **Not required** | **1 × 8+ GB VRAM GPU** (shared across whole org) |
| Storage | ~2 GB (models + SQLite / Postgres) | ~4-5 GB (+ Phi-3.5-mini Q4_K_M) |
| Tidus licence fee | $0 | $0 |
| Hardware investment | $0 | One-time GPU purchase OR hourly cloud GPU |
| Confidential recall | **89.2% measured** (E1 baseline, cross-family IRR) | **95-97% design goal** (not yet empirically verified with T5 in production) |

Both deployments are fully air-gapped. Customer messages never leave the
deployment boundary in either tier — the GPU is for local LLM inference,
not for any remote call.

## The two tiers

### CPU-only tier — small teams, pilots, evaluation

**Runs on any modern server or developer machine.** T5 is disabled via config;
the cascade terminates at T4 (cheap stack).

Recall implication: confidential-classification recall is the cross-family IRR
baseline of **89.2%** (see `findings.md`). The ~10.8% gap is specifically
**topic-bearing confidentials without PII entity markers** — e.g., a first-person
financial-hardship disclosure, an HR-complaint narrative, or a credential-request
message that happens to carry no real names, SSNs, or secrets. Without T5 these
classify as non-confidential and route via the tenant's normal external-LLM
pipeline.

For these cases Tidus emits `confidence_warning: true` on the classification,
so audit logs flag the missed-signal population explicitly.

**Configuration:**
```yaml
# config/policies.yaml
classify_tier5_enabled: false
```

Appropriate for:
- Pilots, POCs, evaluations
- Small internal teams (<500 users)
- Deployments where the 10.8% topic-bearing gap is acceptable
- Environments where GPU is not yet provisioned (add later, no re-deploy)

### Enterprise tier — compliance SLA, full-recall target

**Adds one GPU to the deployment.** T5 is enabled; LLM escalation runs locally
on the GPU via Ollama. Full cascade engages.

Recall implication: **design target is 95-97%** confidential recall; the
6 topic-bearing confidentials that slip past T4 are the T5 LLM's target class.
This target is a plan.md design goal, **not yet empirically verified with T5
live on production traffic** — current measured ceiling is the 89.2% E1
baseline (no-T5). The T5 uplift will be measured once enterprise traffic
accumulates.

**Configuration:**
```yaml
# config/policies.yaml
classify_tier5_enabled: true
classify_tier5_model: "phi3.5:3.8b-mini-instruct-q4_K_M"  # default
```

> **Why Phi-3.5-mini remains the T5 default:** per plan.md, Phi is chosen
> for MMLU 69 / BBH 69 (top accuracy-per-dollar in 3 B class), MIT licence,
> and Q4_K_M (~2.4 GB) fits well below the 8 GB VRAM floor. CPU latency was
> disqualified on 2026-04-21 — hence the GPU-only positioning — but Phi's
> **GPU latency has not been benched by the Tidus team**. The 150-500 ms
> estimate is extrapolated from community benchmarks (llama.cpp, Ollama
> perf threads). An alternative T5 default (e.g., llama3.2:3b, gemma4:e2b)
> is not ruled out but requires a separate dual-gate bench on GPU.

Appropriate for:
- HIPAA / SOC 2 / financial-services compliance targets
- Enterprise deployments with recall SLAs in their data-loss-prevention policy
- Tenants with existing GPU inventory (most AI-hosting enterprises already have this)
- Multi-tenant SaaS offerings where one GPU serves the whole org

## GPU specification for T5

### Minimum

**8 GB VRAM.** Phi-3.5-mini Q4_K_M fits in ~4 GB; the 8 GB floor allows
headroom for KV cache, Ollama model-keep-alive, and future model swaps
(larger Q5 / Q8 variants) without re-provisioning.

### Recommended SKUs

| Tier | Example SKUs | Typical cost | VRAM |
|---|---|---|---|
| **Entry** | RTX 3060 12 GB / RTX 4060 Ti 16 GB | $300-500 one-time | 12-16 GB |
| **Cloud baseline** | AWS L4 / Azure NVads-V710-v5 | $0.50-0.80 / hr | 24 GB |
| **Production** | RTX 4070 Ti / NVIDIA A10G | $800-1,200 one-time / $1.00 / hr | 16-24 GB |
| **High-volume** | A100 40 GB / A100 80 GB | $2.00-3.00 / hr | 40-80 GB |

> ⚠️ **Latency on these GPUs is pre-deployment estimated at 150-500 ms p95
> for Phi-3.5-mini Q4_K_M** based on published llama.cpp / Ollama perf
> threads (see appendix for sources). No Tidus team bench has been run on
> GPU yet — **verify on your hardware before publishing an SLA.** Tidus
> ships `scripts/bench_gemma4_latency.py` (despite the name, accepts any
> Ollama model via `--model`) for this. Target budget: p95 ≤ 500 ms for
> T5 calls.

### Why so few GPU-hours are needed

T5 handles **~2% of traffic**. Throughput math for one GPU instance running
Ollama:

- 60 k users × 30 msgs/user/business-day × 10% peak-hour concentration
  = **50 msgs/sec peak** at busiest hour
- 2% of those hit T5 = **~1 T5 call/sec peak**
- At the estimated 400 ms/call (pre-deployment — see SKU table caveat),
  sustained queue depth stays below 0.5
- **One GPU instance is sufficient** at 60 k users — no per-tenant allocation
  needed

> The math depends on GPU p95 actually landing near 400 ms. If your measured
> p95 is 2× that, queue depth roughly doubles — still comfortably sub-one at
> 60 k users, but tighter at 100 k+. Verify before committing to one GPU at
> 100 k-user scale.

For compliance reasons, some tenants will want isolated GPU allocations anyway.
The cascade still allows that; it just isn't architecturally required for
throughput.

## Why Tidus was measured against CPU-only T5 historically

Earlier plan.md revisions claimed Phi-3.5-mini Q4_K_M hit p95 ≤ 500 ms on 4-vCPU
CPU. That claim was based on extrapolated published numbers from marketing
materials and other authors' benchmarks, not measurement on representative
4-vCPU x86 hardware.

Bench on 2026-04-21 on Intel i7-9700KF (8C/8T @ 3.6 GHz, DDR4, Ollama 0.20.5
Q4_K_M) measured:

| Model | Params | Measured p95 | Measured tok/s |
|---|---|---|---|
| phi3.5:3.8b-mini-instruct-q4_K_M | 3.8 B | 34.7 s | 3.88 |
| llama3.1:8b | 8 B | 23.4 s | 2.56 |
| gemma4:e4b-it-q4_K_M | 8 B actual / 4.5 B PLE | 44.2 s | 1.98 |

CPU LLM inference at Q4 is memory-bandwidth-bound. No practical upgrade path
from consumer CPU reaches ≤ 500 ms for 30-60 token JSON responses:
DDR4→DDR5 is roughly 2×, AVX2→AVX-512 adds 1.5× for prompt eval — combined
~3-4× on high-end desktop, still 8-10× over budget. **GPU inference is
~50-100× faster than CPU for this workload** due to memory bandwidth and
parallelism. A $400 consumer GPU does what no $5 k CPU can.

See `tests/classification/t5_bench_results.md` for the full bench record.

## Falling back: what happens without a GPU

If `classify_tier5_enabled: true` is set but the GPU is unreachable (crash,
model not pulled, Ollama server down), Tidus degrades gracefully:

1. The classifier uses the T4 encoder output directly
2. `confidence_warning: true` is set on the response
3. The request is logged to the active-learning review queue
4. Audit log flags the missed-T5 opportunity for later offline review

No request fails from a T5 outage. The 10.8% topic-bearing gap is the only
quality cost during degraded operation.

## Frequently asked: if we have the GPU, why not use it for every message?

A reasonable question from enterprise buyers who are already paying for
GPU hardware: **"We bought the GPU anyway — why not route everything
through the local LLM and skip T0–T4?"**

This would be **cascade replacement**. It is explicitly out of scope for
Tidus, for three concrete reasons. The numbers below are for a 60 k-user
deployment at 50 msgs/sec peak.

### 1. User-perceived latency gets 5–7× worse

| Approach | Per-message latency (weighted avg) |
|---|---|
| **Cascade** (T1 30–40% @ 5–10 ms, T2 50–60% @ ~50 ms, T5 2% @ ~400 ms) | **~60–80 ms** |
| **GPU-only** (100% @ ~400 ms) | **~400 ms** |

T1 regex fast-path handles 30–40% of traffic in single-digit milliseconds.
Feeding that traffic through an LLM instead multiplies each of those
messages' latency by 40–80×. Even a 400 ms GPU call — fast by LLM
standards — is slow by fast-path standards.

### 2. Throughput requires 20× more GPU inventory

One GPU serves ~2.5 LLM calls per second (at 400 ms each, serialized by
Ollama per model instance). The math:

| Approach | T5-class calls/sec | GPU instances needed |
|---|---|---|
| **Cascade** | ~1/sec (2% of 50) | **1 GPU** |
| **GPU-only** | 50/sec (100% of 50) | **20 GPUs** |

Cloud economics at ~$1,500/mo/GPU: **$1,500/mo cascade vs. $30,000/mo
GPU-only per enterprise.** Same throughput, 20× the hardware spend.

### 3. Accuracy drops on the tasks regex was perfect at

The cascade uses each tier where it is strongest:

| Detection class | Cascade performance | GPU-only LLM performance |
|---|---|---|
| SSN / credit card / API key (regex, Tier 1) | **100%** exact-match | ~85% (LLM can misread formats) |
| Topic-bearing confidentials (hardship, HR complaint) | 95–97% via T5 | 85% (LLM alone) |
| Short messages (<100 chars) | Fast-path correct | LLM wastes tokens on trivial input |

Regex is perfect on its narrow domain; LLMs are not. Replacing T1 regex
with an LLM **lowers recall** on the exact-match PII that is most
compliance-load-bearing (credentials, SSNs, credit-card numbers).

### The correct framing

**The GPU is sized for the 2% of traffic that T0–T4 can't handle.** That
is what makes one modest GPU sufficient for a 60 k-user enterprise.
Repurposing that GPU for 100% of traffic changes the hardware budget,
the latency profile, and the recall story — it is not the same product.

Enterprises aren't paying for "AI compute for classification." They are
paying for **one GPU slot dedicated to the 2% of messages that need
LLM-level topic understanding.** The cascade does the rest, for free,
on CPU that was already provisioned.
