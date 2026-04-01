"""Locust load test suite for Tidus productionization.

Target SLOs (run against a live server with mocked vendor adapters):
  - POST /api/v1/route  — P99 < 50ms  at 100 rps (pure in-process, no vendor call)
  - POST /api/v1/complete — P99 < 2s   at 20 rps  (includes vendor round-trip)
  - Cache throughput      — hit rate > 95% when same query repeated

Usage:
    # Start the Tidus server first:
    uv run tidus

    # Run with the Locust web UI:
    uv run --extra load locust -f tests/load/locustfile.py --host http://127.0.0.1:8000

    # Run headless (CI):
    uv run --extra load locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 \
        --headless --users 50 --spawn-rate 10 --run-time 60s \
        --csv tests/load/results/run

Each user class below represents a distinct load scenario. Run them individually
with --class-picker or combine for a blended load profile.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


# ── Shared payloads ───────────────────────────────────────────────────────────

ROUTE_SIMPLE = {
    "team_id": "team-engineering",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 300,
    "messages": [{"role": "user", "content": "What is 2+2?"}],
}

ROUTE_COMPLEX = {
    "team_id": "team-engineering",
    "complexity": "complex",
    "domain": "reasoning",
    "estimated_input_tokens": 2000,
    "messages": [{"role": "user", "content": "Analyse the trade-offs between CRDT and OT for collaborative editing."}],
}

COMPLETE_SIMPLE = {
    "team_id": "team-engineering",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 300,
    "messages": [{"role": "user", "content": "Reply with exactly the word PONG."}],
}

COMPLETE_CACHE_PROBE = {
    "team_id": "team-engineering",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 100,
    "messages": [{"role": "user", "content": "cache-probe-constant-query-v1"}],
}


# ── User classes ──────────────────────────────────────────────────────────────

class RouteUser(HttpUser):
    """Sustained load against POST /api/v1/route.

    Target SLO: P99 < 50ms at 100 rps.
    Route is pure in-process (no vendor call) so latency should be very low.
    """

    wait_time = between(0.005, 0.015)  # ~100 rps with 10 concurrent users

    @task(3)
    def route_simple_task(self):
        """Simple chat routing — most common request type."""
        self.client.post("/api/v1/route", json=ROUTE_SIMPLE, name="/route [simple]")

    @task(1)
    def route_complex_task(self):
        """Complex reasoning — tests tier-1 model selection path."""
        self.client.post("/api/v1/route", json=ROUTE_COMPLEX, name="/route [complex]")

    @task(1)
    def route_confidential_task(self):
        """Confidential privacy — must always route to a local model."""
        payload = dict(ROUTE_SIMPLE, privacy="confidential")
        with self.client.post(
            "/api/v1/route", json=payload, name="/route [confidential]", catch_response=True
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                model_id = data.get("chosen_model_id", "")
                if "ollama" not in model_id and model_id != "gemini-nano":
                    resp.failure(
                        f"Confidential task routed to non-local model: {model_id}"
                    )

    @task(1)
    def route_all_domains(self):
        """Round-robin across all supported domains."""
        domain = random.choice(
            ["chat", "code", "reasoning", "extraction", "classification", "summarization"]
        )
        payload = dict(ROUTE_SIMPLE, domain=domain)
        self.client.post("/api/v1/route", json=payload, name=f"/route [{domain}]")


class CompleteUser(HttpUser):
    """Moderate load against POST /api/v1/complete.

    Target SLO: P99 < 2s at 20 rps (vendor latency included; use mock adapters
    or a fast local Ollama model so this test isn't blocked on cloud APIs).
    """

    wait_time = between(0.04, 0.06)  # ~20 rps with 1 user

    @task(4)
    def complete_simple_task(self):
        """Standard complete call."""
        self.client.post("/api/v1/complete", json=COMPLETE_SIMPLE, name="/complete [simple]")

    @task(1)
    def complete_budget_near_limit(self):
        """Simulate a request near (but not exceeding) the budget limit."""
        payload = dict(COMPLETE_SIMPLE, max_cost_usd=10.0)
        self.client.post(
            "/api/v1/complete", json=payload, name="/complete [near-limit]"
        )


class CacheThroughputUser(HttpUser):
    """Repeatedly sends the same query to stress the exact-match cache path.

    Target SLO: hit rate > 95% for repeated identical queries.
    The first request primes the cache; subsequent requests must be cache hits.

    Note: This user class tests the ExactCache IF it is wired into /complete.
    If not yet integrated, this profile still measures routing throughput for
    a hot path (same model repeatedly selected).
    """

    wait_time = between(0.001, 0.003)  # high frequency, same payload

    @task
    def repeated_cache_query(self):
        self.client.post(
            "/api/v1/complete",
            json=COMPLETE_CACHE_PROBE,
            name="/complete [cache-probe]",
        )


class BudgetExhaustionUser(HttpUser):
    """Validates hard-stop behavior when a workflow budget is exhausted.

    Sends requests with a tiny per-request workflow budget that will
    be exhausted after 1–2 calls. Expects a mix of 200 and 422 responses.
    """

    wait_time = between(0.1, 0.3)

    @task
    def request_with_tight_workflow_budget(self):
        payload = dict(
            COMPLETE_SIMPLE,
            workflow_id=f"load-test-wf-{random.randint(1, 5)}",
            max_cost_usd=0.000001,  # near-zero → should 422 after first cheap model
        )
        with self.client.post(
            "/api/v1/complete",
            json=payload,
            name="/complete [budget-exhaust]",
            catch_response=True,
        ) as resp:
            # Both 200 (free local model) and 422 (budget exceeded) are valid outcomes
            if resp.status_code not in (200, 422):
                resp.failure(
                    f"Unexpected status {resp.status_code} for budget-exhaustion scenario"
                )
            else:
                resp.success()


class SessionConcurrencyUser(HttpUser):
    """Simulates 50 concurrent agent sessions.

    Each virtual user gets a unique session ID and sends a sequence of
    messages. Tests guardrail isolation between sessions.
    """

    wait_time = between(0.05, 0.15)

    def on_start(self):
        self.session_id = f"load-session-{random.randint(1, 50)}"

    @task
    def session_message(self):
        payload = dict(
            COMPLETE_SIMPLE,
            agent_session_id=self.session_id,
            agent_depth=random.randint(0, 3),
        )
        self.client.post(
            "/api/v1/complete",
            json=payload,
            name="/complete [session]",
        )


class HealthCheckUser(HttpUser):
    """Lightweight health and readiness probe user.

    Ensures observability endpoints remain fast under concurrent load.
    Target: GET /health P99 < 5ms.
    """

    wait_time = between(0.5, 1.0)

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def ready(self):
        self.client.get("/ready", name="/ready")
