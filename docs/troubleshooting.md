# Troubleshooting

Quick fixes for the 10 most common issues when setting up or running Tidus.

---

## 1. "No models available" — routing always fails

**Symptom:** Every routing request returns a `ModelSelectionError` or `422 No capable model found`.

**Cause:** No vendor API keys are configured, so all cloud models are disabled.

**Fix:**
```bash
# Check which keys are set
env | grep -E "(ANTHROPIC|OPENAI|GOOGLE|MISTRAL|DEEPSEEK|XAI|MOONSHOT)_API_KEY"

# Add at least one to .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env

# Restart
docker compose restart   # or: uvicorn tidus.main:app --reload
```

For free testing with no API key, run Ollama locally:
```bash
ollama pull llama4
# Tidus will detect it automatically at http://localhost:11434
```

---

## 2. HTTP 500 on startup — "OIDC_ISSUER_URL not configured"

**Symptom:** Every request returns `500 Internal Server Error` with the message `OIDC_ISSUER_URL is not configured but ENVIRONMENT=production`.

**Cause:** `ENVIRONMENT=production` (the default) requires a real OIDC provider. Dev mode is blocked in production for security.

**Fix A — Use dev mode for local testing:**
```bash
echo 'ENVIRONMENT=development' >> .env
```

**Fix B — Configure OIDC for real production:**
```env
OIDC_ISSUER_URL=https://your-okta.example.com/oauth2/default
OIDC_CLIENT_ID=your-client-id
OIDC_TEAM_CLAIM=tid       # JWT claim that contains the team_id
OIDC_ROLE_CLAIM=role      # JWT claim that contains the Tidus role
```

---

## 3. Budget always blocked — team is immediately hard-stopped

**Symptom:** Requests are rejected with `budget_hard_stop` even though the team should have budget remaining.

**Cause:** The in-memory spend counter accumulated spend from a previous test run. Monthly resets happen on the 1st of the month; there is no automatic reset between process restarts for mid-month testing.

**Fix — Restart the server to reset in-memory counters:**
```bash
docker compose restart tidus
# or: Ctrl+C, then: uvicorn tidus.main:app --reload
```

For a one-time test reset without restart, call the admin reset endpoint:
```bash
# Manual counter reset is not yet a public API endpoint.
# Fastest fix: restart the server process.
```

---

## 4. Dashboard shows no data — all panels are blank

**Symptom:** Dashboard loads but all KPIs show zeros, the chart is empty, and the savings panel shows "—".

**Cause:** No `/api/v1/complete` requests have been sent yet. The DB is empty.

**Fix — Send a test request:**
```bash
curl -s -X POST http://localhost:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-default",
    "complexity": "simple",
    "domain": "chat",
    "estimated_input_tokens": 500,
    "messages": [{"role": "user", "content": "Hello!"}]
  }' | python -m json.tool
```

The dashboard auto-refreshes every 30 seconds. After the first request, all panels populate.

---

## 5. Health probe auto-disabling models — models going offline unexpectedly

**Symptom:** Models appear disabled in the registry health panel even though the API key is valid.

**Cause:** The health probe sends a minimal `"hi"` prompt to every enabled model every 5 minutes. After 3 consecutive failures, it auto-disables the model.

**Likely causes:**
- Ollama is not running (`ollama serve`)
- Temporary vendor API outage
- Rate limit on the vendor API

**Fix:**
```bash
# Check server logs
docker compose logs tidus | grep "health_probe"

# Re-enable a specific model via the API
curl -X PATCH http://localhost:8000/api/v1/models/llama4-maverick-ollama \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# Or restart Ollama
ollama serve
```

---

## 6. CORS errors in the browser

**Symptom:** The dashboard loads but API calls fail with `Access to fetch blocked by CORS policy`.

**Cause:** In production mode, CORS is locked down to the origins listed in `CORS_ALLOWED_ORIGINS`. If this is empty, no origins are allowed.

**Fix:**
```env
# In .env — comma-separated list of allowed origins
CORS_ALLOWED_ORIGINS=https://dashboard.yourdomain.com,https://app.yourdomain.com
```

For local development:
```env
ENVIRONMENT=development   # enables wildcard CORS automatically
```

---

## 7. High latency on first request — Ollama cold start

**Symptom:** The first request to a local Ollama model takes 30–120 seconds.

**Cause:** Ollama loads the model weights from disk on the first request. Subsequent requests use the in-memory model (< 2s).

**Fix — Pre-load the model after starting Ollama:**
```bash
# Pre-warm the model (sends a quick request to load weights into memory)
ollama run llama4 "hi" 2>&1 | head -1

# Verify model is loaded
ollama ps
```

---

## 8. ModelSelectionError on confidential tasks — "no local model configured"

**Symptom:** Requests with `"privacy": "confidential"` fail with `ModelSelectionError` at Stage 2 (hard constraints).

**Cause:** Confidential tasks require a local (`is_local=true`) model. If Ollama is not running or no local model is enabled, no model survives Stage 2.

**Fix:**
```bash
# Start Ollama
ollama serve
ollama pull llama4   # or another local model

# Verify local models are detected
curl http://localhost:8000/api/v1/models | python -m json.tool | grep '"is_local": true'
```

---

## 9. Scheduler not running — health probes and price sync not firing

**Symptom:** Logs show no `health_probe_run` or `price_sync_run` entries. Models are not being health-checked.

**Cause:** Running multiple Uvicorn worker processes (`--workers N`) with SQLite causes APScheduler to start on each worker, leading to duplicate jobs and potential lock contention. With SQLite, use a single worker.

**Fix — Single worker (SQLite):**
```bash
uvicorn tidus.main:app --workers 1 --reload
```

**Fix — Multiple workers (PostgreSQL):**
```bash
# Switch to PostgreSQL for multi-worker deployments
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/tidus uvicorn tidus.main:app --workers 4
```

---

## 10. 429 Too Many Requests — hitting the rate limit

**Symptom:** After many rapid requests, the API returns `429 Too Many Requests`.

**Cause:** Tidus enforces a per-IP rate limit of **60 requests/minute** on `/complete` and **120 requests/minute** on `/route` to prevent runaway loops and accidental DDoS.

**Fix — Space out requests or increase the limit:**
```env
# (future setting — current limits are code-level defaults)
```

**For load testing**, use the dedicated load test profile which bypasses rate limits on localhost:
```bash
uv run locust --headless -u 10 -r 2 -t 60s --host http://localhost:8000
```

If you legitimately need higher throughput in production, this is typically a sign that your workload should use the `/route` endpoint for bulk routing decisions (which has a higher limit) and only call `/complete` for actual execution.
