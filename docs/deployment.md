# Deployment

## Docker Compose (Recommended)

### Prerequisites
- Docker Engine 24+ and Docker Compose v2
- `.env` file with vendor API keys (copy from `.env.example`)

### Start Tidus

```bash
git clone https://github.com/kensterinvest/tidus.git
cd tidus
cp .env.example .env
# Edit .env with your API keys
docker compose up -d
```

Tidus starts on **http://localhost:8000**.

- API: `http://localhost:8000/api/v1/route`
- Dashboard: `http://localhost:8000/dashboard/`
- API docs: `http://localhost:8000/docs`

### With Local Ollama (Free Inference)

```bash
docker compose --profile ollama up -d
```

This starts Ollama alongside Tidus. Pull models:

```bash
docker exec tidus-ollama-1 ollama pull llama4
docker exec tidus-ollama-1 ollama pull mistral
```

Tidus will automatically route confidential or cost-sensitive tasks to local models.

### Config Without Rebuild

Mount `config/` as a volume (already configured in `docker-compose.yml`):

```bash
# Edit config/models.yaml to adjust pricing or enable/disable models
# Tidus picks up changes on the next request after a restart
docker compose restart tidus
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Optional | — | GPT-4/5/o3/Codex access |
| `ANTHROPIC_API_KEY` | Optional | — | Claude Opus/Sonnet/Haiku |
| `GOOGLE_API_KEY` | Optional | — | Gemini Pro/Flash |
| `MISTRAL_API_KEY` | Optional | — | Mistral/Codestral |
| `DEEPSEEK_API_KEY` | Optional | — | DeepSeek R1/V3 |
| `XAI_API_KEY` | Optional | — | Grok 3 |
| `MOONSHOT_API_KEY` | Optional | — | Kimi K2.5 |
| `OLLAMA_BASE_URL` | Optional | `http://localhost:11434` | Local Ollama endpoint |
| `DATABASE_URL` | Optional | SQLite in `/app/data` | SQLAlchemy async URL |
| `ENVIRONMENT` | Optional | `production` | `development` or `production` |
| `LOG_LEVEL` | Optional | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TIDUS_TIER` | Optional | `community` | `community`, `pro`, `business`, `enterprise` |
| `CORS_ALLOWED_ORIGINS` | Optional | `""` | Comma-separated allowed origins (e.g. `https://app.example.com`) |
| `OIDC_ISSUER_URL` | Production required | `""` | e.g. `https://your-okta.example.com/oauth2/default` |
| `OIDC_CLIENT_ID` | Optional | `""` | JWT audience claim value |
| `OIDC_TEAM_CLAIM` | Optional | `tid` | JWT claim holding the `team_id` |
| `OIDC_ROLE_CLAIM` | Optional | `role` | JWT claim holding the Tidus role |

You only need API keys for the vendors you want to use. Tidus gracefully skips vendors with no key.

## Development (Local)

```bash
# Install dependencies
uv sync

# Set environment
cp .env.example .env && edit .env

# Run with auto-reload
uvicorn tidus.main:app --reload
```

## SQLite → PostgreSQL

For production with multiple instances, switch to PostgreSQL:

```bash
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/tidus
```

Install the async driver: `pip install asyncpg`

Tidus uses SQLAlchemy's async interface so no application code changes are needed — only the `DATABASE_URL` changes.

## Health Checks

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness — always returns `{"status": "ok"}` if process is alive |
| `GET /ready` | Readiness — returns `{"status": "ready"}` after startup completes |

The Docker `healthcheck` in `docker-compose.yml` uses `/health` with 30s interval and 3 retries.

## Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name tidus.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Automated Pricing Sync

Tidus includes a two-layer system that keeps model prices accurate automatically.
Accurate prices matter because the routing score weights cost at **70%**.

### Layer 1 — Server-side sync (built-in)

`TidusScheduler` runs `POST /api/v1/sync/prices` automatically every Sunday at 02:00 UTC.
No configuration needed — it starts with the server.

To trigger manually:
```bash
curl -X POST http://localhost:8000/api/v1/sync/prices \
  -H "Authorization: Bearer <admin-token>"
```

To adjust the schedule, edit `config/policies.yaml`:
```yaml
pricing_sync:
  day_of_week: 6      # 0=Monday … 6=Sunday
  hour_utc: 2
  change_threshold: 0.05
```

### Layer 2 — External host script (optional, recommended)

For maintainers who push pricing updates to GitHub, a standalone `sync_pricing.py`
can be scheduled locally (e.g. Windows Task Scheduler, cron) to:
- Fetch live prices from the OpenRouter public API
- Update `config/models.yaml` and `tidus/sync/price_sync.py`
- Git commit and push to GitHub so all deployments stay in sync

See [docs/pricing-sync.md](pricing-sync.md) for the full architecture and setup guide.

## Production Checklist

- [ ] Run `tidus-setup --defaults` to auto-configure before first start
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure `OIDC_ISSUER_URL` (required in production mode)
- [ ] Set `CORS_ALLOWED_ORIGINS` to your dashboard domain(s)
- [ ] Use PostgreSQL for `DATABASE_URL` (multi-instance)
- [ ] Set all required vendor API keys
- [ ] Configure `config/budgets.yaml` with team limits
- [ ] Set up reverse proxy with TLS
- [ ] Monitor `/health` and `/api/v1/dashboard/summary`
- [ ] Review `config/policies.yaml` guardrail limits
- [ ] Verify pricing sync is running (`GET /api/v1/audit/events?action=price_change`)
- [ ] Schedule monthly savings reports: `GET /api/v1/reports/monthly`
