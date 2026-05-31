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

### Layer 1 — GitHub Actions workflow (canonical, runs twice a week)

The repo ships [`.github/workflows/weekly-sync.yml`](../.github/workflows/weekly-sync.yml),
which runs `scripts/weekly_full_sync.py` on **Sundays and Wednesdays at 02:00 UTC**.
Each run: pulls live prices, creates a new DB revision if anything moved 5%+,
writes a snapshot row, regenerates the pricing report (md + html), emails active
subscribers, regenerates `index.html`, and pushes DB + reports + landing-page
changes back to `main`. GitHub Pages (served from the `tidus` repo's main
branch, root) picks up the updated `index.html` within ~1-2 minutes and
publishes to <https://kensterinvest.github.io/tidus/>.

Manual fire from the Actions tab: click "Run workflow" on the "Pricing Sync
(Sun + Wed)" workflow. Or hit the API directly on your Tidus server:

```bash
curl -X POST http://localhost:8000/api/v1/sync/prices \
  -H "Authorization: Bearer <admin-token>"
```

**To change the cadence** — edit the `cron:` line in the workflow file
(e.g. `'0 2 * * 1,4'` for Mon + Thu). Standard 5-field GitHub Actions cron
in UTC.

### Layer 1b — In-process APScheduler (disabled by default)

`TidusScheduler` in `tidus/sync/scheduler.py` carries an APScheduler-driven
`pricing_sync` job that would run inside the FastAPI worker. It is
**disabled by default** (`pricing_sync.enabled: false` in `config/policies.yaml`)
because the GitHub Actions workflow above covers the same work and keeps the
DB commit trail on `main`. Self-hosted deployments without GitHub Actions can
flip `enabled: true`:

```yaml
pricing_sync:
  enabled: true
  day_of_week: 6      # 0=Monday … 6=Sunday (single day — edit scheduler.py for multi-day)
  hour_utc: 2
  change_threshold: 0.05
```

### Layer 2 — External host script (optional, maintainer-only)

For maintainers who push pricing updates from a workstation, a standalone
`sync_pricing.py` can be scheduled locally. The reference setup at
`C:\Users\OWNER\scripts\tidus\setup_windows_schedule.ps1` uses Windows Task
Scheduler with a Sun + Wed 03:00 local trigger. The script:

- Fetches live prices from the OpenRouter public API
- Updates `config/models.yaml` and `tidus/sync/pricing/hardcoded_source.py`
- Git commits and pushes to GitHub so all deployments stay in sync

This layer is redundant with Layer 1 and exists for belt-and-braces scenarios
where the maintainer wants a local commit trail independent of GitHub Actions.

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

---

## Reference: Production VPS Deployment (z-tidus, 2026-05-21)

The canonical Tidus production deployment lives on the shared z-tidus VPS at `77.68.95.23`, fronted by Caddy at `https://ai-router.z-tidus.com`. Only the **subscribe API + landing page** are exposed publicly; the full router/classify/complete surface stays internal.

### Architecture

```
Internet → :443 Caddy ──┬─→ /api/v1/subscribe ─→ 127.0.0.1:9000 (tidus-web.service)
                        └─→ /                  ─→ /opt/tidus/index.html (file_server)

systemd: tidus-sync.timer (Sun + Wed 02:00 UTC) → tidus-sync.service
         ↓
         deploy/sync_wrapper.sh
           1. flock /var/lock/tidus-sync.lock
           2. stash config/subscribers.yaml (writes from web service)
           3. git fetch + reset --hard origin/main
           4. restore stashed subscribers.yaml
           5. uv run python scripts/weekly_full_sync.py
           6. commit tidus.db + reports/ + config/models.auto.yaml + config/subscribers.yaml
           7. git push origin main (via deploy key)
```

### What lives where

| Path | Purpose |
|---|---|
| `/opt/tidus/` | Git checkout of `kensterinvest/tidus` (owned by `tidus` user) |
| `/opt/tidus/.venv/` | uv-managed venv with frozen deps |
| `/opt/tidus/.ssh/id_ed25519` | Deploy key with write access on the repo |
| `/etc/tidus/env` | Secrets (mode 0640, owned `root:tidus`) — RESEND_API_KEY, ANTHROPIC_API_KEY, TIDUS_TELEGRAM_BOT_TOKEN, TIDUS_TELEGRAM_CHAT_ID, vendor keys |
| `/etc/systemd/system/tidus-web.service` | uvicorn on `127.0.0.1:9000` |
| `/etc/systemd/system/tidus-sync.{service,timer}` | Magazine pipeline runner |
| `/var/log/tidus/` | Wrapper logs |
| Caddyfile entry | `ai-router.z-tidus.com` block in `/etc/caddy/Caddyfile` |

### Conventions

- **Web app** = `tidus.web_main:app` — NOT `tidus.main:app`. The slim variant mounts only the subscribe router, has no DB lifespan, no scheduler, no metrics.
- **Port**: `9000` — slot +0 of the `9000–9010` tidus block per `PORTS.md`. Bind on `127.0.0.1` only.
- **Rate-limit**: in-process middleware in `web_main.py`, 5 POSTs/min/IP, X-Forwarded-For trusted (Caddy is the only upstream).
- **No bash-source of `/etc/tidus/env`** — systemd's `EnvironmentFile=` directive parses the file safely; bash-sourcing breaks on values containing `<` or `>` (e.g. `TIDUS_SMTP_FROM`).

### Common operations

```bash
ssh ionos                                                    # admin shell
sudo systemctl status tidus-web.service                      # check web
sudo systemctl status tidus-sync.timer                       # check timer
sudo systemctl list-timers tidus-sync.timer                  # next fire time
sudo systemctl start tidus-sync.service                      # fire magazine on demand
sudo journalctl -u tidus-sync.service -n 100                 # sync logs
sudo nano /etc/tidus/env && sudo systemctl restart tidus-web # rotate secrets
```

### Magazine delivery on this VPS

Direct self-hosted email is **not viable** here: IONOS blocks outbound port 25
(so a local MTA can't reach recipient mail servers), the IP's rDNS isn't a mail
FQDN, and a fresh single-IP sender lands in Gmail's spam folder. Options that work:

- **Telegram (recommended)** — outbound 443 is open and reputation-free. Create a
  bot via @BotFather, then add to `/etc/tidus/env`:
  ```
  TIDUS_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
  TIDUS_TELEGRAM_CHAT_ID=987654321
  ```
  The sync reads env at fire time — no restart needed. Verify with
  `sudo systemctl start tidus-sync.service`. Delivery is additive + fail-open.
- **Email via relay** — port 587 is open, so `RESEND_API_KEY` or `TIDUS_SMTP_*`
  pointed at an authenticated smarthost (Gmail SMTP, Brevo, SMTP2GO, …) also works.

### First-time setup (idempotent)

```bash
ssh ionos
sudo bash /opt/tidus/deploy/install.sh
```

The installer creates the `tidus` system user, generates a deploy key (pauses for you to paste it into the repo's deploy-keys page), clones the repo, builds the venv, writes a placeholder `/etc/tidus/env`, and installs + starts the systemd units. The Caddy block must be added manually to `/etc/caddy/Caddyfile` from `deploy/Caddyfile.snippet`, then `sudo systemctl reload caddy`.

### Emergency fallback

If the VPS is down or you need an out-of-band magazine fire, trigger the GitHub Actions workflow manually:

```bash
gh workflow run weekly-sync.yml -R kensterinvest/tidus
```

The cloud runner does the same pipeline but pushes to `main` directly (no subscriber preservation race because GH Actions doesn't share working-tree state with the web service).
