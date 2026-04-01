# Enterprise: Data Residency

*This feature is on the roadmap for the Enterprise tier.*

Contact lapkei01@gmail.com to discuss early access or to share requirements.

---

## Design Intent

Tidus data residency controls let enterprises choose where routing logs, cost records, and usage data are stored — and optionally ensure no request data ever leaves a specific geographic region or network boundary.

### Data Tidus Stores

| Data Type | Where Stored | Sensitivity |
|-----------|-------------|-------------|
| `cost_records` | DB (`cost_records` table) | Medium — contains team_id, model, token counts, cost |
| `routing_decisions` | DB (`routing_decisions` table) | Low — contains model_id, score, rejection reasons |
| `price_change_log` | DB (`price_change_log` table) | Low — vendor pricing history |
| `budget_policies` | DB + `config/budgets.yaml` | Medium — team spend limits |
| Request messages | **Never stored** | N/A — messages are not logged or persisted |

Message content is never written to the database. Only token counts, costs, and routing metadata are persisted.

### Deployment Options

| Mode | Data Location | Message Privacy |
|------|--------------|----------------|
| Cloud (default) | Your chosen database host | Messages not stored anywhere |
| Self-hosted / on-prem | Your infrastructure | Full control — no external calls except to chosen vendors |
| Air-gapped | Local DB + Ollama only | Zero external network access |

### On-Prem Deployment

The `Dockerfile` and `docker-compose.yml` are designed for on-prem deployment today. Set `DATABASE_URL` to your internal PostgreSQL instance and configure only local (Ollama) models to ensure no data leaves your network:

```yaml
# docker-compose.yml — on-prem configuration
services:
  tidus:
    environment:
      DATABASE_URL: postgresql+asyncpg://user:pass@internal-db:5432/tidus
      OLLAMA_BASE_URL: http://internal-ollama:11434
      # No external API keys = no external vendor calls
```

### Geographic Data Residency (Roadmap)

For regulated industries (GDPR, HIPAA, financial services), planned controls include:
- Per-team routing restrictions: team-legal may only route to EU-hosted models
- Audit log export to region-specific storage (S3 bucket, Azure Blob) with encryption at rest
- Per-request data residency headers for auditors

---

## Current State

On-prem deployment is fully functional today via Docker Compose — see [Deployment](../deployment.md). Geographic routing restrictions and compliance export are on the Enterprise roadmap.
