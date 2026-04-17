# Enterprise: Role-Based Access Control (RBAC)

*Available in the Enterprise tier and shipped with Tidus v1.1.0 (Phase 8).*

Tidus enforces a five-role hierarchy on every API endpoint. Role is extracted
from the OIDC token's role claim (configurable via `OIDC_ROLE_CLAIM`) and
checked by FastAPI's `require_role` dependency. Admin is a super-role — it
satisfies any required role list.

---

## How It Works

Every protected endpoint lists the roles permitted to call it. A caller whose
token role is not in that list is rejected with HTTP 403. Non-admin callers
are additionally scoped to their own team on the dashboard, guardrails session,
and budget-creation endpoints.

### Role Model

| Role | Scope | Permissions |
|------|-------|-------------|
| `admin` | Global | Full access: configure models, budgets, policies, view all logs |
| `team_manager` | Team | Manage team budgets, view team usage, create team sessions |
| `developer` | Team | Call `/route` and `/complete`, view own team's usage |
| `read_only` | Team | View dashboard and usage metrics — no execution rights |
| `service_account` | Workflow | Scoped to a specific `workflow_id` — used by automated agents |

### Planned Enforcement Points

- `POST /api/v1/complete` — requires `developer` or `service_account` role
- `PATCH /api/v1/models/{id}` — requires `admin` role
- `POST /api/v1/budgets` — requires `admin` or `team_manager`
- `GET /api/v1/usage/summary` — requires `team_manager` or `admin`
- Model-level restrictions: specific roles can be locked to tier ceilings (e.g., `developer` role limited to tier 2–4 models only)

### Integration Path

RBAC will integrate with the existing `get_current_team()` stub in `tidus/api/deps.py` — replacing it with a JWT/OIDC token validator that extracts `team_id`, `role`, and `permissions` from the token. No changes to the router, adapter, or cost engine are needed. All enforcement points are already wired.

---

## Current State

`get_current_team()` in `tidus/api/deps.py` returns a static team ID today. Swapping it for a real identity provider is a single-file change — the rest of the system already treats it as the authoritative source of caller identity.
