# Runbook: Override Abuse Response

This runbook covers detecting and responding to unauthorized or abusive use of the
Tidus override system — for example, a team_manager escalating their own model tier,
a compromised API key creating overrides outside their team's scope, or an insider
using overrides to redirect traffic.

---

## Detection

### Unusual audit log entries

```sql
SELECT actor_sub, action, metadata, created_at
FROM audit_logs
WHERE action LIKE 'registry.override.%'
  AND created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;
```

Red flags:
- `actor_sub` creating overrides for `scope_id` that isn't their own team
- High volume of override creates/deletes from a single actor
- `override_type = 'emergency_freeze_revision'` not correlated with an incident

### Override listing

```bash
curl https://<host>/api/v1/registry/overrides \
  -H "Authorization: Bearer <admin-token>"
```

Look for:
- Overrides with `scope: global` created by non-admin accounts
- Overrides on models belonging to other teams
- Active overrides with no obvious business justification

### Prometheus alert

If the number of active overrides spikes unexpectedly, check:

```promql
# No built-in metric, but you can monitor audit log rate via log aggregation
```

---

## Response Steps

### 1. Revoke suspicious overrides

```bash
curl -X DELETE https://<host>/api/v1/registry/overrides/<override-id> \
  -H "Authorization: Bearer <admin-token>"
```

Document each revocation in your incident ticket.

### 2. Review the actor's full history

```sql
SELECT action, metadata, created_at
FROM audit_logs
WHERE actor_sub = '<suspect-actor>'
  AND action LIKE 'registry.%'
ORDER BY created_at DESC;
```

Identify the full scope of unauthorized changes.

### 3. Rotate the compromised credential

If an API key was compromised, rotate it through your SSO/identity provider.
All JWTs signed with the old key will fail validation immediately (assuming short
`exp` claims). If the key had a long expiry, add it to the revocation list.

After rotation, re-audit for any overrides created between the suspected compromise
date and the rotation date.

### 4. Verify team-scoped isolation

Team managers can only create overrides where `scope_id == their own team_id`.
This is enforced in `OverrideManager.create()`. If a team manager created an override
on another team's scope, this indicates either a code bug or a privilege escalation —
escalate to the security team immediately.

### 5. Cross-team scope violation

If `scope_id` on an override doesn't match the actor's `team_id` and the actor's role
is `team_manager` (not `admin`), this is a privilege escalation:

1. Revoke the override immediately (Step 1)
2. Suspend the actor's API key
3. File a security incident
4. Review `audit_logs` for the actor going back 30 days

---

## Notes

- All override operations (create, delete, expiry) write to `audit_logs` with the full
  `actor_sub`, `team_id`, and request metadata. The audit log is append-only.
- The `deactivated_by` field distinguishes manual revocation (`actor_sub`),
  expiry (`system_expiry`), and drift engine removal (`drift_engine`).
- Override export (`GET /api/v1/registry/overrides/export`) produces an HMAC-SHA256 signed
  YAML bundle for offline review and GitOps workflows.
