# Runbook: Emergency Registry Freeze

Use this runbook when a registry revision must be stopped from propagating — for example,
after discovering a bad price change, a validation bypass, or a mid-promotion incident.

The `emergency_freeze_revision` override blocks all revision promotions and suspends the
merge layer: while a freeze is active, every model returns its base catalog spec unchanged —
no overrides applied, no telemetry applied.

---

## Steps

### 1. Activate the freeze

```bash
curl -X POST https://<host>/api/v1/registry/overrides \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "override_type": "emergency_freeze_revision",
    "scope": "global",
    "justification": "<describe the incident>"
  }'
```

Save the returned `override_id` — you need it to lift the freeze.

### 2. Verify freeze is active

```bash
curl https://<host>/api/v1/registry/overrides \
  -H "Authorization: Bearer <admin-token>"
```

Confirm the `emergency_freeze_revision` override is present and `is_active: true`.

### 3. Investigate

Check open drift events:
```bash
curl https://<host>/api/v1/registry/drift \
  -H "Authorization: Bearer <admin-token>"
```

Inspect recent pricing ingestion runs:
```sql
SELECT run_id, started_at, status, error_message
FROM pricing_ingestion_runs
ORDER BY started_at DESC
LIMIT 10;
```

Review the active and recent revisions:
```bash
curl https://<host>/api/v1/registry/revisions \
  -H "Authorization: Bearer <admin-token>"
```

### 4. Rollback (if needed)

If the currently active revision is bad, re-promote a known-good superseded revision:

```bash
curl -X POST https://<host>/api/v1/registry/revisions/<good-revision-id>/activate \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"justification": "rolling back to known-good revision during freeze"}'
```

Note: Rollback re-promotes using Tier 1 + Tier 2 validation only (Tier 3 canary is skipped
for re-promotions of previously-verified revisions).

### 5. Lift the freeze

Once the incident is resolved:

```bash
curl -X DELETE https://<host>/api/v1/registry/overrides/<freeze-override-id> \
  -H "Authorization: Bearer <admin-token>"
```

### 6. Post-incident

Write an audit log entry explaining root cause and resolution. Verify the registry is
serving correct specs:

```bash
curl https://<host>/api/v1/models \
  -H "Authorization: Bearer <any-valid-token>"
```

Confirm `tidus_registry_active_revision_id` metric changed if you performed a rollback.

---

## Notes

- The freeze does NOT prevent read operations. `GET /api/v1/models` continues to work.
- All override create/delete actions are written to `audit_logs`.
- Only one `emergency_freeze_revision` override can be active at a time per the conflict rules.
- Lifting the freeze does not automatically trigger a new price sync — run one manually if
  needed with `POST /api/v1/sync/prices`.
