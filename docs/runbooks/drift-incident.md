# Runbook: Drift Incident Response

This runbook covers responding to a model drift detection alert — situations where a model's
runtime behaviour (latency, context usage, tokenization accuracy, or pricing) has diverged
from the catalog spec.

The Tidus drift engine runs every 5 minutes and classifies detections as `warning` or
`critical`. Critical detections auto-disable the model via a `hard_disable_model` override.

---

## Alert Sources

| Alert | Metric | Trigger |
|---|---|---|
| `TidusProbeHighFailureRate` | `tidus_probe_live_calls_total{result="fail"}` | >50% live probes failing for 5 min |
| Slack/email from drift engine | `tidus_registry_drift_events_total` | New drift event logged |

---

## Steps

### 1. Identify open drift events

```bash
curl https://<host>/api/v1/registry/drift \
  -H "Authorization: Bearer <admin-token>"
```

Note the `model_id`, `drift_type`, `severity`, `metric_value`, and `threshold_value` for each
open event.

### 2. Check if model was auto-disabled

```bash
curl https://<host>/api/v1/models \
  -H "Authorization: Bearer <any-valid-token>"
```

A critical drift event creates a `hard_disable_model` override. Verify the model is absent
from the enabled list.

Check the override:
```bash
curl "https://<host>/api/v1/registry/overrides?model_id=<model-id>" \
  -H "Authorization: Bearer <admin-token>"
```

Look for an override with `created_by: drift_engine`.

### 3. Investigate by drift type

**Latency drift** (`drift_type: latency`):
- Check `model_telemetry` for the P50 trend over the last 24h
- Compare `metric_value` (measured ratio) to `threshold_value` (configured critical_ratio)
- Likely causes: provider degradation, regional outage, rate limiting

**Context drift** (`drift_type: context`):
- Review `cost_records` for the model — high `input_tokens` values indicate prompts near
  the context window ceiling
- Check if the catalog `max_context` field is correct for the current model version
- Likely causes: model version change by provider (context window shrunk), application
  sending unexpectedly long prompts

**Tokenization drift** (`drift_type: tokenization`):
- High `token_delta_pct` in `model_telemetry` suggests tokenizer mismatch
- Likely causes: provider updated the model's tokenizer without notice; Tidus adapter
  uses a different tokenizer than the current model version

**Price drift** (`drift_type: price`):
- Many price changes in 30 days, or a large deviation from the last logged price
- Check `price_change_log` for the model
- Likely causes: provider changed prices; pricing feed is incorrect

### 4. Resolution options

**Option A — Wait for auto-recovery (latency spikes)**

The drift engine checks the last 3 telemetry readings every 5 minutes. If all 3 are
healthy, the override is automatically removed and the drift event is auto_resolved.

**Option B — Adjust threshold (false positive)**

If the drift is a known false positive (e.g., you intentionally changed the context
window), update `config/policies.yaml`:

```yaml
drift:
  latency:
    critical_ratio: 3.0   # was 2.5
  context:
    critical_rate: 0.20   # was 0.15
```

Restart Tidus or wait for the next scheduler cycle. The next DriftEngine run will use
the updated thresholds.

**Option C — Manual resolution**

After investigating and confirming the drift is resolved:

```bash
curl -X POST https://<host>/api/v1/registry/drift/<event-id>/resolve \
  -H "Authorization: Bearer <developer-token>" \
  -H "Content-Type: application/json" \
  -d '{"resolution_notes": "<describe what was done>"}'
```

If the model was auto-disabled and you want to re-enable it immediately (without waiting
for auto-recovery):

```bash
curl -X DELETE https://<host>/api/v1/registry/overrides/<drift-engine-override-id> \
  -H "Authorization: Bearer <admin-token>"
```

### 5. Post-incident verification

Confirm model is healthy and routing again:

```bash
curl https://<host>/api/v1/models \
  -H "Authorization: Bearer <any-valid-token>"
```

Confirm drift event is resolved:

```bash
curl https://<host>/api/v1/registry/drift \
  -H "Authorization: Bearer <admin-token>"
# Should show drift_status: auto_resolved or manually_resolved
```

---

## Notes

- The drift engine writes `active_revision_id` into each drift event at detection time.
  This field links the event to the exact registry state active when the drift was observed —
  useful for correlating drift events with recent revision promotions.
- Warning-level drift escalates the model to Tier A probe priority (probed every cycle)
  but does NOT create a disable override.
- The rolling window for auto-recovery is `_RECOVERY_HEALTHY_CYCLES = 3` consecutive
  healthy probe results. Change this constant in `tidus/sync/drift/engine.py` if needed.
