# Tidus Monitoring

## Alerting Rules

`alerting-rules.yaml` contains Prometheus alerting rules for the Tidus registry.

### Setup

Add the rules file to your Prometheus configuration:

```yaml
# prometheus.yml
rule_files:
  - "alerting-rules.yaml"
```

Verify the rules are valid before loading:

```bash
promtool check rules monitoring/alerting-rules.yaml
```

### Alert Summary

| Alert | Severity | Condition |
|---|---|---|
| `TidusRegistrySyncStale` | warning | No successful sync in >2 days |
| `TidusRegistrySyncCriticallyStale` | critical | No successful sync in >7 days |
| `TidusRegistryStaleModelCount` | warning | >10 models with price data >8 days old |
| `TidusActiveRevisionAged` | info | Active revision older than 30 days |
| `TidusProbeHighFailureRate` | warning | Live probe failure rate >50% for 5 min |

## Grafana Dashboard

`grafana-dashboard.json` contains the Tidus Grafana dashboard.

### Import

1. Open Grafana → Dashboards → Import
2. Upload `grafana-dashboard.json`
3. Select your Prometheus data source

### Registry Health Panels

The dashboard includes a **Registry Health** panel group with:

- **Sync Age** — time since last successful pricing sync
  (`tidus_registry_last_successful_sync_timestamp`)
- **Stale Model Count** — number of models with stale price data
  (`tidus_registry_models_stale_count`)
- **Model Confidence Distribution** — histogram of per-model confidence scores
  (`tidus_registry_model_confidence`)
- **Active Revision Age** — time since the active revision was promoted
  (`tidus_registry_active_revision_activated_timestamp`)
- **Probe Call Rate** — live vs synthetic probe calls per minute
  (`tidus_probe_live_calls_total`, `tidus_probe_synthetic_calls_total`)
- **Probe Failure Rate** — fraction of live probes returning failure
  (`tidus_probe_live_calls_total{result="fail"}`)

## Metrics Reference

### Gauges (refreshed every 5 minutes by MetricsUpdater)

| Metric | Labels | Description |
|---|---|---|
| `tidus_registry_last_successful_sync_timestamp` | — | Unix timestamp of last successful price sync |
| `tidus_registry_active_revision_activated_timestamp` | — | Unix timestamp when active revision was promoted |
| `tidus_registry_model_last_price_update_timestamp` | `model_id` | Unix timestamp of last price update for model |
| `tidus_registry_model_confidence` | `model_id` | Price data confidence: 1.0 (fresh) or 0.5 (stale >8 days) |
| `tidus_registry_active_revision_id` | — | Deterministic integer hash of active revision UUID |
| `tidus_registry_models_stale_count` | — | Count of models with stale price data |

### Counters (incremented at point of operation)

| Metric | Labels | Description |
|---|---|---|
| `tidus_probe_live_calls_total` | `model_id`, `result` | Live health probe calls; result=`success`\|`fail` |
| `tidus_probe_synthetic_calls_total` | `model_id`, `result` | Synthetic probe calls; result=`success`\|`fail` |
| `tidus_registry_drift_events_total` | `model_id`, `drift_type`, `severity` | Drift events detected |

### Scrape Interval

Recommended Prometheus scrape interval: **30 seconds**.
MetricsUpdater refreshes Gauges every 5 minutes — more frequent scraping wastes resources
without improving data freshness.
