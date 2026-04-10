"""Tests for registry_metrics.py — metric names, label sets, and hash function."""



def test_gauge_metric_names():
    from tidus.observability.registry_metrics import (
        REGISTRY_ACTIVE_REVISION_ID,
        REGISTRY_ACTIVE_REVISION_TS,
        REGISTRY_LAST_SYNC_TS,
        REGISTRY_MODEL_CONFIDENCE,
        REGISTRY_MODEL_PRICE_UPDATE_TS,
        REGISTRY_STALE_MODEL_COUNT,
    )

    assert REGISTRY_LAST_SYNC_TS._name == "tidus_registry_last_successful_sync_timestamp"
    assert REGISTRY_ACTIVE_REVISION_TS._name == "tidus_registry_active_revision_activated_timestamp"
    assert REGISTRY_MODEL_PRICE_UPDATE_TS._name == "tidus_registry_model_last_price_update_timestamp"
    assert REGISTRY_MODEL_CONFIDENCE._name == "tidus_registry_model_confidence"
    assert REGISTRY_ACTIVE_REVISION_ID._name == "tidus_registry_active_revision_id"
    assert REGISTRY_STALE_MODEL_COUNT._name == "tidus_registry_models_stale_count"


def test_counter_metric_names():
    """prometheus_client strips _total from _name; verify the base names are correct.

    The full wire name (e.g. tidus_probe_live_calls_total) is assembled at
    exposition time as {_name}_total.  Testing _name is sufficient and avoids
    a prometheus_client version sensitivity in how _total is handled.
    """
    from tidus.observability.registry_metrics import (
        DRIFT_EVENTS,
        PROBE_LIVE_CALLS,
        PROBE_SYNTHETIC_CALLS,
    )

    assert PROBE_LIVE_CALLS._name == "tidus_probe_live_calls"
    assert PROBE_SYNTHETIC_CALLS._name == "tidus_probe_synthetic_calls"
    assert DRIFT_EVENTS._name == "tidus_registry_drift_events"


def test_model_metrics_have_model_id_label():
    from tidus.observability.registry_metrics import (
        REGISTRY_MODEL_CONFIDENCE,
        REGISTRY_MODEL_PRICE_UPDATE_TS,
    )

    assert "model_id" in REGISTRY_MODEL_CONFIDENCE._labelnames
    assert "model_id" in REGISTRY_MODEL_PRICE_UPDATE_TS._labelnames


def test_probe_counters_have_model_id_and_result_labels():
    from tidus.observability.registry_metrics import PROBE_LIVE_CALLS, PROBE_SYNTHETIC_CALLS

    for metric in (PROBE_LIVE_CALLS, PROBE_SYNTHETIC_CALLS):
        assert "model_id" in metric._labelnames
        assert "result" in metric._labelnames


def test_drift_events_counter_labels():
    from tidus.observability.registry_metrics import DRIFT_EVENTS

    assert "model_id" in DRIFT_EVENTS._labelnames
    assert "drift_type" in DRIFT_EVENTS._labelnames
    assert "severity" in DRIFT_EVENTS._labelnames


def test_revision_id_to_int_deterministic():
    from tidus.observability.registry_metrics import revision_id_to_int

    revision_id = "12345678-1234-5678-1234-567812345678"
    result1 = revision_id_to_int(revision_id)
    result2 = revision_id_to_int(revision_id)
    assert result1 == result2


def test_revision_id_to_int_range():
    """Hash must stay within JS safe integer range (2**53)."""
    import uuid

    from tidus.observability.registry_metrics import revision_id_to_int

    for _ in range(20):
        h = revision_id_to_int(str(uuid.uuid4()))
        assert 0 <= h < 2**53


def test_revision_id_to_int_changes_on_different_ids():
    import uuid

    from tidus.observability.registry_metrics import revision_id_to_int

    ids = [str(uuid.uuid4()) for _ in range(10)]
    hashes = [revision_id_to_int(i) for i in ids]
    # All hashes should be distinct (collision extremely unlikely with 20-bit space)
    assert len(set(hashes)) == len(hashes)
