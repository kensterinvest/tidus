from types import SimpleNamespace

from tidus.sync.anthropic_client import SyncTokenLedger, build_sync_anthropic_client


def _usage(inp, out):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def test_ledger_aggregates_and_estimates():
    led = SyncTokenLedger()
    led.record("discovery", _usage(1_000_000, 200_000), web_searches=3)
    led.record("magazine", _usage(500_000, 100_000))
    s = led.summary()
    assert s["total_input_tokens"] == 1_500_000
    assert s["total_output_tokens"] == 300_000
    assert s["web_searches"] == 3
    # 1.5M in @ $3/1M + 0.3M out @ $15/1M + 3 searches @ $0.01
    assert round(led.estimated_usd(), 2) == round(4.5 + 4.5 + 0.03, 2)


def test_ledger_budget_guard():
    led = SyncTokenLedger()
    led.record("discovery", _usage(10_000_000, 10_000_000))
    assert led.over_budget(2.00) is True
    assert SyncTokenLedger().over_budget(2.00) is False


def test_factory_returns_none_without_key(monkeypatch):
    # Patch the name as bound inside anthropic_client's own module — patching
    # tidus.settings.get_settings would not affect this module's already-imported
    # reference, and could accidentally pass if a real sync key happens to be set
    # in the environment.
    import tidus.sync.anthropic_client as client_mod

    monkeypatch.setattr(
        client_mod, "get_settings", lambda: SimpleNamespace(tidus_sync_anthropic_key="")
    )
    assert build_sync_anthropic_client() is None
