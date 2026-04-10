"""Unit tests for BillingCsvParser.

Covers:
  - Valid CSV parses correctly
  - Missing required columns raises BillingParseError
  - Invalid date format raises BillingParseError
  - Invalid float raises BillingParseError
  - Negative cost raises BillingParseError
  - Empty file raises BillingParseError
  - UTF-8 BOM stripped transparently
  - Duplicate (model_id, date) — last row wins
  - Rows with empty model_id are skipped
"""

from __future__ import annotations

from datetime import date

import pytest

from tidus.billing.csv_parser import BillingParseError, parse

VALID_CSV = """\
model_id,date,provider_cost_usd
gpt-4o,2026-04-01,89.20
claude-opus-4-6,2026-04-01,127.45
"""


def test_valid_csv_returns_rows():
    rows = parse(VALID_CSV)
    assert len(rows) == 2
    assert rows[0].model_id == "gpt-4o"
    assert rows[0].date == date(2026, 4, 1)
    assert abs(rows[0].provider_cost_usd - 89.20) < 1e-6


def test_valid_csv_bytes():
    rows = parse(VALID_CSV.encode("utf-8"))
    assert len(rows) == 2


def test_utf8_bom_stripped():
    bom_csv = b"\xef\xbb\xbf" + VALID_CSV.encode("utf-8")
    rows = parse(bom_csv)
    assert len(rows) == 2
    assert rows[0].model_id == "gpt-4o"


def test_missing_column_raises():
    bad_csv = "model_id,date\ngpt-4o,2026-04-01\n"
    with pytest.raises(BillingParseError, match="missing required columns"):
        parse(bad_csv)


def test_empty_file_raises():
    with pytest.raises(BillingParseError, match="empty"):
        parse("")


def test_empty_bytes_raises():
    with pytest.raises(BillingParseError, match="empty"):
        parse(b"")


def test_invalid_date_raises():
    bad = "model_id,date,provider_cost_usd\ngpt-4o,not-a-date,10.0\n"
    with pytest.raises(BillingParseError, match="invalid date"):
        parse(bad)


def test_invalid_float_raises():
    bad = "model_id,date,provider_cost_usd\ngpt-4o,2026-04-01,not_a_number\n"
    with pytest.raises(BillingParseError, match="invalid provider_cost_usd"):
        parse(bad)


def test_negative_cost_raises():
    bad = "model_id,date,provider_cost_usd\ngpt-4o,2026-04-01,-5.00\n"
    with pytest.raises(BillingParseError, match="non-negative"):
        parse(bad)


def test_zero_cost_allowed():
    csv = "model_id,date,provider_cost_usd\ngpt-4o,2026-04-01,0.0\n"
    rows = parse(csv)
    assert len(rows) == 1
    assert rows[0].provider_cost_usd == 0.0


def test_duplicate_model_date_last_row_wins():
    csv = (
        "model_id,date,provider_cost_usd\n"
        "gpt-4o,2026-04-01,10.0\n"
        "gpt-4o,2026-04-01,20.0\n"  # duplicate — this one wins
    )
    rows = parse(csv)
    assert len(rows) == 1
    assert rows[0].provider_cost_usd == 20.0


def test_empty_model_id_rows_skipped():
    csv = (
        "model_id,date,provider_cost_usd\n"
        ",2026-04-01,10.0\n"          # empty model_id — skipped
        "gpt-4o,2026-04-01,89.20\n"
    )
    rows = parse(csv)
    assert len(rows) == 1
    assert rows[0].model_id == "gpt-4o"


def test_column_order_independent():
    csv = "date,provider_cost_usd,model_id\n2026-04-01,50.0,gpt-4o\n"
    rows = parse(csv)
    assert len(rows) == 1
    assert rows[0].model_id == "gpt-4o"
