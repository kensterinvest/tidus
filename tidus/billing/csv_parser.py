"""BillingCsvParser — validates and parses normalized provider billing CSVs.

Expected format (UTF-8 or UTF-8-BOM, header row required):

    model_id,date,provider_cost_usd
    claude-opus-4-6,2026-04-01,127.45
    gpt-4o,2026-04-01,89.20

Rules:
  - Exactly three columns in any order (model_id, date, provider_cost_usd)
  - date must be ISO 8601 (YYYY-MM-DD)
  - provider_cost_usd must be a non-negative float
  - Rows with empty model_id are skipped
  - Duplicate (model_id, date) pairs: last row wins
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date


_REQUIRED_COLUMNS = {"model_id", "date", "provider_cost_usd"}


class BillingParseError(ValueError):
    """Raised when the CSV is malformed and cannot be parsed at all."""


@dataclass
class BillingRow:
    model_id: str
    date: date
    provider_cost_usd: float


def parse(content: bytes | str) -> list[BillingRow]:
    """Parse a normalized billing CSV.

    Args:
        content: Raw bytes (handles UTF-8 BOM) or a string.

    Returns:
        List of BillingRow objects; duplicates resolved (last row wins).

    Raises:
        BillingParseError: CSV is empty, missing required columns, or has
            unparseable values.
    """
    if isinstance(content, bytes):
        # Strip UTF-8 BOM if present
        content = content.lstrip(b"\xef\xbb\xbf").decode("utf-8")

    content = content.strip()
    if not content:
        raise BillingParseError("CSV file is empty")

    reader = csv.DictReader(io.StringIO(content))

    if reader.fieldnames is None:
        raise BillingParseError("CSV has no header row")

    actual_columns = {f.strip().lower() for f in reader.fieldnames if f}
    missing = _REQUIRED_COLUMNS - actual_columns
    if missing:
        raise BillingParseError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    rows: dict[tuple[str, date], BillingRow] = {}
    for line_num, raw in enumerate(reader, start=2):
        # Normalise column names to lowercase
        row = {k.strip().lower(): (v or "").strip() for k, v in raw.items()}

        model_id = row.get("model_id", "")
        if not model_id:
            continue

        date_str = row.get("date", "")
        try:
            parsed_date = date.fromisoformat(date_str)
        except ValueError:
            raise BillingParseError(
                f"Line {line_num}: invalid date '{date_str}' — expected YYYY-MM-DD"
            )

        cost_str = row.get("provider_cost_usd", "")
        try:
            cost = float(cost_str)
        except (ValueError, TypeError):
            raise BillingParseError(
                f"Line {line_num}: invalid provider_cost_usd '{cost_str}'"
            )
        if cost < 0:
            raise BillingParseError(
                f"Line {line_num}: provider_cost_usd must be non-negative, got {cost}"
            )

        rows[(model_id, parsed_date)] = BillingRow(
            model_id=model_id,
            date=parsed_date,
            provider_cost_usd=cost,
        )

    return list(rows.values())
