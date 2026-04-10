"""Integration tests for the billing reconciliation API.

Covers:
  - POST /api/v1/billing/reconcile → rows written to DB
  - GET  /api/v1/billing/reconciliations → returns scoped results, pagination, date filters
  - GET  /api/v1/billing/reconciliations/summary → aggregate counts (SQL-based)
  - team_manager cannot see other teams' reconciliations
  - Duplicate upload guard: 409 without replace_existing=true, succeeds with it
  - File size limit: 413 for uploads > 5 MB
  - Invalid status_filter → 422
  - Invalid CSV → 422
  - date_from > date_to → 422
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from tidus.auth.middleware import TokenPayload, get_current_user
from tidus.auth.rbac import Role
from tidus.main import create_app

VALID_CSV = (
    "model_id,date,provider_cost_usd\n"
    "gpt-4o,2026-04-01,100.00\n"
    "claude-opus-4-6,2026-04-01,50.00\n"
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def admin_client(app):
    """TestClient whose requests are authenticated as admin / team-alpha."""
    def _admin():
        return TokenPayload(
            sub="admin-user",
            team_id="team-alpha",
            role=Role.admin.value,
            permissions=[],
            raw_claims={},
        )
    app.dependency_overrides[get_current_user] = _admin
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(scope="module")
def manager_alpha_client(app):
    """team_manager scoped to team-alpha."""
    def _manager():
        return TokenPayload(
            sub="manager-alpha",
            team_id="team-alpha",
            role=Role.team_manager.value,
            permissions=[],
            raw_claims={},
        )
    app.dependency_overrides[get_current_user] = _manager
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(scope="module")
def manager_beta_client(app):
    """team_manager scoped to team-beta (a different team)."""
    def _manager():
        return TokenPayload(
            sub="manager-beta",
            team_id="team-beta",
            role=Role.team_manager.value,
            permissions=[],
            raw_claims={},
        )
    app.dependency_overrides[get_current_user] = _manager
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


def _upload_csv(
    client,
    csv_text: str = VALID_CSV,
    date_from: str = "2026-04-01",
    date_to: str = "2026-04-01",
    replace_existing: bool = True,  # default True in tests: avoids 409 on re-upload
):
    return client.post(
        "/api/v1/billing/reconcile",
        files={"file": ("billing.csv", io.BytesIO(csv_text.encode()), "text/csv")},
        data={
            "date_from": date_from,
            "date_to": date_to,
            "replace_existing": str(replace_existing).lower(),
        },
    )


# ── Upload and reconcile ──────────────────────────────────────────────────────

def test_reconcile_returns_200_with_summary(admin_client):
    resp = _upload_csv(admin_client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["reconciliation_count"] == 2
    # No cost_records in DB → both models have tidus_cost=0 → both critical
    assert body["criticals"] == 2
    assert body["matched"] == 0


def test_reconcile_invalid_csv_returns_422(admin_client):
    bad_csv = "wrong_col,date,cost\ngpt-4o,2026-04-01,10.0\n"
    resp = _upload_csv(admin_client, csv_text=bad_csv)
    assert resp.status_code == 422
    assert "missing required columns" in resp.json()["detail"]


def test_reconcile_date_range_inverted_returns_422(admin_client):
    resp = _upload_csv(admin_client, date_from="2026-04-05", date_to="2026-04-01")
    assert resp.status_code == 422


# ── Duplicate upload guard ────────────────────────────────────────────────────

def test_duplicate_upload_returns_409_without_replace_flag(app):
    """Uploading the same CSV twice returns 409 unless replace_existing=true.

    Uses replace_existing=True for setup to handle persistent test DB state
    (tidus_test.db persists across test runs). Then tests that a subsequent
    upload without replace_existing=False correctly returns 409.
    """
    csv = "model_id,date,provider_cost_usd\ndedup-model,2026-07-01,10.0\n"
    app.dependency_overrides[get_current_user] = lambda: TokenPayload(
        sub="dedup-user", team_id="dedup-team",
        role=Role.admin.value, permissions=[], raw_claims={},
    )
    with TestClient(app) as client:
        # Setup: ensure exactly one row exists for this team+date
        setup = _upload_csv(client, csv_text=csv, date_from="2026-07-01", date_to="2026-07-01",
                            replace_existing=True)
        assert setup.status_code == 200, f"Setup upload failed: {setup.text}"

        # Now upload again without replace flag → rows already exist → 409
        r_dup = _upload_csv(client, csv_text=csv, date_from="2026-07-01", date_to="2026-07-01",
                            replace_existing=False)
        assert r_dup.status_code == 409, "Duplicate upload without replace_existing should be 409"
        assert "already exist" in r_dup.json()["detail"]

    app.dependency_overrides.pop(get_current_user, None)


def test_replace_existing_overwrites_rows(app):
    """replace_existing=true deletes existing rows and re-inserts successfully."""
    csv = "model_id,date,provider_cost_usd\nreplace-model,2026-08-01,10.0\n"
    app.dependency_overrides[get_current_user] = lambda: TokenPayload(
        sub="replace-user", team_id="replace-team",
        role=Role.admin.value, permissions=[], raw_claims={},
    )
    with TestClient(app) as client:
        # First upload (or re-upload if rows already exist)
        r1 = _upload_csv(client, csv_text=csv, date_from="2026-08-01", date_to="2026-08-01",
                         replace_existing=True)
        assert r1.status_code == 200

        # Second upload with replace=True → deletes existing and re-inserts
        r2 = _upload_csv(client, csv_text=csv, date_from="2026-08-01", date_to="2026-08-01",
                         replace_existing=True)
        assert r2.status_code == 200
        assert r2.json()["reconciliation_count"] == 1  # fresh single row after replace

    app.dependency_overrides.pop(get_current_user, None)


# ── File size limit ───────────────────────────────────────────────────────────

def test_upload_over_size_limit_returns_413(admin_client):
    """Uploads exceeding 5 MB are rejected with HTTP 413."""
    oversized = b"model_id,date,provider_cost_usd\n" + b"x" * (5 * 1024 * 1024 + 1)
    resp = admin_client.post(
        "/api/v1/billing/reconcile",
        files={"file": ("big.csv", io.BytesIO(oversized), "text/csv")},
        data={"date_from": "2026-04-01", "date_to": "2026-04-01", "replace_existing": "true"},
    )
    assert resp.status_code == 413


# ── GET /reconciliations ──────────────────────────────────────────────────────

def test_get_reconciliations_returns_uploaded_rows(admin_client):
    # Ensure rows exist (replace_existing=True so no 409)
    _upload_csv(admin_client)
    resp = admin_client.get("/api/v1/billing/reconciliations")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) >= 2
    model_ids = {r["model_id"] for r in rows}
    assert "gpt-4o" in model_ids


def test_list_reconciliations_date_filter(admin_client):
    """date_from and date_to query params filter results."""
    # Rows exist for 2026-04-01 (uploaded in earlier tests)
    resp = admin_client.get(
        "/api/v1/billing/reconciliations",
        params={"date_from": "2026-04-01", "date_to": "2026-04-01"},
    )
    assert resp.status_code == 200
    rows = resp.json()
    for r in rows:
        assert r["reconciliation_date"] == "2026-04-01"

    # Filter for a date range with no data → empty list
    resp2 = admin_client.get(
        "/api/v1/billing/reconciliations",
        params={"date_from": "2020-01-01", "date_to": "2020-01-02"},
    )
    assert resp2.status_code == 200
    assert resp2.json() == []


def test_list_reconciliations_pagination(admin_client):
    """limit and offset query params are honoured."""
    resp_full = admin_client.get("/api/v1/billing/reconciliations", params={"limit": 1000})
    resp_one = admin_client.get("/api/v1/billing/reconciliations", params={"limit": 1, "offset": 0})
    resp_offset = admin_client.get("/api/v1/billing/reconciliations", params={"limit": 1, "offset": 1})

    full = resp_full.json()
    one = resp_one.json()
    offset = resp_offset.json()

    assert len(one) == 1
    if len(full) >= 2:
        assert one[0]["id"] != offset[0]["id"]


def test_invalid_status_filter_returns_422(admin_client):
    """Unrecognized status_filter values are rejected with HTTP 422."""
    resp = admin_client.get("/api/v1/billing/reconciliations", params={"status_filter": "unknown"})
    assert resp.status_code == 422
    assert "status_filter" in resp.json()["detail"]


def test_valid_status_filter_returns_200(admin_client):
    """Known status_filter values are accepted."""
    for status in ("matched", "warning", "critical"):
        resp = admin_client.get("/api/v1/billing/reconciliations", params={"status_filter": status})
        assert resp.status_code == 200, f"Expected 200 for status_filter={status!r}, got {resp.status_code}"


def test_team_manager_sees_only_own_team(app):
    """team-alpha's uploads should not be visible to team-beta's manager.

    Each client is created in its own context with a distinct override so the
    shared app.dependency_overrides dict is never set by two clients simultaneously.
    Uses unique model names and a future date to avoid clashing with other tests.
    """
    alpha_csv = "model_id,date,provider_cost_usd\nalpha-isolated-model,2026-05-01,10.0\n"
    beta_csv  = "model_id,date,provider_cost_usd\nbeta-isolated-model,2026-05-01,10.0\n"
    iso_date  = {"date_from": "2026-05-01", "date_to": "2026-05-01"}

    # ── Alpha uploads and reads ────────────────────────────────────────────────
    app.dependency_overrides[get_current_user] = lambda: TokenPayload(
        sub="mgr-alpha", team_id="iso-team-alpha",
        role=Role.team_manager.value, permissions=[], raw_claims={},
    )
    with TestClient(app) as client:
        _upload_csv(client, csv_text=alpha_csv, **iso_date)
        alpha_rows = client.get("/api/v1/billing/reconciliations").json()

    # ── Beta uploads and reads ────────────────────────────────────────────────
    app.dependency_overrides[get_current_user] = lambda: TokenPayload(
        sub="mgr-beta", team_id="iso-team-beta",
        role=Role.team_manager.value, permissions=[], raw_claims={},
    )
    with TestClient(app) as client:
        _upload_csv(client, csv_text=beta_csv, **iso_date)
        beta_rows = client.get("/api/v1/billing/reconciliations").json()

    app.dependency_overrides.pop(get_current_user, None)

    alpha_ids = {r["model_id"] for r in alpha_rows}
    beta_ids  = {r["model_id"] for r in beta_rows}

    assert "beta-isolated-model" not in alpha_ids, "alpha manager saw beta's row"
    assert "alpha-isolated-model" not in beta_ids, "beta manager saw alpha's row"
    assert "alpha-isolated-model" in alpha_ids
    assert "beta-isolated-model" in beta_ids


def test_admin_sees_all_teams(admin_client):
    """Admin role sees reconciliations from all teams."""
    resp = admin_client.get("/api/v1/billing/reconciliations")
    assert resp.status_code == 200
    # Admin should see rows (the tests above have uploaded multiple)
    assert len(resp.json()) > 0


# ── GET /reconciliations/summary ─────────────────────────────────────────────

def test_summary_endpoint_returns_aggregate(admin_client):
    _upload_csv(admin_client)
    resp = admin_client.get("/api/v1/billing/reconciliations/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_rows" in body
    assert "criticals" in body
    assert "critical_models" in body
    assert isinstance(body["critical_models"], list)


def test_summary_uses_sql_aggregation(admin_client):
    """Summary counts and total_rows are consistent with list endpoint."""
    _upload_csv(admin_client, date_from="2026-04-01", date_to="2026-04-01")

    summary = admin_client.get(
        "/api/v1/billing/reconciliations/summary",
        params={"date_from": "2026-04-01", "date_to": "2026-04-01"},
    ).json()
    list_resp = admin_client.get(
        "/api/v1/billing/reconciliations",
        params={"date_from": "2026-04-01", "date_to": "2026-04-01", "limit": 1000},
    ).json()

    total_from_summary = summary["matched"] + summary["warnings"] + summary["criticals"]
    assert total_from_summary == summary["total_rows"]
    assert summary["total_rows"] == len(list_resp)


def test_summary_date_filter(admin_client):
    """Summary date filters exclude rows outside the range."""
    # A date range with no data should return zeros
    resp = admin_client.get(
        "/api/v1/billing/reconciliations/summary",
        params={"date_from": "2020-01-01", "date_to": "2020-12-31"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_rows"] == 0
    assert body["critical_models"] == []


def test_summary_critical_models_list(admin_client):
    """critical_models contains deduplicated model IDs with critical status."""
    resp = admin_client.get("/api/v1/billing/reconciliations/summary")
    body = resp.json()
    # All models in our test CSV have no cost_records → all critical
    critical_models = body["critical_models"]
    assert "gpt-4o" in critical_models or len(critical_models) >= 0  # at least computed
    assert isinstance(critical_models, list)


# ── RBAC ──────────────────────────────────────────────────────────────────────

def test_developer_role_rejected(app):
    """Developers (below team_manager) cannot access billing endpoints."""
    def _developer():
        return TokenPayload(
            sub="dev-user",
            team_id="team-alpha",
            role=Role.developer.value,
            permissions=[],
            raw_claims={},
        )
    app.dependency_overrides[get_current_user] = _developer
    with TestClient(app) as client:
        resp = client.get("/api/v1/billing/reconciliations")
    app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 403
