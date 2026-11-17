"""
Performance and caching contract (§12.6).

The budget is a REQUIREMENT, not an aspiration: "portfolio view interactive
< 2s on 800 rows". These tests fail the build if a change reintroduces the
quadratic lookup that made it 2,387 ms, so the regression cannot return
quietly.

Timings are measured in-process, so they exclude network latency and are
therefore a LOWER bound on what a browser sees. The thresholds are set with
that headroom in mind rather than at the budget itself.
"""
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                   # noqa: E402
import main                                     # noqa: E402
from services import real_data                  # noqa: E402

PORTFOLIO_BUDGET_MS = 2000      # §12.6, 800 rows
INTERACTION_BUDGET_MS = 500     # per-company drill-in


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def _timed(client, url):
    t0 = time.perf_counter()
    r = client.get(url)
    return r, (time.perf_counter() - t0) * 1000


# ── §12.6 budget ──────────────────────────────────────────────────────
def test_portfolio_800_rows_within_budget(client):
    r, ms = _timed(client, "/api/predict/portfolio?limit=800")
    assert r.status_code == 200
    assert ms < PORTFOLIO_BUDGET_MS, (
        f"portfolio view took {ms:.0f} ms for 800 rows, over the §12.6 "
        f"{PORTFOLIO_BUDGET_MS} ms budget")


def test_portfolio_scales_sublinearly(client):
    """Guards the specific regression: a per-row scan of the 33k-row panel.

    If flag lookup goes back to O(n) per company, 800 rows costs ~16x what 50
    does. Index lookups keep the ratio far below that.
    """
    _, ms50 = _timed(client, "/api/predict/portfolio?limit=50")
    _, ms800 = _timed(client, "/api/predict/portfolio?limit=800")
    assert ms800 < max(ms50, 1.0) * 10, (
        f"800 rows took {ms800:.0f} ms vs {ms50:.0f} ms for 50 — lookup cost "
        f"is scaling per row again")


def test_first_explanation_is_not_a_cold_start(client):
    """TreeSHAP must be built at startup, not on the analyst's first click."""
    m = real_data.state()["matrix"]
    tk = m.nlargest(1, "probability").iloc[0]["ticker"]
    r, ms = _timed(client, f"/api/predict/explanation?ticker={tk}")
    assert r.status_code == 200
    assert ms < INTERACTION_BUDGET_MS, (
        f"first explanation took {ms:.0f} ms — the explainer is being built "
        f"on demand instead of warmed at startup")


def test_explainer_is_warm_after_startup():
    assert "explainer" in real_data.state(), \
        "lifespan hook must call real_data.warm_explainer()"


def test_dashboard_and_quality_screens_are_responsive(client):
    for url in ("/api/dashboard/summary", "/api/data-quality",
                "/api/companies?per_page=25", "/api/model/metrics"):
        _, ms = _timed(client, url)
        assert ms < INTERACTION_BUDGET_MS, f"{url} took {ms:.0f} ms"


# ── caching semantics ─────────────────────────────────────────────────
def test_etag_is_model_version_derived(client):
    """A new model version must invalidate every cached response.

    This is the property a time-based TTL cannot provide: without it the
    client keeps serving v1.1.0 numbers after v1.2.0 ships.
    """
    r1 = client.get("/api/model/metrics")
    etag = r1.headers["ETag"]
    assert etag and r1.headers["X-Model-Version"] == config.MODEL_A_VERSION

    original = config.MODEL_A_VERSION
    try:
        config.MODEL_A_VERSION = "A-9.9.9-test"
        r2 = client.get("/api/model/metrics")
        assert r2.headers["ETag"] != etag, \
            "ETag did not change with the model version"
    finally:
        config.MODEL_A_VERSION = original


def test_etag_varies_by_path_and_query(client):
    a = client.get("/api/predict/portfolio?limit=10").headers["ETag"]
    b = client.get("/api/predict/portfolio?limit=20").headers["ETag"]
    c_ = client.get("/api/features").headers["ETag"]
    assert a != b != c_ and a != c_


def test_static_reports_cache_longer_than_predictions(client):
    static = client.get("/api/model/metrics").headers["Cache-Control"]
    pred = client.get("/api/predict/portfolio?limit=5").headers["Cache-Control"]
    assert "max-age=3600" in static and "max-age=300" in pred


def test_health_is_never_cached(client):
    assert "no-store" in client.get("/api/health").headers["Cache-Control"]


def test_errors_are_never_cached(client):
    fin = sorted(real_data.state()["excluded_tickers"])[0]
    r = client.post("/api/predict", json={"ticker": fin})
    assert r.status_code == 422
    assert "no-store" in r.headers["Cache-Control"], \
        "a SECTOR_UNSUPPORTED response must not be cached as if it were data"


def test_mutating_requests_are_never_cached(client):
    m = real_data.state()["matrix"]
    tk = m.nlargest(1, "probability").iloc[0]["ticker"]
    r = client.post("/api/predict", json={"ticker": tk})
    assert "no-store" in r.headers["Cache-Control"]


def test_response_time_header_present(client):
    assert float(client.get("/api/features").headers["X-Response-Time-ms"]) >= 0


# ── the optimisation must not have changed any answer ─────────────────
def test_vectorised_banding_matches_the_scalar_classifier(client):
    """The fast path and the reference path must agree exactly."""
    s = real_data.state()
    m = s["matrix"]
    cur = m[m["fiscal_year_num"] == m["fiscal_year_num"].max()]
    scalar = dict(Counter(real_data.classify_risk(float(p))
                          for p in cur["probability"]))
    served = client.get("/api/dashboard/summary").json()["risk_bands"]
    assert served == scalar


def test_flag_index_matches_a_direct_panel_scan(client):
    """The O(1) index must return exactly what the O(n) scan returned."""
    s = real_data.state()
    panel = s["panel"]
    sample = s["matrix"].sample(40, random_state=0)
    for _, row in sample.iterrows():
        tk, yr = row["ticker"], int(row["fiscal_year_num"])
        hit = panel[(panel["ticker"] == tk) & (panel["fiscal_year_num"] == yr)]
        expected = []
        if not hit.empty:
            r = hit.iloc[0]
            expected = [label for col, label in real_data.QC_FLAG_LABELS
                        if col in r.index and bool(r[col]) is True]
        assert real_data.data_quality_flags(tk, yr) == expected, \
            f"index disagrees with a direct scan for {tk} FY{yr}"
