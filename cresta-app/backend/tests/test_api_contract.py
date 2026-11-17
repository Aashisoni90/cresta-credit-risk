"""
Contract tests for the CRESTA API (Phase 4).

These are not smoke tests. Each one pins a property the project's invariants
depend on, so that a later refactor cannot quietly undo them:

  * Track B must never return a number (it would be synthetic)
  * financial-sector issuers must answer SECTOR_UNSUPPORTED, not a score
  * "out of scope" and "no data" must stay distinguishable
  * SHAP attributions must add up to the probability shown beside them
  * risk bands must not contradict the flag shown next to them
  * the human-in-the-loop disclaimer must travel with every estimate
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main                                    # noqa: E402
from services import real_data                 # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def state(client):
    return real_data.state()


@pytest.fixture(scope="module")
def live_ticker(state):
    m = state["matrix"]
    return m[m.fiscal_year_num == m.fiscal_year_num.max()] \
        .nlargest(1, "probability").iloc[0]["ticker"]


@pytest.fixture(scope="module")
def financial_ticker(state):
    return sorted(state["excluded_tickers"])[0]


# ── service identity ──────────────────────────────────────────────────
def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_serves_the_real_model_not_the_synthetic_one(client):
    info = client.get("/api/model/info").json()
    assert info["version"].endswith("-real"), \
        "API must serve a real-data model, never the synthetic artefact"
    # Asserted against the artefact, not a literal: a hardcoded 16 failed the
    # day the §5b look-ahead removal took the count to 14, which is a change
    # the pipeline is allowed to make and this test is not here to police.
    assert info["n_features"] == len(info["features"])
    assert info["n_features"] > 0
    assert info["track_b_available"] is False


def test_no_point_in_time_feature_is_served(client):
    """§7.4 / invariant 2. cur_mkt_cap and volatility_260d are real-time
    Bloomberg market fields: the EQY_FUND_YEAR override does not apply to them,
    so they carry the pull-date value on every historical row. Serving either
    one means the API is publishing a look-ahead score."""
    served = set(client.get("/api/model/info").json()["features"])
    banned = {"log_mkt_cap", "equity_volatility"}
    leaked = banned & served
    assert not leaked, f"look-ahead features reached the served model: {leaked}"


def test_registry_records_the_served_version(client):
    r = client.get("/api/model/registry").json()
    assert r["serving"] in [v["version"] for v in r["versions"]]


# ── invariant 6: Track A works with zero ESG, always ──────────────────
def test_track_b_never_returns_a_number(client, live_ticker):
    d = client.post("/api/predict", json={"ticker": live_ticker}).json()
    assert d["model_b"] is None, \
        "a Model B number here would be synthetic — see TrackBUnavailable"
    assert d["track_b_status"]["available"] is False
    assert d["track_b_status"]["code"] == "TRACK_B_UNAVAILABLE"


def test_model_manager_refuses_track_b():
    from ml.model_manager import get_model_b, TrackBUnavailable
    with pytest.raises(TrackBUnavailable):
        get_model_b()


def test_esg_fields_are_null_not_fabricated(client, live_ticker):
    detail = client.get(f"/api/companies/{live_ticker}").json()
    assert detail.get("esg") is None
    assert client.get("/api/data-quality").json()["esg_coverage"] is None


# ── invariant 5 / §5.6: financial-sector issuers are out of scope ─────
def test_financial_issuer_is_sector_unsupported(client, financial_ticker):
    r = client.post("/api/predict", json={"ticker": financial_ticker})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "SECTOR_UNSUPPORTED"


def test_out_of_scope_is_distinct_from_no_data(client, financial_ticker):
    """A bank and a typo must not produce the same answer."""
    a = client.post("/api/predict", json={"ticker": financial_ticker})
    b = client.post("/api/predict", json={"ticker": "ZZZNOTREAL IN Equity"})
    assert a.status_code == 422 and b.status_code == 404
    assert a.json()["detail"]["code"] != b.json()["detail"]["code"]


def test_sector_unsupported_also_blocks_explanations(client, financial_ticker):
    for url in (f"/api/predict/explanation?ticker={financial_ticker}",
                f"/api/predict/counterfactual?ticker={financial_ticker}"):
        assert client.get(url).status_code == 422


# ── prediction semantics ──────────────────────────────────────────────
def test_prediction_states_its_horizon(client, live_ticker):
    d = client.post("/api/predict", json={"ticker": live_ticker}).json()
    assert str(d["fiscal_year"] + 1) in d["prediction_horizon"], \
        "a probability must never be readable as a statement about year t"


def test_every_estimate_carries_the_human_in_the_loop_disclaimer(client, live_ticker):
    d = client.post("/api/predict", json={"ticker": live_ticker}).json()
    assert "human review" in d["disclaimer"].lower()
    assert "not investment advice" in d["disclaimer"].lower()


def test_threshold_basis_is_stated(client, live_ticker):
    d = client.post("/api/predict", json={"ticker": live_ticker}).json()
    assert "cost" in d["threshold_basis"].lower()


def test_flag_and_risk_band_never_contradict(client, state):
    """A company the model flags must not be labelled 'low risk'."""
    rows = client.get("/api/predict/portfolio?limit=200").json()["companies"]
    for r in rows:
        if r["flagged"]:
            assert r["risk_category"] != "low", \
                f"{r['ticker']} is flagged but banded 'low'"


def test_portfolio_is_sorted_by_risk(client):
    p = [c["probability"] for c in
         client.get("/api/predict/portfolio?limit=50").json()["companies"]]
    assert p == sorted(p, reverse=True)


# ── §8.8 explainability ───────────────────────────────────────────────
def test_shap_additivity_holds_per_request(client, live_ticker):
    e = client.get(f"/api/predict/explanation?ticker={live_ticker}").json()
    assert e["additivity_ok"], \
        "the waterfall must add up to the probability displayed beside it"
    assert e["additivity_error"] < 1e-3


def test_explainability_responses_carry_the_causation_caveat(client, live_ticker):
    for url in ("/api/model/shap/global", "/api/model/shap/blocks",
                f"/api/predict/explanation?ticker={live_ticker}"):
        assert "not the world" in client.get(url).json()["caveat"]


def test_counterfactual_peer_is_actually_healthy(client, live_ticker, state):
    q = client.get(f"/api/predict/counterfactual?ticker={live_ticker}").json()
    assert q["peer"]["probability"] < state["threshold"], \
        "the contrast must be against a healthy peer, not a lower-scored one"


# ── §8.5 / §12.4 evidence surface ─────────────────────────────────────
def test_metrics_disclose_the_persistence_inflation(client):
    d = client.get("/api/model/metrics").json()
    assert "persistence" in d["headline_caveat"].lower(), \
        "serving pooled AUC without the persistence caveat overstates the model"


def test_early_warning_stratum_is_reported(client):
    names = [s["stratum"] for s in client.get("/api/model/strata").json()["strata"]]
    assert "healthy_at_t_early_warning" in names


def test_rolling_origin_reports_a_distribution(client):
    d = client.get("/api/model/rolling-origin").json()
    assert len(d["folds"]) >= 4
    assert d["summary"]["xgboost"]["sd"] >= 0


def test_threshold_sweep_includes_the_operating_point(client):
    d = client.get("/api/threshold?n_points=20").json()
    assert any(r["is_operating_point"] for r in d["sweep"])
    assert d["cost_ratio_fn_fp"] == 10.0


# ── data quality (§12.5) ──────────────────────────────────────────────
def test_data_quality_states_the_no_zero_fill_policy(client):
    d = client.get("/api/data-quality").json()
    assert "never zero-filled" in d["missingness_policy"].lower()
    assert d["universe"]["n_sector_unsupported_tickers"] > 0


def test_failed_tickers_are_reported_not_hidden(client):
    assert client.get("/api/data-quality").json()["n_failed_tickers"] > 0
