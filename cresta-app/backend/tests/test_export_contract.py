"""
Export contract (§12.6) and accessibility-adjacent guarantees.

An exported file outlives the session that made it. These tests pin the
properties that let someone reading the CSV in six months know what produced
it, and stop an export quietly becoming a decision record.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                   # noqa: E402
import main                                     # noqa: E402
from services import real_data                  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def portfolio_csv(client):
    r = client.get("/api/export/portfolio.csv?limit=25")
    assert r.status_code == 200
    return r


def _header(text):
    return [ln for ln in text.splitlines() if ln.startswith("#")]


def test_export_is_stamped_with_model_version_and_timestamp(portfolio_csv):
    """§12.6 states both explicitly."""
    h = "\n".join(_header(portfolio_csv.text))
    assert f"model_version: {real_data.state()['version']}" in h
    assert "generated_utc:" in h
    assert "trained_utc:" in h


def test_export_stamp_pins_the_exact_pipeline(portfolio_csv):
    """Config hash and commit, so the file is traceable to a build."""
    h = "\n".join(_header(portfolio_csv.text))
    reg = next(r for r in real_data.registry()
               if r["version"] == real_data.state()["version"])
    assert reg["config_hash"] in h
    assert "git_commit:" in h


def test_export_carries_the_threshold_and_cost_assumption(portfolio_csv):
    h = "\n".join(_header(portfolio_csv.text))
    assert "operating_threshold:" in h
    assert str(config.COST_RATIO_FN_FP) in h


def test_export_carries_the_human_review_constraint(portfolio_csv):
    """A spreadsheet circulating without §14.5 is how an estimate becomes a
    decision."""
    h = "\n".join(_header(portfolio_csv.text)).lower()
    assert "human review" in h
    assert "not investment advice" in h
    assert "not a credit rating" in h


def test_export_states_what_is_out_of_scope(portfolio_csv):
    h = "\n".join(_header(portfolio_csv.text)).lower()
    assert "financial-sector" in h          # §5.6 exclusion
    assert "not a realised default" in h    # the label is a proxy
    assert "unavailable" in h               # Track B


def test_export_parses_with_the_documented_recipe(portfolio_csv):
    """The preamble must not break the file it annotates."""
    df = pd.read_csv(io.StringIO(portfolio_csv.text), comment="#")
    assert len(df) == 25
    for col in ("ticker", "probability", "risk_band", "flagged", "threshold",
                "prediction_horizon", "data_quality_flags"):
        assert col in df.columns


def test_every_row_carries_the_threshold_it_was_flagged_against(portfolio_csv):
    """A filtered copy of the file must still say what the flag meant."""
    df = pd.read_csv(io.StringIO(portfolio_csv.text), comment="#")
    import numpy as np
    thr = real_data.state()["threshold"]
    # Series == pytest.approx does not compare elementwise; use allclose.
    assert np.allclose(df["threshold"].to_numpy(), thr)
    assert ((df["probability"] >= thr).astype(int) == df["flagged"]).all()


def test_export_is_never_cached(portfolio_csv):
    assert "no-store" in portfolio_csv.headers["Cache-Control"]


def test_export_downloads_as_a_file(portfolio_csv):
    assert "attachment" in portfolio_csv.headers["Content-Disposition"]
    assert portfolio_csv.headers["content-type"].startswith("text/csv")


def test_financial_issuer_cannot_be_exported(client):
    fin = sorted(real_data.state()["excluded_tickers"])[0]
    r = client.get(f"/api/export/company.csv?ticker={fin}")
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "SECTOR_UNSUPPORTED"


def test_company_export_includes_the_model_inputs(client):
    m = real_data.state()["matrix"]
    tk = m.nlargest(1, "probability").iloc[0]["ticker"]
    r = client.get(f"/api/export/company.csv?ticker={tk}")
    assert r.status_code == 200
    df = pd.read_csv(io.StringIO(r.text), comment="#")
    feats = real_data.state()["features"]
    present = [f for f in feats if f in df.columns]
    assert len(present) == len(feats), (
        "the company export must carry the model's own inputs so the estimate "
        "can be audited rather than taken on trust")


def test_manifest_matches_the_csv_stamp(client, portfolio_csv):
    man = client.get("/api/export/manifest").json()
    h = "\n".join(_header(portfolio_csv.text))
    assert man["model_version"] in h
    assert man["config_hash"] in h
