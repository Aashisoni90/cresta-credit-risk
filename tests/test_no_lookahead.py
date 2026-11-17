"""
test_no_lookahead.py
====================
Pins §7.4 / invariant 2 at the data level, not just the model level.

These tests exist because the defect they describe shipped. v1.1.0 trained
`log_mkt_cap` and `equity_volatility` as features for eleven fiscal years when
both were a single pull-date value repeated down every row. Nothing caught it:
the split assertion checks that the *years* do not overlap, which they did not,
while the leakage sat inside a column.

The general lesson is the shape of the check. An out-of-time split guarantees
nothing if a feature's value is dated outside its row's fiscal year, and that
is a property of the column, not of the split.
"""
import numpy as np
import pandas as pd
import pytest

from build_training_matrix import POINT_IN_TIME_FEATURES

MATRIX = "data/processed/training_matrix.csv"
PANEL = "data/processed/bloomberg_panel_scored_gated.csv"

# A field that honours Bloomberg's EQY_FUND_YEAR override varies year to year
# for essentially every firm. The two that do not varied for under 5%. The gate
# sits at 50%: far above anything a real fundamental series produces, far below
# the 95%+ the broken fields show, so it catches the defect without tripping on
# a genuinely sticky ratio.
MAX_CONSTANT_TICKER_SHARE = 0.50
MIN_YEARS_REPORTED = 5


@pytest.fixture(scope="module")
def matrix():
    return pd.read_csv(MATRIX, low_memory=False)


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(PANEL, low_memory=False)


def test_point_in_time_features_are_absent_from_the_matrix(matrix):
    leaked = [c for c in POINT_IN_TIME_FEATURES if c in matrix.columns]
    assert not leaked, (
        f"{leaked} reached the training matrix. These are real-time Bloomberg "
        f"market fields; EQY_FUND_YEAR does not apply to them, so each row "
        f"carries the pull-date value regardless of its fiscal year.")


def test_the_offending_panel_fields_are_still_detectably_constant(panel):
    """The source defect is a property of the extract, so it is still there.

    This test does not fail on it — it asserts that the diagnostic still fires,
    so that if a later BDH re-pull fixes the panel, this test fails loudly and
    whoever re-pulls knows the two features can be restored.
    """
    for field in ("cur_mkt_cap", "volatility_260d"):
        g = panel.groupby("ticker")[field]
        eligible = g.count() >= MIN_YEARS_REPORTED
        share = ((g.nunique() <= 1) & eligible).sum() / max(eligible.sum(), 1)
        assert share > 0.9, (
            f"{field} is no longer constant per ticker ({share:.1%}). If the "
            f"panel was re-pulled with BDH, remove {field}'s derived feature "
            f"from POINT_IN_TIME_FEATURES and retrain — the exclusion is now "
            f"costing real signal.")


@pytest.mark.parametrize("field", [
    "total_assets", "current_assets", "current_liabilities", "retained_earnings",
    "ebit", "total_equity_book", "total_liabilities", "roa", "cur_ratio",
    "net_income", "total_revenue", "cash_from_operations",
])
def test_every_surviving_fundamental_actually_varies_by_year(panel, field):
    """The complement of the test above: the fields we DID keep are historical.

    Without this, the fix is a pair of names on a list rather than a claim
    about the panel.
    """
    g = panel.groupby("ticker")[field]
    eligible = g.count() >= MIN_YEARS_REPORTED
    assert eligible.sum() > 100, f"too few tickers report {field} to judge"
    share = ((g.nunique() <= 1) & eligible).sum() / eligible.sum()
    assert share < MAX_CONSTANT_TICKER_SHARE, (
        f"{field} is constant across years for {share:.1%} of tickers — it "
        f"looks like a point-in-time pull, not a fundamental series.")


def test_split_is_out_of_time_and_ordered(matrix):
    """Invariant 2, restated at the matrix level rather than inside the trainer."""
    bounds = {s: (int(g.fiscal_year_num.min()), int(g.fiscal_year_num.max()))
              for s, g in matrix.groupby("split")}
    assert bounds["train"][1] < bounds["validation"][0]
    assert bounds["validation"][1] < bounds["test"][0]


def test_no_ticker_year_is_duplicated(matrix):
    """A duplicated firm-year can land on both sides of the split boundary."""
    dup = matrix.duplicated(subset=["ticker", "fiscal_year_num"]).sum()
    assert dup == 0, f"{dup} duplicated ticker-years in the matrix"
