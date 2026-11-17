"""
altman_zscore.py
================
Computes the Altman Z" (Z double-prime) score — the emerging-market /
non-manufacturing variant of Altman's distress score. This is the correct
variant for this project (see Foundations Guide, Section 2.2): it excludes
the asset-turnover ratio (X5) that the original 1968 model needed, because
turnover varies too much across industries/countries to be reliable outside
US manufacturing.

Formula (Altman, Hartzell & Peck, 1995):
    Z" = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Book Value of Equity / Total Liabilities
         (NOTE: this uses BOOK value of equity, not market value — unlike
         the original 1968 model's X4. Verify which convention your source
         paper uses if comparing published cutoffs; some emerging-market
         adaptations use market value instead. State explicitly which one
         you use in your methodology.)

    EMS (emerging market score) = Z" + 3.25   (maps onto a rating-equivalent scale)

INPUT: expects a pandas DataFrame with these columns (rename your Bloomberg
export to match, or edit COLUMN_MAP below):
    total_assets, current_assets, current_liabilities, retained_earnings,
    ebit, total_equity_book, total_liabilities

⚠️ SCALING TRAP (Foundations Guide §2.2): this implementation expects ratios
as DECIMALS (e.g. 0.20, not 20). If your ratios come out of Bloomberg as
percentages, divide by 100 before calling these functions. Sanity-check a
few known companies' scores against expected zones before trusting the
column at scale.

⚠️ NEGATIVE BOOK EQUITY (confirmed on real data — Suzlon FY2015-FY2021):
X4 = total_equity_book / total_liabilities goes NEGATIVE when book equity is
negative. That is CORRECT and is deliberately preserved: a firm whose
liabilities exceed its assets should score deep in the distress zone, and
suppressing that would remove the single clearest balance-sheet signal in
the sample. What is NOT safe is dividing by a non-positive denominator, so
X4 is set to NaN when total_liabilities <= 0 (see _safe_divide below).
Rows with negative equity are flagged in `altman_flag_negative_equity` so
you can report on them and so `altman_flag_check_scaling` does not confuse
legitimate deep distress with a units error upstream.
"""

import pandas as pd
import numpy as np


COLUMN_MAP = {
    "total_assets": "total_assets",
    "current_assets": "current_assets",
    "current_liabilities": "current_liabilities",
    "retained_earnings": "retained_earnings",
    "ebit": "ebit",
    "total_equity_book": "total_equity_book",
    "total_liabilities": "total_liabilities",
}

REQUIRED_COLUMNS = list(COLUMN_MAP.values())


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for Z\" calculation: {missing}\n"
            f"Expected columns (rename your data or edit COLUMN_MAP): {REQUIRED_COLUMNS}"
        )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divides, returning NaN wherever the denominator is <= 0 or missing.

    A non-positive denominator does not just risk inf — it silently FLIPS the
    sign of the ratio, which for X4 turns a solvency measure into nonsense
    (and for a distressed firm, into a spuriously *favourable* number). NaN
    is the honest answer; the row is then visible in sanity_check_report()
    instead of quietly polluting the Z" column.

    Note the numerator is deliberately unconstrained: a NEGATIVE numerator
    (negative book equity, negative retained earnings, negative EBIT) is real,
    meaningful and must flow through to the score.
    """
    denom = denominator.where(denominator > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denom


def compute_altman_zscore_components(df: pd.DataFrame) -> pd.DataFrame:
    """Adds X1-X4, Z'', EMS, a zone label and diagnostic flags. Returns a copy."""
    _check_columns(df)
    out = df.copy()

    out["X1_working_cap_to_assets"] = _safe_divide(
        out["current_assets"] - out["current_liabilities"], out["total_assets"]
    )
    out["X2_retained_earnings_to_assets"] = _safe_divide(
        out["retained_earnings"], out["total_assets"]
    )
    out["X3_ebit_to_assets"] = _safe_divide(out["ebit"], out["total_assets"])
    # X4: negative equity flows through (correctly punitive); only a
    # non-positive LIABILITIES denominator is treated as uncomputable.
    out["X4_equity_to_liabilities"] = _safe_divide(
        out["total_equity_book"], out["total_liabilities"]
    )

    out["altman_z_double_prime"] = (
        6.56 * out["X1_working_cap_to_assets"]
        + 3.26 * out["X2_retained_earnings_to_assets"]
        + 6.72 * out["X3_ebit_to_assets"]
        + 1.05 * out["X4_equity_to_liabilities"]
    )

    out["altman_ems_score"] = out["altman_z_double_prime"] + 3.25

    out["altman_zone"] = pd.cut(
        out["altman_z_double_prime"],
        bins=[-np.inf, 1.1, 2.6, np.inf],
        labels=["distress", "grey_zone", "safe"],
    )

    # ── Diagnostic flags — do NOT silently trust extreme values ──────────
    eq = out["total_equity_book"]
    out["altman_flag_negative_equity"] = eq.notna() & (eq < 0)

    # Keep a single negative_equity_flag column for modelling. If the
    # ingestion step (load_bloomberg_export.py) already created one, don't
    # clobber it — just verify the two agree, because a disagreement means
    # the equity column changed between ingestion and scoring.
    if "negative_equity_flag" in out.columns:
        derived = out["altman_flag_negative_equity"].astype(float)
        existing = pd.to_numeric(out["negative_equity_flag"], errors="coerce")
        both_present = derived.notna() & existing.notna()
        n_disagree = int((derived[both_present] != existing[both_present]).sum())
        if n_disagree:
            print(f"[altman_zscore] WARNING: negative_equity_flag disagrees with "
                  f"total_equity_book on {n_disagree} row(s) — the equity column has "
                  f"changed since ingestion. Investigate before trusting these scores.")
    else:
        out["negative_equity_flag"] = np.where(
            eq.isna(), np.nan, out["altman_flag_negative_equity"].astype(float)
        )

    out["altman_flag_x4_uncomputable"] = (
        out["total_equity_book"].notna() & out["X4_equity_to_liabilities"].isna()
    )  # equity was known but liabilities were <= 0 or missing

    out["altman_flag_extreme_value"] = out["altman_z_double_prime"].abs() > 50

    # An extreme score is only *suspicious* when negative equity doesn't
    # already explain it. A firm with equity of -3,500 against liabilities of
    # 17,500 legitimately scores far below the distress cutoff; flagging that
    # as a scaling error would send you hunting a bug that isn't there.
    out["altman_flag_check_scaling"] = (
        out["altman_flag_extreme_value"] & ~out["altman_flag_negative_equity"]
    )

    return out


def sanity_check_report(df: pd.DataFrame) -> None:
    """Prints a quick summary — run this on every new batch before trusting it."""
    n = len(df)
    n_valid = int(df["altman_z_double_prime"].notna().sum())
    print(f"Rows: {n} | Valid Z'' computed: {n_valid} | Missing: {n - n_valid}")

    if n_valid < n:
        # Which input drove the NaN — this is what tells you whether it's a
        # financial-sector row (issue 1) or a genuine gap in the pull.
        missing = df[df["altman_z_double_prime"].isna()]
        culprits = {c: int(missing[c].isna().sum())
                    for c in REQUIRED_COLUMNS if missing[c].isna().any()}
        if culprits:
            print(f"  Missing-input breakdown for those rows: {culprits}")
            fin_pattern = {"current_assets", "current_liabilities", "ebit"}
            if fin_pattern & set(culprits):
                print("  -> current_assets/current_liabilities/ebit missing together is the "
                      "financial-sector signature (banks/NBFCs). Those companies should be "
                      "excluded from this model, not imputed.")

    if "altman_flag_negative_equity" in df:
        n_neg = int(df["altman_flag_negative_equity"].sum())
        if n_neg:
            print(f"  Negative book equity: {n_neg} row(s) — X4 is negative there BY DESIGN "
                  "(real insolvency signal, not an error).")
    if "altman_flag_x4_uncomputable" in df:
        n_bad = int(df["altman_flag_x4_uncomputable"].sum())
        if n_bad:
            print(f"  X4 uncomputable: {n_bad} row(s) — total_liabilities <= 0 or missing. "
                  "Check the pull for these rows.")
    if "altman_flag_check_scaling" in df:
        n_scale = int(df["altman_flag_check_scaling"].sum())
        print(f"  Flagged |Z''| > 50 with NO negative-equity explanation "
              f"(likely units/scaling error): {n_scale}")

    if "altman_zone" in df:
        print(df["altman_zone"].value_counts(dropna=False))


if __name__ == "__main__":
    # ── Self-test with synthetic data mimicking real company shapes ────────
    # A "healthy" firm and a "distressed" firm, values in decimal-ratio-safe
    # absolute rupee terms (crores), to confirm the formula behaves sensibly
    # before you ever point it at real Bloomberg exports.
    # NegEquityCo mirrors the real Suzlon rows: book equity below zero, which
    # must produce a negative X4 and a distress-zone score WITHOUT tripping
    # the scaling-error flag. ZeroLiabCo has an impossible denominator and
    # must come out NaN rather than inf.
    test_df = pd.DataFrame({
        "company": ["HealthyCo", "DistressedCo", "GreyZoneCo", "NegEquityCo", "ZeroLiabCo"],
        "total_assets":        [10000, 5000, 8000, 14000, 6000],
        "current_assets":      [4000,  800,  2500,  3000, 2000],
        "current_liabilities": [1500,  2200, 2000,  9000, 1000],
        "retained_earnings":   [3000, -1500,  400, -9000,  500],
        "ebit":                [1200,  -600,  350,  -800,  400],
        "total_equity_book":   [5000,   300, 2200, -3500, 3000],
        "total_liabilities":   [5000,  4700, 5800, 17500,     0],
    }).set_index("company")

    result = compute_altman_zscore_components(test_df)
    print(result[["X4_equity_to_liabilities", "altman_z_double_prime", "altman_ems_score",
                  "altman_zone", "altman_flag_negative_equity",
                  "altman_flag_x4_uncomputable", "altman_flag_check_scaling"]].to_string())
    print()
    sanity_check_report(result)

    # ── assertions ──────────────────────────────────────────────────────
    assert result.loc["HealthyCo", "altman_zone"] == "safe"
    assert result.loc["DistressedCo", "altman_zone"] == "distress"
    assert result.loc["GreyZoneCo", "altman_zone"] == "grey_zone"

    # negative equity: X4 negative, deep distress, correctly attributed
    assert result.loc["NegEquityCo", "X4_equity_to_liabilities"] < 0, \
        "negative book equity must produce a negative X4, not a suppressed one"
    assert result.loc["NegEquityCo", "altman_zone"] == "distress"
    assert bool(result.loc["NegEquityCo", "altman_flag_negative_equity"])
    assert not bool(result.loc["NegEquityCo", "altman_flag_check_scaling"]), \
        "negative equity explains the low score — must NOT be flagged as a scaling error"
    assert result.loc["NegEquityCo", "negative_equity_flag"] == 1.0

    # non-positive liabilities: NaN, not inf, and flagged
    assert pd.isna(result.loc["ZeroLiabCo", "X4_equity_to_liabilities"]), \
        "total_liabilities == 0 must give NaN, never inf"
    assert pd.isna(result.loc["ZeroLiabCo", "altman_z_double_prime"])
    assert bool(result.loc["ZeroLiabCo", "altman_flag_x4_uncomputable"])
    assert np.isfinite(result["altman_z_double_prime"].dropna()).all(), \
        "no inf may survive into the Z'' column"

    print("\n✅ Self-test passed: zones correct, negative equity flows through to a "
          "negative X4 without a false scaling flag, non-positive liabilities give NaN.")
