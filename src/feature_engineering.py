"""
feature_engineering.py
=======================
Phase 2 — Feature blocks 1-7 from CRESTA PRD §6.

Builds the model-ready feature matrix from the tidy Bloomberg panel
(output of load_filled_template.py or load_bloomberg_export.py).

FEATURE BLOCKS IMPLEMENTED
───────────────────────────
Block 1 — Leverage & Solvency        §6.1   ← fully computable from current panel
Block 2 — Liquidity                  §6.2   ← fully computable
Block 3 — Profitability              §6.3   ← mostly computable; notes missing fields
Block 4 — Debt Servicing Capacity    §6.4   ← fully computable from current panel
Block 5 — Classical Composite        §6.5   ← Altman Z″ (computed here or brought in)
Block 6 — Temporal Dynamics          §6.6   ← fully computable (only needs history)
Block 7 — Size, Sector & Controls    §6.7   ← sector dummy; size proxy only

FIELDS NEEDED BUT NOT YET IN PANEL (BLOCKER 3)
───────────────────────────────────────────────
SHORT_AND_LONG_TERM_DEBT   → debt_to_assets (§6.1), net_debt_to_ebitda (§6.1)
SALES_REV_TURN             → asset_turnover (§6.2), revenue growth (§6.3)
CF_CASH_FROM_OPER          → cfo_to_debt, cfo_to_assets (§6.3)
NET_INCOME                 → net_profit_margin (§6.3)
IS_INT_EXPENSE             → interest_expense raw (§6.4 alt computation)
VOLATILITY_260D            → equity_volatility (§6.7)

Features that require those fields are computed as NaN and marked in
FEATURE_NEEDS_EXPANDED_PULL — they will populate automatically once the
expanded Bloomberg template arrives.

CIRCULARITY (D5 / PRD §7.5)
────────────────────────────
interest_coverage and negative_equity_flag appear in both Block 1/4 (features)
and §7.5 (label criteria). If using the fallback label, these must be removed
from the training matrix. Use:
    excl = circular_features_for(criteria)  # from fallback_label.py
    features = [f for f in ALL_FEATURES if f not in excl]

OUT-OF-TIME SPLIT INVARIANT (D8 / PRD §8.5)
────────────────────────────────────────────
All normalisation, imputation, and encoding must be fitted on the TRAINING
window only and applied to validation and test. This module computes raw
features only; do NOT fit any scaler/imputer here.

USAGE
─────
    from feature_engineering import build_feature_matrix, FEATURE_GROUPS

    panel = pd.read_csv("data/processed/bloomberg_panel.csv")
    features = build_feature_matrix(panel)
    features.to_csv("data/processed/feature_matrix.csv", index=False)

    python src/feature_engineering.py data/processed/bloomberg_panel.csv \\
        --out data/processed/feature_matrix.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from altman_zscore import compute_altman_zscore_components


# ══════════════════════════════════════════════════════════════════════
# Safe arithmetic helpers
# ══════════════════════════════════════════════════════════════════════

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """NaN where denominator ≤ 0 or missing (D2 / PRD §5.5).

    A non-positive denominator FLIPS the sign of the ratio — for leverage
    and solvency measures this converts the most distressed firms into the
    most apparently healthy ones, the worst possible silent failure.
    """
    d = denom.where(denom > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / d


def _cap(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Winsorise to [lo, hi]; preserves NaN."""
    return s.clip(lower=lo, upper=hi)


# ══════════════════════════════════════════════════════════════════════
# Feature group catalogue
# ══════════════════════════════════════════════════════════════════════

# Columns in the expanded Bloomberg pull that are not yet present.
# Features computed from these will be all-NaN until the expanded template lands.
FEATURE_NEEDS_EXPANDED_PULL: set = {
    "short_and_long_term_debt",   # SHORT_AND_LONG_TERM_DEBT
    "sales_rev_turn",             # SALES_REV_TURN
    "cf_cash_from_oper",          # CF_CASH_FROM_OPER
    "net_income",                 # NET_INCOME
    "is_int_expense",             # IS_INT_EXPENSE
    "volatility_260d",            # VOLATILITY_260D
}

# Names of all features this module can produce (populated at end of file).
FEATURE_GROUPS: dict = {}   # {block_name: [feature_col, ...]}
ALL_FEATURES: list = []     # flat list, in block order


# ══════════════════════════════════════════════════════════════════════
# Block 1 — Leverage & Solvency (§6.1)
# ══════════════════════════════════════════════════════════════════════

def build_block1_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features computed:
        liabilities_to_assets      TL / TA          — primary leverage proxy when
                                                       debt_to_equity is undefined
        equity_to_assets           TE / TA          — positive ↔ solvency buffer
        negative_equity_flag       1 if TE < 0, else 0  ← circular if fallback label
        debt_to_assets             total_debt / TA  ← NEEDS short_and_long_term_debt
        net_debt_to_ebitda         net_debt / EBITDA ← NEEDS short_and_long_term_debt

    Design notes:
        debt_to_equity is already in the panel (Bloomberg field). We keep it
        for continuity but DO NOT compute it here — the negative-equity NaN
        rule is enforced at ingestion (D3); we just inherit that column.

        liabilities_to_assets is the recommended leverage substitute when
        debt_to_equity is undefined (D3). It is well-defined even under
        negative equity, making it the only leverage measure that survives
        the entire panel without imputation.
    """
    out = pd.DataFrame(index=df.index)

    # ── always computable ─────────────────────────────────────────────
    out["liabilities_to_assets"] = _safe_div(df["total_liabilities"], df["total_assets"])
    out["equity_to_assets"] = _safe_div(df["total_equity_book"], df["total_assets"])

    # negative_equity_flag already set at ingestion; inherit it
    if "negative_equity_flag" in df.columns:
        out["negative_equity_flag"] = df["negative_equity_flag"]
    else:
        out["negative_equity_flag"] = np.where(
            df["total_equity_book"].isna(), np.nan,
            (df["total_equity_book"] < 0).astype(float)
        )

    # ── requires expanded pull ─────────────────────────────────────────
    if "short_and_long_term_debt" in df.columns:
        out["debt_to_assets"] = _safe_div(
            df["short_and_long_term_debt"], df["total_assets"]
        )
        # EBITDA proxy = EBIT + estimated DA (not available → use EBIT as lower bound)
        # Full EBITDA requires depreciation field; note this approximation in methodology
        ebitda_proxy = df["ebit"]   # underestimates EBITDA by ~DA; document
        net_debt = df["short_and_long_term_debt"] - df.get("cash", pd.Series(np.nan, index=df.index))
        out["net_debt_to_ebitda_proxy"] = _safe_div(net_debt, ebitda_proxy)
        out["_debt_source"] = "short_and_long_term_debt"
    else:
        out["debt_to_assets"] = np.nan
        out["net_debt_to_ebitda_proxy"] = np.nan

    return out


# ══════════════════════════════════════════════════════════════════════
# Block 2 — Liquidity (§6.2)
# ══════════════════════════════════════════════════════════════════════

def build_block2_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features computed:
        cur_ratio                  current_assets / current_liabilities
                                   (already in panel — inherited)
        working_capital_to_assets  (CA - CL) / TA  = Altman X1
        cash_ratio                 (CA - Inventory) / CL  ← inventory not available;
                                   use CA / CL × 0.8 as a conservative proxy;
                                   document in methodology
        asset_turnover             Revenue / TA   ← NEEDS sales_rev_turn
    """
    out = pd.DataFrame(index=df.index)

    # cur_ratio already in panel — inherit
    if "cur_ratio" in df.columns:
        out["cur_ratio"] = df["cur_ratio"]
    else:
        out["cur_ratio"] = _safe_div(df["current_assets"], df["current_liabilities"])

    wc = df["current_assets"] - df["current_liabilities"]
    out["working_capital_to_assets"] = _safe_div(wc, df["total_assets"])  # = Altman X1

    # Conservative cash-ratio proxy (no inventory field; ×0.8 ≈ removing ~1/5 CA)
    # Document: this is a rough estimate; refine when inventory is available.
    out["cur_ratio_conservative"] = _safe_div(
        df["current_assets"] * 0.8, df["current_liabilities"]
    )

    # ── requires expanded pull ─────────────────────────────────────────
    if "sales_rev_turn" in df.columns:
        out["asset_turnover"] = _safe_div(df["sales_rev_turn"], df["total_assets"])
    else:
        out["asset_turnover"] = np.nan

    return out


# ══════════════════════════════════════════════════════════════════════
# Block 3 — Profitability (§6.3)
# ══════════════════════════════════════════════════════════════════════

def build_block3_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features computed:
        roa                        Return on Assets (already in panel — inherited)
        ebit_margin                EBIT / Revenue   ← NEEDS sales_rev_turn
        net_profit_margin          Net Income / Revenue ← NEEDS both
        cfo_to_assets              Operating CF / TA  ← NEEDS cf_cash_from_oper
        cfo_to_debt                Operating CF / total_debt ← NEEDS both
        revenue_growth_yoy         ΔRevenue / Revenue(t-1) ← NEEDS sales_rev_turn
        retained_earnings_to_assets  RE / TA  = Altman X2 (computable now)
    """
    out = pd.DataFrame(index=df.index)

    # ── always computable ─────────────────────────────────────────────
    if "roa" in df.columns:
        out["roa"] = df["roa"]
    else:
        out["roa"] = _safe_div(df["ebit"], df["total_assets"]) * 100

    out["retained_earnings_to_assets"] = _safe_div(
        df["retained_earnings"], df["total_assets"]
    )  # = Altman X2

    # ── requires sales_rev_turn ────────────────────────────────────────
    if "sales_rev_turn" in df.columns:
        rev = df["sales_rev_turn"]
        out["ebit_margin"] = _safe_div(df["ebit"], rev)
        out["revenue_growth_yoy"] = (
            df.groupby("ticker")["sales_rev_turn"]
            .transform(lambda s: s.pct_change())
        )
        if "net_income" in df.columns:
            out["net_profit_margin"] = _safe_div(df["net_income"], rev)
        else:
            out["net_profit_margin"] = np.nan
    else:
        out["ebit_margin"] = np.nan
        out["revenue_growth_yoy"] = np.nan
        out["net_profit_margin"] = np.nan

    # ── requires cash-flow field ─────────────────────────────────────
    if "cf_cash_from_oper" in df.columns:
        cfo = df["cf_cash_from_oper"]
        out["cfo_to_assets"] = _safe_div(cfo, df["total_assets"])
        if "short_and_long_term_debt" in df.columns:
            out["cfo_to_debt"] = _safe_div(cfo, df["short_and_long_term_debt"])
        else:
            out["cfo_to_debt"] = np.nan
    else:
        out["cfo_to_assets"] = np.nan
        out["cfo_to_debt"] = np.nan

    return out


# ══════════════════════════════════════════════════════════════════════
# Block 4 — Debt Servicing Capacity (§6.4)
# ══════════════════════════════════════════════════════════════════════

def build_block4_debt_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features computed:
        interest_coverage          EBIT / Interest expense (already in panel)
        interest_coverage_capped   winsorised to [-10, 50]
        interest_coverage_below_1_flag  1 if IC < 1.0   ← circular if fallback label
        ebit_to_assets             EBIT / TA = Altman X3 (computable now)

    CIRCULARITY NOTE:
        interest_coverage and interest_coverage_below_1_flag are label criteria
        in §7.5. If using the fallback label, BOTH must be excluded from the
        training matrix (D5). This is enforced by CIRCULAR_FEATURE_MAP in
        fallback_label.py — run circular_features_for(criteria) to get the list.
    """
    out = pd.DataFrame(index=df.index)

    # ── always computable ─────────────────────────────────────────────
    if "interest_coverage" in df.columns:
        out["interest_coverage"] = df["interest_coverage"]
    else:
        # If raw interest expense is available, compute it
        if "is_int_expense" in df.columns:
            out["interest_coverage"] = _safe_div(df["ebit"], df["is_int_expense"])
        else:
            out["interest_coverage"] = np.nan

    # Winsorise to remove extreme outliers while preserving the sign
    out["interest_coverage_capped"] = _cap(out["interest_coverage"], lo=-10.0, hi=50.0)

    # Binary flag (below 1.0 = interest not covered by operating earnings)
    ic = out["interest_coverage"]
    out["interest_coverage_below_1_flag"] = np.where(
        ic.isna(), np.nan, (ic < 1.0).astype(float)
    )

    # EBIT / TA = Altman X3 — always computable
    out["ebit_to_assets"] = _safe_div(df["ebit"], df["total_assets"])  # = Altman X3

    return out


# ══════════════════════════════════════════════════════════════════════
# Block 5 — Classical Composite Score (§6.5)
# ══════════════════════════════════════════════════════════════════════

def build_block5_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Altman Z″ (EM variant). Brings in all diagnostic flags."""
    z = compute_altman_zscore_components(df)

    # Non-circular 3-term variant. Under the §7.5 financial-criteria label the
    # X4 = equity/liabilities term restates label criterion 2 exactly
    # (sign(X4) == book equity < 0 on 100.00% of rows), so canonical Z'' cannot
    # be used as a FEATURE. Dropping X4 also improves AUC (0.8069 vs 0.7661)
    # because X4 is unbounded and carries the thin-liabilities degeneracy.
    # Canonical Z'' is still emitted here — it remains the §8.1 benchmark — and
    # is excluded at matrix-assembly time, not here.
    # See Project_Context.md §17.1.
    z["altman_z_ex_x4"] = (
        6.56 * z["X1_working_cap_to_assets"]
        + 3.26 * z["X2_retained_earnings_to_assets"]
        + 6.72 * z["X3_ebit_to_assets"]
    )

    cols = [
        "X1_working_cap_to_assets",
        "X2_retained_earnings_to_assets",
        "X3_ebit_to_assets",
        "X4_equity_to_liabilities",
        "altman_z_double_prime",
        "altman_ems_score",
        "altman_z_ex_x4",
        "altman_flag_negative_equity",
        "altman_flag_x4_uncomputable",
        "altman_flag_check_scaling",
    ]
    return z[[c for c in cols if c in z.columns]]


# ══════════════════════════════════════════════════════════════════════
# Block 6 — Temporal Dynamics (§6.6)
# ══════════════════════════════════════════════════════════════════════

def build_block6_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year changes and trend features.

    All features in this block require the panel to be in chronological order
    within each ticker (guaranteed by sort in build_feature_matrix). They do
    NOT require any new Bloomberg fields — everything computes from the existing
    12-field panel.

    Features:
        roa_yoy_change             ΔROa: roa(t) - roa(t-1)
        ic_yoy_change              ΔIC: interest_coverage(t) - ic(t-1)
        liab_to_assets_yoy_change  ΔLeverage year-over-year
        altman_z_yoy_change        ΔZ″ — captures trend toward/away from distress
        altman_z_3yr_trend         OLS slope of Z″ over a 3-year rolling window
                                   (positive = improving, negative = deteriorating)
        consecutive_loss_years     number of consecutive years with EBIT < 0
        consecutive_ic_below1_years  consecutive years with IC < 1.0
    """
    df = df.sort_values(["ticker", "fiscal_year_num"]).copy()
    out = pd.DataFrame(index=df.index)

    def _yoy(col: str, name: str) -> None:
        if col not in df.columns:
            out[name] = np.nan
            return
        out[name] = df.groupby("ticker")[col].transform(lambda s: s.diff())

    _yoy("roa", "roa_yoy_change")
    _yoy("interest_coverage", "ic_yoy_change")
    _yoy("liabilities_to_assets", "liab_to_assets_yoy_change")

    # Z″ yoy change — requires Z″ to already be in df (passed in after block5)
    if "altman_z_double_prime" in df.columns:
        out["altman_z_yoy_change"] = df.groupby("ticker")["altman_z_double_prime"].transform(
            lambda s: s.diff()
        )
    else:
        out["altman_z_yoy_change"] = np.nan

    # 3-year rolling OLS slope of Z″ (min 2 observations; NaN if fewer)
    # A negative slope signals deterioration — companies headed toward distress
    if "altman_z_double_prime" in df.columns:

        def _rolling_slope(s: pd.Series, w: int = 3) -> pd.Series:
            """OLS slope of y on t over a rolling window of `w` years."""
            result = pd.Series(np.nan, index=s.index)
            for i in range(w - 1, len(s)):
                y = s.iloc[max(0, i - w + 1): i + 1].dropna()
                n = len(y)
                if n < 2:
                    continue
                x = np.arange(n, dtype=float)
                # Simple closed-form OLS slope
                xm, ym = x.mean(), y.mean()
                slope = ((x - xm) * (y.values - ym)).sum() / ((x - xm) ** 2).sum()
                result.iloc[i] = slope
            return result

        out["altman_z_3yr_trend"] = df.groupby("ticker")["altman_z_double_prime"].transform(
            _rolling_slope
        )
    else:
        out["altman_z_3yr_trend"] = np.nan

    # Non-circular counterparts (see Block 5). Without these, dropping the
    # canonical Z'' derivatives for circularity would leave Block 6 with no
    # distress-trajectory feature at all.
    if "altman_z_ex_x4" in df.columns:
        out["altman_z_ex_x4_yoy_change"] = df.groupby("ticker")["altman_z_ex_x4"].transform(
            lambda s: s.diff()
        )
        out["altman_z_ex_x4_3yr_trend"] = df.groupby("ticker")["altman_z_ex_x4"].transform(
            _rolling_slope
        )
    else:
        out["altman_z_ex_x4_yoy_change"] = np.nan
        out["altman_z_ex_x4_3yr_trend"] = np.nan

    # Consecutive loss years (EBIT < 0)
    if "ebit" in df.columns:

        def _consecutive_negative(s: pd.Series) -> pd.Series:
            """Rolling count of consecutive years with EBIT < 0.
            NaN EBIT breaks the run (missing data ≠ confirmed loss).
            """
            flag = (s < 0).where(s.notna(), False)
            # cumsum trick (same approach as _consecutive_true in fallback_label.py)
            run = flag.groupby((~flag).cumsum()).cumsum()
            return run.where(s.notna(), np.nan)

        out["consecutive_loss_years"] = df.groupby("ticker")["ebit"].transform(
            _consecutive_negative
        )
    else:
        out["consecutive_loss_years"] = np.nan

    # Consecutive IC below 1.0 years
    if "interest_coverage" in df.columns:

        def _consec_ic_below1(s: pd.Series) -> pd.Series:
            flag = (s < 1.0).where(s.notna(), False)
            run = flag.groupby((~flag).cumsum()).cumsum()
            return run.where(s.notna(), np.nan)

        out["consecutive_ic_below1_years"] = df.groupby("ticker")["interest_coverage"].transform(
            _consec_ic_below1
        )
    else:
        out["consecutive_ic_below1_years"] = np.nan

    return out


# ══════════════════════════════════════════════════════════════════════
# Block 7 — Size, Sector & Controls (§6.7)
# ══════════════════════════════════════════════════════════════════════

def build_block7_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features:
        log_total_assets           log(TA) — size proxy; log-transforms skew
        log_mkt_cap                log(market cap) — market size proxy
        is_financials_sector       1 if GICS = Financials (tagging, not exclusion)
        equity_volatility          260-day equity return vol ← NEEDS volatility_260d
        fiscal_year_num            calendar year (for temporal fixed effects)

    SECTOR DUMMIES:
        Not created here — dummies should be fitted on the TRAINING window
        only to avoid look-ahead in the category set. Create them in train_model.py
        using sklearn's OrdinalEncoder / pd.get_dummies on the training split.
    """
    out = pd.DataFrame(index=df.index)

    # Size (log-transform total assets; NaN-safe)
    out["log_total_assets"] = np.log(
        df["total_assets"].where(df["total_assets"] > 0)
    )

    if "cur_mkt_cap" in df.columns:
        out["log_mkt_cap"] = np.log(
            df["cur_mkt_cap"].where(df["cur_mkt_cap"] > 0)
        )
    else:
        out["log_mkt_cap"] = np.nan

    # Sector tag — inherit from panel if present, else mark as unknown
    if "gics_sector" in df.columns:
        out["gics_sector"] = df["gics_sector"]
        out["is_financials_sector"] = (
            df["gics_sector"].astype(str).str.strip().str.title() == "Financials"
        ).astype(float)
    elif "is_financials_sector" in df.columns:
        out["is_financials_sector"] = df["is_financials_sector"]
    else:
        out["is_financials_sector"] = np.nan

    # Equity volatility — requires VOLATILITY_260D field
    if "volatility_260d" in df.columns:
        out["equity_volatility"] = df["volatility_260d"]
    else:
        out["equity_volatility"] = np.nan

    # Year (for temporal fixed effects in panel regression)
    if "fiscal_year_num" in df.columns:
        out["fiscal_year_num"] = df["fiscal_year_num"]

    return out


# ══════════════════════════════════════════════════════════════════════
# Master assembly
# ══════════════════════════════════════════════════════════════════════

def build_block8_esg(df: pd.DataFrame) -> pd.DataFrame:
    """PRD §6.8 — ESG Track B Features."""
    b = pd.DataFrame(index=df.index)
    for col in ["e_score", "s_score", "g_score", "carbon_intensity"]:
        if col in df.columns:
            b[col] = df[col]
    return b

def build_feature_matrix(
    panel: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assemble all seven feature blocks into a single model-ready frame.

    Returns a frame with (ticker, company_name, fiscal_year, fiscal_year_num)
    as keys plus all feature columns. Rows retain the panel's original index.

    NOTE: this function does NOT scale, impute, or encode anything. Those
    steps must be fitted on the training window only (D8 / PRD §8.5).
    """
    panel = panel.sort_values(["ticker", "fiscal_year_num"]).copy()

    id_cols = [c for c in ["ticker", "company_name", "fiscal_year", "fiscal_year_num",
                            "fiscal_year_end", "period_end_date"] if c in panel.columns]
    out = panel[id_cols].copy()

    # Build blocks sequentially; Block 6 needs Z″ from Block 5
    b1 = build_block1_leverage(panel)
    b2 = build_block2_liquidity(panel)
    b3 = build_block3_profitability(panel)
    b4 = build_block4_debt_service(panel)
    b5 = build_block5_zscore(panel)

    # Attach Z″ to panel for Block 6
    panel_with_z = panel.copy()
    for c in ["altman_z_double_prime", "altman_ems_score", "altman_z_ex_x4"]:
        if c in b5.columns:
            panel_with_z[c] = b5[c].values

    # Block 6 also needs liabilities_to_assets from Block 1
    if "liabilities_to_assets" in b1.columns:
        panel_with_z["liabilities_to_assets"] = b1["liabilities_to_assets"].values

    b6 = build_block6_temporal(panel_with_z)
    b7 = build_block7_controls(panel)

    b8 = build_block8_esg(panel)

    for name, blk in [("b1_leverage", b1), ("b2_liquidity", b2),
                      ("b3_profitability", b3), ("b4_debt_service", b4),
                      ("b5_zscore", b5), ("b6_temporal", b6),
                      ("b7_controls", b7), ("b8_esg", b8)]:
        # Drop internal helper columns (_prefixed)
        blk = blk[[c for c in blk.columns if not c.startswith("_")]]
        # Don't duplicate ID columns already in out
        blk = blk[[c for c in blk.columns if c not in out.columns]]
        out = pd.concat([out, blk], axis=1)

    if verbose:
        _coverage_report(out)
        _missing_fields_report(panel)

    return out


def _coverage_report(features: pd.DataFrame) -> None:
    """Print per-feature null rates so gaps are visible before training."""
    id_cols = {"ticker", "company_name", "fiscal_year", "fiscal_year_num",
               "fiscal_year_end", "period_end_date", "gics_sector"}
    feat_cols = [c for c in features.columns if c not in id_cols]

    n = len(features)
    null_rates = features[feat_cols].isna().mean().sort_values(ascending=False)
    all_null = null_rates[null_rates == 1.0]
    partial_null = null_rates[(null_rates > 0) & (null_rates < 1.0)]

    print(f"\n[feature_engineering] Built {len(feat_cols)} features from {n} rows.")
    if len(all_null):
        print(f"  All-NaN ({len(all_null)} features — waiting for expanded pull):")
        for f in all_null.index:
            print(f"    {f}")
    if len(partial_null):
        print(f"  Partial coverage ({len(partial_null)} features):")
        for f, rate in partial_null.head(10).items():
            print(f"    {f}: {rate:.1%} null")


def _missing_fields_report(panel: pd.DataFrame) -> None:
    """Remind what expanded-pull fields are missing and which blocks they unlock."""
    missing = [f for f in FEATURE_NEEDS_EXPANDED_PULL if f not in panel.columns]
    if not missing:
        return
    print("\n[feature_engineering] ── EXPANDED PULL NEEDED ──────────────────")
    needed = {
        "short_and_long_term_debt": "Block 1: debt_to_assets, net_debt_to_ebitda_proxy",
        "sales_rev_turn":           "Blocks 2 & 3: asset_turnover, ebit_margin, revenue_growth",
        "cf_cash_from_oper":        "Block 3: cfo_to_assets, cfo_to_debt",
        "net_income":               "Block 3: net_profit_margin",
        "is_int_expense":           "Block 4: alternative interest_coverage computation",
        "volatility_260d":          "Block 7: equity_volatility (Merton DtD proxy)",
    }
    for f in missing:
        print(f"  {f:30s} → {needed.get(f, 'unknown')}")
    print("  Add these to the Bloomberg vertical template and re-run.\n")


# ══════════════════════════════════════════════════════════════════════
# Populate catalogues (after all functions are defined)
# ══════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "block1_leverage":      [
        "liabilities_to_assets", "equity_to_assets", "negative_equity_flag",
        "debt_to_assets", "net_debt_to_ebitda_proxy",
    ],
    "block2_liquidity":     [
        "cur_ratio", "working_capital_to_assets", "cur_ratio_conservative",
        "asset_turnover",
    ],
    "block3_profitability": [
        "roa", "retained_earnings_to_assets", "ebit_margin",
        "revenue_growth_yoy", "net_profit_margin", "cfo_to_assets", "cfo_to_debt",
    ],
    "block4_debt_service":  [
        "interest_coverage", "interest_coverage_capped",
        "interest_coverage_below_1_flag", "ebit_to_assets",
    ],
    "block5_zscore":        [
        "X1_working_cap_to_assets", "X2_retained_earnings_to_assets",
        "X3_ebit_to_assets", "X4_equity_to_liabilities",
        "altman_z_double_prime", "altman_ems_score", "altman_z_ex_x4",
        "altman_z_ex_x4_yoy_change", "altman_z_ex_x4_3yr_trend",
    ],
    "block6_temporal":      [
        "roa_yoy_change", "ic_yoy_change", "liab_to_assets_yoy_change",
        "altman_z_yoy_change", "altman_z_3yr_trend",
        "consecutive_loss_years", "consecutive_ic_below1_years",
    ],
    "block7_controls":      [
        "log_total_assets", "log_mkt_cap", "is_financials_sector",
        "equity_volatility", "fiscal_year_num",
    ],
}
ALL_FEATURES = [f for feats in FEATURE_GROUPS.values() for f in feats]


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("panel", help="Tidy panel CSV (from load_filled_template.py)")
    ap.add_argument("--out", default="data/processed/feature_matrix.csv")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    print(f"[feature_engineering] Loaded {len(panel)} rows from {args.panel}")

    features = build_feature_matrix(panel, verbose=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    print(f"[feature_engineering] Wrote {len(features)} rows × "
          f"{len(features.columns)} cols → {out}")


if __name__ == "__main__":
    main()
