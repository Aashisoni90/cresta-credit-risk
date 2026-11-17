"""
build_training_matrix.py
=========================
Assembles the model-ready training matrix for Track A: feature blocks 1-7,
the PRD §7.5 forward-looking label, circularity enforcement, and the
out-of-time split assignment.

This is the Phase 2 deliverable. It makes three decisions explicit in code
rather than leaving them to a notebook (Project_Context.md §17):

1. LABEL — §7.5 financial criteria, forward-looking FY(t) -> FY(t+1).
   The §7.2 ratings label is unavailable: the census found 4/3,000 tickers
   rated (0.13%), with CRISIL returning #N/A Authorization on 2,999 of 3,000
   (an account entitlement gap). See §16.1.

2. ALTMAN Z'' — the feature set uses the NON-CIRCULAR 3-term `altman_z_ex_x4`,
   not canonical Z''. Under a financial-criteria label the X4 = equity/
   liabilities term is circular: sign(X4) == (book equity < 0) on 100.00% of
   rows, which is label criterion 2 exactly. Dropping it costs nothing and
   helps: AUC 0.8069 ex-X4 vs 0.7661 full, corr 0.9943. Canonical Z'' is
   preserved in the panel and remains the §8.1 Tier-1 BENCHMARK — it is simply
   not used as a feature. See §17.1.

3. SPLIT — out-of-time, never random (§8.5, invariant #2). PRD §8.5's literal
   windows do not fit the data once the FY convention is resolved (§14.5): the
   test window overruns the panel by a year and FY2014-15 falls outside every
   window. The split below keeps §8.5's shape (long train / 1 validation /
   multi-year test) while using all available years.

NO PREPROCESSING HAPPENS HERE. No scaling, imputation or encoding — those must
be fitted on the training window only (§7.4, invariant #3). This module emits
raw features plus a `split` column; the fitting happens downstream.

USAGE
    python src/build_training_matrix.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_feature_matrix
from fallback_label import (build_fallback_labels, make_forward_looking,
                            circular_features_for, assert_no_circular_leak,
                            DEFAULT_CRITERIA, PARTIALLY_CIRCULAR)

ID_COLS = ["ticker", "company_name", "fiscal_year", "fiscal_year_num"]

UNIVERSE_PATH = "data/processed/universe_800.csv"

# §5.6 / invariant 5 — financial-sector issuers are excluded from the primary
# model. Their balance sheets have no operating cycle, so working capital, the
# current ratio and EBIT/assets are not merely noisy but undefined in meaning.
#
# Measured on this panel: current_assets and current_liabilities are null on
# 86.3% of financial-sector rows versus 23.5% elsewhere, while EBIT is reported
# at a similar rate to other sectors (19.8% vs 22.0%). So the exclusion is not
# cosmetic — three of sixteen features are structurally absent for these firms,
# and XGBoost's native NaN branch can isolate them on that absence alone.
#
# Name fallback: gics_sector is missing for 692 of 2,999 tickers. The regex
# below matches 62.7% of KNOWN financials and only 0.7% of known
# non-financials, so it is high-precision and moderate-recall. Applied ONLY to
# sector-unknown tickers, where the alternative is leaving a known-nonzero
# number of lenders in the training data. Excluding a handful of genuine
# non-financials is the conservative direction of error; disable with
# --keep-financials or --no-name-heuristic if you disagree.
# §7.4 / invariant 2 — POINT-IN-TIME FIELDS ARE LOOK-AHEAD AND ARE DROPPED.
#
# The Bloomberg pull applied `FUND_PER="Y"` + `EQY_FUND_YEAR=<year>` to every
# field uniformly. That override governs FUNDAMENTAL fields only. CUR_MKT_CAP
# and VOLATILITY_260D are real-time MARKET fields: Bloomberg ignores the
# override and returns the value as of the pull date for every requested year.
#
# Measured on this panel (tickers with >=5 reported years):
#     cur_mkt_cap       98.4% of tickers carry ONE value across all years
#     volatility_260d   95.4%                 "
#     every other field  <5%, and the accounting fields 0.0%
#
# So `log_mkt_cap` on a FY2014 row is the firm's market capitalisation as at
# the 2026 pull date — a value that postdates the FY2015 label it is used to
# predict by eleven years. The leakage signature is visible directly: the
# feature alone scores AUC 0.598 against the FY2014 label and rises
# monotonically to 0.672 against FY2023 as the gap to the pull date closes.
# A genuinely contemporaneous size control does not trend that way.
#
# This is not recoverable by rescaling or by a missingness indicator. The
# correct historical series requires a BDH time-series pull, which is future
# extraction work (see Data_Extraction_Guide_v2.md Part 2). Until then the two
# features are dropped. `log_total_assets` survives: it comes from BS_TOT_ASSET,
# varies per year on 100% of tickers, and remains a legitimate size control.
#
# --keep-point-in-time restores them for the leakage sensitivity run ONLY.
POINT_IN_TIME_FEATURES = ["log_mkt_cap", "equity_volatility"]

FINANCIAL_NAME_RE = re.compile(
    r"(bank|financ|capital|invest|securit|insur|nbfc|housing fin|leasing"
    r"|credit|asset manag|broking|fincorp|finvest)", re.I)


def flag_financial_sector(df, universe_path=UNIVERSE_PATH, use_name_heuristic=True):
    """Return (mask, reason) marking financial-sector rows for exclusion."""
    u = (pd.read_csv(universe_path, usecols=["ticker", "gics_sector", "company_name"])
         .drop_duplicates("ticker"))
    meta = df[["ticker"]].merge(u, on="ticker", how="left")
    meta.index = df.index
    sector = meta["gics_sector"]
    reason = pd.Series("", index=df.index, dtype="object")
    reason[sector == "Financials"] = "gics_sector == Financials"
    if use_name_heuristic:
        nm = meta["company_name"].fillna("").astype(str)
        hit = sector.isna() & nm.str.contains(FINANCIAL_NAME_RE)
        reason[hit] = "sector unknown; company name matches financial pattern"
    return reason.ne(""), reason

# Exact duplicates produced under two names by different modules. Keeping both
# adds perfect collinearity and splits SHAP attribution across twins, making the
# explanation layer (the project's whole purpose) harder to read.
REDUNDANT_ALIASES = {
    "X1_working_cap_to_assets": "working_capital_to_assets",
    "X2_retained_earnings_to_assets": "retained_earnings_to_assets",
    "X3_ebit_to_assets": "ebit_to_assets",
    "altman_ems_score": "altman_z_double_prime",   # EMS = Z'' + 3.25, perfectly collinear
}

# Out-of-time split on fiscal_year_num (Indian FY START year).
# Features at FY(t) predict the label at FY(t+1), so t runs 2014..2023.
SPLIT_BOUNDS = {"train": (2014, 2020), "validation": (2021, 2021), "test": (2022, 2023)}


def assign_split(year: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=year.index, dtype="object")
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        out[(year >= lo) & (year <= hi)] = name
    return out


def build(panel_path, out_path, strict_altman=True, verbose=True,
          exclude_financials=True, use_name_heuristic=True,
          drop_point_in_time=True,
          excluded_log="data/processed/excluded_financials.csv"):
    panel = pd.read_csv(panel_path)
    if "company_name" not in panel.columns:
        panel["company_name"] = ""
    if verbose:
        print(f"[build] panel: {len(panel)} rows, {panel.ticker.nunique()} tickers")

    feats = build_feature_matrix(panel, verbose=False)
    labels = make_forward_looking(build_fallback_labels(panel, verbose=False), verbose=False)

    df = feats.merge(
        labels[["ticker", "fiscal_year_num", "label_binary_next", "distress_label_next"]],
        on=["ticker", "fiscal_year_num"], how="inner",
    )
    if verbose:
        print(f"[build] joined to forward label: {len(df)} rows")

    # ---- §5.6 financial-sector exclusion (invariant 5) --------------------
    # Logged to a file, never silently dropped (invariant 4's principle: the
    # rows you remove are exactly the ones a reviewer will ask about).
    if exclude_financials:
        mask, reason = flag_financial_sector(df, use_name_heuristic=use_name_heuristic)
        if mask.any():
            out = df.loc[mask, ["ticker", "fiscal_year_num"]].copy()
            out["exclusion_reason"] = reason[mask]
            Path(excluded_log).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(excluded_log, index=False)
            if verbose:
                print(f"[build] §5.6 EXCLUDED {int(mask.sum()):,} rows / "
                      f"{out.ticker.nunique():,} financial-sector tickers "
                      f"-> {excluded_log}")
                for r, n in reason[mask].value_counts().items():
                    print(f"           {n:>6,}  {r}")
        df = df.loc[~mask].copy()
    elif verbose:
        print("[build] WARNING: financial-sector rows RETAINED — this violates "
              "§5.6 / invariant 5. Only valid for the retain-vs-exclude comparison.")

    # carry the coverage marker so unusable rows can be excluded, not silently kept
    if "qc_below_coverage" in panel.columns:
        cov = panel[["ticker", "fiscal_year_num", "qc_below_coverage"]]
        df = df.merge(cov, on=["ticker", "fiscal_year_num"], how="left")

    # ---- drop circular features -------------------------------------------
    banned = set(circular_features_for(DEFAULT_CRITERIA))
    if strict_altman:
        for c in DEFAULT_CRITERIA:
            banned |= set(PARTIALLY_CIRCULAR.get(c, []))
    dropped_circ = sorted(c for c in df.columns if c in banned)
    df = df.drop(columns=dropped_circ)

    # ---- drop point-in-time (look-ahead) features --------------------------
    dropped_pit = []
    if drop_point_in_time:
        dropped_pit = [c for c in POINT_IN_TIME_FEATURES if c in df.columns]
        df = df.drop(columns=dropped_pit)
    elif verbose:
        print("[build] WARNING: point-in-time features RETAINED — "
              f"{POINT_IN_TIME_FEATURES} are look-ahead (see module header). "
              "Only valid for the leakage sensitivity comparison.")

    # ---- drop redundant aliases -------------------------------------------
    dropped_alias = [a for a in REDUNDANT_ALIASES if a in df.columns]
    df = df.drop(columns=dropped_alias)

    # ---- drop all-empty features ------------------------------------------
    meta = set(ID_COLS) | {"label_binary_next", "distress_label_next", "qc_below_coverage"}
    empty = [c for c in df.columns if c not in meta and df[c].notna().sum() == 0]
    df = df.drop(columns=empty)

    feature_cols = [c for c in df.columns if c not in meta]

    # ---- HARD GUARD: nothing circular may reach the matrix -----------------
    assert_no_circular_leak(feature_cols, DEFAULT_CRITERIA, strict=strict_altman)

    df["split"] = assign_split(df["fiscal_year_num"])

    if verbose:
        _report(df, feature_cols, dropped_circ, dropped_alias, empty, strict_altman,
                dropped_pit)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\n[build] wrote {len(df)} rows x {len(df.columns)} cols -> {out_path}")
    return df, feature_cols


def _report(df, feats, dropped_circ, dropped_alias, empty, strict, dropped_pit=()):
    print(f"\n  dropped circular{' (+Altman-derived, strict)' if strict else ''}: "
          f"{len(dropped_circ)}")
    for c in dropped_circ:
        print(f"      - {c}")
    print(f"  dropped point-in-time (look-ahead): {len(dropped_pit)}  {list(dropped_pit)}")
    print(f"  dropped redundant aliases: {len(dropped_alias)}  {dropped_alias}")
    print(f"  dropped all-empty: {len(empty)}")
    print(f"\n  FINAL FEATURE COUNT: {len(feats)}")

    print("\n  out-of-time split (fiscal_year_num = Indian FY start year):")
    hdr = f"    {'split':<11s} {'FY range':<22s} {'rows':>7s} {'pos':>6s} {'rate':>7s} {'tickers':>8s}"
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for name in ["train", "validation", "test"]:
        s = df[df.split == name]
        if not len(s):
            continue
        y = s.label_binary_next
        lo, hi = int(s.fiscal_year_num.min()), int(s.fiscal_year_num.max())
        rng = f"FY{lo}-{str(lo+1)[-2:]} .. FY{hi}-{str(hi+1)[-2:]}"
        print(f"    {name:<11s} {rng:<22s} {len(s):>7,} {int(y.sum()):>6,} "
              f"{y.mean()*100:>6.1f}% {s.ticker.nunique():>8,}")
    un = df[df.split.isna()]
    if len(un):
        print(f"    {'UNASSIGNED':<11s} {'':<22s} {len(un):>7,}")

    if "qc_below_coverage" in df.columns:
        usable = df[df.qc_below_coverage != True]
        print(f"\n  rows passing the coverage gate: {len(usable):,} of {len(df):,} "
              f"({len(usable)/len(df)*100:.1f}%)")
        for name in ["train", "validation", "test"]:
            s = usable[usable.split == name]
            if len(s):
                print(f"      {name:<11s} {len(s):>7,} rows, "
                      f"{int(s.label_binary_next.sum()):>5,} positives "
                      f"({s.label_binary_next.mean()*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/bloomberg_panel_scored_gated.csv")
    ap.add_argument("--out", default="data/processed/training_matrix.csv")
    ap.add_argument("--keep-financials", action="store_true",
                    help="RETAIN financial-sector rows (violates §5.6 / invariant 5; "
                         "only for the retain-vs-exclude comparison)")
    ap.add_argument("--no-name-heuristic", action="store_true",
                    help="exclude only tickers GICS-tagged Financials, leaving "
                         "sector-unknown lenders in the matrix")
    ap.add_argument("--allow-altman", action="store_true",
                    help="keep canonical Z'' and its derivatives as features (NOT recommended; "
                         "they embed the circular X4 term)")
    ap.add_argument("--keep-point-in-time", action="store_true",
                    help="RETAIN log_mkt_cap and equity_volatility. They are "
                         "point-in-time Bloomberg market fields repeated across "
                         "every year, i.e. look-ahead (see module header). Only "
                         "for the leakage sensitivity comparison.")
    a = ap.parse_args()
    build(a.panel, a.out, strict_altman=not a.allow_altman,
          exclude_financials=not a.keep_financials,
          use_name_heuristic=not a.no_name_heuristic,
          drop_point_in_time=not a.keep_point_in_time)


if __name__ == "__main__":
    main()
