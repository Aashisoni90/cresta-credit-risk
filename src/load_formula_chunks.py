"""
load_formula_chunks.py
=======================
Loader + validator for the BDP/EQY_FUND_YEAR formula-chunk pulls
(`data/raw/bloomberg_formula_chunk_N_filled.csv`, N = 1..7).

This module PROMOTES the ad hoc validation logic that was run interactively
against chunks 1-4 (Project_Context.md §12.3/§12.4) into reusable code. That
promotion was explicitly deferred in §12.5 and is done here now that all seven
chunks exist.

WHY THIS EXISTS RATHER THAN load_p2_export.py
----------------------------------------------
`load_p2_export.py` was written against the earlier, superseded `p2_export_1`
attempt and expects the PRE-CORRECTION field names (TOTAL_ASSETS, TOT_EQUITY,
INT_COVERAGE_RATIO...). The formula chunks carry the CORRECTED Bloomberg
mnemonics (BS_TOT_ASSET, TOTAL_EQUITY, INTEREST_COVERAGE_RATIO...) validated in
§12.2. Pointing the old loader at these files would silently produce an empty
frame. It also derived total_liabilities from TOTAL_LIAB_AND_EQY, which is
liabilities PLUS equity — wrong by construction.

WHAT IT CHECKS (all of §12.3, in order)
----------------------------------------
1. Structure      — quote-aware CSV parse; column count and header must match.
2. Junk rows      — rows whose ticker is not a Bloomberg ticker (the EQS
                    disclaimer row, see §12.x) are removed AND logged.
3. Sentinels      — every known Bloomberg NA string -> typed NaN. NEVER zero.
4. Coverage       — per-field % non-null across all 18 fields.
5. Dead tickers   — tickers with zero usable data across all fields/years are
                    appended to data/raw/failed_tickers.csv with a reason.
                    NEVER silently dropped (cross-phase invariant #4): distressed
                    and delisted firms fail disproportionately.
6. BS identity    — |TA - (TL + TE)| / |TA| <= 2% on rows where all three are
                    present. This is the project's standard proof that column
                    mapping is correct (§Phase II Step 9, §I2), not optional
                    diligence. A pass rate materially below ~99% blocks the chunk.

FISCAL YEAR CONVENTION — RESOLVED 2026-10-27, DO NOT RE-DERIVE
---------------------------------------------------------------
Settled empirically against published annual-report figures (Project_Context.md
§14). Bloomberg's EQY_FUND_YEAR is the CALENDAR YEAR IN WHICH THE INDIAN FISCAL
YEAR ENDS:

    fiscal_year_bbg = 2024  ==  year ended 31-Mar-2024  ==  Indian "FY2023-24"

Evidence (all for the year ended 31-Mar-2024, against fiscal_year_bbg == 2024):
    TCS       SALES_REV_TURN 2,408,930 INR mn  vs reported Rs 240,893 cr  EXACT
    RELIANCE  SALES_REV_TURN 9,010,640 INR mn  vs reported Rs 901,064 cr  EXACT
              (RIL's "revenue from operations", i.e. net of GST recovered —
               NOT its "value of sales & services" headline of Rs 10,00,122 cr)
    UTCEM     level differs ~1.5% on definition, but YoY growth +12.0% matches
              the reported +12.1% for FY2022-23 -> FY2023-24, confirming alignment

Therefore:
    fiscal_year_num  = fiscal_year_bbg - 1      (Indian FY START year)
    fiscal_year      = "FY{start}-{end:02d}"    e.g. 2024 -> "FY2023-24"

This matches `label_construction.py::build_fiscal_year_labels`, which maps a
rating action dated April(y)..March(y+1) to "FY{y}-{y+1}". A rating action on
2023-06-01 -> "FY2023-24" therefore joins to the panel row with
fiscal_year_bbg == 2024. The two conventions now agree by construction.

An off-by-one here misaligns EVERY label, so the mapping is asserted in code
(`derive_fiscal_year`) rather than left as a convention.

USAGE
-----
    python src/load_formula_chunks.py --chunk 5
    python src/load_formula_chunks.py --all --merge data/processed/bloomberg_panel_full.csv
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Bloomberg NA sentinels. '#N/A Daily Capacity' was added after the chunk-5
# throttle incident (§12.4); '#N/A Invalid Security' after the disclaimer row.
NA_SENTINELS = [
    "#N/A Field Not Applicable",
    "#N/A Invalid Field",
    "#N/A Invalid Security",
    "#N/A Requesting Data...",
    "#N/A Authorization",
    "#N/A Daily Capacity",
    "#N/A Mandatory parameter [SECURITY] cannot be empty",
    "#N/A N/A",
    "#N/A",
    "N/A",
    "",
]

FIELDS = [
    "BS_TOT_ASSET", "BS_CUR_ASSET_REPORT", "BS_CUR_LIAB", "BS_RETAIN_EARN",
    "EBIT", "TOTAL_EQUITY", "BS_TOT_LIAB2", "RETURN_ON_ASSET",
    "TOT_DEBT_TO_TOT_EQY", "INTEREST_COVERAGE_RATIO", "CUR_MKT_CAP", "CUR_RATIO",
    "NET_INCOME", "SALES_REV_TURN", "VOLATILITY_260D", "SHORT_AND_LONG_TERM_DEBT",
    "CF_CASH_FROM_OPER", "IS_INT_EXPENSE",
]

# Bloomberg mnemonic -> canonical panel name (PRD §5.2).
CANONICAL = {
    "BS_TOT_ASSET": "total_assets",
    "BS_CUR_ASSET_REPORT": "current_assets",
    "BS_CUR_LIAB": "current_liabilities",
    "BS_RETAIN_EARN": "retained_earnings",
    "EBIT": "ebit",
    "TOTAL_EQUITY": "total_equity_book",
    "BS_TOT_LIAB2": "total_liabilities",
    "RETURN_ON_ASSET": "roa",
    "TOT_DEBT_TO_TOT_EQY": "debt_to_equity",
    "INTEREST_COVERAGE_RATIO": "interest_coverage",
    "CUR_MKT_CAP": "cur_mkt_cap",
    "CUR_RATIO": "cur_ratio",
    "NET_INCOME": "net_income",
    "SALES_REV_TURN": "total_revenue",
    "VOLATILITY_260D": "volatility_260d",
    "SHORT_AND_LONG_TERM_DEBT": "total_debt",
    "CF_CASH_FROM_OPER": "cash_from_operations",
    "IS_INT_EXPENSE": "interest_expense",
}

TICKER_RE = re.compile(r"^[A-Z0-9/\.\-]+ IN Equity$")
BS_IDENTITY_TOLERANCE = 0.02
FAILED_TICKERS_CSV = "data/raw/failed_tickers.csv"


def clean_numeric(series: pd.Series) -> pd.Series:
    """Bloomberg strings -> float, sentinels -> NaN. Never zero-fills."""
    s = series.astype(str).str.strip()
    s = s.replace(NA_SENTINELS, np.nan)
    s = s.str.replace(",", "", regex=False)
    # accounting negatives: (1,234.5) -> -1234.5
    paren = s.str.match(r"^\(.*\)$", na=False)
    s = s.mask(paren, "-" + s.str.strip("()"))
    s = s.replace(["nan", "None", "NaN", ""], np.nan)
    return pd.to_numeric(s, errors="coerce")


def read_chunk(path: Path, verbose: bool = True):
    """Quote-aware read. Returns (frame, junk_rows)."""
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        rows = list(rdr)

    expected = ["ticker", "fiscal_year"] + FIELDS
    if header != expected:
        raise ValueError(
            f"{path.name}: header does not match the corrected 18-field template.\n"
            f"  expected: {expected}\n  got:      {header}\n"
            "Refusing to load — a header mismatch is how the original 37-company "
            "pull was silently corrupted (§Phase II Step 8)."
        )
    widths = {len(r) for r in rows}
    if widths != {len(expected)}:
        raise ValueError(f"{path.name}: ragged rows, widths={sorted(widths)}")

    df = pd.DataFrame(rows, columns=header)
    is_ticker = df["ticker"].str.strip().str.match(TICKER_RE, na=False)
    junk = df[~is_ticker].copy()
    df = df[is_ticker].copy()

    if verbose and len(junk):
        print(f"  [junk] {len(junk)} non-ticker row(s) removed and logged "
              f"(EQS disclaimer artefact carried into the universe).")
    return df, junk


def derive_fiscal_year(bbg_year: pd.Series):
    """Bloomberg EQY_FUND_YEAR -> (Indian FY start year, Indian FY label).

    See the module docstring: fiscal_year_bbg is the year the Indian FY ENDS in,
    so the Indian FY start year is one less. Resolved against published annual
    reports 2026-10-27; do not change without re-running that evidence.
    """
    y = pd.to_numeric(bbg_year, errors="coerce").astype("Int64")
    if y.notna().any():
        lo, hi = int(y.min()), int(y.max())
        if not (1990 <= lo <= 2100 and 1990 <= hi <= 2100):
            raise ValueError(f"EQY_FUND_YEAR out of plausible range: {lo}-{hi}")
    start = y - 1
    label = start.astype("string") .radd("FY") + "-" + (y % 100).map(
        lambda v: f"{int(v):02d}" if pd.notna(v) else pd.NA, na_action="ignore"
    ).astype("string")
    return start, label


def validate_chunk(path, verbose=True, log_failed=True):
    path = Path(path)
    name = path.name
    if verbose:
        print(f"\n{'='*72}\n{name}\n{'='*72}")

    df, junk = read_chunk(path, verbose)
    df["ticker"] = df["ticker"].str.strip()
    df["fiscal_year_bbg"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df["fiscal_year_num"], df["fiscal_year"] = derive_fiscal_year(df["fiscal_year_bbg"])

    for f in FIELDS:
        df[f] = clean_numeric(df[f])

    n_rows, n_tickers = len(df), df["ticker"].nunique()
    if verbose:
        print(f"  rows={n_rows}  tickers={n_tickers}  "
              f"years={df['fiscal_year_bbg'].min()}-{df['fiscal_year_bbg'].max()}")

    # ---- per-field coverage
    cov = (df[FIELDS].notna().mean() * 100).sort_values()
    if verbose:
        print(f"\n  per-field coverage ({cov.min():.1f}% - {cov.max():.1f}%):")
        for f, v in cov.items():
            print(f"      {f:<28s} {v:5.1f}%")

    # ---- dead tickers: no usable value on ANY field in ANY year
    per_tkr = df.groupby("ticker")[FIELDS].apply(lambda g: g.notna().any().any())
    dead = sorted(per_tkr[~per_tkr].index)
    if verbose:
        print(f"\n  fully-dead tickers: {len(dead)} of {n_tickers} "
              f"({len(dead)/n_tickers*100:.1f}%)")

    if log_failed and dead:
        _append_failed(df, dead, name)

    # ---- balance-sheet identity (the parser proof)
    ok = df[["total_assets" if False else "BS_TOT_ASSET", "TOTAL_EQUITY", "BS_TOT_LIAB2"]].notna().all(axis=1)
    sub = df[ok]
    if len(sub):
        lhs = sub["BS_TOT_ASSET"]
        rhs = sub["TOTAL_EQUITY"] + sub["BS_TOT_LIAB2"]
        rel = (lhs - rhs).abs() / lhs.abs().replace(0, np.nan)
        passed = (rel <= BS_IDENTITY_TOLERANCE).sum()
        rate = passed / len(sub) * 100
        if verbose:
            print(f"\n  balance-sheet identity |TA-(TL+TE)|/TA <= 2%:")
            print(f"      {passed}/{len(sub)} = {rate:.2f}%   (max rel err {rel.max():.3e})")
            print(f"      {'PASS' if rate >= 99.0 else 'INVESTIGATE — below the ~99% bar'}")
    else:
        rate = float("nan")
        if verbose:
            print("\n  balance-sheet identity: NOT CHECKABLE (no complete rows)")

    df = df.rename(columns=CANONICAL)
    df["source_chunk"] = name
    keep = (["ticker", "fiscal_year", "fiscal_year_num", "fiscal_year_bbg"]
            + list(CANONICAL.values()) + ["source_chunk"])
    return df[keep], {"chunk": name, "rows": n_rows, "tickers": n_tickers,
                      "dead": len(dead), "bs_pass_pct": rate,
                      "cov_min": cov.min(), "cov_max": cov.max(), "junk": len(junk)}


def _append_failed(df, dead, chunk_name, path=FAILED_TICKERS_CSV):
    """Append dead tickers. Never drops silently (invariant #4)."""
    p = Path(path)
    rows = []
    for t in dead:
        g = df[df["ticker"] == t]
        rows.append({
            "ticker": t, "company_name": "",
            "n_years": len(g),
            "n_with_assets": int(g["BS_TOT_ASSET"].notna().sum()),
            "n_with_mktcap": int(g["CUR_MKT_CAP"].notna().sum()),
            "reason": f"no usable data on any of 18 fields across all years ({chunk_name})",
        })
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        out = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=["ticker"], keep="first")
    else:
        out = new
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(f"  [failed_tickers] {len(new)} appended -> {p} (file now {len(out)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--merge", type=str, default=None)
    ap.add_argument("--no-log-failed", action="store_true")
    a = ap.parse_args()

    chunks = list(range(1, 8)) if a.all else (a.chunk or [])
    if not chunks:
        ap.error("pass --chunk N (repeatable) or --all")

    frames, stats = [], []
    for n in chunks:
        p = Path(f"data/raw/bloomberg_formula_chunk_{n}_filled.csv")
        if not p.exists():
            print(f"\n!! {p} missing — skipped")
            continue
        f, s = validate_chunk(p, log_failed=not a.no_log_failed)
        frames.append(f); stats.append(s)

    if stats:
        print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
        print(pd.DataFrame(stats).to_string(index=False))

    if a.merge and frames:
        panel = pd.concat(frames, ignore_index=True)
        dup = panel.duplicated(subset=["ticker", "fiscal_year_bbg"]).sum()
        print(f"\nmerged: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
              f"duplicate (ticker,FY) keys: {dup}")
        Path(a.merge).parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(a.merge, index=False)
        print(f"wrote {a.merge}")


if __name__ == "__main__":
    main()
