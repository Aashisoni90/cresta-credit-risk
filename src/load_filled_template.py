"""
load_filled_template.py
========================
Loads the FILLED vertical Bloomberg template (bloomberg_pull_template_filled)
into the tidy panel the rest of the pipeline expects.

This is a different input shape from load_bloomberg_export.py (which handles
the wide column-block export). The vertical template is already one row per
company-year, so no reshaping is needed — but it has three real defects that
must be handled, all confirmed against the 37-company filled file.

⚠️ DEFECT 1 — HEADER/DATA OFF-BY-ONE. THIS IS WHY THE FILE IS READ BY POSITION.
The header row names 16 columns plus a trailing empty one:

    ticker, company_name, fiscal_year, date_range_used, total_assets, ...

but every DATA row carries an extra, UNLABELLED period-end-date column in
position 5:

    JETIN IN Equity, Jet Airways, 2015, 1/1/2015 - 12/31/2015, 12/31/2015, 191608.2, ...
                                                               ^^^^^^^^^^ unnamed

Both rows have 17 fields, so pandas raises NO error — it just assigns
total_assets = "12/31/2015" and shifts EVERY subsequent field by one place.
That is a silent, total corruption of the dataset: it would train a model on
garbage without a single warning.

The corrected positional mapping (COLUMN_ORDER below) was verified against
the balance-sheet identity Total Assets == Total Equity + Total Liabilities,
which holds to a maximum relative error of 1.4e-05 across all 331 complete
rows. verify_balance_sheet_identity() re-runs that proof on every load, so if
a future export fixes the header (changing the alignment) you find out
immediately instead of silently.

⚠️ DEFECT 2 — UNIT/TICKER OUTLIERS. In the 37-company file, 'CCL IN Equity'
(labelled "Coffee Day Enterprises") has a median total_assets of 7.02 against
4.1e4-4.0e6 for every other company — 4-5 orders of magnitude off. Coffee Day
trades as COFFEE IN; CCL IN is a different company. flag_unit_outliers()
detects this class of error by order-of-magnitude rather than by hardcoding
the ticker.

⚠️ DEFECT 3 — DEAD TICKERS ARE PREFERENTIALLY DISTRESSED. Three companies
returned no fundamentals at all (Videocon — pulled under a US ticker,
Essar Steel, ABG Shipyard) and all three are distressed-history names.
Dropping them quietly biases the sample toward survivors and destroys the
positive class. They are written to data/raw/failed_tickers.csv for ticker
repair, NOT silently discarded.

FISCAL YEAR CONVENTION
----------------------
The template pulls with "Per=CY", so the `fiscal_year` column is the CALENDAR
YEAR IN WHICH THE FISCAL YEAR ENDS. Indian companies almost all close on
31 March, so fiscal_year=2020 means the year ending 31-Mar-2020, which is
"FY2019-20" in label_construction.py's notation — a -1 offset. Getting this
backwards misaligns every label by a full year, so it is applied in exactly
one place (fiscal_year_end_to_indian_fy) and is worth verifying against one
known company in the Terminal before the final run.

USAGE
-----
    python src/load_filled_template.py "data/raw/bloomberg_pull_template_filled - Sheet1.csv"
    python src/load_filled_template.py <file.xlsx> --sheet Sheet1 -o data/processed/panel.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_bloomberg_export import clean_numeric, add_negative_equity_handling  # noqa: E402


# ── Positional layout ───────────────────────────────────────────────────
# The authoritative column order, INCLUDING the unlabelled period-end-date
# column the header omits. Read by position; the header row is discarded.
COLUMN_ORDER = [
    "ticker",
    "company_name",
    "fiscal_year_end",       # calendar year the fiscal year ENDS in
    "date_range_used",
    "period_end_date",       # <- the column the header forgets to name
    "total_assets",
    "current_assets",
    "current_liabilities",
    "retained_earnings",
    "ebit",
    "total_equity_book",
    "total_liabilities",
    "roa",
    "debt_to_equity",
    "interest_coverage",
    "cur_mkt_cap",
    "cur_ratio",
]

NUMERIC_COLUMNS = COLUMN_ORDER[5:]

# Header layout we EXPECT to see. If a future export differs, the positional
# assumption may no longer hold, so we warn rather than assume.
EXPECTED_HEADER_START = ["ticker", "company_name", "fiscal_year", "date_range_used"]


def _read_raw(path: Path, sheet=None) -> pd.DataFrame:
    """Reads the file with NO header interpretation at all."""
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet or 0, header=None, dtype=object)
    else:
        import csv
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            rows = list(csv.reader(fh))
        width = max(len(r) for r in rows)
        raw = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=object)
    return raw


def _check_header_assumption(header_row, verbose=True):
    """Confirms the off-by-one defect is still present before relying on it."""
    got = [str(h).strip().lower() for h in header_row[:4]]
    if got != EXPECTED_HEADER_START:
        print(f"[load_filled] WARNING: unexpected header start {got}, expected "
              f"{EXPECTED_HEADER_START}. Verify COLUMN_ORDER still matches this export.")
        return

    # position 4 named 'total_assets' is the signature of the off-by-one
    fifth = str(header_row[4]).strip().lower() if len(header_row) > 4 else ""
    if fifth == "total_assets" and verbose:
        print("[load_filled] Header/data off-by-one detected (header names "
              "'total_assets' at position 5, data holds the period-end date there). "
              "Reading by POSITION — header names discarded. See defect 1.")
    elif fifth in {"period_end_date", "date"} and verbose:
        print("[load_filled] NOTE: this export appears to name the period-end-date "
              "column, i.e. the off-by-one has been fixed upstream. COLUMN_ORDER "
              "still matches, but re-check the balance-sheet identity result below.")


def fiscal_year_end_to_indian_fy(year_end):
    """Calendar year of fiscal-year END -> (Indian FY label, FY start year).

    Per=CY + a 31-March year end means fiscal_year 2020 covers Apr-2019 to
    Mar-2020 = 'FY2019-20', start year 2019. The start year is what
    label_construction.py sorts on and what train_model.py should use as
    time_col, so both are returned together and never derived separately.
    """
    if pd.isna(year_end):
        return None, None
    try:
        y = int(float(year_end))   # xlsx gives 2015.0, csv gives '2015'
    except (TypeError, ValueError):
        return None, None
    if not (1990 <= y <= 2100):
        return None, None
    start = y - 1
    return f"FY{start}-{str(start + 1)[-2:]}", start


# ── validation ──────────────────────────────────────────────────────────
def verify_balance_sheet_identity(df: pd.DataFrame, tolerance=1e-3, verbose=True) -> bool:
    """Proves the positional column mapping is right: TA == TE + TL.

    This is the single check that distinguishes a correctly-parsed file from
    a silently shifted one, because a one-column shift destroys the identity
    immediately. Run on EVERY load — it costs nothing and it is the only
    thing standing between you and a model trained on misaligned columns.
    """
    need = ["total_assets", "total_equity_book", "total_liabilities"]
    if not all(c in df.columns for c in need):
        print("[load_filled] Cannot verify balance-sheet identity — columns missing.")
        return False

    m = df[need].notna().all(axis=1)
    if m.sum() == 0:
        print("[load_filled] Cannot verify balance-sheet identity — no complete rows.")
        return False

    diff = (df.loc[m, "total_assets"]
            - (df.loc[m, "total_equity_book"] + df.loc[m, "total_liabilities"])).abs()
    rel = diff / df.loc[m, "total_assets"].abs().replace(0, np.nan)
    ok_frac = float((rel < tolerance).mean())

    if verbose:
        print(f"[load_filled] Balance-sheet identity TA == TE + TL on {int(m.sum())} "
              f"complete rows: {ok_frac*100:.1f}% within {tolerance:.1%}, "
              f"max relative error {rel.max():.2e}")

    if ok_frac < 0.95:
        print("  *** FAILED. The column mapping is almost certainly misaligned — "
              "the export's layout has changed. Do NOT use this data until "
              "COLUMN_ORDER is corrected. ***")
        return False
    return True


def flag_unit_outliers(df: pd.DataFrame, max_log10_gap=2.5, verbose=True) -> pd.DataFrame:
    """Flags companies whose reporting scale is orders of magnitude off.

    Catches wrong-security and wrong-currency pulls (defect 2) generically:
    compares each company's median total_assets to the cross-sectional median
    on a log10 scale. A genuine size difference between a small cap and a
    mega cap is ~2 orders; 2.5+ means a different unit or a different company.
    """
    out = df.copy()
    med = out.groupby("ticker")["total_assets"].median()
    med = med[med > 0]
    if med.empty:
        out["unit_outlier_flag"] = False
        return out

    log_med = np.log10(med)
    centre = log_med.median()
    gap = (log_med - centre).abs()
    outliers = gap[gap > max_log10_gap]

    out["unit_outlier_flag"] = out["ticker"].isin(outliers.index)

    if verbose and len(outliers):
        print(f"[load_filled] UNIT/TICKER OUTLIERS — {len(outliers)} company(ies) report on a "
              f"scale >{max_log10_gap} orders of magnitude from the sample median "
              f"(10^{centre:.1f}):")
        for tkr, g in outliers.sort_values(ascending=False).items():
            nm = out.loc[out.ticker == tkr, "company_name"].iloc[0]
            print(f"    {tkr:22s} {str(nm)[:32]:34s} median TA={med[tkr]:,.2f}  "
                  f"({g:.1f} orders off)")
        print("    -> almost always the WRONG SECURITY or a different reporting "
              "currency. Verify the ticker before including these rows.")
    return out


def report_dead_tickers(df: pd.DataFrame, failed_csv="data/raw/failed_tickers.csv",
                        verbose=True) -> pd.DataFrame:
    """Finds tickers with no fundamentals at all and writes them for repair.

    These are NOT dropped silently (defect 3): delisted-after-bankruptcy names
    are precisely the positive class, so losing them preferentially would
    gut the model's ability to learn distress at all.
    """
    per = df.groupby(["ticker", "company_name"]).agg(
        n_years=("fiscal_year_end", "size"),
        n_with_assets=("total_assets", "count"),
        n_with_mktcap=("cur_mkt_cap", "count"),
    ).reset_index()

    dead = per[per["n_with_assets"] == 0].copy()
    if len(dead):
        dead["reason"] = np.where(
            dead["n_with_mktcap"] > 0,
            "market data only — no fundamentals (delisted? wrong ticker?)",
            "no data at all — likely Invalid Security / wrong exchange suffix",
        )
        Path(failed_csv).parent.mkdir(parents=True, exist_ok=True)
        dead.to_csv(failed_csv, index=False)
        if verbose:
            print(f"[load_filled] DEAD TICKERS — {len(dead)} company(ies) returned no "
                  f"fundamentals, written to {failed_csv}:")
            for _, r in dead.iterrows():
                print(f"    {r['ticker']:22s} {str(r['company_name'])[:32]:34s} {r['reason']}")
            print("    -> these are disproportionately DISTRESSED names. Repair the "
                  "tickers rather than dropping them, or you bias the sample toward "
                  "survivors and lose the positive class.")
    return dead


# ── main loader ─────────────────────────────────────────────────────────
def load_filled_template(path, sheet=None, failed_csv="data/raw/failed_tickers.csv",
                         drop_unit_outliers=False, verbose=True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Filled template not found: {path}")

    raw = _read_raw(path, sheet)
    if raw.shape[1] < len(COLUMN_ORDER):
        raise ValueError(
            f"Export has {raw.shape[1]} columns, expected at least {len(COLUMN_ORDER)}. "
            f"Layout has changed — re-check COLUMN_ORDER against the file."
        )

    _check_header_assumption(list(raw.iloc[0]), verbose=verbose)

    body = raw.iloc[1:, :len(COLUMN_ORDER)].copy()
    body.columns = COLUMN_ORDER
    body = body.reset_index(drop=True)

    # strings first, so blank trailing rows can be identified before typing
    for c in ["ticker", "company_name"]:
        body[c] = body[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    body = body[body["ticker"].notna()].copy()

    for c in NUMERIC_COLUMNS:
        body[c] = body[c].map(clean_numeric)

    fy = body["fiscal_year_end"].map(fiscal_year_end_to_indian_fy)
    body["fiscal_year"] = [f[0] for f in fy]
    body["fiscal_year_num"] = [f[1] for f in fy]
    body["fiscal_year_end"] = pd.to_numeric(body["fiscal_year_end"], errors="coerce").astype("Int64")

    bad_fy = int(body["fiscal_year"].isna().sum())
    if bad_fy:
        print(f"[load_filled] Dropping {bad_fy} row(s) with an unparseable fiscal year.")
        body = body[body["fiscal_year"].notna()].copy()

    body["period_end_date"] = pd.to_datetime(body["period_end_date"], errors="coerce",
                                             format="mixed", dayfirst=False)

    ordered = (["ticker", "company_name", "fiscal_year", "fiscal_year_num",
                "fiscal_year_end", "period_end_date"] + NUMERIC_COLUMNS)
    ordered = [c for c in ordered if c in body.columns and c != "date_range_used"]
    panel = body[ordered].sort_values(["ticker", "fiscal_year_num"]).reset_index(drop=True)

    if verbose:
        print(f"[load_filled] Parsed {len(panel)} row(s), {panel['ticker'].nunique()} "
              f"companies, {panel['fiscal_year'].min()} to {panel['fiscal_year'].max()}.")

    verify_balance_sheet_identity(panel, verbose=verbose)
    panel = add_negative_equity_handling(panel, verbose=verbose)
    panel = flag_unit_outliers(panel, verbose=verbose)
    report_dead_tickers(panel, failed_csv=failed_csv, verbose=verbose)

    if drop_unit_outliers:
        n = int(panel["unit_outlier_flag"].sum())
        panel = panel[~panel["unit_outlier_flag"]].copy()
        if verbose:
            print(f"[load_filled] Dropped {n} unit-outlier row(s) (--drop-unit-outliers).")

    return panel


def coverage_by_year(panel: pd.DataFrame) -> None:
    n_co = panel["ticker"].nunique()
    print("\n=== total_assets coverage by fiscal year ===")
    for fy, g in panel.groupby("fiscal_year"):
        print(f"  {fy}  {g['total_assets'].notna().sum():3d}/{n_co}")
    print("\n=== field coverage overall ===")
    for c in NUMERIC_COLUMNS:
        if c in panel:
            print(f"  {c:22s} {panel[c].notna().mean()*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("-o", "--out", default="data/processed/bloomberg_panel.csv")
    ap.add_argument("--failed-csv", default="data/raw/failed_tickers.csv")
    ap.add_argument("--drop-unit-outliers", action="store_true",
                    help="Exclude companies flagged as reporting on the wrong scale")
    args = ap.parse_args()

    panel = load_filled_template(args.path, sheet=args.sheet,
                                 failed_csv=args.failed_csv,
                                 drop_unit_outliers=args.drop_unit_outliers)
    coverage_by_year(panel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    print(f"\nWrote {len(panel)} row(s) x {panel.shape[1]} col(s) -> {out}")


if __name__ == "__main__":
    main()
