"""
load_bloomberg_export.py
=========================
Reshapes a REAL Bloomberg column-block export (the "Data Pull" sheet shape
produced by the pull template) into the long/tidy panel that
altman_zscore.py, label_construction.py and train_model.py all expect.

INPUT SHAPE (wide, one column-block per company)
------------------------------------------------
    row 1:  ticker          RELIANCE IN                       TCS IN
    row 2:  company name    Reliance Industries Ltd           Tata Consultancy...
    row 3:  field headers   Date | total_assets | ebit |...   Date | total_assets |...
    row 4+: data            31/03/2015 | 700000 | 40000 |...  31/03/2015 | ...

The ticker/company name usually appear ONLY in the first column of each
block (merged cells export as one value + blanks), and blocks may or may
not be separated by an empty spacer column. This parser handles both, and
does not care how many companies, fields or years there are — it is written
against the SHAPE, not against a specific file.

OUTPUT SHAPE (long/tidy)
-------------------------
    ticker | company_name | fiscal_year | fiscal_year_num | period_end_date |
    total_assets | current_assets | ... | negative_equity_flag

    fiscal_year      "FY2019-20"  (same string convention as label_construction.py,
                                   so the two tables join on ticker+fiscal_year)
    fiscal_year_num  2019         (FY START year — matches label_construction's
                                   FY(\\d{4}) sort key; use this as train_model's
                                   time_col)

TWO REAL DATA ISSUES THIS FILE HANDLES (both confirmed on the 5-company test pull)
----------------------------------------------------------------------------------
1. FINANCIAL-SECTOR COMPANIES (banks/NBFCs) return #N/A on current_assets,
   current_liabilities, ebit, interest_coverage and cur_ratio — confirmed
   identically on HDFC Bank and Yes Bank. Banks have no meaningful
   current/non-current balance-sheet split, so these fields do not exist
   rather than failing to fetch. Consistent with Altman's original
   methodology, financial-sector companies are EXCLUDED from the main
   distress model: pass --universe <csv> --exclude-financial (needs a
   `sector_type` column in the universe CSV).

2. NEGATIVE BOOK EQUITY makes debt_to_equity come back #N/A (confirmed on
   Suzlon FY2015-FY2021). This is mathematically expected, NOT a fetch
   error: a debt/equity ratio against a negative denominator is not
   interpretable. Handling here:
     (a) `negative_equity_flag` (1/0) is created as a real feature — for a
         distressed-firm model this is one of the most informative single
         bits on the balance sheet, so it must not be thrown away;
     (b) debt_to_equity is left as NaN in those rows and is NEVER zero-filled
         — zero-filling would encode "no leverage" for the most levered
         firms in the sample, inverting the signal;
     (c) altman_zscore.py handles the X4 = equity/liabilities consequence
         separately (see the guard added there).

USAGE
-----
    python src/load_bloomberg_export.py data/raw/bloomberg_export.csv
    python src/load_bloomberg_export.py data/raw/bloomberg_pull_template.xlsx \\
        --sheet "Data Pull" \\
        --universe data/raw/company_universe_starter.csv --exclude-financial \\
        -o data/processed/bloomberg_panel.csv

    or from your pipeline:
        from src.load_bloomberg_export import load_bloomberg_export
        panel = load_bloomberg_export("data/raw/bloomberg_export.csv")
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Row layout of the wide export ───────────────────────────────────────
# 0-indexed positions of the three header rows described in the docstring.
# Override via --ticker-row/--name-row/--field-row if your export differs.
DEFAULT_TICKER_ROW = 0
DEFAULT_NAME_ROW = 1
DEFAULT_FIELD_ROW = 2

# ── Field-name normalisation ────────────────────────────────────────────
# Maps whatever the export calls a field -> the canonical name the rest of
# the pipeline uses. Keys are matched after lowercasing and stripping all
# non-alphanumerics, so "Total Assets", "TOTAL_ASSETS" and "BS_TOT_ASSET"
# all land on the same entry. EXTEND THIS as you meet new headers; unknown
# fields are kept (slugified) and reported, never silently dropped.
FIELD_ALIASES = {
    # --- balance sheet ---
    "totalassets": "total_assets",
    "bstotasset": "total_assets",
    "currentassets": "current_assets",
    "bscurassetreport": "current_assets",
    "totalcurrentassets": "current_assets",
    "currentliabilities": "current_liabilities",
    "bscurliab": "current_liabilities",
    "totalcurrentliabilities": "current_liabilities",
    "retainedearnings": "retained_earnings",
    "retainearn": "retained_earnings",
    "bsretainearn": "retained_earnings",
    "totalequitybook": "total_equity_book",
    "totalequity": "total_equity_book",
    "totalcommonequity": "total_equity_book",
    "bstotalequity": "total_equity_book",
    "totalliabilities": "total_liabilities",
    "bstotliab2": "total_liabilities",
    "shortandlongtermdebt": "total_debt",
    "totaldebt": "total_debt",
    # --- income statement ---
    "ebit": "ebit",
    "isoperinc": "ebit",
    "operatingincome": "ebit",
    "netincome": "net_income",
    "revenue": "revenue",
    "salesrevturn": "revenue",
    "totalrevenue": "revenue",
    # --- ratios ---
    "debttoequity": "debt_to_equity",
    "totdebttototeqy": "debt_to_equity",
    "totaldebttototalequity": "debt_to_equity",
    "interestcoverage": "interest_coverage",
    "interestcoverageratio": "interest_coverage",
    "ebittointerestexpn": "interest_coverage",
    "curratio": "cur_ratio",
    "currentratio": "cur_ratio",
    # --- market ---
    "marketcap": "market_cap",
    "curmktcap": "market_cap",
}

# Header text that identifies the per-block DATE/PERIOD column rather than a
# data field. Matched against the normalised (alphanumeric-only) header.
DATE_HEADER_TOKENS = {"date", "dates", "period", "perioddate", "periodend",
                      "periodenddate", "fiscalyear", "fy", "year", "asofdate"}

# Bloomberg's ways of saying "no value". All become NaN — never 0.
NA_STRINGS = {
    "", "-", "--", "n/a", "na", "nan", "none", "null",
    "#n/a", "#n/a n/a", "#n/a na", "#n/a field not applicable",
    "#n/a invalid security", "#n/a requesting data...", "#n/a history",
    "#value!", "#name?", "#ref!", "#div/0!",
}

# Fields confirmed to be structurally unavailable for banks/NBFCs — used
# only to explain the pattern in the coverage report.
FINANCIAL_SECTOR_NA_FIELDS = [
    "current_assets", "current_liabilities", "ebit",
    "interest_coverage", "cur_ratio",
]


# ── small helpers ───────────────────────────────────────────────────────
def _norm_header(s) -> str:
    """'Total Assets (INR mn)' -> 'totalassetsinrmn'; used for alias lookup."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def _slugify(s) -> str:
    """Fallback name for headers with no alias entry."""
    out = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return out or "unnamed_field"


def _is_blank(s) -> bool:
    return s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == ""


def canonical_field_name(raw_header) -> tuple:
    """Returns (canonical_name, was_recognised)."""
    key = _norm_header(raw_header)
    if key in FIELD_ALIASES:
        return FIELD_ALIASES[key], True
    return _slugify(raw_header), False


def clean_numeric(value):
    """Bloomberg cell -> float or NaN.

    Handles: #N/A variants, thousands separators, trailing/leading currency
    or unit text, percent signs, and accounting-style negatives '(1,234)'.
    Anything unparseable becomes NaN and is counted in the coverage report —
    it is NEVER coerced to 0, because 0 is a meaningful financial value and
    silently inventing it corrupts every downstream ratio.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return np.nan if pd.isna(value) else float(value)

    s = str(value).strip()
    if s.lower() in NA_STRINGS:
        return np.nan
    if s.lower().startswith("#n/a"):
        return np.nan

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]

    s = s.replace(",", "").replace("%", "").strip()
    m = re.search(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        out = float(m.group(0))
    except ValueError:
        return np.nan
    return -out if negative else out


def parse_fiscal_year(value, fiscal_year_start_month: int = 4):
    """Period cell -> (fiscal_year_label, fiscal_year_start, period_end_date).

    Accepts a real date (Bloomberg yearly periodicity returns period-END
    dates, e.g. 31/03/2020), a bare year ('2020'), or an already-formatted
    'FY2019-20' / 'FY20' string.

    Indian FY runs April(y) -> March(y+1), so a 31/03/2020 period end is
    FY2019-20 — the SAME convention label_construction.py uses, which is
    what makes the two tables joinable on ticker+fiscal_year. Getting this
    off by one silently misaligns every label by a year, so it is derived in
    exactly one place: here.
    """
    if _is_blank(value):
        return None, None, None

    s = str(value).strip()

    # Already a fiscal-year label: FY2019-20 / FY2019-2020 / FY19-20
    m = re.fullmatch(r"FY\s*(\d{2}|\d{4})\s*[-/]\s*(\d{2}|\d{4})", s, flags=re.I)
    if m:
        start = int(m.group(1))
        if start < 100:
            start += 2000
        return f"FY{start}-{str(start + 1)[-2:]}", start, None

    # A bare year, with or without an FY prefix: 2020 / FY2020
    m = re.fullmatch(r"(?:FY\s*)?(\d{4})", s, flags=re.I)
    if m:
        year = int(m.group(1))
        # A bare year on a yearly Bloomberg pull is the year the fiscal year
        # ENDS, matching the period-end-date convention below.
        start = year - 1
        return f"FY{start}-{str(start + 1)[-2:]}", start, None

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        return None, None, None

    start = dt.year if dt.month >= fiscal_year_start_month else dt.year - 1
    return f"FY{start}-{str(start + 1)[-2:]}", start, dt.normalize()


# ── reading ─────────────────────────────────────────────────────────────
def _read_ragged_csv(path: Path) -> pd.DataFrame:
    """Reads a CSV whose rows have DIFFERENT column counts.

    Real exports are ragged: the ticker and company-name rows often stop at
    the last non-empty cell while the data rows run wider, and trailing
    spacer commas are inconsistent. pandas' C parser raises
    'Expected N fields, saw M' on exactly this. So parse with the stdlib csv
    reader (which handles quoting/embedded commas correctly) and pad every
    row out to the widest one. Everything stays a string here; typing is
    clean_numeric's job.
    """
    import csv

    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        raise ValueError(f"Export is empty: {path}")

    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded, dtype=object)


# ── block detection ─────────────────────────────────────────────────────
def _detect_blocks(raw: pd.DataFrame, ticker_row: int, name_row: int, field_row: int):
    """Splits the wide sheet into per-company column blocks.

    A block starts at any column carrying a ticker value. Merged-cell
    exports put the ticker in the block's first column only, so a column
    with a blank ticker but a non-blank field header belongs to the block on
    its left. Fully-blank spacer columns end a block.
    """
    blocks = []
    current = None

    for col in range(raw.shape[1]):
        ticker = raw.iat[ticker_row, col] if ticker_row < len(raw) else None
        name = raw.iat[name_row, col] if name_row < len(raw) else None
        field = raw.iat[field_row, col] if field_row < len(raw) else None

        if not _is_blank(ticker):
            current = {
                "ticker": str(ticker).strip(),
                "company_name": None if _is_blank(name) else str(name).strip(),
                "cols": [],
            }
            blocks.append(current)

        if _is_blank(field):
            if _is_blank(ticker):
                current = None  # spacer column terminates the open block
            continue

        if current is None:
            continue  # a stray labelled column before any ticker — ignore
        if current["company_name"] is None and not _is_blank(name):
            current["company_name"] = str(name).strip()
        current["cols"].append((col, field))

    return [b for b in blocks if b["cols"]]


def _block_to_long(raw, block, field_row, fiscal_year_start_month, unknown_headers):
    """Turns one company's column block into a list of tidy row-dicts."""
    date_col = None
    field_cols = []

    for col, header in block["cols"]:
        if date_col is None and _norm_header(header) in DATE_HEADER_TOKENS:
            date_col = col
            continue
        canonical, recognised = canonical_field_name(header)
        if not recognised:
            unknown_headers.setdefault(str(header).strip(), canonical)
        field_cols.append((col, canonical))

    # No explicit date header -> fall back to the block's first column, which
    # is where Bloomberg puts the period on every pull shape we've seen.
    if date_col is None and block["cols"]:
        date_col = block["cols"][0][0]
        field_cols = [(c, n) for c, n in field_cols if c != date_col]

    rows = []
    for r in range(field_row + 1, len(raw)):
        fy_label, fy_num, period_end = parse_fiscal_year(
            raw.iat[r, date_col], fiscal_year_start_month
        )
        if fy_label is None:
            continue  # blank tail row, or a footnote line under the data

        record = {
            "ticker": block["ticker"],
            "company_name": block["company_name"],
            "fiscal_year": fy_label,
            "fiscal_year_num": fy_num,
            "period_end_date": period_end,
        }
        for col, canonical in field_cols:
            record[canonical] = clean_numeric(raw.iat[r, col])

        # Drop rows that carried a date but no data at all (common trailing
        # artefact when a template pre-fills more year rows than were pulled).
        if all(pd.isna(record.get(n)) for _, n in field_cols):
            continue
        rows.append(record)

    return rows


# ── negative-equity handling ────────────────────────────────────────────
def add_negative_equity_handling(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Creates negative_equity_flag and enforces the debt_to_equity NaN rule.

    See issue 2 in the module docstring. Two separate things happen here and
    it matters that they stay separate:

      * negative_equity_flag is a FEATURE (1 = book equity below zero). It is
        computed from total_equity_book, so it is available even in rows
        where debt_to_equity itself was returned as #N/A.
      * debt_to_equity is forced to NaN wherever equity is negative, even if
        the export happened to return a number there. A negative denominator
        produces a negative "leverage" ratio that sorts alongside the
        LEAST-levered firms in the sample — worse than missing. NaN is
        honest; XGBoost handles it natively, and any model that cannot should
        be given the flag plus an explicit imputation you document.
    """
    out = df.copy()

    if "total_equity_book" not in out.columns:
        if verbose:
            print("[load_bloomberg] total_equity_book absent — negative_equity_flag "
                  "not created. Check the pull template covered it.")
        return out

    eq = out["total_equity_book"]
    out["negative_equity_flag"] = np.where(eq.isna(), np.nan, (eq < 0).astype(float))

    n_neg = int((out["negative_equity_flag"] == 1).sum())
    if "debt_to_equity" in out.columns:
        overwritten = int(((out["negative_equity_flag"] == 1)
                           & out["debt_to_equity"].notna()).sum())
        out.loc[out["negative_equity_flag"] == 1, "debt_to_equity"] = np.nan
        if verbose and overwritten:
            print(f"[load_bloomberg] Forced {overwritten} debt_to_equity value(s) to NaN "
                  f"where book equity is negative (uninterpretable denominator).")

    if verbose:
        print(f"[load_bloomberg] negative_equity_flag=1 on {n_neg} row(s).")
        if n_neg:
            per_ticker = (out[out["negative_equity_flag"] == 1]
                          .groupby("ticker")["fiscal_year"]
                          .agg(lambda s: f"{s.min()}..{s.max()} ({len(s)})"))
            for tkr, span in per_ticker.items():
                print(f"    {tkr}: {span}")
            print("    -> expected for genuinely distressed firms; debt_to_equity left "
                  "NaN (not zero-filled) in these rows.")

    return out


# ── sector merge / financial exclusion ──────────────────────────────────
def attach_sector_type(panel: pd.DataFrame, universe_csv: str,
                       exclude_financial: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
    """Joins sector_type from the universe CSV and optionally drops financials.

    Matches on ticker_hint, tolerating the Bloomberg suffix ('YESBANK IN'
    vs 'YESBANK'). If the universe CSV has no sector_type column yet, this
    warns and returns the panel untouched rather than failing — the tagging
    lives in the universe file, not here.
    """
    uni_path = Path(universe_csv)
    if not uni_path.exists():
        print(f"[load_bloomberg] Universe CSV not found: {uni_path} — skipping sector merge.")
        return panel

    uni = pd.read_csv(uni_path)
    if "sector_type" not in uni.columns:
        print(f"[load_bloomberg] WARNING: {uni_path.name} has no 'sector_type' column "
              f"(found: {list(uni.columns)}).")
        print("  -> Tag each company financial/non_financial in the universe CSV first; "
              "no sector merge or financial exclusion performed.")
        return panel

    ticker_col = "ticker_hint" if "ticker_hint" in uni.columns else "ticker"
    if ticker_col not in uni.columns:
        print("[load_bloomberg] WARNING: universe CSV has no ticker_hint/ticker column — "
              "skipping sector merge.")
        return panel

    def base(t):
        return re.split(r"\s+", str(t).strip().upper())[0]

    lookup = (uni.assign(_key=uni[ticker_col].map(base))
                 .drop_duplicates("_key")
                 .set_index("_key")["sector_type"]
                 .astype(str).str.strip().str.lower())

    out = panel.copy()
    out["sector_type"] = out["ticker"].map(base).map(lookup)

    unmatched = sorted(out.loc[out["sector_type"].isna(), "ticker"].unique())
    if unmatched:
        print(f"[load_bloomberg] No sector_type match for: {unmatched}")
        print("  -> Check the ticker_hint spelling in the universe CSV; these rows are "
              "kept but cannot be filtered by sector.")

    if exclude_financial:
        is_fin = out["sector_type"] == "financial"
        dropped_tickers = sorted(out.loc[is_fin, "ticker"].unique())
        out = out[~is_fin].copy()
        if verbose:
            print(f"[load_bloomberg] Excluded {int(is_fin.sum())} row(s) from "
                  f"{len(dropped_tickers)} financial-sector company(ies): {dropped_tickers}")
            print("  -> Banks/NBFCs have no meaningful current/non-current split, so "
                  "X1 and X3 of the Altman Z\" are undefined for them (issue 1).")

    return out


# ── main entry point ────────────────────────────────────────────────────
def load_bloomberg_export(
    path: str,
    sheet: str = None,
    ticker_row: int = DEFAULT_TICKER_ROW,
    name_row: int = DEFAULT_NAME_ROW,
    field_row: int = DEFAULT_FIELD_ROW,
    fiscal_year_start_month: int = 4,
    universe_csv: str = None,
    exclude_financial: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Wide column-block export -> tidy (ticker, fiscal_year, features...) panel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Export not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet or 0, header=None, dtype=object)
    else:
        raw = _read_ragged_csv(path)

    if len(raw) <= field_row:
        raise ValueError(
            f"Export has only {len(raw)} row(s) — expected header rows at "
            f"{ticker_row}/{name_row}/{field_row} plus data below."
        )

    blocks = _detect_blocks(raw, ticker_row, name_row, field_row)
    if not blocks:
        raise ValueError(
            "No company column-blocks detected. Expected tickers on row "
            f"{ticker_row + 1}, company names on row {name_row + 1} and field "
            f"headers on row {field_row + 1} (1-indexed). Pass --ticker-row/"
            "--name-row/--field-row if your export is laid out differently."
        )

    unknown_headers = {}
    records = []
    for block in blocks:
        records.extend(_block_to_long(raw, block, field_row,
                                      fiscal_year_start_month, unknown_headers))

    if not records:
        raise ValueError("Blocks were detected but no data rows parsed — check that the "
                         "first column of each block holds parseable period dates.")

    panel = pd.DataFrame(records)
    panel = panel.sort_values(["ticker", "fiscal_year_num"]).reset_index(drop=True)

    # Collapse accidental duplicate columns (same field pulled twice in a block)
    panel = panel.loc[:, ~panel.columns.duplicated()]

    if verbose:
        print(f"[load_bloomberg] Parsed {len(blocks)} company block(s) -> "
              f"{len(panel)} row(s), {panel['fiscal_year'].nunique()} fiscal year(s) "
              f"({panel['fiscal_year'].min()} to {panel['fiscal_year'].max()}).")
        if unknown_headers:
            print("[load_bloomberg] Unrecognised field headers (kept under slugified "
                  "names — add them to FIELD_ALIASES if the pipeline should use them):")
            for original, slug in unknown_headers.items():
                print(f"    {original!r} -> {slug}")

    panel = add_negative_equity_handling(panel, verbose=verbose)

    if universe_csv:
        panel = attach_sector_type(panel, universe_csv,
                                   exclude_financial=exclude_financial, verbose=verbose)

    return panel


# ── coverage report ─────────────────────────────────────────────────────
def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker, per-field non-null coverage. Run this on EVERY new pull
    before trusting it — it is what surfaced both known data issues."""
    meta = {"ticker", "company_name", "fiscal_year", "fiscal_year_num",
            "period_end_date", "sector_type"}
    fields = [c for c in panel.columns if c not in meta]

    cov = (panel.groupby("ticker")[fields]
                .apply(lambda g: g.notna().mean() * 100)
                .round(1))

    print("\n=== Field coverage by ticker (% of fiscal years with a value) ===")
    print(cov.to_string())

    print("\n=== Overall field coverage ===")
    overall = (panel[fields].notna().mean() * 100).round(1).sort_values()
    for field, pct in overall.items():
        note = ""
        if field in FINANCIAL_SECTOR_NA_FIELDS and pct < 100:
            note = "  <- expected #N/A for banks/NBFCs (issue 1)"
        if field == "debt_to_equity" and pct < 100:
            note = "  <- expected NaN where book equity is negative (issue 2)"
        print(f"  {field:28s} {pct:5.1f}%{note}")

    zero_cov = [f for f, p in overall.items() if p == 0]
    if zero_cov:
        print(f"\n  WARNING: {zero_cov} came back empty for EVERY company — that is a "
              "pull/template problem, not a data problem. Re-check those fields.")

    return cov


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Wide Bloomberg export (.csv or .xlsx)")
    ap.add_argument("-o", "--out", default="data/processed/bloomberg_panel.csv",
                    help="Output tidy CSV path")
    ap.add_argument("--sheet", default=None, help="Sheet name for .xlsx input (e.g. 'Data Pull')")
    ap.add_argument("--ticker-row", type=int, default=DEFAULT_TICKER_ROW + 1,
                    help="1-indexed row holding tickers (default 1)")
    ap.add_argument("--name-row", type=int, default=DEFAULT_NAME_ROW + 1,
                    help="1-indexed row holding company names (default 2)")
    ap.add_argument("--field-row", type=int, default=DEFAULT_FIELD_ROW + 1,
                    help="1-indexed row holding field headers (default 3)")
    ap.add_argument("--fy-start-month", type=int, default=4,
                    help="Fiscal year start month (default 4 = Indian FY)")
    ap.add_argument("--universe", default=None,
                    help="company_universe_starter.csv, to attach sector_type")
    ap.add_argument("--exclude-financial", action="store_true",
                    help="Drop sector_type=='financial' companies (requires --universe)")
    args = ap.parse_args()

    if args.exclude_financial and not args.universe:
        ap.error("--exclude-financial requires --universe")

    panel = load_bloomberg_export(
        args.path,
        sheet=args.sheet,
        ticker_row=args.ticker_row - 1,
        name_row=args.name_row - 1,
        field_row=args.field_row - 1,
        fiscal_year_start_month=args.fy_start_month,
        universe_csv=args.universe,
        exclude_financial=args.exclude_financial,
    )

    coverage_report(panel)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    print(f"\nWrote {len(panel)} row(s) x {panel.shape[1]} column(s) -> {out_path}")
    print("Next: feed this to compute_altman_zscore_components(), then join to "
          "distress_labels.csv on ticker + fiscal_year.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # ── Self-test against a synthetic export that reproduces BOTH real
        # data issues, so the handling is proven before the real CSV lands.
        print("No path given — running the built-in self-test.\n")
        import csv as _csv
        import tempfile

        # Three column-blocks reproducing BOTH confirmed real data issues:
        #   RELIANCE — clean non-financial, all fields present
        #   YESBANK  — financial sector: current_assets/EBIT structurally #N/A (issue 1)
        #   SUZLON   — negative book equity in 2 of 3 years, debt_to_equity #N/A there (issue 2)
        # Built as explicit per-block column lists rather than hand-counted
        # commas: an off-by-one here silently shifts a ticker onto the previous
        # block's last field, which is exactly the kind of misalignment this
        # parser exists to get right.
        blocks_spec = [
            ("RELIANCE IN", "Reliance Industries Ltd",
             ["Date", "Total Assets", "Current Assets", "EBIT", "Total Equity", "Total Liabilities"],
             [["31/03/2020", "700,000", "250,000", "60,000", "400,000", "300,000"],
              ["31/03/2021", "780,000", "270,000", "65,000", "450,000", "330,000"],
              ["31/03/2022", "850,000", "290,000", "72,000", "500,000", "350,000"]]),
            ("YESBANK IN", "Yes Bank Ltd",
             ["Date", "Total Assets", "Current Assets", "EBIT", "Total Equity", "Total Liabilities"],
             [["31/03/2020", "257,000", "#N/A N/A", "#N/A Field Not Applicable", "21,000", "236,000"],
              ["31/03/2021", "273,000", "#N/A N/A", "#N/A Field Not Applicable", "24,000", "249,000"],
              ["31/03/2022", "318,000", "#N/A N/A", "#N/A Field Not Applicable", "33,000", "285,000"]]),
            ("SUZLON IN", "Suzlon Energy Ltd",
             ["Date", "Total Assets", "Total Equity", "Total Liabilities", "Debt to Equity"],
             [["31/03/2020", "14,000", "(3,500)", "17,500", "#N/A N/A"],
              ["31/03/2021", "13,200", "(4,100)", "17,300", "#N/A N/A"],
              ["31/03/2022", "12,900", "1,200", "11,700", "9.75"]]),
        ]

        n_years = 3
        ticker_row, name_row, field_row = [""], [""], [""]   # leading spacer column
        data_rows = [[""] for _ in range(n_years)]
        for tkr, nm, headers, values in blocks_spec:
            ticker_row += [tkr] + [""] * (len(headers) - 1)
            name_row += [nm] + [""] * (len(headers) - 1)
            field_row += headers
            for i in range(n_years):
                data_rows[i] += values[i]
            # blank spacer column between blocks — real templates have these
            for row in (ticker_row, name_row, field_row, *data_rows):
                row.append("")

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(ticker_row)
            w.writerow(name_row)
            w.writerow(field_row)
            w.writerows(data_rows)
            w.writerow([])            # trailing blank line, as exports have
            tmp_path = fh.name

        panel = load_bloomberg_export(tmp_path)
        print()
        print(panel.to_string(index=False))
        coverage_report(panel)

        # ── assertions: the self-test must actually PROVE the handling works
        rel = panel[panel["ticker"] == "RELIANCE IN"]
        yes = panel[panel["ticker"] == "YESBANK IN"]
        suz = panel[panel["ticker"] == "SUZLON IN"]

        assert len(panel) == 9, f"expected 9 rows (3 companies x 3 years), got {len(panel)}"
        assert list(rel["fiscal_year"]) == ["FY2019-20", "FY2020-21", "FY2021-22"], \
            "31/03/2020 must map to FY2019-20 (Indian FY), matching label_construction.py"
        assert rel["total_assets"].tolist() == [700000.0, 780000.0, 850000.0], \
            "thousands separators must parse"
        # every block's LAST field must survive — this is where an off-by-one
        # in block detection shows up first (the next ticker overwriting it)
        assert rel["total_liabilities"].tolist() == [300000.0, 330000.0, 350000.0], \
            "last field of a block must not be swallowed by the next block's ticker"
        assert rel["current_assets"].tolist() == [250000.0, 270000.0, 290000.0]
        assert rel["ebit"].tolist() == [60000.0, 65000.0, 72000.0]
        assert rel["total_equity_book"].tolist() == [400000.0, 450000.0, 500000.0]
        # issue 1: bank fields structurally #N/A, never zero-filled
        assert yes["current_assets"].isna().all() and yes["ebit"].isna().all(), \
            "financial-sector #N/A fields must be NaN, not 0"
        assert yes["total_assets"].tolist() == [257000.0, 273000.0, 318000.0], \
            "bank fields that DO exist must survive"
        assert yes["total_equity_book"].tolist() == [21000.0, 24000.0, 33000.0]
        assert yes["total_liabilities"].tolist() == [236000.0, 249000.0, 285000.0]
        assert (yes["negative_equity_flag"] == 0).all(), \
            "present, positive equity must give flag=0 (not NaN)"
        # issue 2: negative equity flagged, debt_to_equity kept NaN
        assert suz["total_equity_book"].tolist() == [-3500.0, -4100.0, 1200.0], \
            "accounting-style negatives '(3,500)' must parse as negative"
        assert suz["negative_equity_flag"].tolist() == [1.0, 1.0, 0.0], \
            "negative_equity_flag must be 1 exactly where book equity < 0"
        assert suz["debt_to_equity"].isna().tolist() == [True, True, False], \
            "debt_to_equity must stay NaN under negative equity and survive otherwise"
        assert suz["debt_to_equity"].iloc[2] == 9.75

        print("\n✅ Self-test passed: block parsing, FY mapping, #N/A -> NaN, "
              "negative-equity flag and debt_to_equity NaN rule all behave as specified.")
