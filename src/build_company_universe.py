"""
build_company_universe.py
==========================
Two entry points:

  1. build_universe_csv()  — generates the 51-company seed list (already done;
                             kept for reproducing data/raw/company_universe_starter.csv)

  2. process_eqs_export()  — processes the Bloomberg EQS export into the clean
                             ~800-company universe needed for Phase 1 completion.

     EQS EXPORT REQUIREMENTS
     ━━━━━━━━━━━━━━━━━━━━━━
     The export CSV must contain these columns (exact Bloomberg field names may
     differ — see EQS_COLUMN_ALIASES below for accepted variants):

         Ticker / Security ID        ← primary key
         Company Name                ← display label
         GICS Sector Name            ← for Financials exclusion tag (not filter)
         Trading Status              ← Active / Inactive / Delisted
         Delisting Date              ← blank for active; date for dead tickers
         Current Market Cap (M)      ← display-only; filtering done HERE in code
         ICRA Long Term Issuer Rating
         CRISIL Long Term Rating Lookup  ← instrument-level proxy (document this)

     NOT-SUPPORTED IN SCREEN:
         - Market-cap floor at screen time (Current basis deletes delisted firms;
           see D6 in Project_Context.md §8). Apply it here instead.
         - Primary-security dedup (-PP, -RT, -W, -BE suffixes; see D10).
         - Financials exclusion filter (keep them; tag them; PRD §6.5 needs sector).

     SURVIVORSHIP INVARIANT (PRD §5.7 / D6):
         The market-cap floor is applied to ACTIVE companies only.
         ALL inactive/delisted companies are retained regardless of size.
         Rationale: distressed companies are disproportionately delisted; a size
         filter on current cap rebuilds survivorship bias from the other direction.

     DEDUPLICATION (D10):
         Bloomberg carries partly-paid (-PP), rights (-RT), warrants (-W),
         and book-entry (-BE) shares as separate securities under the same root
         ticker. Keep the PRIMARY share class; remove the rest.
         Priority: highest current market cap among same-ISIN or same root-ticker
         duplicates (where cap is zero/NaN for delisted, use TRADING_STATUS = Active
         as primary signal, then largest cap).

USAGE
-----
    # Process a raw EQS CSV export:
    python src/build_company_universe.py \\
        data/raw/eqs_export.csv \\
        --mktcap-floor 5000 \\          # INR millions; 5000 = ₹500 crore
        --out data/processed/universe_800.csv \\
        --census data/processed/rating_census.csv

    # Regenerate the 51-company seed list:
    python src/build_company_universe.py --seed-only
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── column-name aliases ────────────────────────────────────────────────
# EQS export headers vary by Bloomberg version and export format.
# Each key is the canonical internal name; values are known Bloomberg variants.
# The first match wins; all comparisons are case-insensitive, stripped.
EQS_COLUMN_ALIASES = {
    "ticker":          ["ticker", "security id", "id", "bbg ticker", "bloomberg ticker"],
    "company_name":    ["name", "company name", "security name", "issuer name"],
    "gics_sector":     ["gics sector name", "gics sector", "sector name", "sector"],
    "trading_status":  ["trading status", "status", "trading status (current)", "trd status"],
    "delist_date":     ["delisting date", "delist date", "date delisted",
                        "dvd_dvd_dvd_date_delist"],
    "mkt_cap":         ["cur mkt cap", "current market cap", "mkt cap (m)",
                        "market cap", "cur_mkt_cap", "mkt_cap",
                        "cur mkt cap (usd)", "cur mkt cap (inr)"],
    "icra_rating":     ["icra long term issuer rating", "icra lt issuer rating",
                        "icra_lt_issuer_rating", "icra rating", "icra lt rating",
                        "icra lt issr rtg"],
    "crisil_rating":   ["crisil long term rating lookup", "crisil lt rating lookup",
                        "crisil_lt_rating_lookup", "crisil rating", "crisil lt rating",
                        "crisil lt rtg lu"],
}

# Secondary-share suffixes to strip/remove. Sorted longest first so
# "-INDUSTR" is not accidentally matched by "-INDU-RT".
SECONDARY_SUFFIXES = [
    "-INDU-PP", "-INDU-RT",       # Reliance Industries variants
    "-PP", "-RT", "-W", "-BE",    # generic partly-paid, rights, warrants, book-entry
    "-DVR",                        # differential voting rights (separate security)
]

# GICS sector value for financial-sector tagging. Not excluded — TAGGED.
# (Exclusion from model is downstream in feature engineering and training.)
FINANCIALS_GICS = "Financials"

# Rating strings that mean "not rated" — excluded from coverage count.
NR_VALUES = {"NR", "N.R.", "N/R", "NOT RATED", "NR (ISSUER)", "", "NAN", "#N/A",
             "#N/A N/A", "#N/A FIELD NOT APPLICABLE", "#N/A INVALID SECURITY", "--"}

# Trading-status strings that mean "active" (others → inactive/delisted).
ACTIVE_VALUES = {"ACTIVE", "TRADING", "LISTED", "NORMAL", "Y"}


# ═══════════════════════════════════════════════════════════════════════
# I. Column normalisation
# ═══════════════════════════════════════════════════════════════════════

def _normalise_header(raw: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", str(raw).lower().strip())


def _map_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Rename EQS export columns to canonical internal names.

    Raises ValueError listing every column that could not be matched so the
    caller knows exactly what to fix rather than silently losing fields.
    """
    normalised = {_normalise_header(c): c for c in df.columns}

    mapping = {}   # canonical_name → original_col_name
    missing = []

    for canonical, aliases in EQS_COLUMN_ALIASES.items():
        matched = None
        for alias in aliases:
            if _normalise_header(alias) in normalised:
                matched = normalised[_normalise_header(alias)]
                break
        if matched is not None:
            mapping[matched] = canonical
        else:
            if canonical not in ["delist_date", "icra_rating", "crisil_rating"]:
                missing.append(canonical)

    if missing:
        available = sorted(df.columns.tolist())
        raise ValueError(
            f"[universe] Cannot map these required columns: {missing}\n"
            f"  Available columns in export: {available}\n"
            f"  Add the correct Bloomberg header to EQS_COLUMN_ALIASES in this file."
        )

    out = df.rename(columns=mapping)
    extra = [c for c in out.columns if c not in EQS_COLUMN_ALIASES]
    if extra and verbose:
        print(f"[universe] {len(extra)} extra column(s) in export (kept as-is): "
              f"{extra[:8]}{'...' if len(extra) > 8 else ''}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# II. Deduplication  (D10 — must happen in code, not in the Terminal)
# ═══════════════════════════════════════════════════════════════════════

def _root_ticker(ticker: str) -> str:
    """Extract the deduplication key from a Bloomberg full ticker string.

    Bloomberg full ticker format:
        '<MNEMONIC> <EXCHANGE_CODE> <SECURITY_TYPE>'
        e.g.  'RELIANCE INDUSTR IN Equity'   ← ordinary shares
              'RELIANCE INDU-PP IN Equity'   ← partly-paid  }  same company
              'RELIANCE INDU-RT IN Equity'   ← rights        }

    The dedup key is the MNEMONIC token only (first word), stripped of
    secondary-class suffixes so all three collapse to 'RELIANCE':

        'RELIANCE INDUSTR IN Equity'  → 'RELIANCE'
        'RELIANCE INDU-PP IN Equity'  → 'RELIANCE'   (INDU-PP → INDU → strip 'INDU'?)

    Wait — 'INDU-PP' is part of 'INDUSTR' (the mnemonic token), not the full
    string. Correct strategy: take the FIRST SPACE-DELIMITED token, then strip
    any trailing secondary suffixes from that token.

        Token: 'RELIANCE'   → no secondary suffix → stays 'RELIANCE'
        Token: 'INDUSTR'    → no secondary suffix
        Token: 'INDU-PP'    → strip '-PP' → 'INDU'  (close enough for grouping)

    For grouping purposes we also normalise to just the first token (before the
    exchange code / security type), since securities for the same company share
    the same leading mnemonic.
    """
    t = str(ticker).upper().strip()
    # Take only the first word (mnemonic) — drops 'IN Equity', 'US Equity' etc.
    mnemonic = t.split()[0] if t else t
    for suf in SECONDARY_SUFFIXES:
        mnemonic_suf = suf.lstrip("-")   # '-PP' → 'PP' so we can try both forms
        if mnemonic.endswith(suf.upper()):
            mnemonic = mnemonic[: -len(suf)]
        elif mnemonic.endswith("-" + mnemonic_suf):
            mnemonic = mnemonic[: -(len(mnemonic_suf) + 1)]
    return mnemonic.strip().rstrip("-")


def deduplicate_secondary_shares(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep only the primary share class for each company.

    Strategy (in priority order):
      1. Among securities sharing the same root ticker, prefer the one
         WITHOUT any secondary suffix (the ordinary share).
      2. If multiple ordinary shares remain, prefer ACTIVE over delisted.
      3. Finally, highest market cap.

    Removed rows are logged to data/raw/dedup_removed.csv for audit.
    """
    df = df.copy()
    df["_root_ticker"] = df["ticker"].apply(_root_ticker)
    df["_is_secondary"] = df.apply(
        lambda r: any(s in str(r["ticker"]).upper() for s in SECONDARY_SUFFIXES), axis=1
    )
    df["_is_active"] = (
        df["trading_status"].astype(str).str.upper().str.strip().isin(ACTIVE_VALUES)
    )
    df["_mkt_cap_num"] = pd.to_numeric(df["mkt_cap"], errors="coerce").fillna(0.0)

    # Sort so the PRIMARY row floats to top: ordinary first, active first, largest cap first
    df = df.sort_values(
        ["_root_ticker", "_is_secondary", "_is_active", "_mkt_cap_num"],
        ascending=[True, True, False, False],
    )

    before = len(df)
    removed = df[df.duplicated(subset=["_root_ticker"], keep="first")]
    df = df.drop_duplicates(subset=["_root_ticker"], keep="first")
    after = len(df)

    if verbose:
        n_removed = before - after
        if n_removed:
            print(f"[universe] Dedup: removed {n_removed} secondary-class securities "
                  f"(partly-paid, rights, warrants, DVR).")
            # Save audit log
            out_path = Path("data/raw/dedup_removed.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            removed[["ticker", "company_name", "_root_ticker",
                      "_is_secondary", "trading_status", "mkt_cap"]].to_csv(
                out_path, index=False
            )
            print(f"[universe]   Removed securities logged → {out_path}")
        else:
            print("[universe] Dedup: no secondary-class duplicates found.")

    # Drop internal helper columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# III. Survivorship-safe market-cap floor  (D6)
# ═══════════════════════════════════════════════════════════════════════

def apply_mktcap_floor(
    df: pd.DataFrame,
    floor_inr_m: float = 5000.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply market-cap floor to ACTIVE companies only.

    Inactive/delisted companies pass unconditionally — they are the positive
    class and excluding them by current market cap is a survivorship failure
    mode (D6, Project_Context.md §8).

    The floor is in INR millions:
        ₹500 crore  = 5,000 M INR  (minimum recommended by PRD §5.7)
        ₹1,000 crore = 10,000 M INR

    If the export's market cap is in USD, convert before calling
    (1 USD ≈ 83 INR as of 2024; use the rate at the time of pull).
    """
    df = df.copy()
    df["_mkt_cap_num"] = pd.to_numeric(df["mkt_cap"], errors="coerce")
    df["_is_active"] = (
        df["trading_status"].astype(str).str.upper().str.strip().isin(ACTIVE_VALUES)
    )

    # Only active companies with a KNOWN cap below the floor get removed.
    # Active companies with NaN cap: keep (give benefit of doubt; mark separately).
    below_floor_mask = (
        df["_is_active"]
        & df["_mkt_cap_num"].notna()
        & (df["_mkt_cap_num"] < floor_inr_m)
    )
    active_no_cap = df["_is_active"] & df["_mkt_cap_num"].isna()

    n_active = int(df["_is_active"].sum())
    n_inactive = int((~df["_is_active"]).sum())
    n_below = int(below_floor_mask.sum())
    n_active_no_cap = int(active_no_cap.sum())

    if verbose:
        print(f"[universe] Market-cap floor: {floor_inr_m:,.0f} M INR "
              f"(≈ ₹{floor_inr_m/100:.0f} crore)")
        print(f"  Active companies:               {n_active}")
        print(f"  Inactive/delisted (kept all):   {n_inactive}")
        print(f"  Active below floor (removed):   {n_below}")
        print(f"  Active with unknown cap (kept): {n_active_no_cap}  ← investigate these")

    removed = df[below_floor_mask].copy()
    if len(removed):
        out_path = Path("data/raw/below_mktcap_floor.csv")
        removed[["ticker", "company_name", "trading_status",
                 "_mkt_cap_num"]].to_csv(out_path, index=False)
        print(f"[universe]   Below-floor active tickers logged → {out_path}")

    df = df[~below_floor_mask].drop(columns=["_mkt_cap_num", "_is_active"])
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# IV. Sector tagging (not filtering — PRD §6.5 needs sector as a feature)
# ═══════════════════════════════════════════════════════════════════════

def tag_financial_sector(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add is_financials_sector flag. Does NOT remove financial companies.

    Financial-sector companies (banks, NBFCs, insurers) are excluded from
    Model A training because the standard ratio set doesn't apply — but they
    must remain in the universe file so:
      - The exclusion is countable and reported (§12.5 Data Quality screen).
      - Their sector column is available as a Block 7 feature for the companies
        that do enter training.
    """
    df = df.copy()
    df["is_financials_sector"] = (
        df["gics_sector"].astype(str).str.strip().str.title() == FINANCIALS_GICS
    )
    if verbose:
        n = int(df["is_financials_sector"].sum())
        print(f"[universe] Tagged {n} Financials-sector companies "
              f"(will be excluded from Model A training, not from this file).")
    return df


# ═══════════════════════════════════════════════════════════════════════
# V. Rating coverage census  (BLOCKER 2 — determines primary label)
# ═══════════════════════════════════════════════════════════════════════

def _is_rated(val) -> bool:
    """True if the rating value represents a real rating (not NR/blank/error)."""
    s = str(val).strip().upper()
    return s not in NR_VALUES and not s.startswith("#N/A")


def rating_coverage_census(
    df: pd.DataFrame,
    out_path: str = "data/processed/rating_census.csv",
    verbose: bool = True,
) -> dict:
    """Count non-NR entries for each rating agency and report coverage.

    ⚠️ These are POINT-IN-TIME snapshots from EQS, not ratings history.
    They tell you WHETHER a company has ever had a rating with this agency —
    a coverage CENSUS — not the label itself.  The label requires rated history
    pulled via a separate Bloomberg template.

    Returns a dict with coverage statistics (for the label-strategy decision):
        {
          "total_companies": int,
          "icra_rated": int,
          "crisil_rated": int,
          "either_rated": int,
          "coverage_pct": float,   # either_rated / total * 100
          "label_strategy": str,   # "ratings" (>=60%) or "fallback" (<60%)
        }
    """
    df = df.copy()

    has_icra = "icra_rating" in df.columns
    has_crisil = "crisil_rating" in df.columns

    if not has_icra and not has_crisil:
        print("[universe] WARNING: No rating columns found in export. "
              "Add ICRA_LT_ISSUER_RATING and CRISIL_LT_RATING_LOOKUP to the "
              "EQS results grid before running the census.")
        return {"total_companies": len(df), "coverage_pct": 0.0,
                "label_strategy": "fallback"}

    n_total = len(df)

    if has_icra:
        df["icra_is_rated"] = df["icra_rating"].apply(_is_rated)
    else:
        df["icra_is_rated"] = False

    if has_crisil:
        df["crisil_is_rated"] = df["crisil_rating"].apply(_is_rated)
    else:
        df["crisil_is_rated"] = False

    df["either_rated"] = df["icra_is_rated"] | df["crisil_is_rated"]

    n_icra = int(df["icra_is_rated"].sum()) if has_icra else 0
    n_crisil = int(df["crisil_is_rated"].sum()) if has_crisil else 0
    n_either = int(df["either_rated"].sum())
    pct = n_either / n_total * 100 if n_total else 0.0

    # PRD §7 threshold: ≥60% coverage → ratings-based label as primary.
    # Below 60% → promote §7.5 fallback, with circularity treatment (D5).
    COVERAGE_THRESHOLD = 60.0
    strategy = "ratings" if pct >= COVERAGE_THRESHOLD else "fallback"

    if verbose:
        print(f"\n[universe] ═══ RATING COVERAGE CENSUS ═══")
        print(f"  Total companies in universe:    {n_total}")
        if has_icra:
            print(f"  ICRA rated (non-NR):           {n_icra}  ({n_icra/n_total*100:.1f}%)")
        if has_crisil:
            print(f"  CRISIL rated (non-NR):         {n_crisil}  ({n_crisil/n_total*100:.1f}%)")
        print(f"  Rated by EITHER agency:         {n_either}  ({pct:.1f}%)")
        print(f"  Coverage threshold (§7):        {COVERAGE_THRESHOLD:.0f}%")
        print(f"  ⮞ LABEL STRATEGY:  {strategy.upper()}")
        if strategy == "ratings":
            print("    → Use label_construction.py as primary. No circularity.")
        else:
            print("    → Promote fallback_label.py §7.5 to primary.")
            print("      MANDATORY: exclude circular features (D5). See fallback_label.py.")
        print(f"  NOTE: EQS fields are point-in-time snapshots. This census tells you")
        print(f"  WHICH companies have ever been rated, not their full rating history.")
        print(f"  Ratings history for labelling requires a separate Bloomberg template pull.\n")

    # Save per-company census
    census_cols = ["ticker", "company_name", "gics_sector", "trading_status",
                   "icra_rating", "crisil_rating", "icra_is_rated", "crisil_is_rated",
                   "either_rated"]
    save_cols = [c for c in census_cols if c in df.columns]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[save_cols].to_csv(out, index=False)
    if verbose:
        print(f"[universe] Per-company rating census → {out}")

    return {
        "total_companies": n_total,
        "icra_rated": n_icra,
        "crisil_rated": n_crisil,
        "either_rated": n_either,
        "coverage_pct": pct,
        "label_strategy": strategy,
    }


# ═══════════════════════════════════════════════════════════════════════
# VI. Main EQS processing pipeline
# ═══════════════════════════════════════════════════════════════════════

def _read_bloomberg_csv(path: Path) -> pd.DataFrame:
    """Read CSV, automatically skipping Bloomberg's top metadata rows."""
    # Read first 20 rows to find the header
    head = pd.read_csv(path, nrows=20, header=None, dtype=str)
    header_idx = 0
    for i, row in head.iterrows():
        row_str = " ".join(str(x).lower() for x in row.dropna())
        if "ticker" in row_str or "security id" in row_str:
            header_idx = i
            break
            
    df = pd.read_csv(path, skiprows=header_idx, dtype=str)
    # Drop Bloomberg's summary rows like "None (3000 securities)"
    df = df[~df.iloc[:, 0].astype(str).str.contains(r"None \(.*\)", na=False, case=False)]
    return df

def process_eqs_export(
    eqs_paths: list[str],
    out_path: str = "data/processed/universe_800.csv",
    census_path: str = "data/processed/rating_census.csv",
    mktcap_floor_inr_m: float = 5000.0,
    verify_row_count: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Full EQS export → clean ~800-company universe."""
    if isinstance(eqs_paths, str):
        eqs_paths = [eqs_paths]
        
    dfs = []
    for path_str in eqs_paths:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(
                f"EQS export not found: {p}\n"
                f"Export from Bloomberg EQS<GO> and place at that path."
            )
        dfs.append(_read_bloomberg_csv(p))

    raw = pd.concat(dfs, ignore_index=True)
    raw = raw.drop_duplicates()  # in case user's chunks overlap
    
    n_raw = len(raw)
    if verbose:
        print(f"[universe] Loaded {n_raw} unique rows from {len(eqs_paths)} file(s)")

    # Step 1: normalise column names
    df = _map_columns(raw, verbose=verbose)

    # Step 2: row-count sanity check
    if verify_row_count and n_raw < 3500:
        print(
            f"\n[universe] ⚠️  WARNING: Only {n_raw} rows in export.\n"
            f"  Bloomberg EQS results grid shows '3000 of 6640 Securities' — if the\n"
            f"  export was done while the display was capped at 3,000, you only have\n"
            f"  the top half (largest caps). The lost rows include small-caps and\n"
            f"  delisted names — exactly where distress concentrates.\n"
            f"  Action: re-export after clicking 'Export All' or increasing the row cap.\n"
        )
    elif verbose and n_raw >= 5000:
        print(f"[universe] ✅ Row count {n_raw} ≥ 5,000 — export appears complete.")

    # Step 3: deduplication
    df = deduplicate_secondary_shares(df, verbose=verbose)
    if verbose:
        print(f"[universe] After dedup: {len(df)} companies")

    # Step 4: market-cap floor (active only)
    df = apply_mktcap_floor(df, floor_inr_m=mktcap_floor_inr_m, verbose=verbose)
    if verbose:
        n_active = (df["trading_status"].astype(str).str.upper().str.strip()
                    .isin(ACTIVE_VALUES)).sum()
        n_inactive = len(df) - n_active
        print(f"[universe] After floor: {len(df)} companies "
              f"({n_active} active, {n_inactive} inactive/delisted)")

    # Step 5: financial-sector tag
    df = tag_financial_sector(df, verbose=verbose)

    # Step 6: rating census
    census = rating_coverage_census(df, out_path=census_path, verbose=verbose)

    # Step 7: write output
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if verbose:
        print(f"[universe] Clean universe ({len(df)} companies) → {out}")
        _sector_summary(df)

    return df, census


def _sector_summary(df: pd.DataFrame) -> None:
    """Print sector composition and survivorship-bias exposure stats."""
    print("\n[universe] ── Composition ──────────────────────────────────")
    if "gics_sector" in df.columns:
        print(df["gics_sector"].value_counts(dropna=False).to_string())
    if "trading_status" in df.columns:
        print("\n[universe] Trading status:")
        print(df["trading_status"].value_counts(dropna=False).to_string())
    n_fin = int(df.get("is_financials_sector", pd.Series(dtype=bool)).sum())
    print(f"\n[universe] Financials-sector (excluded from Model A): {n_fin}")
    n_model = len(df) - n_fin
    print(f"[universe] Non-financial (eligible for Model A):      {n_model}")
    print("[universe] ──────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════
# VII. 51-company seed-list generator  (original, preserved)
# ═══════════════════════════════════════════════════════════════════════

distressed = [
    ("IL&FS (Infrastructure Leasing & Financial Services)", "IL&FS", "Financials/NBFC",
     "2018 default, triggered NBFC liquidity crisis",
     "Group entity structure — verify which listed arm, if any"),
    ("Jet Airways (India)", "JETAIRWAYS IN", "Industrials/Airlines",
     "CIRP since 2019", "Grounded operations; verify current listing status"),
    ("Dewan Housing Finance Corporation (DHFL)", "DHFL IN", "Financials/Housing Finance",
     "2019 default, resolved via IBC (acquired by Piramal)",
     "Delisted post-resolution — verify"),
    ("Reliance Capital", "RELCAPITAL IN", "Financials/NBFC",
     "Insolvency proceedings 2021+", "Verify current listing status"),
    ("Reliance Communications", "RCOM IN", "Telecom",
     "Defaulted, insolvency proceedings", "Verify current listing status"),
    ("Reliance Naval and Engineering", "RNAVAL IN", "Industrials/Shipbuilding",
     "Insolvency, later acquired", "Verify current listing status"),
    ("Videocon Industries", "VIDEOIND IN", "Consumer Durables",
     "IBC resolution 2021 (Vedanta group acquisition)",
     "Verify current listing status"),
    ("Bhushan Steel", "BHUSANSTL IN (legacy)", "Materials/Steel",
     "One of first major IBC cases (2017-18), acquired by Tata Steel",
     "SURVIVORSHIP: now part of Tata Steel — legacy ticker likely delisted"),
    ("Bhushan Power & Steel", "BPSL (legacy, unlisted/private)", "Materials/Steel",
     "IBC case, acquired by JSW Steel",
     "May not have Bloomberg equity coverage — verify"),
    ("Essar Steel India", "ESSARSTL (legacy)", "Materials/Steel",
     "Major IBC case, acquired by ArcelorMittal",
     "SURVIVORSHIP: now ArcelorMittal Nippon Steel India, largely unlisted — verify"),
    ("Amtek Auto", "AMTEKAUTO IN", "Industrials/Auto Components",
     "Early, prominent IBC case", "Verify current listing status"),
    ("Jaypee Infratech", "JPINFRATEC IN", "Real Estate/Infrastructure",
     "IBC case, NBCC resolution", "Verify current listing status"),
    ("Alok Industries", "ALOKTEXT IN", "Consumer Discretionary/Textiles",
     "IBC case, acquired by Reliance-JM Financial ARC consortium",
     "Verify current listing status"),
    ("Electrosteel Steels", "ELECTHERM (legacy)", "Materials/Steel",
     "IBC case, acquired by Vedanta", "Verify current listing status"),
    ("Monnet Ispat & Energy", "MONNETISPA IN", "Materials/Steel",
     "IBC case", "Verify current listing status"),
    ("ABG Shipyard", "ABGSHIP (legacy)", "Industrials/Shipbuilding",
     "Default, later fraud case, liquidation",
     "Verify Bloomberg coverage — may be delisted/liquidated"),
    ("CG Power and Industrial Solutions", "CGPOWER IN",
     "Industrials/Electrical Equipment",
     "Accounting irregularities, severe distress 2019-20, later recovered",
     "Good 'recovered from distress' case — check post-recovery years show label improvement"),
    ("Future Retail", "FRETAIL IN", "Consumer Discretionary/Retail",
     "Defaulted, insolvency 2022", "Verify current listing status"),
    ("Future Enterprises", "FEL IN", "Consumer Discretionary/Retail",
     "Related distress, insolvency", "Verify current listing status"),
    ("Sintex Plastics Technology", "SINTEXPLA IN", "Materials/Plastics",
     "Defaulted, IBC", "Verify current listing status"),
    ("Coffee Day Enterprises", "CCL IN", "Consumer Discretionary/Restaurants",
     "Severe financial stress post-2019", "Verify current listing status"),
    ("Yes Bank", "YESBANK IN", "Financials/Banks",
     "RBI moratorium March 2020, reconstruction scheme",
     "Not a full IBC case — useful as 'label 1' in many fiscal years"),
    ("SREI Infrastructure Finance", "SREINFRA IN", "Financials/NBFC",
     "RBI superseded board, insolvency 2021", "Verify current listing status"),
    ("Suzlon Energy", "SUEL IN", "Industrials/Renewable Energy",
     "Severe stress, multiple restructurings, later recovery",
     "Good long-running 'grey zone' example across many years"),
    ("GTL Infrastructure", "GTLINFRA IN", "Telecom/Infrastructure",
     "Defaulted on debt obligations", "Verify current listing status"),
    ("IVRCL", "IVRCLINFR IN", "Industrials/Construction",
     "IBC/liquidation", "Verify current listing status"),
]

stable = [
    ("Reliance Industries", "RELIANCE IN", "Energy/Conglomerate"),
    ("Tata Consultancy Services", "TCS IN", "Information Technology"),
    ("Infosys", "INFY IN", "Information Technology"),
    ("HDFC Bank", "HDFCBANK IN", "Financials/Banks"),
    ("ICICI Bank", "ICICIBC IN", "Financials/Banks"),
    ("Kotak Mahindra Bank", "KMB IN", "Financials/Banks"),
    ("State Bank of India", "SBIN IN", "Financials/Banks"),
    ("Larsen & Toubro", "LT IN", "Industrials/Construction"),
    ("Tata Steel", "TATA IN", "Materials/Steel"),
    ("JSW Steel", "JSTL IN", "Materials/Steel"),
    ("Hindalco Industries", "HNDL IN", "Materials/Metals"),
    ("Bharti Airtel", "BHARTI IN", "Telecom"),
    ("Maruti Suzuki India", "MSIL IN", "Consumer Discretionary/Auto"),
    ("Mahindra & Mahindra", "MM IN", "Industrials/Auto"),
    ("Titan Company", "TTAN IN", "Consumer Discretionary/Retail"),
    ("Asian Paints", "APNT IN", "Materials/Chemicals"),
    ("Nestle India", "NEST IN", "Consumer Staples"),
    ("Hindustan Unilever", "HUVR IN", "Consumer Staples"),
    ("Sun Pharmaceutical Industries", "SUNP IN", "Healthcare/Pharma"),
    ("Dr Reddy's Laboratories", "DRRD IN", "Healthcare/Pharma"),
    ("Adani Ports and Special Economic Zone", "ADSEZ IN",
     "Industrials/Infrastructure"),
    ("Power Grid Corporation of India", "PWGR IN", "Utilities"),
    ("NTPC", "NTPC IN", "Utilities/Power"),
    ("Bajaj Finance", "BAF IN", "Financials/NBFC"),
    ("Britannia Industries", "BRIT IN", "Consumer Staples"),
]


def build_universe_csv(out_path: str = "data/raw/company_universe_starter.csv") -> pd.DataFrame:
    """Regenerate the 51-company seed CSV. Run once to bootstrap; then use
    process_eqs_export() for the production ~800-company universe."""
    rows = []
    for name, ticker_hint, sector, event, note in distressed:
        rows.append({
            "company_name": name,
            "ticker_hint": ticker_hint,
            "sector": sector,
            "category": "distressed_history",
            "distress_event": event,
            "note": note,
        })
    for name, ticker_hint, sector in stable:
        rows.append({
            "company_name": name,
            "ticker_hint": ticker_hint,
            "sector": sector,
            "category": "stable_contrast",
            "distress_event": "",
            "note": "Verify Bloomberg ticker format before pulling",
        })
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} companies to {out_path}")
    print(f"  Distressed-history: {len(distressed)}")
    print(f"  Stable-contrast:    {len(stable)}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("eqs_export", nargs="*",
                    help="Path(s) to Bloomberg EQS export CSV(s) "
                         "(omit with --seed-only to just regenerate the seed list)")
    ap.add_argument("--seed-only", action="store_true",
                    help="Regenerate data/raw/company_universe_starter.csv and exit")
    ap.add_argument("--out", default="data/processed/universe_800.csv",
                    help="Output path for the clean universe CSV")
    ap.add_argument("--census", default="data/processed/rating_census.csv",
                    help="Output path for the rating coverage census CSV")
    ap.add_argument("--mktcap-floor", type=float, default=5000.0,
                    help="Market-cap floor in INR millions for ACTIVE companies "
                         "(default: 5000 = ₹500 crore). Set to 0 to disable.")
    ap.add_argument("--no-row-check", action="store_true",
                    help="Suppress the 3,000-row truncation warning")
    args = ap.parse_args()

    if args.seed_only:
        build_universe_csv()
        return

    if not args.eqs_export:
        ap.error("At least one eqs_export path is required unless --seed-only is specified.")

    df, census = process_eqs_export(
        eqs_paths=args.eqs_export,
        out_path=args.out,
        census_path=args.census,
        mktcap_floor_inr_m=args.mktcap_floor,
        verify_row_count=not args.no_row_check,
        verbose=True,
    )
    print(f"\n✅ Universe built: {len(df)} companies")
    print(f"   Label strategy: {census['label_strategy'].upper()}")
    print(f"   Coverage: {census['coverage_pct']:.1f}% either-rated")
    if census["label_strategy"] == "fallback":
        print("\n⚠️  FALLBACK LABEL IS PRIMARY — read circularity warning in fallback_label.py")
        print("   Run: python src/fallback_label.py data/processed/bloomberg_panel.csv --forward")


if __name__ == "__main__":
    main()
