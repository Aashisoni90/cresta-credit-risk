"""
label_construction.py
======================
Builds the graded distress label described in the Foundations Guide (§5.5):

    0 = Stable      -> no downgrade in the observation window; rating stable/upgraded
    1 = Deteriorating -> downgraded, but still investment grade
    2 = Distressed  -> downgraded to below investment grade, defaulted, or entered IBC

⚠️ DOMESTIC vs INTERNATIONAL RATING SCALES — read before running anything.
CRISIL / ICRA / CARE / India Ratings / Brickwork ratings are on INDIA'S
DOMESTIC scale. Moody's / S&P / Fitch (global) ratings are on the
INTERNATIONAL scale. These are NOT the same scale and must not be pooled
without an explicit mapping. This script defaults to filtering to
DOMESTIC-scale agencies only (see DOMESTIC_AGENCIES below) — this is the
recommended approach from the Foundations Guide (§5.5): it's the
economically meaningful scale for Indian credit markets and maximizes your
usable sample.

INPUT: a long-format ratings-history CSV/DataFrame with columns:
    ticker, rating_date, agency, rating

The 'rating' values must be strings from a recognized scale (see
RATING_SCALE_MAP below). If your Bloomberg export uses different exact
strings, extend the map — don't silently drop unmapped ratings without
checking what they are first.

OUTPUT: a fiscal-year-level dataframe:
    ticker, fiscal_year, distress_label, label_definition, driving_rating,
    driving_agency
"""

import pandas as pd
import numpy as np


DOMESTIC_AGENCIES = {"CRISIL", "ICRA", "CARE", "INDIA RATINGS", "IND-RA", "BRICKWORK", "CARE RATINGS"}

# Domestic-scale long-term rating notches, ordered BEST -> WORST.
# This list follows the standard CRISIL/ICRA/CARE/India Ratings notation.
# Extend if you encounter variants not listed (e.g. some agencies omit '+'/'-'
# modifiers at certain notches) — but verify against the agency's own scale
# definition before adding, don't guess.
DOMESTIC_SCALE_ORDER = [
    "AAA",
    "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",   # <- investment grade cutoff is here (BBB- and above)
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "C", "D",                # D = default
]
RATING_RANK = {r: i for i, r in enumerate(DOMESTIC_SCALE_ORDER)}
INVESTMENT_GRADE_CUTOFF_RANK = RATING_RANK["BBB-"]  # ranks <= this are investment grade


def _clean_rating_string(raw: str) -> str:
    """Strips agency name prefixes like 'CRISIL AAA' -> 'AAA', and whitespace."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    for agency in DOMESTIC_AGENCIES:
        s = s.replace(agency + " ", "").replace(agency, "")
    # Strip common suffixes like "(Stable)", "(Negative)" outlook tags
    s = s.split("(")[0].strip()
    return s


def filter_domestic_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only domestic-scale agency ratings. Logs what was dropped."""
    df = ratings_df.copy()
    df["agency_upper"] = df["agency"].astype(str).str.upper().str.strip()
    is_domestic = df["agency_upper"].isin(DOMESTIC_AGENCIES)

    n_total, n_kept = len(df), is_domestic.sum()
    n_dropped = n_total - n_kept
    if n_dropped > 0:
        dropped_agencies = df.loc[~is_domestic, "agency_upper"].unique()
        print(f"[label_construction] Dropped {n_dropped}/{n_total} rows from non-domestic "
              f"or unrecognized agencies: {list(dropped_agencies)}")
        print("  -> If any of these are domestic agencies under a different spelling, "
              "add them to DOMESTIC_AGENCIES rather than silently losing the data.")

    return df[is_domestic].drop(columns=["agency_upper"])


def rating_to_rank(rating_raw: str):
    """Maps a raw rating string to its ordinal rank (0=best). Returns None if unmapped."""
    cleaned = _clean_rating_string(rating_raw)
    return RATING_RANK.get(cleaned, None)


def build_fiscal_year_labels(ratings_df: pd.DataFrame, fiscal_year_start_month: int = 4) -> pd.DataFrame:
    """
    ratings_df columns required: ticker, rating_date (datetime), agency, rating

    Logic per ticker-fiscal_year:
      - Take the WORST (highest rank number = worst) rating observed during
        that fiscal year, across all domestic agencies covering the company.
      - If that rating is 'D' (default) -> label 2
      - Else if rank > investment grade cutoff (i.e. below BBB-) -> label 2
      - Else if the worst rating during the year is a DOWNGRADE relative to
        the previous fiscal year's worst rating, but still investment grade -> label 1
      - Else -> label 0
    """
    df = filter_domestic_ratings(ratings_df).copy()
    df["rating_date"] = pd.to_datetime(df["rating_date"])
    df["rating_clean"] = df["rating"].apply(_clean_rating_string)
    df["rating_rank"] = df["rating_clean"].apply(lambda r: RATING_RANK.get(r, np.nan))

    unmapped = df[df["rating_rank"].isna()]["rating_clean"].unique()
    if len(unmapped) > 0:
        print(f"[label_construction] WARNING: unmapped rating strings found (excluded): {list(unmapped)}")
        print("  -> Inspect these — they may be valid ratings missing from DOMESTIC_SCALE_ORDER.")
    df = df.dropna(subset=["rating_rank"])

    # Indian fiscal year: April(y) - March(y+1) = "FY y-(y+1)"
    def to_fiscal_year(dt):
        if dt.month >= fiscal_year_start_month:
            return f"FY{dt.year}-{str(dt.year + 1)[-2:]}"
        else:
            return f"FY{dt.year - 1}-{str(dt.year)[-2:]}"

    df["fiscal_year"] = df["rating_date"].apply(to_fiscal_year)

    # Worst (max rank) rating per ticker per fiscal year
    worst_per_year = (
        df.sort_values("rating_rank", ascending=False)
        .groupby(["ticker", "fiscal_year"], as_index=False)
        .first()[["ticker", "fiscal_year", "rating_clean", "rating_rank", "agency"]]
        .rename(columns={"rating_clean": "driving_rating", "agency": "driving_agency"})
    )

    # Sort chronologically per ticker to compute year-over-year downgrade detection
    worst_per_year["fy_sort_key"] = worst_per_year["fiscal_year"].str.extract(r"FY(\d{4})").astype(int)
    worst_per_year = worst_per_year.sort_values(["ticker", "fy_sort_key"])
    worst_per_year["prev_rank"] = worst_per_year.groupby("ticker")["rating_rank"].shift(1)

    def assign_label(row):
        if row["driving_rating"] == "D":
            return 2, "Default (rating = D)"
        if row["rating_rank"] > INVESTMENT_GRADE_CUTOFF_RANK:
            return 2, "Below investment grade"
        if pd.notna(row["prev_rank"]) and row["rating_rank"] > row["prev_rank"]:
            return 1, "Downgraded year-over-year, still investment grade"
        return 0, "Stable or upgraded, investment grade"

    labels = worst_per_year.apply(assign_label, axis=1, result_type="expand")
    worst_per_year["distress_label"] = labels[0]
    worst_per_year["label_definition"] = labels[1]

    return worst_per_year[[
        "ticker", "fiscal_year", "distress_label", "label_definition",
        "driving_rating", "driving_agency"
    ]]


if __name__ == "__main__":
    # ── Self-test with synthetic ratings history ────────────────────────
    synthetic = pd.DataFrame([
        # A stable company
        {"ticker": "STABLECO", "rating_date": "2020-06-01", "agency": "CRISIL", "rating": "CRISIL AAA"},
        {"ticker": "STABLECO", "rating_date": "2021-06-01", "agency": "CRISIL", "rating": "CRISIL AAA"},
        {"ticker": "STABLECO", "rating_date": "2022-06-01", "agency": "CRISIL", "rating": "CRISIL AAA"},
        # A company that deteriorates then defaults
        {"ticker": "FALLINGCO", "rating_date": "2019-05-01", "agency": "ICRA", "rating": "ICRA A"},
        {"ticker": "FALLINGCO", "rating_date": "2020-05-01", "agency": "ICRA", "rating": "ICRA BBB"},
        {"ticker": "FALLINGCO", "rating_date": "2021-05-01", "agency": "ICRA", "rating": "ICRA BB"},
        {"ticker": "FALLINGCO", "rating_date": "2022-05-01", "agency": "ICRA", "rating": "ICRA D"},
        # A row with a non-domestic agency that should be filtered out
        {"ticker": "FALLINGCO", "rating_date": "2022-05-01", "agency": "Moody's", "rating": "Baa2"},
    ])

    result = build_fiscal_year_labels(synthetic)
    print(result.to_string(index=False))
