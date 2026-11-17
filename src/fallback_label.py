"""
fallback_label.py
==================
Builds the FINANCIAL-CRITERIA distress label specified in CRESTA PRD §7.5,
as the documented contingency for when rating coverage is inadequate.

PRD §7.5 criteria, implemented verbatim:
    1. Interest coverage ratio below 1.0 for two or more consecutive years
    2. Negative book equity (balance-sheet insolvency)
    3. Documented entry into IBC / CIRP proceedings
    4. Auditor going-concern qualification, where retrievable
       -> NOT IMPLEMENTED: not retrievable from the Bloomberg field set.
          Recorded as a known gap rather than silently omitted.

Output contract matches label_construction.py exactly (ticker, fiscal_year,
distress_label, label_definition, ...) so the two are interchangeable
downstream and can be compared against each other where ratings do exist.

═══════════════════════════════════════════════════════════════════════════
⚠️ CIRCULARITY WARNING — READ BEFORE USING THIS LABEL
═══════════════════════════════════════════════════════════════════════════
The PRD specifies, in two different sections, that the same quantities are
both FEATURES and LABEL CRITERIA:

    §6.1 Block 1 — Leverage & Solvency
         negative_equity_flag  ....... a FEATURE
    §6.4 Block 4 — Debt Servicing
         interest_coverage ........... a FEATURE
         interest_coverage_below_1_flag  a FEATURE

    §7.5 Fallback label
         negative book equity ........ a LABEL CRITERION
         interest coverage < 1 for 2+ years .. a LABEL CRITERION

Using both at once makes the label a near-deterministic function of the
features. The model would score an AUC close to 1.0 while having learned
nothing except how to read back its own target, and SHAP would attribute
the prediction almost entirely to the defining features — which would look
like a strong result and be worthless.

The forward-looking design in §7.4 (features from FY(t), label from FY(t+1))
softens this but does NOT remove it: negative equity and sub-1 coverage are
highly persistent year over year, so FY(t) values predict FY(t+1) status
nearly mechanically.

This module therefore refuses to hide the problem:
  * `circular_features_for(criteria)` returns the feature list that MUST be
    excluded from training for a given criteria set;
  * `build_fallback_labels(...)` prints that exclusion list every time;
  * criteria can be disabled individually so you can build a label that is
    independent of the features you actually want to use.

Recommended usage, in order of preference:
  1. Ratings-based label (label_construction.py) as primary — no circularity.
  2. This label restricted to `ibc_event` only — an external, non-circular
     criterion — with the financial criteria used as a ROBUSTNESS CHECK.
  3. Full financial criteria WITH the circular features excluded from the
     model, and that exclusion stated explicitly in the methodology.
═══════════════════════════════════════════════════════════════════════════
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Which features become circular if a given criterion is switched on.
# Keys are criterion names; values are feature columns that must then be
# dropped from the training matrix.
# TEST FOR INCLUSION: is the feature a DETERMINISTIC FUNCTION of a label
# criterion? Not "does it predict well". `ebit_to_assets` (single-feature AUC
# 0.870), `roa` (0.839) and `consecutive_loss_years` (0.806) are economically
# correlated with distress and are legitimate, retained signal. A feature that
# RESTATES a criterion is not.
#
# ⚠️ EXPANDED 2026-10-27 after an empirical audit found the guard was being
# bypassed by NAME DRIFT. This map was written in fallback_label.py before
# feature_engineering.py existed; that module then introduced aliases and
# derivatives under different names, which the guard never saw. Worst case:
# `consecutive_ic_below1_years` — literally criterion 1 ("IC < 1 for 2+
# consecutive years") as a feature — scored single-feature AUC 0.941 against
# the forward label and was ALLOWED THROUGH. See Project_Context.md §16.
# Use `assert_no_circular_leak()` below rather than trusting this list alone.
CIRCULAR_FEATURE_MAP = {
    "negative_equity": [
        "negative_equity_flag",
        "total_equity_book",
        "equity_to_liabilities",
        "X4_equity_to_liabilities",
        "liabilities_to_assets",
        # --- added 2026-10-27 (name drift) ---
        "altman_flag_negative_equity",   # altman_zscore.py duplicate of negative_equity_flag
        "equity_to_assets",              # sign(TE/TA) IS the criterion (TA>0 post sign-gate); AUC 0.721
        "liab_to_assets_yoy_change",     # derivative of liabilities_to_assets
    ],
    "low_interest_coverage": [
        "interest_coverage",
        "interest_coverage_capped",
        "interest_coverage_below_1_flag",
        "interest_coverage_yoy_change",
        # --- added 2026-10-27 (name drift) ---
        "consecutive_ic_below1_years",   # IS criterion 1 restated. AUC 0.941. The worst leak.
        "ic_yoy_change",                 # feature_engineering.py's name for interest_coverage_yoy_change
    ],
    "ibc_event": [],   # external criterion — nothing circular
}

# Features that embed a criterion only PARTIALLY, where exclusion is a
# methodology judgement rather than a clear-cut leak. Altman Z'' carries the
# X4 = equity/liabilities term (weight 1.05) out of four terms, so it is
# partly contaminated by the negative-equity criterion (single-feature AUC
# 0.766) while also being a genuine, PRD-mandated predictor (§6.5, Block 5).
# NOT excluded automatically. Decide explicitly and state the choice in the
# methodology; `assert_no_circular_leak(..., strict=True)` will flag them.
PARTIALLY_CIRCULAR = {
    "negative_equity": [
        "altman_z_double_prime",
        "altman_z_double_prime_capped",
        "altman_ems_score",
        "altman_z_yoy_change",
        "altman_z_3yr_trend",
    ],
}

DEFAULT_CRITERIA = ("negative_equity", "low_interest_coverage", "ibc_event")


def circular_features_for(criteria) -> list:
    """Feature columns that must be excluded from training, given criteria."""
    out = []
    for c in criteria:
        out.extend(CIRCULAR_FEATURE_MAP.get(c, []))
    return sorted(set(out))


def assert_no_circular_leak(feature_columns, criteria=DEFAULT_CRITERIA,
                            strict: bool = False, known_safe=None):
    """Fail loudly if a circular feature reached the training matrix.

    Exists because the exclusion LIST alone proved insufficient: new feature
    names introduced in feature_engineering.py bypassed it silently
    (Project_Context.md §16). Call this immediately before fitting, with the
    actual column list of the matrix you are about to train on.

    strict=True additionally rejects the PARTIALLY_CIRCULAR set (Altman Z'' and
    its derivatives, which embed the X4 equity/liabilities term).

    Raises ValueError listing every offending column. Does not warn and
    continue: a silent inflated AUC is the exact failure this guards against.
    """
    cols = set(feature_columns)
    banned = set(circular_features_for(criteria))
    label = "circular"
    if strict:
        for c in criteria:
            banned |= set(PARTIALLY_CIRCULAR.get(c, []))
        label = "circular or partially-circular"
    for c in (known_safe or []):
        banned.discard(c)

    hits = sorted(cols & banned)
    if hits:
        raise ValueError(
            f"{len(hits)} {label} feature(s) present in the training matrix "
            f"while criteria {list(criteria)} are enabled:\n"
            + "".join(f"    - {h}\n" for h in hits)
            + "These are functions of the label. Training on them yields an "
              "inflated AUC that means nothing (PRD §6.1/§6.4 vs §7.5)."
        )
    return True


def _consecutive_true(flags: pd.Series, n: int = 2) -> pd.Series:
    """True where `flags` has been True for `n` consecutive rows ending here.

    Rows must already be in chronological order within a ticker. NaN coverage
    values are treated as NOT satisfying the criterion (a missing ratio is not
    evidence of distress), and they break a run rather than extending it.
    """
    f = flags.fillna(False).astype(bool)
    # Run length must count only the True elements. Using cumcount() here would
    # include the False row that OPENS each group, so the first True after any
    # break would score a run of 2 and be graded "distressed" (label 2) after a
    # single bad year instead of "deteriorating" (label 1). Caught by
    # tests/test_foundation.py::test_consecutive_run_helper.
    run = f.groupby((~f).cumsum()).cumsum()
    return f & (run >= n)


def build_fallback_labels(
    panel: pd.DataFrame,
    criteria=DEFAULT_CRITERIA,
    coverage_threshold: float = 1.0,
    consecutive_years: int = 2,
    ibc_events: dict = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Panel (from the loader) -> graded distress label per company-year.

    ibc_events: optional {ticker: first_distress_fiscal_year_num}. Years at or
    after that year are labelled 2 on the external criterion.

    Grading (PRD §7.2):
        2 = Distressed     any hard criterion met
        1 = Deteriorating  early-warning: coverage below threshold for a
                           single year (run not yet at `consecutive_years`)
        0 = Stable
    """
    criteria = tuple(criteria)
    unknown = set(criteria) - set(CIRCULAR_FEATURE_MAP)
    if unknown:
        raise ValueError(f"Unknown criteria: {sorted(unknown)}. "
                         f"Valid: {sorted(CIRCULAR_FEATURE_MAP)}")

    df = panel.sort_values(["ticker", "fiscal_year_num"]).copy()

    # --- criterion 1: interest coverage below threshold, N consecutive years
    if "interest_coverage" in df.columns:
        below = df["interest_coverage"] < coverage_threshold
        below = below.where(df["interest_coverage"].notna())
        df["_cov_below"] = below.fillna(False).astype(bool)
        df["_cov_sustained"] = (
            df.groupby("ticker", group_keys=False)["_cov_below"]
              .apply(lambda s: _consecutive_true(s, consecutive_years))
        )
    else:
        df["_cov_below"] = False
        df["_cov_sustained"] = False
        if "low_interest_coverage" in criteria and verbose:
            print("[fallback_label] interest_coverage column absent — that criterion is inert.")

    # --- criterion 2: negative book equity
    if "total_equity_book" in df.columns:
        df["_neg_equity"] = (df["total_equity_book"] < 0).fillna(False)
    else:
        df["_neg_equity"] = False

    # --- criterion 3: documented IBC / CIRP entry (external)
    df["_ibc"] = False
    if ibc_events:
        for tkr, first_year in ibc_events.items():
            if first_year is None:
                continue
            df.loc[(df["ticker"] == tkr) & (df["fiscal_year_num"] >= first_year), "_ibc"] = True

    # --- combine only the ENABLED criteria
    hard = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index)

    def _add(mask, text):
        nonlocal hard, reasons
        hard = hard | mask
        reasons = reasons.where(~mask | (reasons != ""), text)
        both = mask & (reasons != "") & (~reasons.str.contains(text, regex=False))
        reasons = reasons.mask(both, reasons + "; " + text)

    if "low_interest_coverage" in criteria:
        _add(df["_cov_sustained"],
             f"Interest coverage < {coverage_threshold} for {consecutive_years}+ consecutive years")
    if "negative_equity" in criteria:
        _add(df["_neg_equity"], "Negative book equity (balance-sheet insolvency)")
    if "ibc_event" in criteria:
        _add(df["_ibc"], "Documented IBC/CIRP proceedings")

    soft = pd.Series(False, index=df.index)
    if "low_interest_coverage" in criteria:
        soft = df["_cov_below"] & ~hard

    df["distress_label"] = np.select([hard, soft], [2, 1], default=0)
    df["label_definition"] = np.select(
        [hard, soft],
        [reasons.replace("", "Distress criterion met"),
         f"Interest coverage < {coverage_threshold} for a single year (early warning)"],
        default="No financial distress criterion met",
    )
    df["label_binary"] = (df["distress_label"] == 2).astype(int)
    df["label_source"] = "financial_criteria_prd_7_5"

    # rows with no usable inputs must not be scored as "stable"
    inputs = [c for c in ["interest_coverage", "total_equity_book"] if c in df.columns]
    if inputs:
        no_data = df[inputs].isna().all(axis=1) & ~df["_ibc"]
        df.loc[no_data, ["distress_label", "label_binary"]] = np.nan
        df.loc[no_data, "label_definition"] = "Insufficient data to evaluate criteria"

    if verbose:
        _report(df, criteria)

    return df[["ticker", "company_name", "fiscal_year", "fiscal_year_num",
               "distress_label", "label_binary", "label_definition",
               "label_source"]].reset_index(drop=True)


def _report(df, criteria):
    print(f"[fallback_label] Criteria enabled: {list(criteria)}")
    n = len(df)
    scored = int(df["distress_label"].notna().sum())
    print(f"[fallback_label] {scored}/{n} company-years scored "
          f"({n - scored} lacked the inputs to evaluate).")
    vc = df["distress_label"].value_counts(dropna=True).sort_index()
    names = {0: "stable", 1: "deteriorating", 2: "distressed"}
    for lbl, cnt in vc.items():
        print(f"    label {int(lbl)} ({names.get(int(lbl),'?'):13s}): {cnt:4d}  "
              f"({cnt/scored*100:.1f}%)")
    if 2 in vc.index:
        print(f"[fallback_label] Positive (binary) rate: {vc.get(2,0)/scored*100:.1f}% "
              f"— PRD §8.3 anticipates 5-15%.")

    excl = circular_features_for(criteria)
    if excl:
        print("\n  *** CIRCULARITY: with these criteria the following features are a "
              "near-deterministic function of the label and MUST be excluded from "
              "the training matrix (see module docstring): ***")
        for f in excl:
            print(f"        - {f}")
        print("      Failing to exclude them yields an inflated AUC that means nothing.")


def make_forward_looking(labels: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Shifts the label one year EARLIER so features FY(t) predict label FY(t+1).

    PRD §7.4: "The model predicts distress occurring in the following fiscal
    year using features observable at the current year end. Contemporaneous
    prediction is operationally useless and will be criticised as trivial."

    The join key stays (ticker, fiscal_year) but the label now describes the
    NEXT year, so this frame merges directly onto the feature panel. The last
    observed year per company necessarily has no forward label and is dropped
    — that is correct, not a bug: there is no outcome to learn from yet.
    """
    out = labels.sort_values(["ticker", "fiscal_year_num"]).copy()
    for c in ["distress_label", "label_binary", "label_definition"]:
        out[f"{c}_next"] = out.groupby("ticker")[c].shift(-1)

    before = len(out)
    out = out[out["label_binary_next"].notna()].copy()
    if verbose:
        print(f"[fallback_label] Forward-looking target: {len(out)} rows retain an "
              f"FY(t+1) outcome ({before - len(out)} final-year rows dropped — expected).")
        rate = out["label_binary_next"].mean() * 100
        print(f"[fallback_label] Forward positive rate: {rate:.1f}%")
    return out


def ibc_events_from_universe(universe_csv: str, verbose: bool = True) -> dict:
    """Extracts {ticker: first_distress_year} from the universe CSV's free text.

    The universe file records distress events as prose ("2018 default",
    "CIRP since 2019"), so the year is parsed out of the text. This is a
    curation aid, NOT an authority — PRD §5.7 requires IBC cases to be
    curated from public record, and the extracted years should be checked
    against IBBI before the final run.
    """
    p = Path(universe_csv)
    if not p.exists():
        print(f"[fallback_label] Universe CSV not found: {p} — no IBC events loaded.")
        return {}

    uni = pd.read_csv(p)
    col = "distress_event" if "distress_event" in uni.columns else None
    tcol = "ticker_hint" if "ticker_hint" in uni.columns else "ticker"
    if col is None or tcol not in uni.columns:
        print("[fallback_label] Universe CSV lacks distress_event/ticker columns.")
        return {}

    years = uni[col].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
    events = {}
    for tkr, yr in zip(uni[tcol], years):
        if pd.notna(yr):
            events[str(tkr).strip().upper().split()[0]] = int(yr)
    if verbose:
        print(f"[fallback_label] Parsed {len(events)} distress-event year(s) from "
              f"{p.name}. VERIFY these against IBBI records before final use.")
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel", help="Tidy panel CSV from the loader")
    ap.add_argument("-o", "--out", default="data/processed/distress_labels_fallback.csv")
    ap.add_argument("--universe", default="data/raw/company_universe_starter.csv")
    ap.add_argument("--criteria", nargs="+", default=list(DEFAULT_CRITERIA),
                    choices=list(CIRCULAR_FEATURE_MAP))
    ap.add_argument("--forward", action="store_true",
                    help="Also emit the FY(t+1) forward-looking target (PRD 7.4)")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    events = ibc_events_from_universe(args.universe)
    # universe tickers are bare ('JETAIRWAYS'); panel tickers are full
    # Bloomberg strings ('JETIN IN Equity') — match on the leading token.
    panel["_base"] = panel["ticker"].astype(str).str.upper().str.split().str[0]
    mapped = {t: events[b] for t, b in zip(panel["ticker"], panel["_base"]) if b in events}
    panel = panel.drop(columns="_base")
    print(f"[fallback_label] Matched {len(set(mapped))} panel ticker(s) to a distress event year.")

    labels = build_fallback_labels(panel, criteria=args.criteria, ibc_events=mapped)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)
    print(f"\nWrote {len(labels)} label row(s) -> {out}")

    if args.forward:
        fwd = make_forward_looking(labels)
        fout = out.with_name(out.stem + "_forward.csv")
        fwd.to_csv(fout, index=False)
        print(f"Wrote {len(fwd)} forward-looking row(s) -> {fout}")


if __name__ == "__main__":
    main()
