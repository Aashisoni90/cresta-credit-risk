"""
quality_gates.py
=================
Implements all seven data quality gates from CRESTA PRD §5.4.

Governing principle (§5.4): "Ingestion fails loudly rather than silently degrading.
Each gate below blocks the load and reports the offending rows."

This module exists because every serious defect found in this project so far was
SILENT — the header off-by-one produced no error, a `[:-0]` slice dropped every
numeric column without complaint, and a self-test passed while its own fixture was
misaligned. Loud failure is the design goal, not a nicety.

THE SEVEN GATES (§5.4)
----------------------
    Gate                 Rule                                      On failure
    ------------------------------------------------------------------------------
    primary_key          (ticker, fiscal_year) unique, non-null    HARD FAIL BATCH
    accounting_identity  |TA - (TL + TE)| within 2%                FLAG row
    sign_sanity          TA > 0; cur_ratio >= 0; mkt_cap > 0       HARD FAIL ROW
    range_plausibility   ROA in [-200,200]; int_cov in [-1e3,1e3]  FLAG outlier, retain
    coverage             >= 70% non-null on core features          EXCLUDE from training
    temporal_continuity  no unexplained gap > 2 years per ticker   WARN
    scale_sanity         |Z"| <= 50                                FLAG

SEVERITY SEMANTICS — deliberately distinct, do not collapse them
----------------------------------------------------------------
    HARD_FAIL_BATCH  the whole batch is rejected; nothing loads
    HARD_FAIL_ROW    the offending rows are removed from the batch
    EXCLUDE_TRAINING rows are retained and loaded, but marked unusable for training
    FLAG             rows are retained and loaded, with a marker column
    WARN             reported only; no data effect

The distinction between FLAG and HARD_FAIL_ROW matters. A row failing the accounting
identity might be a genuine restatement; a row with negative total assets is
impossible and must go. Treating them alike either discards good data or admits
corrupt data.

USAGE
-----
    from quality_gates import run_all_gates, GateReport

    report = run_all_gates(panel)
    report.summary()                 # prints the per-gate table
    clean = report.apply(panel)      # returns the loadable subset
    if not report.passed:            # batch-level rejection
        raise SystemExit(1)

    python src/quality_gates.py data/processed/bloomberg_panel.csv
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ── severities ──────────────────────────────────────────────────────────
HARD_FAIL_BATCH = "HARD_FAIL_BATCH"
HARD_FAIL_ROW = "HARD_FAIL_ROW"
EXCLUDE_TRAINING = "EXCLUDE_TRAINING"
FLAG = "FLAG"
WARN = "WARN"

# Core features for the coverage gate (§5.4). These are the Altman Z" inputs plus
# the ratios the model cannot meaningfully run without.
CORE_FEATURES = [
    "total_assets", "current_assets", "current_liabilities",
    "retained_earnings", "ebit", "total_equity_book", "total_liabilities",
]

# §5.4 thresholds, named so they are greppable and citable in the methodology.
ACCOUNTING_IDENTITY_TOLERANCE = 0.02      # 2%
COVERAGE_MINIMUM = 0.70                   # 70%
ROA_RANGE = (-200.0, 200.0)
INTEREST_COVERAGE_RANGE = (-1000.0, 1000.0)
MAX_YEAR_GAP = 2
ALTMAN_SCALE_LIMIT = 50.0


@dataclass
class GateResult:
    name: str
    severity: str
    n_offending: int
    n_checked: int
    detail: str
    offending_index: pd.Index = field(default_factory=lambda: pd.Index([]))
    marker_column: str = None

    @property
    def passed(self) -> bool:
        return self.n_offending == 0

    @property
    def blocks_batch(self) -> bool:
        return self.severity == HARD_FAIL_BATCH and self.n_offending > 0


@dataclass
class GateReport:
    results: list

    @property
    def passed(self) -> bool:
        """False only if a BATCH-level gate failed. Row-level gates do not
        invalidate the batch — they remove or mark the offending rows."""
        return not any(r.blocks_batch for r in self.results)

    def get(self, name):
        return next((r for r in self.results if r.name == name), None)

    def summary(self) -> None:
        print("\n" + "=" * 78)
        print("DATA QUALITY GATE REPORT (CRESTA PRD §5.4)")
        print("=" * 78)
        print(f"{'Gate':22s} {'Severity':17s} {'Offending':>10s} {'Checked':>9s}  Detail")
        print("-" * 78)
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"{r.name:22s} {r.severity:17s} {r.n_offending:10d} {r.n_checked:9d}  "
                  f"[{mark}] {r.detail}")
        print("-" * 78)
        if self.passed:
            print("BATCH ACCEPTED — no batch-level gate failed.")
        else:
            blocking = [r.name for r in self.results if r.blocks_batch]
            print(f"*** BATCH REJECTED — blocking gate(s): {blocking} ***")
            print("    Nothing loads until these are resolved (§5.4).")

    def apply(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Returns the loadable subset with marker columns attached.

        HARD_FAIL_ROW rows are removed. EXCLUDE_TRAINING and FLAG rows are
        retained with a boolean marker — retaining them matters because
        dropping rows silently is exactly the survivorship failure §5.7 warns
        about, and a flagged row is still useful for coverage reporting.
        """
        out = df.copy()
        drop_idx = pd.Index([])

        for r in self.results:
            if r.marker_column:
                out[r.marker_column] = False
                if len(r.offending_index):
                    out.loc[r.offending_index, r.marker_column] = True
            if r.severity == HARD_FAIL_ROW and len(r.offending_index):
                drop_idx = drop_idx.union(r.offending_index)

        if len(drop_idx):
            out = out.drop(index=drop_idx)
            if verbose:
                print(f"\n[quality_gates] Removed {len(drop_idx)} row(s) failing "
                      f"HARD_FAIL_ROW gates.")

        excl = self.get("coverage")
        if excl is not None and verbose and excl.n_offending:
            print(f"[quality_gates] {excl.n_offending} row(s) marked "
                  f"'{excl.marker_column}' — retained but not usable for training.")
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "gate": r.name, "severity": r.severity, "offending": r.n_offending,
            "checked": r.n_checked, "passed": r.passed, "detail": r.detail,
        } for r in self.results])


# ── the gates ───────────────────────────────────────────────────────────
def gate_primary_key(df) -> GateResult:
    """(ticker, fiscal_year) unique and non-null. Batch-blocking.

    A duplicate primary key means either the export double-counted a company
    (the -PP/-RT rights-and-partly-paid duplication seen in EQS) or two
    different companies collided on a ticker. Either corrupts every groupby
    downstream, so the batch cannot load.
    """
    keys = ["ticker", "fiscal_year"]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        return GateResult("primary_key", HARD_FAIL_BATCH, len(df), len(df),
                          f"key column(s) absent: {missing}", df.index)

    null_mask = df[keys].isna().any(axis=1)
    dup_mask = df.duplicated(subset=keys, keep=False) & ~null_mask
    bad = df.index[null_mask | dup_mask]

    if len(bad):
        dups = df.loc[dup_mask, "ticker"].unique()[:5]
        detail = (f"{int(null_mask.sum())} null key(s), {int(dup_mask.sum())} duplicate(s)"
                  + (f"; e.g. {list(dups)}" if len(dups) else ""))
    else:
        detail = "unique and non-null"
    return GateResult("primary_key", HARD_FAIL_BATCH, len(bad), len(df), detail, bad)


def gate_accounting_identity(df) -> GateResult:
    """|TA - (TL + TE)| within 2%. Flags rows; does not block.

    Deviations can be genuine (restatements, minority interest treatment), so
    this flags rather than rejects. But a WIDESPREAD failure means the column
    mapping is wrong — which is exactly how the header off-by-one was caught.
    """
    need = ["total_assets", "total_equity_book", "total_liabilities"]
    if not all(c in df.columns for c in need):
        return GateResult("accounting_identity", FLAG, 0, 0,
                          "skipped — columns absent", marker_column="qc_identity_fail")

    m = df[need].notna().all(axis=1)
    checked = int(m.sum())
    if checked == 0:
        return GateResult("accounting_identity", FLAG, 0, 0, "no complete rows",
                          marker_column="qc_identity_fail")

    diff = (df.loc[m, "total_assets"]
            - (df.loc[m, "total_equity_book"] + df.loc[m, "total_liabilities"])).abs()
    rel = diff / df.loc[m, "total_assets"].abs().replace(0, np.nan)
    bad = rel.index[rel > ACCOUNTING_IDENTITY_TOLERANCE]

    ok_pct = (1 - len(bad) / checked) * 100
    detail = f"{ok_pct:.1f}% within {ACCOUNTING_IDENTITY_TOLERANCE:.0%}, max rel err {rel.max():.2e}"
    if ok_pct < 95:
        detail += "  <<< WIDESPREAD FAILURE — SUSPECT COLUMN MISALIGNMENT"
    return GateResult("accounting_identity", FLAG, len(bad), checked, detail, bad,
                      "qc_identity_fail")


def gate_sign_sanity(df) -> GateResult:
    """total_assets > 0; cur_ratio >= 0; cur_mkt_cap > 0. Removes rows.

    These are impossible values, not extreme ones. Negative total assets cannot
    occur in any accounting framework, so the row is corrupt rather than
    interesting. Note NaN is NOT a violation — missingness is handled by the
    coverage gate and the §5.5 missing-data doctrine, not here.
    """
    bad = pd.Index([])
    parts = []
    for col, rule, ok in [("total_assets", "> 0", lambda s: s > 0),
                          ("cur_ratio", ">= 0", lambda s: s >= 0),
                          ("cur_mkt_cap", "> 0", lambda s: s > 0)]:
        if col not in df.columns:
            continue
        present = df[col].notna()
        viol = df.index[present & ~ok(df[col])]
        if len(viol):
            parts.append(f"{col} {rule}: {len(viol)}")
        bad = bad.union(viol)
    detail = "; ".join(parts) if parts else "all present values within sign rules"
    return GateResult("sign_sanity", HARD_FAIL_ROW, len(bad), len(df), detail, bad,
                      "qc_sign_fail")


def gate_range_plausibility(df) -> GateResult:
    """ROA in [-200,200]; interest_coverage in [-1000,1000]. Flags, retains.

    Retention is deliberate (§5.4 "retain with marker"): an interest coverage of
    -234 is implausible as a ratio but is a real signal of a firm with negative
    EBIT and trivial interest expense. Removing it would delete distress cases.
    """
    bad = pd.Index([])
    parts = []
    for col, (lo, hi) in [("roa", ROA_RANGE), ("interest_coverage", INTEREST_COVERAGE_RANGE)]:
        if col not in df.columns:
            continue
        present = df[col].notna()
        viol = df.index[present & ((df[col] < lo) | (df[col] > hi))]
        if len(viol):
            parts.append(f"{col} outside [{lo:g},{hi:g}]: {len(viol)}")
        bad = bad.union(viol)
    detail = "; ".join(parts) if parts else "all present values in range"
    return GateResult("range_plausibility", FLAG, len(bad), len(df), detail, bad,
                      "qc_range_outlier")


def gate_coverage(df) -> GateResult:
    """>= 70% non-null across CORE_FEATURES. Excludes from training, retains row.

    Retaining is important: a bank row fails coverage structurally (§5.6) and a
    delisted company's final years fail it naturally. Both are informative for
    the data-quality report even though neither can be trained on.
    """
    cols = [c for c in CORE_FEATURES if c in df.columns]
    if not cols:
        return GateResult("coverage", EXCLUDE_TRAINING, 0, 0, "no core columns present",
                          marker_column="qc_below_coverage")

    frac = df[cols].notna().mean(axis=1)
    bad = df.index[frac < COVERAGE_MINIMUM]
    detail = (f"{len(bad)} row(s) below {COVERAGE_MINIMUM:.0%} across {len(cols)} core "
              f"features (mean coverage {frac.mean():.1%})")
    return GateResult("coverage", EXCLUDE_TRAINING, len(bad), len(df), detail, bad,
                      "qc_below_coverage")


def gate_temporal_continuity(df) -> GateResult:
    """No gap > 2 years within a ticker's series. Warns only.

    A gap usually means a ticker change, a merger, or a suspension — all worth
    investigating, none of which justify deleting data automatically.
    """
    if "fiscal_year_num" not in df.columns or "ticker" not in df.columns:
        return GateResult("temporal_continuity", WARN, 0, 0, "skipped — columns absent")

    offenders, bad = [], pd.Index([])
    for tkr, g in df.dropna(subset=["fiscal_year_num"]).groupby("ticker"):
        yrs = g.sort_values("fiscal_year_num")
        gaps = yrs["fiscal_year_num"].diff()
        hit = yrs.index[gaps > MAX_YEAR_GAP]
        if len(hit):
            offenders.append(f"{tkr}({int(gaps.max())}y)")
            bad = bad.union(hit)

    detail = (f"gaps > {MAX_YEAR_GAP}y in: {', '.join(offenders[:5])}"
              + (" ..." if len(offenders) > 5 else "")) if offenders else "no gaps"
    return GateResult("temporal_continuity", WARN, len(bad), df["ticker"].nunique(),
                      detail, bad)


def gate_scale_sanity(df) -> GateResult:
    """|Z''| <= 50. Flags near-certain units/scaling errors upstream.

    Only meaningful once Altman Z" has been computed, so it self-skips before
    that. Excludes negative-equity rows, which legitimately produce extreme
    scores — see altman_zscore.altman_flag_check_scaling.
    """
    col = "altman_z_double_prime"
    if col not in df.columns:
        return GateResult("scale_sanity", FLAG, 0, 0,
                          "skipped — Z'' not computed yet", marker_column="qc_scale_suspect")

    present = df[col].notna()
    extreme = present & (df[col].abs() > ALTMAN_SCALE_LIMIT)
    if "altman_flag_negative_equity" in df.columns:
        extreme = extreme & ~df["altman_flag_negative_equity"].fillna(False).astype(bool)

    bad = df.index[extreme]
    detail = (f"{len(bad)} row(s) with |Z''| > {ALTMAN_SCALE_LIMIT:g} unexplained by "
              f"negative equity" if len(bad) else f"all |Z''| <= {ALTMAN_SCALE_LIMIT:g}")
    return GateResult("scale_sanity", FLAG, len(bad), int(present.sum()), detail, bad,
                      "qc_scale_suspect")


ALL_GATES = [
    gate_primary_key,
    gate_accounting_identity,
    gate_sign_sanity,
    gate_range_plausibility,
    gate_coverage,
    gate_temporal_continuity,
    gate_scale_sanity,
]


def run_all_gates(df: pd.DataFrame) -> GateReport:
    return GateReport([g(df) for g in ALL_GATES])


def write_rejection_log(df, report, path="data/processed/quality_rejections.csv"):
    """Persists every offending row with the gate that caught it.

    §5.7 and §5.4 both require that rejected rows are logged rather than
    dropped. This file is also the source for the Data Quality dashboard's
    'quality-gate failure log' (§12.5).
    """
    rows = []
    id_cols = [c for c in ["ticker", "company_name", "fiscal_year"] if c in df.columns]
    for r in report.results:
        for idx in r.offending_index:
            rec = {c: df.at[idx, c] for c in id_cols}
            rec.update({"gate": r.name, "severity": r.severity, "detail": r.detail})
            rows.append(rec)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame(rows)
    log.to_csv(out, index=False)
    print(f"[quality_gates] Wrote {len(log)} rejection record(s) -> {out}")
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel")
    ap.add_argument("-o", "--out", default="data/processed/bloomberg_panel_gated.csv")
    ap.add_argument("--log", default="data/processed/quality_rejections.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    print(f"[quality_gates] Loaded {len(df)} row(s) from {args.panel}")

    report = run_all_gates(df)
    report.summary()
    write_rejection_log(df, report, args.log)

    clean = report.apply(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out, index=False)
    print(f"[quality_gates] Wrote {len(clean)} row(s) -> {args.out}")

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
