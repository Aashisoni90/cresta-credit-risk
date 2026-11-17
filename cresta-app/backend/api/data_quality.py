"""
Data-quality endpoint (§12.5) — makes limitations visible rather than tacit.

This screen is a governance requirement and an honesty mechanism: it reports
what was rejected, what failed to resolve, and what is missing, from the real
logs rather than from a summary written by hand.
"""
import pandas as pd
from fastapi import APIRouter

import config
from services import real_data

router = APIRouter()


def _count(path):
    return int(len(pd.read_csv(path))) if path.exists() else None


@router.get("/data-quality")
async def data_quality():
    s = real_data.state()
    panel, matrix = s["panel"], s["matrix"]

    # Gates as a LIST with a verdict, not a bare count map: §12.5 asks for the
    # failure log with reasons, and a count alone does not say whether a gate
    # blocks a row or merely warns. qc_below_coverage warns — those rows are
    # kept in training and marked, which is the opposite of a hard reject.
    GATE_META = {
        "qc_identity_fail": ("balance-sheet identity", "hard",
                             "assets must equal equity + liabilities within 2%"),
        "qc_sign_fail": ("sign checks", "hard",
                         "quantities that cannot be negative"),
        "qc_range_outlier": ("range outliers", "warn",
                             "values far outside the plausible band"),
        "qc_below_coverage": ("field coverage", "warn",
                              "kept in training and marked, never dropped"),
        "qc_scale_suspect": ("scale/units", "warn",
                             "possible order-of-magnitude error"),
    }
    gates = []
    for col in [c for c in panel.columns if c.startswith("qc_")]:
        label, kind, why = GATE_META.get(col, (col, "warn", ""))
        n = int(panel[col].fillna(False).astype(bool).sum())
        gates.append({"gate": col, "name": label, "kind": kind,
                      "description": why, "n_offending": n,
                      "status": "warning" if n else "ok"})
    gates.sort(key=lambda g: -g["n_offending"])

    n_raw = int(len(panel))
    n_usable = int(len(matrix))
    n_coverage_flagged = int(
        panel["qc_below_coverage"].fillna(False).astype(bool).sum()
        if "qc_below_coverage" in panel.columns else 0)

    feats = s["features"]
    nulls = sorted(
        ({"feature": f,
          "null_rate_pct": round(float(matrix[f].isna().mean()) * 100, 2),
          "n_null": int(matrix[f].isna().sum())} for f in feats),
        key=lambda r: -r["null_rate_pct"])

    coverage = (matrix.groupby("fiscal_year_num")
                .apply(lambda d: 100 * (1 - d[feats].isna().mean().mean()),
                       include_groups=False))

    return {
        "universe": {
            "n_tickers_scored": int(matrix["ticker"].nunique()),
            "n_company_years": int(len(matrix)),
            "n_sector_unsupported_tickers": len(s["excluded_tickers"]),
            "sector_unsupported_note": (
                "Financial-sector issuers are OUT OF SCOPE (§5.6), not missing "
                "data. They were excluded at matrix assembly and logged."),
        },
        "pipeline_flow": {
            "raw_company_years": n_raw,
            "removed": n_raw - n_usable,
            "usable_company_years": n_usable,
            "coverage_flagged_kept": n_coverage_flagged,
            "note": (f"{n_raw:,} scored panel rows reduce to {n_usable:,} model "
                     f"rows: financial-sector exclusion (§5.6), the forward-label "
                     f"join that drops the final year, and hard gate failures. "
                     f"{n_coverage_flagged:,} coverage-flagged rows were KEPT and "
                     f"marked rather than dropped."),
        },
        "quality_gates": gates,
        "n_quality_rejections": _count(config.QUALITY_REJECTIONS_PATH),
        "n_failed_tickers": _count(config.FAILED_TICKERS_PATH),
        "failed_ticker_note": (
            "Dead and unresolvable tickers are logged, never silently dropped: "
            "distressed and delisted firms fail disproportionately, so dropping "
            "them would reintroduce survivorship bias (invariant 4)."),
        "feature_null_rates": nulls,
        "coverage_by_year_pct": {int(k): round(float(v), 2)
                                 for k, v in coverage.items()},
        "missingness_policy": (
            "Financial ratios are NEVER zero-filled. A missing ratio is "
            "information — negative equity makes debt/equity undefined — so it "
            "is carried as NaN and paired with an explicit missingness "
            "indicator (invariant 1)."),
        "esg_coverage": None,
        "esg_note": "Track B (ESG / BRSR) not extracted — see /api/model/info.",
    }
