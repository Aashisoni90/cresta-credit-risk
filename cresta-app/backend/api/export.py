"""
Export endpoints (§12.6).

"CSV and PDF from portfolio and company views, with model version and
generation timestamp stamped on the export."

WHY THE STAMP IS NOT OPTIONAL METADATA
---------------------------------------
An exported CSV outlives the session that produced it. Six months later the
only way to know which model, which threshold and which cost assumption
produced a flag is what the file itself says. So every export carries a
commented preamble with the model version, the config hash, the generation
timestamp, the operating threshold and the cost ratio — and the §14.5
human-review constraint, because a spreadsheet circulating without it is
exactly how an analytical estimate gets read as a decision.

Comment lines are prefixed with `#`, which Excel and pandas both skip
(`pd.read_csv(..., comment="#")`).

PDF IS DELIBERATELY NOT IMPLEMENTED HERE
-----------------------------------------
The PDF half of §12.6 is better served by the browser's own print pipeline
against a print stylesheet than by a second rendering engine on the server
that would drift from what the screen shows. `/export/manifest` gives the
front end the same stamp to put in the print header. Recorded as a deviation
rather than silently dropped.
"""
import csv
import io
import json
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

import config
from services import real_data

router = APIRouter()


def _stamp() -> dict:
    s = real_data.state()
    reg = next((r for r in real_data.registry()
                if r.get("version") == s["version"]), {})
    return {
        "model_version": s["version"],
        "config_hash": reg.get("config_hash"),
        "git_commit": reg.get("git_commit"),
        "trained_utc": s["payload"].get("trained_utc"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operating_threshold": round(s["threshold"], 4),
        "cost_ratio_fn_fp": config.COST_RATIO_FN_FP,
        "track_b_esg": "unavailable (deferred)",
        "scope_note": ("Financial-sector issuers are excluded from this model "
                       "(PRD §5.6) and do not appear in any export."),
        "label_note": ("Target is the §7.5 forward-looking financial-criteria "
                       "distress proxy at FY(t+1) — not a realised default."),
        "use_note": ("ANALYTICAL ESTIMATE, NOT A DECISION. Not investment "
                     "advice, not a credit rating. Requires qualified human "
                     "review before any credit, lending or investment action "
                     "(PRD §14.5)."),
    }


def _csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    stamp = _stamp()
    buf.write("# CRESTA export\n")
    for k, v in stamp.items():
        buf.write(f"# {k}: {v}\n")
    buf.write("#\n# Read with: pd.read_csv(path, comment='#')\n")
    df.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # An export is a point-in-time artefact; caching one would hand
            # back a stale timestamp on the next download.
            "Cache-Control": "no-store",
            "X-Model-Version": stamp["model_version"],
        },
    )


@router.get("/export/manifest")
async def manifest():
    """The same stamp the CSVs carry, for the print/PDF header."""
    return _stamp()


@router.get("/export/portfolio.csv")
async def export_portfolio(fiscal_year: int | None = None,
                           sector: str | None = None,
                           min_probability: float = 0.0,
                           limit: int = Query(5000, ge=1, le=50000)):
    m = real_data.state()["matrix"]
    year = fiscal_year if fiscal_year is not None else int(m["fiscal_year_num"].max())
    d = m[m["fiscal_year_num"] == year]
    if sector and "gics_sector" in d.columns:
        d = d[d["gics_sector"] == sector]
    d = d[d["probability"] >= min_probability].nlargest(limit, "probability")
    if d.empty:
        raise HTTPException(status_code=404, detail={
            "code": "NO_DATA", "message": f"No scored companies for FY{year}."})

    thr = real_data.state()["threshold"]
    out = pd.DataFrame({
        "ticker": d["ticker"],
        "company_name": [real_data.display_name(t) for t in d["ticker"]],
        "sector": d.get("gics_sector"),
        "fiscal_year": d["fiscal_year_num"].astype(int),
        "fiscal_year_label": [f"FY{int(y)}-{str(int(y)+1)[-2:]}"
                              for y in d["fiscal_year_num"]],
        "prediction_horizon": [f"FY{int(y)+1}-{str(int(y)+2)[-2:]}"
                               for y in d["fiscal_year_num"]],
        "probability": d["probability"].round(6),
        "risk_band": [real_data.classify_risk(float(p)) for p in d["probability"]],
        "flagged": (d["probability"] >= thr).astype(int),
        # Carried per row so a filtered copy of this file still says what the
        # flag meant.
        "threshold": thr,
        "in_evaluation_population": d["is_evaluation_row"].astype(int),
        "data_quality_flags": [
            "; ".join(real_data.data_quality_flags(t, int(y))) or ""
            for t, y in zip(d["ticker"], d["fiscal_year_num"])],
    })
    return _csv_response(out, f"cresta_portfolio_FY{year}.csv")


@router.get("/export/company.csv")
async def export_company(ticker: str):
    if real_data.sector_unsupported(ticker):
        raise HTTPException(status_code=422, detail={
            "code": config.SECTOR_UNSUPPORTED_CODE,
            "message": config.SECTOR_UNSUPPORTED_MESSAGE, "ticker": ticker})
    hist = real_data.company_history(ticker)
    if hist.empty:
        raise HTTPException(status_code=404, detail={
            "code": "NO_DATA", "message": f"No scored company-year for {ticker}."})

    s = real_data.state()
    thr = s["threshold"]
    cols = {
        "ticker": hist["ticker"],
        "company_name": real_data.display_name(ticker),
        "fiscal_year": hist["fiscal_year_num"].astype(int),
        "probability": hist["probability"].round(6),
        "risk_band": [real_data.classify_risk(float(p)) for p in hist["probability"]],
        "flagged": (hist["probability"] >= thr).astype(int),
        "label_next": hist.get("label_binary_next"),
        "data_quality_flags": [
            "; ".join(real_data.data_quality_flags(ticker, int(y))) or ""
            for y in hist["fiscal_year_num"]],
    }
    # The model's own inputs, so a reader can audit the estimate rather than
    # take it on trust.
    for f in s["features"]:
        if f in hist.columns:
            cols[f] = hist[f]
    return _csv_response(pd.DataFrame(cols),
                         f"cresta_{ticker.split()[0]}_history.csv")
