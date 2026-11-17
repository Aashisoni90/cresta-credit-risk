"""Prediction endpoints — real Model A (Track A only)."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from ml.model_manager import TrackBUnavailable, model_info
from services import real_data

router = APIRouter()


class PredictionRequest(BaseModel):
    ticker: str
    fiscal_year: int | None = None     # None -> most recent year available
    threshold: float | None = None     # None -> the model's validated threshold


@router.post("/predict")
async def predict(req: PredictionRequest):
    # §5.6 — out of scope is a DIFFERENT answer from no data, and the API must
    # not conflate them. 422, not 404: the request is well-formed, the issuer
    # is simply outside what this model is entitled to speak about.
    if real_data.sector_unsupported(req.ticker):
        raise HTTPException(status_code=422, detail={
            "code": config.SECTOR_UNSUPPORTED_CODE,
            "message": config.SECTOR_UNSUPPORTED_MESSAGE,
            "ticker": req.ticker,
            "exclusion_reason": real_data.exclusion_reason(req.ticker),
        })

    row = real_data.company_row(req.ticker, req.fiscal_year)
    if row is None:
        raise HTTPException(status_code=404, detail={
            "code": "NO_DATA",
            "message": (f"No scored company-year for {req.ticker}"
                        + (f" in FY{req.fiscal_year}" if req.fiscal_year else "")
                        + ". The ticker may have failed extraction or fallen "
                          "below the coverage gate."),
            "ticker": req.ticker,
        })

    p = float(row["probability"])
    thr = req.threshold if req.threshold is not None else real_data.state()["threshold"]
    year = int(row["fiscal_year_num"])
    info = model_info()

    return {
        "ticker": req.ticker,
        "company_name": real_data.display_name(req.ticker),
        "sector": real_data.sector_of(req.ticker),
        "fiscal_year": year,
        "fiscal_year_label": f"FY{year}-{str(year + 1)[-2:]}",
        # The model predicts the label at t+1 from features at t. Stating the
        # horizon on every response stops a probability being read as a
        # statement about the current year.
        "prediction_horizon": f"FY{year + 1}-{str(year + 2)[-2:]} (X_t -> Y_t+1)",
        "model_a": {
            "probability": round(p, 4),
            "risk_category": real_data.classify_risk(p),
            "flagged": bool(p >= thr),
            "model_type": "financial_only",
            "model_version": info["version"],
        },
        # Reported as explicitly unavailable, never as a number. See
        # ml/model_manager.TrackBUnavailable.
        "model_b": None,
        "track_b_status": {
            "available": False,
            "code": TrackBUnavailable.CODE,
            "message": TrackBUnavailable.MESSAGE,
        },
        "threshold_used": round(float(thr), 4),
        "threshold_basis": (
            f"cost-optimal on the validation window at a "
            f"{config.COST_RATIO_FN_FP:.0f}:1 false-negative:false-positive "
            f"cost ratio (§8.4, a stated assumption)"),
        "data_quality_flags": real_data.data_quality_flags(req.ticker, year),
        "disclaimer": (
            "Analytical estimate, not a decision. Not investment advice, not a "
            "credit rating, and not a probability of default — the label is a "
            "financial-criteria distress proxy. Requires qualified human review "
            "before any credit, lending or investment action (PRD §14.5)."),
    }


@router.get("/predict/portfolio")
async def portfolio(limit: int = 50, min_probability: float = 0.0,
                    fiscal_year: int | None = None, sector: str | None = None):
    """Ranked portfolio view — the triage screen (§12.2).

    Served from predictions computed once at startup rather than scored per
    request, so the §12.6 under-2s interactivity requirement holds.
    """
    m = real_data.state()["matrix"]
    year = fiscal_year if fiscal_year is not None else int(m["fiscal_year_num"].max())
    d = m[m["fiscal_year_num"] == year]
    if sector:
        d = d[d.get("gics_sector") == sector]
    d = d[d["probability"] >= min_probability].nlargest(limit, "probability")

    thr = real_data.state()["threshold"]
    return {
        "fiscal_year": year,
        "prediction_horizon": f"FY{year + 1}-{str(year + 2)[-2:]}",
        "threshold": round(thr, 4),
        "n_returned": len(d),
        "n_scored_in_year": int((m["fiscal_year_num"] == year).sum()),
        "companies": [{
            "ticker": r["ticker"],
            "company_name": real_data.display_name(r["ticker"]),
            "sector": r.get("gics_sector") if pd.notna(r.get("gics_sector")) else None,
            "probability": round(float(r["probability"]), 4),
            "risk_category": real_data.classify_risk(float(r["probability"])),
            "flagged": bool(r["probability"] >= thr),
            "n_quality_flags": len(real_data.data_quality_flags(
                r["ticker"], int(r["fiscal_year_num"]))),
        } for _, r in d.iterrows()],
        "disclaimer": "Analytical estimates requiring human review (PRD §14.5).",
    }


@router.get("/model/info")
async def info():
    s = real_data.state()
    return {**model_info(),
            "n_scored_rows": int(len(s["matrix"])),
            "n_tickers": int(s["matrix"]["ticker"].nunique()),
            "n_sector_unsupported_tickers": len(s["excluded_tickers"]),
            "track_b_available": False,
            "risk_bands": {k: round(v, 4) for k, v in real_data.risk_bands().items()}}
