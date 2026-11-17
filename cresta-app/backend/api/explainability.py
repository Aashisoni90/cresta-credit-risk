"""
Explainability endpoints — §8.8, real Model A.

Serves all four §8.8 levels. Global and block-level attribution are read from
the Phase 3 artefacts (`reports/phase3/shap_*.csv`), which were produced by the
SAME fitted model this API serves; recomputing them here would risk the API and
the paper disagreeing. Local explanations are computed live, because they are
per-company and cannot be precomputed for every request.

EVERY RESPONSE CARRIES THE CAVEAT. §8.8 names conflating attribution with
causation as a visible and damaging error, so the disclaimer travels with the
payload rather than living in UI copy that can be dropped in a redesign.
"""
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import config
from services import real_data

router = APIRouter()

CAVEAT = ("SHAP explains the MODEL, not the world. A high attribution means the "
          "model relies on that feature; it does not establish causation (PRD §8.8).")

def _get_explainer():
    """TreeSHAP over the served model, built at startup by
    real_data.warm_explainer(). Falls back to building it on demand so the
    endpoint still works if the warm-up hook is ever removed."""
    s = real_data.state()
    if "explainer" not in s:
        return real_data.warm_explainer()
    return s["explainer"]


def _read_csv(path, label):
    if not path.exists():
        raise HTTPException(status_code=503, detail={
            "code": "ARTEFACT_MISSING",
            "message": f"{label} not found at {path}. Run "
                       f"`PYTHONPATH=src python src/explain_model_a.py`."})
    # NaN is not valid JSON. It appears legitimately here — a constant feature
    # has no correlation with its own SHAP — so it becomes null rather than
    # being dropped or coerced to 0.0, which would read as "no relationship"
    # instead of "not defined".
    return pd.read_csv(path).replace({np.nan: None})


@router.get("/model/shap/global")
async def shap_global(top_n: int = Query(20, ge=1, le=100)):
    """Global mean-|SHAP| ranking (§8.8 'Global')."""
    df = _read_csv(config.SHAP_GLOBAL_PATH, "Global SHAP ranking")
    return {
        "model_version": real_data.state()["version"],
        "n_features": len(df),
        "features": df.head(top_n).to_dict("records"),
        "note": ("`mean_signed_shap` gives the direction of the model's typical "
                 "reliance, which mean-|SHAP| discards; "
                 "`corr_feature_vs_shap` gives the sign of the relationship."),
        "caveat": CAVEAT,
    }


@router.get("/model/shap/blocks")
async def shap_blocks():
    """Block-level attribution (§8.8 'Block-level' — the committee view)."""
    df = _read_csv(config.SHAP_BLOCK_PATH, "Block SHAP ranking")
    meta = {}
    if config.SHAP_SUMMARY_PATH.exists():
        meta = json.loads(config.SHAP_SUMMARY_PATH.read_text())
    return {
        "model_version": real_data.state()["version"],
        "blocks": df.to_dict("records"),
        "base_rate": meta.get("base_rate"),
        "note": ("`ebit_to_assets` is catalogued under debt-service (it is the "
                 "numerator of interest coverage) but is attributed to "
                 "profitability here: no interest-coverage feature reached this "
                 "matrix, so reporting it as debt-service would claim the model "
                 "uses a debt-service variable it never sees."),
        "caveat": CAVEAT,
    }


@router.get("/predict/explanation")
async def explanation(ticker: str, fiscal_year: int | None = None,
                      top_n: int = Query(8, ge=1, le=40)):
    """Local per-prediction waterfall (§8.8 'Local'): base rate -> contributions."""
    if real_data.sector_unsupported(ticker):
        raise HTTPException(status_code=422, detail={
            "code": config.SECTOR_UNSUPPORTED_CODE,
            "message": config.SECTOR_UNSUPPORTED_MESSAGE, "ticker": ticker})

    row = real_data.company_row(ticker, fiscal_year)
    if row is None:
        raise HTTPException(status_code=404, detail={
            "code": "NO_DATA",
            "message": f"No scored company-year for {ticker}.", "ticker": ticker})

    s = real_data.state()
    feats = s["features"]
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}], columns=feats) \
        .apply(pd.to_numeric, errors="coerce")
    sv = _get_explainer()(X)
    vals = np.asarray(sv.values)[0]
    base = float(np.atleast_1d(sv.base_values)[0])
    p = float(row["probability"])

    # Additivity: base + sum(contributions) must reproduce the model's margin.
    # Checked per request because a silent mismatch would mean the waterfall
    # shown to an analyst does not add up to the number beside it.
    recon = base + float(vals.sum())
    margin = float(np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12))))
    additivity_error = abs(recon - margin)

    order = np.argsort(-np.abs(vals))
    contributions = [{
        "feature": feats[i],
        "shap_log_odds": round(float(vals[i]), 4),
        "direction": "increases risk" if vals[i] > 0 else "decreases risk",
        "value": (None if pd.isna(X.iloc[0, i]) else round(float(X.iloc[0, i]), 4)),
        "value_missing": bool(pd.isna(X.iloc[0, i])),
    } for i in order[:top_n]]

    year = int(row["fiscal_year_num"])
    return {
        "ticker": ticker,
        "company_name": real_data.display_name(ticker),
        "fiscal_year": year,
        "prediction_horizon": f"FY{year + 1}-{str(year + 2)[-2:]}",
        "probability": round(p, 4),
        "risk_category": real_data.classify_risk(p),
        "base_value_log_odds": round(base, 4),
        "base_rate": round(float(1 / (1 + np.exp(-base))), 4),
        "base_rate_note": ("the model's class-rebalanced prior "
                           "(scale_pos_weight), not the sample prevalence"),
        "contributions": contributions,
        "additivity_error": float(additivity_error),
        "additivity_ok": bool(additivity_error < 1e-3),
        "data_quality_flags": real_data.data_quality_flags(ticker, year),
        "caveat": CAVEAT,
    }


@router.get("/predict/counterfactual")
async def counterfactual(ticker: str, fiscal_year: int | None = None,
                         top_n: int = Query(5, ge=1, le=20)):
    """Nearest healthy peer (§8.8 'Counterfactual').

    The peer is matched on size among companies the model scored BELOW its
    operating threshold, so the contrast is against an actually-healthy
    comparator rather than merely a lower-scored one.
    """
    if real_data.sector_unsupported(ticker):
        raise HTTPException(status_code=422, detail={
            "code": config.SECTOR_UNSUPPORTED_CODE,
            "message": config.SECTOR_UNSUPPORTED_MESSAGE, "ticker": ticker})
    row = real_data.company_row(ticker, fiscal_year)
    if row is None:
        raise HTTPException(status_code=404, detail={
            "code": "NO_DATA", "message": f"No scored company-year for {ticker}."})

    s = real_data.state()
    feats, m, thr = s["features"], s["matrix"], s["threshold"]
    year = int(row["fiscal_year_num"])

    pool = m[(m["fiscal_year_num"] == year) & (m["probability"] < thr / 2)
             & (m["ticker"] != ticker)]
    if pool.empty:
        raise HTTPException(status_code=404, detail={
            "code": "NO_PEER",
            "message": f"No sufficiently healthy peer scored in FY{year}."})

    Z = pd.concat([pool[feats], pd.DataFrame([row[feats]])], ignore_index=True)
    mu, sd = Z.mean(), Z.std().replace(0, 1.0)
    Zs = ((Z - mu) / sd).fillna(0.0)
    target, peers = Zs.iloc[-1], Zs.iloc[:-1]
    if "log_total_assets" in feats:   # size band first, distance second
        band = (peers["log_total_assets"] - target["log_total_assets"]).abs()
        keep = band <= 0.5
        if keep.any():
            peers, pool = peers[keep], pool[keep.values]
    j = ((peers - target) ** 2).sum(axis=1).idxmin()
    peer = pool.iloc[peers.index.get_loc(j)]

    ex = _get_explainer()
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in feats},
                      {f: peer.get(f, np.nan) for f in feats}],
                     columns=feats).apply(pd.to_numeric, errors="coerce")
    vals = np.asarray(ex(X).values)
    gap = vals[0] - vals[1]
    order = np.argsort(-np.abs(gap))

    return {
        "ticker": ticker, "company_name": real_data.display_name(ticker),
        "fiscal_year": year, "probability": round(float(row["probability"]), 4),
        "peer": {"ticker": peer["ticker"],
                 "company_name": real_data.display_name(peer["ticker"]),
                 "probability": round(float(peer["probability"]), 4)},
        "peer_selection": ("nearest neighbour in standardised feature space, "
                           "within a ±0.5 sd size band, among companies scored "
                           f"below {thr/2:.3f} in the same fiscal year"),
        "differences": [{
            "feature": feats[i],
            "delta_shap_log_odds": round(float(gap[i]), 4),
            "company_value": (None if pd.isna(X.iloc[0, i]) else round(float(X.iloc[0, i]), 4)),
            "peer_value": (None if pd.isna(X.iloc[1, i]) else round(float(X.iloc[1, i]), 4)),
        } for i in order[:top_n]],
        "caveat": CAVEAT,
    }
