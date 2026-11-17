"""
Model-performance endpoints (§12.4) — real Model A.

Serves the Phase 3 evidence surface: the §8.1 comparison table, rolling-origin
fold distribution, per-stratum breakdown and the model card. Figures are read
from `reports/phase3/`, not recomputed, so the dashboard and the paper cannot
disagree.
"""
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

import config
from services import real_data

router = APIRouter()


def _require(path, label):
    if not path.exists():
        raise HTTPException(status_code=503, detail={
            "code": "ARTEFACT_MISSING",
            "message": f"{label} not found at {path}."})
    return path


@router.get("/model/metrics")
async def metrics():
    """Headline out-of-time metrics, with the persistence caveat attached.

    The pooled AUC is inflated by label persistence (the §7.5 label is
    unchanged year-over-year on ~90% of rows), so the honest early-warning
    figure is returned in the same payload. Serving the pooled number alone
    would overstate the model to anyone reading the dashboard.
    """
    df = pd.read_csv(_require(config.MODEL_COMPARISON_PATH, "model comparison"))
    s = real_data.state()
    m = s["matrix"]
    # Evaluation population only, so these figures match MODEL_CARD.md exactly.
    te = m[(m["split"] == "test") & m["is_evaluation_row"]]
    return {
        "model_version": s["version"],
        "test_window": [int(te["fiscal_year_num"].min()),
                        int(te["fiscal_year_num"].max())],
        "n_test_rows": int(len(te)),
        "test_prevalence": round(float(te["label_binary_next"].mean()), 4),
        "models": df.replace({np.nan: None}).to_dict("records"),
        "headline_caveat": (
            "Pooled AUC is inflated by label persistence: the §7.5 label is "
            "identical at t and t+1 on roughly 90% of rows, so a naive "
            "'the condition continues' baseline already scores ~0.886. The "
            "honest early-warning figure is the healthy-at-t stratum, where "
            "persistence scores 0.5 by construction. See /api/model/strata."),
        "scored_once": True,
    }


@router.get("/model/strata")
async def strata():
    """Stratified performance — the early-warning number that is not persistence."""
    s = real_data.state()
    m = s["matrix"]
    te = m[(m["split"] == "test") & m["is_evaluation_row"]].copy()
    if "label_now" not in te.columns:
        # label_now is a training-time construct; recompute the stratum from the
        # previous year's label rather than asserting it exists.
        prev = (m[["ticker", "fiscal_year_num", "label_binary_next"]]
                .rename(columns={"label_binary_next": "label_now",
                                 "fiscal_year_num": "_y"}))
        prev["_y"] = prev["_y"] + 1
        te = te.merge(prev.rename(columns={"_y": "fiscal_year_num"}),
                      on=["ticker", "fiscal_year_num"], how="left")

    out = []
    for name, d in [("all_test_rows", te),
                    ("already_distressed_at_t", te[te["label_now"] == 1]),
                    ("healthy_at_t_early_warning", te[te["label_now"] == 0])]:
        y = d["label_binary_next"].dropna()
        if y.nunique() < 2:
            continue
        out.append({
            "stratum": name, "n": int(len(d)),
            "prevalence": round(float(y.mean()), 4),
            "auc": round(float(roc_auc_score(y, d.loc[y.index, "probability"])), 4),
        })
    return {"model_version": s["version"], "strata": out,
            "note": ("Persistence scores 0.5 on the healthy-at-t stratum by "
                     "construction, so lift there is genuine predictive skill.")}


@router.get("/model/rolling-origin")
async def rolling_origin():
    """§8.5 walk-forward fold distribution — stability, not a point estimate."""
    df = pd.read_csv(_require(config.ROLLING_ORIGIN_PATH, "rolling-origin summary"))
    cols = [c for c in df.columns if c not in ("fold", "test_year")]
    return {
        "folds": df.replace({np.nan: None}).to_dict("records"),
        "summary": {c: {"mean": round(float(df[c].mean()), 4),
                        "sd": round(float(df[c].std()), 4)} for c in cols},
        "note": ("Columns suffixed _EW are the healthy-at-t early-warning "
                 "subset. Persistence has no _EW column: it scores 0.5 there "
                 "by construction."),
    }


@router.get("/model/curves")
async def curves(n_points: int = 200):
    """ROC and precision-recall curves on the out-of-time test split (§12.4)."""
    m = real_data.state()["matrix"]
    te = (m[(m["split"] == "test") & m["is_evaluation_row"]]
          .dropna(subset=["label_binary_next"]))
    y, p = te["label_binary_next"].values, te["probability"].values

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    idx = np.linspace(0, len(fpr) - 1, min(n_points, len(fpr))).astype(int)
    jdx = np.linspace(0, len(prec) - 1, min(n_points, len(prec))).astype(int)

    # Decile calibration (§12.4): predicted probability against observed
    # frequency. A model whose probabilities are shown to an analyst has to
    # mean something, not merely rank correctly.
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q")
           .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")))

    return {
        "roc": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx],
        "pr": [{"recall": round(float(rec[j]), 4), "precision": round(float(prec[j]), 4)} for j in jdx],
        "calibration": [{"decile": int(k), "predicted": round(float(r.predicted), 4),
                         "observed": round(float(r.observed), 4), "n": int(r.n)}
                        for k, r in cal.iterrows()],
        "n_test_rows": int(len(te)),
    }


@router.get("/model/card")
async def model_card():
    """The §14.1 model card, served so the governance text travels with the API."""
    p = _require(config.MODEL_CARD_PATH, "model card")
    return {"format": "markdown", "content": p.read_text()}


@router.get("/model/registry")
async def registry():
    return {"versions": real_data.registry(), "serving": real_data.state()["version"]}
