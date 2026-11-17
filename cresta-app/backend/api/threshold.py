"""
Threshold endpoint (§8.4) — cost-ratio sensitivity on the real test split.

The deployed cost ratio is a STATED ASSUMPTION, not a default to leave
unexamined. At this label's ~22% prevalence a 10:1 ratio drives the threshold
toward flagging over half the portfolio, which is useless for triage — so the
sweep is served rather than a single number, and the operating point is
labelled with the assumption behind it.
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

import config
from services import real_data

router = APIRouter()


@router.get("/threshold")
async def threshold_sweep(n_points: int = Query(50, ge=5, le=500)):
    m = real_data.state()["matrix"]
    # Evaluation population only — see real_data.is_evaluation_row.
    te = (m[(m["split"] == "test") & m["is_evaluation_row"]]
          .dropna(subset=["label_binary_next"]))
    y, p = te["label_binary_next"].values.astype(int), te["probability"].values
    operating = real_data.state()["threshold"]

    rows = []
    for t in np.unique(np.concatenate([np.linspace(0.01, 0.99, n_points),
                                       [operating]])):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        rows.append({
            "threshold": round(float(t), 4),
            "recall": round(rec, 4), "precision": round(pre, 4),
            "f1": round(2 * rec * pre / (rec + pre), 4) if (rec + pre) else 0.0,
            # §8.6 reports accuracy "for completeness" only. It is served so
            # the UI can show the accuracy paradox directly: slide the
            # threshold up and accuracy climbs while recall collapses.
            "accuracy": round((tp + tn) / len(y), 4),
            "pct_flagged": round(float(pred.mean()) * 100, 2),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "expected_cost_per_row": round(
                (config.COST_RATIO_FN_FP * fn + fp) / len(y), 4),
            "is_operating_point": bool(np.isclose(t, operating)),
        })

    best_f1 = max(rows, key=lambda r: r["f1"])
    return {
        "operating_threshold": round(operating, 4),
        "cost_ratio_fn_fp": config.COST_RATIO_FN_FP,
        "cost_ratio_basis": (
            "§8.4 default. It is an ASSUMPTION and a consequential one: at this "
            "prevalence a 10:1 ratio flags over half the portfolio. A 3:1 ratio "
            "maximises F1 and is the recommended triage operating point. State "
            "the deployed ratio wherever a flag is shown."),
        "max_f1_point": best_f1,
        "sweep": rows,
        "n_test_rows": int(len(te)),
        "n_positives": int(y.sum()),
    }
