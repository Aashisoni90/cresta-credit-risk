"""
metrics.py
===========
Shared evaluation for every Track A model, so the Altman benchmark, the
logistic baseline and XGBoost are scored by identical code (PRD §8.6).

Implements the §8.6 metric set, plus the §8.4 cost-weighted threshold.

TWO THINGS THIS DELIBERATELY REFUSES TO DO
-------------------------------------------
1. It never selects a threshold on the test set. `choose_threshold()` is meant
   to be called on VALIDATION predictions; the chosen value is then applied
   unchanged to test. Tuning a threshold on test is a silent way of reporting
   an optimistically biased recall.
2. Accuracy is computed but is NOT a headline metric (§8.6) — under 21-26%
   prevalence it is misleading, and it is reported only for completeness.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, confusion_matrix)

# §8.4: a false negative (missing a distress event) costs the lender potentially
# the whole exposure; a false positive costs analyst time. Default 10:1.
DEFAULT_COST_RATIO = 10.0


def ks_statistic(y_true, score) -> float:
    """Kolmogorov-Smirnov: max separation between cumulative good/bad."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def choose_threshold(y_true, proba, cost_ratio: float = DEFAULT_COST_RATIO):
    """Threshold minimising expected cost, with FN `cost_ratio`x worse than FP.

    MUST be called on validation predictions, never on test (§8.5).
    Returns (threshold, expected_cost_per_row).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    best_t, best_c = 0.5, np.inf
    for t in grid:
        pred = (p >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = (cost_ratio * fn + fp) / len(y)
        if cost < best_c:
            best_t, best_c = float(t), float(cost)
    return best_t, best_c


def evaluate(y_true, score, threshold=None, proba=None, label="", n_total=None):
    """Full §8.6 metric set.

    `score`  — any monotone risk score; used for ranking metrics (AUC/PR/KS).
    `proba`  — calibrated probability, if the model produces one. Brier and the
               thresholded metrics use it; omitted for a raw score like Z''.
    `n_total`— rows the model COULD NOT score, for honest coverage reporting.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    out = {
        "model": label,
        "n_scored": len(y),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "auc_roc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "ks": ks_statistic(y, s),
    }
    if n_total is not None:
        out["coverage"] = len(y) / n_total

    if proba is not None:
        p = np.asarray(proba, dtype=float)
        out["brier"] = float(brier_score_loss(y, p))
        if threshold is not None:
            pred = (p >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            out.update({
                "threshold": float(threshold),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
                "accuracy": (tp + tn) / len(y),   # NOT a headline metric (§8.6)
            })
            r, pr = out["recall"], out["precision"]
            out["f1"] = 2 * r * pr / (r + pr) if (r and pr and (r + pr)) else np.nan
    return out


def delong_auc_variance(y_true, s1, s2):
    """DeLong covariance for two correlated ROC curves on the same sample.

    Returns (auc1, auc2, var_of_difference). PRD §8.7 names DeLong's test as
    the formal standard for comparing correlated ROCs.
    """
    y = np.asarray(y_true).astype(int)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan

    def structural(s):
        x, yv = np.asarray(s, float)[pos], np.asarray(s, float)[neg]
        # midrank-based placement values
        v01 = np.array([(np.sum(yv < xi) + 0.5 * np.sum(yv == xi)) / n for xi in x])
        v10 = np.array([(np.sum(x > yj) + 0.5 * np.sum(x == yj)) / m for yj in yv])
        return v01, v10, float(v01.mean())

    a01, a10, auc1 = structural(s1)
    b01, b10, auc2 = structural(s2)
    s01 = np.cov(np.vstack([a01, b01]))
    s10 = np.cov(np.vstack([a10, b10]))
    S = s01 / m + s10 / n
    var_diff = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return auc1, auc2, var_diff


def delong_test(y_true, s1, s2):
    """Two-sided DeLong test. Returns dict with ΔAUC, z, p, and 95% CI."""
    from scipy import stats
    a1, a2, var = delong_auc_variance(y_true, s1, s2)
    if not np.isfinite(var) or var <= 0:
        return {"auc_1": a1, "auc_2": a2, "delta_auc": a2 - a1,
                "z": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    d = a2 - a1
    se = np.sqrt(var)
    z = d / se
    return {"auc_1": a1, "auc_2": a2, "delta_auc": d, "se": se, "z": z,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
            "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se}


def bootstrap_auc_diff(y_true, s1, s2, n_boot=2000, seed=42):
    """Bootstrapped 95% CI on ΔAUC (§8.7's named alternative to DeLong)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        diffs.append(roc_auc_score(y[b], s2[b]) - roc_auc_score(y[b], s1[b]))
    diffs = np.array(diffs)
    return {"delta_auc_mean": float(diffs.mean()),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "n_boot": len(diffs)}
