"""
train_model_a.py — Phase 3, Track A
====================================
Trains and compares the §8.1 model hierarchy on the real panel:

    Tier 1  Benchmark          Altman Z'' with a fitted threshold
    Tier 2  Statistical base   L2 logistic regression, balanced class weight
    Tier 3  Primary model      XGBoost            [--xgb]

Reporting all three together is the result (§8.1). Reporting only the winner is
a methodological failure.

INVARIANTS ENFORCED IN CODE, NOT BY CONVENTION
-----------------------------------------------
* OUT-OF-TIME SPLIT ONLY (§8.5, invariant #2). The `split` column is read from
  the matrix; this module never resamples it. `_assert_out_of_time()` verifies
  train < validation < test with no year overlap and fails otherwise.
* PREPROCESSING FITTED ON TRAIN ONLY (§7.4, invariant #3). The imputer and
  scaler are fit on the training window and applied unchanged downstream.
  `_assert_fitted_on_train_only()` re-fits on train alone and asserts the
  learned statistics are identical — this is the leakage test §15 calls "the
  most valuable test in the suite".
* NEVER ZERO-FILL (§5.5, invariant #1). Missing ratios are median-imputed with
  TRAIN medians and paired with an explicit `*_was_missing` indicator, so the
  model can learn from the missingness instead of being told the ratio was
  zero. XGBoost receives raw NaN and uses its native handling.
* THRESHOLD CHOSEN ON VALIDATION (§8.4). Test is scored once, at the end.

THE ALTMAN BENCHMARK USES CANONICAL Z'', NOT THE FEATURE VARIANT
-----------------------------------------------------------------
`altman_z_double_prime` is read from the scored panel, not the training matrix,
because it was deliberately excluded from the features as circular (§17.1). The
benchmark must be the classical model as published; the feature set uses the
3-term `altman_z_ex_x4`. Lower Z'' means higher risk, so the benchmark score is
negated for ranking metrics.

USAGE
    python src/train_model_a.py            # Tiers 1-2
    python src/train_model_a.py --xgb      # add Tier 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from metrics import (evaluate, choose_threshold, delong_test, bootstrap_auc_diff,
                     DEFAULT_COST_RATIO)
from fallback_label import build_fallback_labels

META = {"ticker", "company_name", "fiscal_year", "fiscal_year_num",
        "label_binary_next", "distress_label_next", "qc_below_coverage", "split"}
LABEL = "label_binary_next"
SEED = 42


# ── split integrity ───────────────────────────────────────────────────
def _assert_out_of_time(df):
    yrs = {s: (df.loc[df.split == s, "fiscal_year_num"].min(),
               df.loc[df.split == s, "fiscal_year_num"].max())
           for s in ["train", "validation", "test"] if (df.split == s).any()}
    if not {"train", "validation", "test"} <= set(yrs):
        raise ValueError(f"missing split(s); found {sorted(yrs)}")
    if not (yrs["train"][1] < yrs["validation"][0] <= yrs["validation"][1] < yrs["test"][0]):
        raise ValueError(
            "SPLIT IS NOT OUT-OF-TIME — train/validation/test years overlap or are "
            f"out of order: {yrs}. A random or overlapping split leaks the future "
            "into training (§8.5)."
        )
    return yrs


# ── preprocessing, fitted on train only ───────────────────────────────
def fit_preprocessor(X_train):
    """Learn TRAIN medians. Returns the stats dict that is applied unchanged."""
    return {"medians": X_train.median(numeric_only=True),
            "columns": list(X_train.columns)}


def apply_preprocessor(X, stats, add_indicators=True):
    """Median-impute + optional missingness indicators. Never zero-fills."""
    X = X[stats["columns"]].copy()
    out = X.copy()
    if add_indicators:
        for c in stats["columns"]:
            if X[c].isna().any():
                out[f"{c}__was_missing"] = X[c].isna().astype(int)
    out[stats["columns"]] = X.fillna(stats["medians"])
    return out


def _assert_fitted_on_train_only(X_train, X_full, stats):
    """THE LEAKAGE TEST (§15). Re-fit on train alone; statistics must match."""
    refit = fit_preprocessor(X_train)
    a, b = refit["medians"], stats["medians"]
    common = a.index.intersection(b.index)
    if not np.allclose(a[common].values, b[common].values, equal_nan=True):
        raise AssertionError("preprocessing statistics differ when re-fit on train "
                             "alone — they were contaminated by validation/test data.")
    full = X_full.median(numeric_only=True)
    drift = (a[common] - full[common]).abs().sum()
    return float(drift)


def _standardise(df_tr, df_other_list):
    mu, sd = df_tr.mean(), df_tr.std().replace(0, 1.0)
    return [(d - mu) / sd for d in [df_tr] + df_other_list]


# ── main ──────────────────────────────────────────────────────────────
def run(matrix_path, panel_path, use_xgb=False, cost_ratio=DEFAULT_COST_RATIO,
        out_dir="reports/phase3"):
    m = pd.read_csv(matrix_path)
    before = len(m)
    m = m[m.get("qc_below_coverage") != True].copy()
    print(f"[data] {before:,} rows -> {len(m):,} after the coverage gate")

    yrs = _assert_out_of_time(m)
    print(f"[split] out-of-time verified: {yrs}")

    # canonical Z'' for the benchmark only (excluded from features as circular)
    panel = pd.read_csv(panel_path, usecols=["ticker", "fiscal_year_num",
                                             "altman_z_double_prime"])
    m = m.merge(panel, on=["ticker", "fiscal_year_num"], how="left")

    # CONTEMPORANEOUS label — benchmark and stratifier only, NEVER a feature.
    # The §7.5 label persists 90.7% year-over-year, so "the state simply
    # continues" is a strong naive predictor (AUC 0.858) that any honest
    # comparison must beat. Altman Z'' does NOT beat it. Omitting this baseline
    # would make every model look far better than it is. See §18.2.
    _pan = pd.read_csv(panel_path)
    if "company_name" not in _pan.columns:
        _pan["company_name"] = ""
    _lab = build_fallback_labels(_pan, verbose=False)[["ticker", "fiscal_year_num",
                                                       "label_binary"]]
    m = m.merge(_lab.rename(columns={"label_binary": "label_now"}),
                on=["ticker", "fiscal_year_num"], how="left")

    feats = [c for c in m.columns
             if c not in META and c not in ("altman_z_double_prime", "label_now")]
    print(f"[data] {len(feats)} features")

    tr, va, te = m[m.split == "train"], m[m.split == "validation"], m[m.split == "test"]
    for nm, d in [("train", tr), ("validation", va), ("test", te)]:
        print(f"   {nm:<11s} {len(d):>6,} rows  {int(d[LABEL].sum()):>5,} pos "
              f"({d[LABEL].mean()*100:.1f}%)")

    results, store = [], {}

    # ── Tier 1 — Altman Z'' benchmark ─────────────────────────────────
    z = te["altman_z_double_prime"]
    ok = z.notna()
    zv, okv = va["altman_z_double_prime"], va["altman_z_double_prime"].notna()
    # Fitted threshold: pick the Z'' cut minimising §8.4 cost on VALIDATION.
    # The grid is taken over a PLAUSIBLE Z'' RANGE, not over raw quantiles. Raw
    # Z'' has a degenerate tail (§15.3 — debt-free shells push the validation
    # 99th percentile to ~260), and a quantile grid over it returns a
    # meaningless cut of "Z'' <= 260", i.e. flag everything. The benchmark cut
    # must be an interpretable score level, so the search is bounded to the
    # range Altman's own zones occupy.
    cand = np.linspace(-10.0, 10.0, 401)
    best_z, best_c = None, np.inf
    yv = va.loc[okv, LABEL].values
    for c in cand:
        pred = (zv[okv].values <= c).astype(int)
        fn = int(((pred == 0) & (yv == 1)).sum()); fp = int(((pred == 1) & (yv == 0)).sum())
        cost = (cost_ratio * fn + fp) / len(yv)
        if cost < best_c:
            best_z, best_c = float(c), cost
    print(f"\n[tier1] Altman Z'' threshold fitted on VALIDATION: Z'' <= {best_z:.4f}")
    r1 = evaluate(te.loc[ok, LABEL], -z[ok], label="Altman Z'' (benchmark)",
                  n_total=len(te))
    pred = (z[ok].values <= best_z).astype(int)
    yt = te.loc[ok, LABEL].values
    tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
    r1.update({"threshold": best_z, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "recall": tp / (tp + fn) if tp + fn else np.nan,
               "precision": tp / (tp + fp) if tp + fp else np.nan,
               "accuracy": (tp + tn) / len(yt)})
    results.append(r1)
    store["altman"] = (te.index[ok], -z[ok].values)

    # Altman at its CONVENTIONAL distress cutoff (Z'' < 1.1), reported alongside
    # the fitted cut because it is the literature-standard, interpretable zone
    # boundary and does not depend on our cost assumption.
    r1b = dict(r1)
    pred = (z[ok].values <= 1.1).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
    r1b.update({"model": "Altman Z'' @ conventional 1.1 cutoff", "threshold": 1.1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "accuracy": (tp + tn) / len(yt)})
    results.append(r1b)

    # ── Tier 0 — persistence ("the condition continues") ──────────────
    pn = te["label_now"]
    okp = pn.notna()
    if okp.any() and te.loc[okp, LABEL].nunique() > 1:
        r0 = evaluate(te.loc[okp, LABEL], pn[okp].astype(float),
                      label="Persistence (label at t)  [naive baseline]",
                      n_total=len(te))
        results.insert(0, r0)
        store["persistence"] = (te.index[okp], pn[okp].astype(float).values)

    # ── Tier 2 — L2 logistic regression ───────────────────────────────
    stats = fit_preprocessor(tr[feats])
    drift = _assert_fitted_on_train_only(tr[feats], m[feats], stats)
    print(f"[leakage] preprocessing re-fit on train alone: statistics identical ✅ "
          f"(train-vs-full median drift {drift:.4f} — non-zero is expected and proves "
          f"train stats are NOT full-sample stats)")

    Xtr, Xva, Xte = (apply_preprocessor(d[feats], stats) for d in (tr, va, te))
    cols = Xtr.columns
    Xva, Xte = Xva.reindex(columns=cols, fill_value=0), Xte.reindex(columns=cols, fill_value=0)
    Str, Sva, Ste = _standardise(Xtr, [Xva, Xte])

    lr = LogisticRegression(class_weight="balanced", max_iter=2000,
                            random_state=SEED)   # L2 is the default; naming it
                                                 # explicitly is deprecated in
                                                 # sklearn 1.8+ and removed in 1.10
    lr.fit(Str, tr[LABEL])
    pv = lr.predict_proba(Sva)[:, 1]
    t_lr, _ = choose_threshold(va[LABEL], pv, cost_ratio)
    pt = lr.predict_proba(Ste)[:, 1]
    print(f"[tier2] logistic threshold chosen on VALIDATION: {t_lr:.3f}")
    results.append(evaluate(te[LABEL], pt, threshold=t_lr, proba=pt,
                            label="Logistic regression (L2, balanced)", n_total=len(te)))
    store["logistic"] = (te.index, pt)

    # ── Tier 3 — XGBoost ──────────────────────────────────────────────
    if use_xgb:
        from xgboost import XGBClassifier
        spw = float((tr[LABEL] == 0).sum() / max((tr[LABEL] == 1).sum(), 1))
        print(f"[tier3] scale_pos_weight (train only) = {spw:.3f}")
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            scale_pos_weight=spw, eval_metric="aucpr", random_state=SEED, n_jobs=4,
        )
        xgb.fit(tr[feats], tr[LABEL])        # raw NaN — native handling, no imputation
        pv = xgb.predict_proba(va[feats])[:, 1]
        t_x, _ = choose_threshold(va[LABEL], pv, cost_ratio)
        px = xgb.predict_proba(te[feats])[:, 1]
        print(f"[tier3] XGBoost threshold chosen on VALIDATION: {t_x:.3f}")
        results.append(evaluate(te[LABEL], px, threshold=t_x, proba=px,
                                label="XGBoost", n_total=len(te)))
        store["xgboost"] = (te.index, px)
        store["_xgb_model"] = xgb
        store["_features"] = feats

    # ── §8.4 cost-ratio sensitivity ───────────────────────────────────
    # "The chosen ratio must be stated as an assumption and its sensitivity
    # reported." At this label's 21-26% prevalence a 10:1 ratio drives the
    # threshold toward flagging everything, so the ratio is material.
    print("\n" + "-" * 100)
    print(f"COST-RATIO SENSITIVITY (§8.4) — logistic model, threshold re-chosen on "
          f"validation at each ratio")
    print("-" * 100)
    print(f"  {'FN:FP':>7s} {'thresh':>8s} {'recall':>8s} {'precision':>10s} "
          f"{'F1':>8s} {'flagged %':>10s}")
    pv_lr = lr.predict_proba(Sva)[:, 1]
    for cr in [1.0, 3.0, 5.0, 10.0, 20.0]:
        t, _ = choose_threshold(va[LABEL], pv_lr, cr)
        pr_ = (pt >= t).astype(int); yte = te[LABEL].values
        tp = int(((pr_ == 1) & (yte == 1)).sum()); fp = int(((pr_ == 1) & (yte == 0)).sum())
        fn = int(((pr_ == 0) & (yte == 1)).sum())
        rec = tp / (tp + fn) if tp + fn else np.nan
        pre = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * rec * pre / (rec + pre) if (rec and pre) else np.nan
        mark = "  <- PRD default" if cr == 10.0 else ""
        print(f"  {cr:>6.0f}:1 {t:>8.3f} {rec:>8.3f} {pre:>10.3f} {f1:>8.3f} "
              f"{pr_.mean()*100:>9.1f}%{mark}")

    # ── report ────────────────────────────────────────────────────────
    res = pd.DataFrame(results)
    show = [c for c in ["model", "n_scored", "coverage", "auc_roc", "pr_auc", "ks",
                        "brier", "threshold", "recall", "precision", "f1", "accuracy"]
            if c in res.columns]
    print("\n" + "=" * 100)
    print("TEST-SET RESULTS (out-of-time, scored once)")
    print("=" * 100)
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # significance vs the benchmark, on the rows both models scored
    print("\n" + "-" * 100)
    print("SIGNIFICANCE vs Altman Z'' benchmark (§8.7)")
    print("-" * 100)
    aidx, ascore = store["altman"]
    for name in [k for k in ("logistic", "xgboost") if k in store]:
        idx, sc = store[name]
        common = aidx.intersection(idx)
        y = m.loc[common, LABEL].values
        s1 = pd.Series(ascore, index=aidx).loc[common].values
        s2 = pd.Series(sc, index=idx).loc[common].values
        d = delong_test(y, s1, s2)
        b = bootstrap_auc_diff(y, s1, s2, n_boot=2000, seed=SEED)
        sig = "SIGNIFICANT" if (d["p_value"] == d["p_value"] and d["p_value"] < 0.05) else "not significant"
        print(f"  {name:<10s} n={len(common):,}  ΔAUC={d['delta_auc']:+.4f}  "
              f"DeLong z={d['z']:+.2f} p={d['p_value']:.3e}  "
              f"95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  -> {sig}")
        print(f"             bootstrap ΔAUC={b['delta_auc_mean']:+.4f} "
              f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] ({b['n_boot']} resamples)")

    # ── stratified evaluation — the honest early-warning number ───────
    # Pooled AUC is inflated by label persistence. The subset that is HEALTHY at
    # FY(t) is the actual early-warning task: persistence scores 0.5 there by
    # construction, so any lift is genuine skill.
    print("\n" + "-" * 100)
    print("STRATIFIED BY CURRENT STATE — pooled AUC is inflated by persistence (§18.2)")
    print("-" * 100)
    strata = [("ALL test rows", te.index),
              ("already distressed at t", te.index[te.label_now == 1]),
              ("HEALTHY at t -> early warning", te.index[te.label_now == 0])]
    for nm, idx in strata:
        row = f"  {nm:<32s} n={len(idx):>5,}"
        y = m.loc[idx, LABEL]
        if y.nunique() < 2:
            print(row + "   (single class)"); continue
        row += f"  pos={int(y.sum()):>4,} ({y.mean()*100:4.1f}%)"
        for name in [k for k in ("altman", "logistic", "xgboost") if k in store]:
            sidx, sc = store[name]
            com = idx.intersection(sidx)
            yy = m.loc[com, LABEL]
            if yy.nunique() < 2:
                continue
            ss = pd.Series(sc, index=sidx).loc[com].values
            from sklearn.metrics import roc_auc_score as _auc
            row += f"   {name}={_auc(yy, ss):.4f}"
        print(row)

    # Artefacts for the §8.8 explainability layer (explain_model_a.py). Exposed
    # rather than re-derived so SHAP explains THE model that produced the table
    # above — a second fit, even with the same seed, is not the same object.
    store["_frames"] = {"train": tr, "validation": va, "test": te}
    store["_matrix"] = m

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    res.to_csv(f"{out_dir}/model_comparison.csv", index=False)
    print(f"\nwrote {out_dir}/model_comparison.csv")
    return res, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/processed/training_matrix.csv")
    ap.add_argument("--panel", default="data/processed/bloomberg_panel_scored_gated.csv")
    ap.add_argument("--xgb", action="store_true")
    ap.add_argument("--cost-ratio", type=float, default=DEFAULT_COST_RATIO)
    a = ap.parse_args()
    run(a.matrix, a.panel, use_xgb=a.xgb, cost_ratio=a.cost_ratio)


if __name__ == "__main__":
    main()
