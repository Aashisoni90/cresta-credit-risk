"""
rolling_origin.py — Phase 3, §8.5 supplementary validation
===========================================================
Walk-forward (rolling-origin) validation. The §8.5 primary protocol gives ONE
test estimate from ONE arbitrary split date; this gives a DISTRIBUTION across
regimes, which is the difference between "the model scores 0.96" and "the model
scores 0.96 ± 0.01 across five independent origins".

FOLD DESIGN — and where it extends the PRD sketch
--------------------------------------------------
§8.5 sketches four folds of the form `train 2015-2019 -> test 2020`. That sketch
has no validation year, yet a cost-optimal threshold (§8.4) has to be chosen
somewhere, and choosing it on the fold's own test set is leakage. Each fold here
therefore holds out the LAST training year as that fold's validation:

    Fold 1:  train 2014-2017   val 2018   test 2019
    Fold 2:  train 2014-2018   val 2019   test 2020
    Fold 3:  train 2014-2019   val 2020   test 2021
    Fold 4:  train 2014-2020   val 2021   test 2022
    Fold 5:  train 2014-2021   val 2022   test 2023

`fiscal_year_num` is the Indian FY START year (§14), so fold 5's test year 2023
is FY2023-24. Ranking metrics (AUC, PR-AUC, KS) do not depend on the threshold,
so they are unaffected by this addition; recall/precision do, and would be
unreportable without it.

WHAT IS RE-FITTED PER FOLD — ALL OF IT
---------------------------------------
Preprocessing statistics, class weights, and both models are re-fitted inside
every fold on that fold's training window alone (§7.4, invariant #3). Nothing is
carried across folds. A single preprocessor fitted once on the whole panel and
reused would leak later years into earlier folds and is the most common way
walk-forward validation is quietly invalidated.

THE PERSISTENCE BASELINE TRAVELS WITH THE MODELS
-------------------------------------------------
§18.2: the §7.5 label persists 90.7% year-over-year, so pooled AUC is largely
"the condition continues". Every fold therefore reports persistence and the
HEALTHY-AT-t early-warning subset alongside the pooled numbers. A fold table
showing only pooled AUC would restate that inflation five times over.

USAGE
    PYTHONPATH=src .venv/bin/python src/rolling_origin.py --xgb
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from metrics import evaluate, choose_threshold, DEFAULT_COST_RATIO
from fallback_label import build_fallback_labels
from train_model_a import (META, LABEL, SEED, fit_preprocessor, apply_preprocessor,
                           _assert_fitted_on_train_only, _standardise)


def make_folds(years, min_train=4):
    """Expanding-window folds: [y0..Y-2] train, Y-1 validation, Y test."""
    years = sorted(years)
    out = []
    for Y in years:
        tr = [y for y in years if y <= Y - 2]
        if len(tr) >= min_train and (Y - 1) in years:
            out.append({"fold": len(out) + 1, "train": tr, "val": Y - 1, "test": Y})
    return out


def _early_warning_auc(frame, score):
    """AUC on rows HEALTHY at t. Persistence is 0.5 here by construction, so a
    lift is genuine skill rather than restating the current state (§18.7)."""
    mask = frame["label_now"] == 0
    y = frame.loc[mask, LABEL]
    if y.nunique() < 2:
        return np.nan, int(mask.sum()), np.nan
    return (float(roc_auc_score(y, pd.Series(score, index=frame.index)[mask])),
            int(mask.sum()), float(y.mean()))


def run(matrix_path, panel_path, use_xgb=False, cost_ratio=DEFAULT_COST_RATIO,
        out_dir="reports/phase3"):
    m = pd.read_csv(matrix_path)
    m = m[m.get("qc_below_coverage") != True].copy()

    panel = pd.read_csv(panel_path)
    if "company_name" not in panel.columns:
        panel["company_name"] = ""
    m = m.merge(panel[["ticker", "fiscal_year_num", "altman_z_double_prime"]],
                on=["ticker", "fiscal_year_num"], how="left")
    lab = build_fallback_labels(panel, verbose=False)[["ticker", "fiscal_year_num",
                                                       "label_binary"]]
    m = m.merge(lab.rename(columns={"label_binary": "label_now"}),
                on=["ticker", "fiscal_year_num"], how="left")

    feats = [c for c in m.columns
             if c not in META and c not in ("altman_z_double_prime", "label_now")]
    folds = make_folds(m["fiscal_year_num"].unique())
    print(f"[rolling] {len(feats)} features, {len(folds)} folds over "
          f"FY{min(m.fiscal_year_num)}-FY{max(m.fiscal_year_num)} "
          f"(fiscal_year_num = Indian FY START year)")

    rows = []
    for f in folds:
        tr = m[m.fiscal_year_num.isin(f["train"])]
        va = m[m.fiscal_year_num == f["val"]]
        te = m[m.fiscal_year_num == f["test"]]
        if te[LABEL].nunique() < 2 or va[LABEL].nunique() < 2:
            print(f"  fold {f['fold']}: skipped (single-class val/test)")
            continue

        # Everything below is fitted inside the fold, on the fold's train only.
        stats = fit_preprocessor(tr[feats])
        _assert_fitted_on_train_only(tr[feats], tr[feats], stats)
        Xtr, Xva, Xte = (apply_preprocessor(d[feats], stats) for d in (tr, va, te))
        cols = Xtr.columns
        Xva = Xva.reindex(columns=cols, fill_value=0)
        Xte = Xte.reindex(columns=cols, fill_value=0)
        Str, Sva, Ste = _standardise(Xtr, [Xva, Xte])

        base = {"fold": f["fold"],
                "train_years": f"{min(f['train'])}-{max(f['train'])}",
                "val_year": f["val"], "test_year": f["test"],
                "n_train": len(tr), "n_test": len(te),
                "test_prevalence": float(te[LABEL].mean())}

        # Tier 0 — persistence
        ok = te["label_now"].notna()
        if ok.any() and te.loc[ok, LABEL].nunique() > 1:
            r = evaluate(te.loc[ok, LABEL], te.loc[ok, "label_now"].astype(float),
                         label="persistence", n_total=len(te))
            rows.append({**base, **r})

        # Tier 1 — Altman Z'' (bounded threshold grid, §18.3)
        z, okz = te["altman_z_double_prime"], te["altman_z_double_prime"].notna()
        zv, okv = va["altman_z_double_prime"], va["altman_z_double_prime"].notna()
        if okz.any() and te.loc[okz, LABEL].nunique() > 1 and okv.any():
            yv = va.loc[okv, LABEL].values
            best_z, best_c = None, np.inf
            for c in np.linspace(-10.0, 10.0, 401):
                pred = (zv[okv].values <= c).astype(int)
                cost = (cost_ratio * ((pred == 0) & (yv == 1)).sum()
                        + ((pred == 1) & (yv == 0)).sum()) / len(yv)
                if cost < best_c:
                    best_z, best_c = float(c), cost
            r = evaluate(te.loc[okz, LABEL], -z[okz], label="altman", n_total=len(te))
            a, n_, p_ = _early_warning_auc(te, -z.fillna(z.median()))
            rows.append({**base, **r, "threshold": best_z,
                         "auc_early_warning": a, "n_early_warning": n_,
                         "prev_early_warning": p_})

        # Tier 2 — logistic
        lr = LogisticRegression(class_weight="balanced",   # L2 is the default
                                max_iter=2000, random_state=SEED)
        lr.fit(Str, tr[LABEL])
        t_lr, _ = choose_threshold(va[LABEL], lr.predict_proba(Sva)[:, 1], cost_ratio)
        pt = lr.predict_proba(Ste)[:, 1]
        r = evaluate(te[LABEL], pt, threshold=t_lr, proba=pt, label="logistic",
                     n_total=len(te))
        a, n_, p_ = _early_warning_auc(te, pt)
        rows.append({**base, **r, "auc_early_warning": a,
                     "n_early_warning": n_, "prev_early_warning": p_})

        # Tier 3 — XGBoost
        if use_xgb:
            from xgboost import XGBClassifier
            spw = float((tr[LABEL] == 0).sum() / max((tr[LABEL] == 1).sum(), 1))
            x = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              scale_pos_weight=spw, eval_metric="aucpr",
                              random_state=SEED, n_jobs=4)
            x.fit(tr[feats], tr[LABEL])      # raw NaN, native handling
            t_x, _ = choose_threshold(va[LABEL], x.predict_proba(va[feats])[:, 1],
                                      cost_ratio)
            px = x.predict_proba(te[feats])[:, 1]
            r = evaluate(te[LABEL], px, threshold=t_x, proba=px, label="xgboost",
                         n_total=len(te))
            a, n_, p_ = _early_warning_auc(te, px)
            rows.append({**base, **r, "auc_early_warning": a,
                         "n_early_warning": n_, "prev_early_warning": p_})

        print(f"  fold {f['fold']}: train {base['train_years']} "
              f"({len(tr):,}) -> val {f['val']} -> test {f['test']} "
              f"({len(te):,}, {base['test_prevalence']*100:.1f}% pos)  ✓")

    res = pd.DataFrame(rows)

    print("\n" + "=" * 112)
    print("ROLLING-ORIGIN FOLDS (§8.5 supplementary) — pooled AUC, and the "
          "early-warning AUC that is not inflated by persistence")
    print("=" * 112)
    piv = res.pivot_table(index=["fold", "test_year"], columns="model",
                          values="auc_roc")
    pive = res.pivot_table(index=["fold", "test_year"], columns="model",
                           values="auc_early_warning")
    pive.columns = [f"{c}_EW" for c in pive.columns]
    comb = piv.join(pive)
    order = [c for c in ["persistence", "altman", "logistic", "xgboost",
                         "altman_EW", "logistic_EW", "xgboost_EW"] if c in comb]
    print(comb[order].to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\nSTABILITY — mean ± sd across folds (the point of this exercise)")
    print("-" * 112)
    print(f"  {'model':<14s} {'pooled AUC':>22s} {'early-warning AUC':>24s} "
          f"{'PR-AUC':>20s} {'folds':>6s}")
    for mdl in order:
        if mdl.endswith("_EW"):
            continue
        s = comb[mdl].dropna()
        e = comb.get(f"{mdl}_EW", pd.Series(dtype=float)).dropna()
        pr = res.loc[res.model == mdl, "pr_auc"].dropna()
        ew = f"{e.mean():.4f} ± {e.std():.4f}" if len(e) else "n/a (0.5 by constr.)"
        print(f"  {mdl:<14s} {s.mean():>10.4f} ± {s.std():.4f}   {ew:>22s}   "
              f"{pr.mean():>10.4f} ± {pr.std():.4f} {len(s):>6d}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Stability figure. Pooled solid, early-warning dashed — plotted on one
    # axis because the gap between the two IS the persistence inflation, and
    # separating them into two charts lets a reader miss it.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = comb.index.get_level_values("test_year")
    for mdl in [c for c in order if not c.endswith("_EW")]:
        ax.plot(xs, comb[mdl], marker="o", label=f"{mdl} (pooled)")
        if f"{mdl}_EW" in comb:
            ax.plot(xs, comb[f"{mdl}_EW"], marker="s", linestyle="--", alpha=0.65,
                    label=f"{mdl} (early warning)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("test fiscal year (FY start)"); ax.set_ylabel("AUC-ROC")
    ax.set_title("Rolling-origin validation — stability across regimes (§8.5)")
    ax.set_xticks(list(xs)); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/figures/rolling_origin_auc.png", dpi=160)
    plt.close(fig)

    res.to_csv(f"{out_dir}/rolling_origin_folds.csv", index=False)
    comb[order].to_csv(f"{out_dir}/rolling_origin_summary.csv")
    print(f"\nwrote {out_dir}/rolling_origin_folds.csv, rolling_origin_summary.csv")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/processed/training_matrix.csv")
    ap.add_argument("--panel", default="data/processed/bloomberg_panel_scored_gated.csv")
    ap.add_argument("--xgb", action="store_true")
    ap.add_argument("--cost-ratio", type=float, default=DEFAULT_COST_RATIO)
    ap.add_argument("--out", default="reports/phase3")
    a = ap.parse_args()
    run(a.matrix, a.panel, use_xgb=a.xgb, cost_ratio=a.cost_ratio, out_dir=a.out)


if __name__ == "__main__":
    main()
