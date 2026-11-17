"""
explain_model_a.py — Phase 3, §8.8 Explainability Layer
========================================================
Produces the four levels of explanation the PRD specifies for Model A:

    Global        mean-|SHAP| ranking + beeswarm over the test set
    Block-level   SHAP aggregated by §6 feature block (leverage, liquidity,
                  profitability, debt-service, Z-score, temporal, controls)
    Local         per-prediction waterfall: base rate -> contributions -> p
    Counterfactual nearest-neighbour contrast against a similar HEALTHY firm

THE CENTRAL CAVEAT, STATED IN CODE AS WELL AS IN THE PAPER
-----------------------------------------------------------
SHAP explains THE MODEL, NOT THE WORLD (§8.8). A large mean-|SHAP| says the
model leans on that feature; it says nothing about causation. Every artefact
this module writes carries that sentence, so it cannot be detached from the
numbers downstream (dashboard copy, report figures).

WHY THIS IMPORTS run() INSTEAD OF RE-FITTING
---------------------------------------------
The explained model must be the one that produced reports/phase3/
model_comparison.csv. Re-fitting "with the same seed" gives a different object
the moment a column order, a pandas version or a dtype shifts, and an
explanation of a model you did not report is worse than no explanation.
train_model_a.run() returns its fitted estimator in `store`; we explain that.

TreeSHAP is exact for tree ensembles, so no background-sample approximation is
involved and the additivity check below must hold to floating-point tolerance.
It is asserted, not assumed.

USAGE
    PYTHONPATH=src .venv/bin/python src/explain_model_a.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_engineering import FEATURE_GROUPS
from train_model_a import run, LABEL

CAVEAT = ("SHAP explains the MODEL, not the world. A high attribution means the "
          "model relies on that feature; it does not establish causation (PRD §8.8).")

# Features that post-date the FEATURE_GROUPS catalogue, or are diagnostic flags
# emitted by the Altman scorer rather than by a block builder. Mapped explicitly
# so nothing silently lands in "unassigned" and quietly distorts the block chart.
EXTRA_BLOCK_MAP = {
    "altman_flag_x4_uncomputable": "block5_zscore",
    "altman_flag_check_scaling": "block5_zscore",
}

# DELIBERATE OVERRIDE, NOT A CATALOGUE EDIT.
# FEATURE_GROUPS files `ebit_to_assets` under block4_debt_service, where it
# belongs as the numerator of interest coverage. But the interest-coverage
# features never reached this matrix (the expanded pull is outstanding), so
# block4 contains that one feature and nothing else. Aggregating it as-is makes
# the committee chart say "debt servicing drives 34% of the model" when the
# model sees no debt-servicing variable at all. EBIT/Total Assets is a return
# measure; for the attribution view it is reported as profitability and the
# substitution is printed with the table so it is never silently applied.
BLOCK_ATTRIBUTION_OVERRIDE = {"ebit_to_assets": "block3_profitability"}


def block_of(feature: str) -> str:
    """Map a feature (or its *__was_missing* partner) to its §6 block."""
    base = feature.replace("__was_missing", "")
    if base in BLOCK_ATTRIBUTION_OVERRIDE:
        return BLOCK_ATTRIBUTION_OVERRIDE[base]
    for blk, feats in FEATURE_GROUPS.items():
        if base in feats:
            return blk
    return EXTRA_BLOCK_MAP.get(base, "unassigned")


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def explain(store, out_dir="reports/phase3", n_local=5, seed=42):
    import shap

    xgb = store["_xgb_model"]
    feats = store["_features"]
    te = store["_frames"]["test"]
    X = te[feats]
    y = te[LABEL]
    _, p_test = store["xgboost"]
    p_test = pd.Series(p_test, index=te.index)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)

    # ── TreeSHAP (exact for tree ensembles; margin/log-odds space) ─────
    expl = shap.TreeExplainer(xgb)
    sv = expl(X)
    S = sv.values
    base = float(np.atleast_1d(sv.base_values)[0])

    # ADDITIVITY: base + sum(contributions) must reproduce the model margin.
    # If this fails the attributions are not a decomposition of the prediction
    # and nothing below them is trustworthy, so it halts rather than warns.
    recon = base + S.sum(axis=1)
    margin = _logit(p_test.values)
    max_err = float(np.abs(recon - margin).max())
    if max_err > 1e-3:
        raise AssertionError(
            f"SHAP additivity violated: max |base+sum(shap) - logit(p)| = {max_err:.2e}. "
            "The attributions do not decompose the reported predictions.")
    print(f"[shap] additivity verified: max reconstruction error {max_err:.2e} "
          f"(log-odds), base value {base:+.4f} = {1/(1+np.exp(-base))*100:.2f}% base rate")

    # ── LEVEL 1 — global ranking ──────────────────────────────────────
    mean_abs = pd.Series(np.abs(S).mean(axis=0), index=feats).sort_values(ascending=False)
    # Signed mean: direction of the model's typical reliance, which mean-|SHAP|
    # deliberately discards. Reported alongside so a feature that pushes risk UP
    # is not confused with one that merely matters.
    signed = pd.Series(S.mean(axis=0), index=feats)
    corr = pd.Series({f: (np.corrcoef(X[f].fillna(X[f].median()), S[:, i])[0, 1]
                          if X[f].nunique(dropna=True) > 1 else np.nan)
                      for i, f in enumerate(feats)})
    glob = pd.DataFrame({"feature": mean_abs.index,
                         "block": [block_of(f) for f in mean_abs.index],
                         "mean_abs_shap": mean_abs.values,
                         "share_pct": 100 * mean_abs.values / mean_abs.sum(),
                         "mean_signed_shap": signed[mean_abs.index].values,
                         "corr_feature_vs_shap": corr[mean_abs.index].values})
    glob.to_csv(out / "shap_global_ranking.csv", index=False)

    print("\n" + "=" * 96)
    print("GLOBAL SHAP RANKING — test set, mean |SHAP| in log-odds")
    print("=" * 96)
    print(glob.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\n{CAVEAT}")

    # ── LEVEL 2 — block-level aggregation ─────────────────────────────
    blocks = pd.Series([block_of(f) for f in feats], index=feats)
    agg = (pd.DataFrame({"block": blocks.values,
                         "mean_abs_shap": np.abs(S).mean(axis=0)})
           .groupby("block")["mean_abs_shap"].sum().sort_values(ascending=False))
    blk = pd.DataFrame({"block": agg.index, "mean_abs_shap": agg.values,
                        "share_pct": 100 * agg.values / agg.sum(),
                        "n_features": [int((blocks == b).sum()) for b in agg.index]})
    blk.to_csv(out / "shap_block_ranking.csv", index=False)
    print("\n" + "-" * 96)
    print("BLOCK-LEVEL SHAP (§8.8 committee view)")
    print("-" * 96)
    print(blk.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    for f, b in BLOCK_ATTRIBUTION_OVERRIDE.items():
        if f in feats:
            cat = next((k for k, v in FEATURE_GROUPS.items() if f in v), "?")
            print(f"  NOTE: '{f}' is catalogued under {cat} but attributed to {b} here "
                  f"— {cat} holds no other feature in this matrix (interest coverage "
                  f"awaits the expanded pull), and EBIT/TA is a return measure.")

    # Features the model never used. Reported, not hidden: a feature carried
    # through the whole pipeline and then split on zero times is either
    # redundant or near-constant, and should be justified or dropped.
    dead = [f for f, v in mean_abs.items() if v == 0.0]
    if dead:
        print(f"  NOTE: {len(dead)} feature(s) received ZERO attribution — the model "
              f"never split on them: {dead}")

    # ── figures ───────────────────────────────────────────────────────
    plt.figure()
    shap.summary_plot(S, X, feature_names=feats, show=False, max_display=20)
    plt.title("Global SHAP — Model A (XGBoost), out-of-time test set", fontsize=10)
    plt.figtext(0.01, 0.005, CAVEAT, fontsize=6, wrap=True)
    plt.tight_layout(); plt.savefig(fig_dir / "shap_beeswarm.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 4))
    plt.barh(blk["block"][::-1], blk["mean_abs_shap"][::-1])
    plt.xlabel("Summed mean |SHAP| (log-odds)")
    plt.title("SHAP by feature block — Model A")
    plt.figtext(0.01, 0.005, CAVEAT, fontsize=6, wrap=True)
    plt.tight_layout(); plt.savefig(fig_dir / "shap_blocks.png", dpi=160); plt.close()

    # ── LEVEL 3 — local waterfalls ────────────────────────────────────
    # Chosen from the EARLY-WARNING stratum (healthy at t, distressed at t+1):
    # these are the predictions the product exists to make, and the only ones
    # where an explanation is not largely restating "it was already distressed".
    early = te.index[(te["label_now"] == 0) & (te[LABEL] == 1)]
    cases = p_test.loc[early].sort_values(ascending=False).head(n_local).index
    local_rows = []
    print("\n" + "-" * 96)
    print(f"LOCAL EXPLANATIONS — {len(cases)} highest-confidence TRUE early warnings "
          f"(healthy at t, distressed at t+1)")
    print("-" * 96)
    for rank, i in enumerate(cases, 1):
        pos = te.index.get_loc(i)
        contrib = pd.Series(S[pos], index=feats).sort_values(key=np.abs, ascending=False)
        name = te.loc[i, "company_name"] if "company_name" in te.columns else ""
        print(f"\n  [{rank}] {te.loc[i,'ticker']}  {str(name)[:38]:<38s} "
              f"FY{int(te.loc[i,'fiscal_year_num'])}  p(distress t+1) = {p_test[i]*100:.1f}%")
        print(f"      base rate {1/(1+np.exp(-base))*100:.2f}%  ->  top drivers (log-odds):")
        for f, v in contrib.head(6).items():
            print(f"        {v:+8.3f}  {f:<34s} = {te.loc[i, f]:,.4f}"
                  if pd.notna(te.loc[i, f]) else
                  f"        {v:+8.3f}  {f:<34s} = NaN (missing — model's native branch)")
        for f, v in contrib.items():
            local_rows.append({"rank": rank, "ticker": te.loc[i, "ticker"],
                               "fiscal_year_num": int(te.loc[i, "fiscal_year_num"]),
                               "p_distress": float(p_test[i]), "feature": f,
                               "block": block_of(f), "shap": float(v),
                               "value": (float(te.loc[i, f]) if pd.notna(te.loc[i, f]) else None)})
        plt.figure()
        shap.plots.waterfall(sv[pos], max_display=12, show=False)
        plt.title(f"{te.loc[i,'ticker']} FY{int(te.loc[i,'fiscal_year_num'])} — "
                  f"p={p_test[i]*100:.1f}%", fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / f"shap_waterfall_{rank}_{str(te.loc[i,'ticker']).split()[0]}.png",
                    dpi=160, bbox_inches="tight")
        plt.close()
    pd.DataFrame(local_rows).to_csv(out / "shap_local_cases.csv", index=False)

    # ── LEVEL 4 — nearest-neighbour counterfactual ────────────────────
    # "What distinguishes this firm from a healthy peer?" The peer is matched on
    # SIZE and SECTOR-FREE scale controls in standardised feature space among
    # firms the model scored LOW and that stayed healthy — so the contrast is
    # against an actually-healthy comparator, not merely a low-scored one.
    cf_rows = []
    healthy = te.index[(te["label_now"] == 0) & (te[LABEL] == 0) &
                       (p_test.loc[te.index] < 0.10)]
    Z = te.loc[:, feats]
    mu, sd = Z.loc[healthy].mean(), Z.loc[healthy].std().replace(0, 1.0)
    Zs = ((Z - mu) / sd).fillna(0.0)
    size_cols = [c for c in ("log_total_assets", "log_mkt_cap") if c in feats]
    print("\n" + "-" * 96)
    print("COUNTERFACTUAL CONTRASTS — nearest healthy peer (matched on size, "
          "Euclidean in standardised feature space)")
    print("-" * 96)
    for rank, i in enumerate(cases, 1):
        pool = healthy
        if size_cols:  # restrict to a +/-0.5 sd size band before distance ranking
            band = (Zs.loc[healthy, size_cols] - Zs.loc[i, size_cols]).abs().max(axis=1)
            pool = band[band <= 0.5].index if (band <= 0.5).any() else healthy
        d = ((Zs.loc[pool] - Zs.loc[i]) ** 2).sum(axis=1)
        j = d.idxmin()
        pos_i, pos_j = te.index.get_loc(i), te.index.get_loc(j)
        gap = pd.Series(S[pos_i] - S[pos_j], index=feats).sort_values(key=np.abs,
                                                                     ascending=False)
        print(f"\n  [{rank}] {te.loc[i,'ticker']} FY{int(te.loc[i,'fiscal_year_num'])} "
              f"(p={p_test[i]*100:.1f}%)  vs peer  {te.loc[j,'ticker']} "
              f"FY{int(te.loc[j,'fiscal_year_num'])} (p={p_test[j]*100:.1f}%)")
        for f, v in gap.head(5).items():
            print(f"        Δshap {v:+7.3f}  {f:<32s}  "
                  f"{te.loc[i,f]:>12,.4f} vs {te.loc[j,f]:>12,.4f}"
                  if pd.notna(te.loc[i, f]) and pd.notna(te.loc[j, f]) else
                  f"        Δshap {v:+7.3f}  {f:<32s}  (value missing on one side)")
            cf_rows.append({"rank": rank, "case": te.loc[i, "ticker"],
                            "peer": te.loc[j, "ticker"], "feature": f,
                            "delta_shap": float(v),
                            "case_value": (float(te.loc[i, f]) if pd.notna(te.loc[i, f]) else None),
                            "peer_value": (float(te.loc[j, f]) if pd.notna(te.loc[j, f]) else None)})
    pd.DataFrame(cf_rows).to_csv(out / "shap_counterfactuals.csv", index=False)

    meta = {"caveat": CAVEAT, "n_test_rows": int(len(X)), "n_features": len(feats),
            "base_value_logodds": base, "base_rate": float(1 / (1 + np.exp(-base))),
            "additivity_max_error": max_err,
            "top_features": glob.head(10)[["feature", "share_pct"]].to_dict("records"),
            "block_shares": blk[["block", "share_pct"]].to_dict("records")}
    (out / "shap_summary.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/shap_global_ranking.csv, shap_block_ranking.csv, "
          f"shap_local_cases.csv, shap_counterfactuals.csv, shap_summary.json")
    print(f"wrote {fig_dir}/ (beeswarm, blocks, {len(cases)} waterfalls)")
    return glob, blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/processed/training_matrix.csv")
    ap.add_argument("--panel", default="data/processed/bloomberg_panel_scored_gated.csv")
    ap.add_argument("--out", default="reports/phase3")
    ap.add_argument("--n-local", type=int, default=5)
    a = ap.parse_args()
    _, store = run(a.matrix, a.panel, use_xgb=True, out_dir=a.out)
    explain(store, out_dir=a.out, n_local=a.n_local)


if __name__ == "__main__":
    main()
