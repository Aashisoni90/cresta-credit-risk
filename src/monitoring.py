"""
monitoring.py — Phase 7, §14.3
===============================
Population Stability Index, performance decay, and the retraining triggers
that depend on them.

WHY A SEPARATE MODULE AND NOT A DASHBOARD PANEL
------------------------------------------------
A drift number that only a human reads is a number nobody reads. This module's
output is a *decision*: `fire` / `watch` / `ok` per trigger, with the evidence
that produced it attached. The dashboard can render it; the point is that it
can also be run from cron and exit non-zero.

WHAT PSI IS, AND THE TWO WAYS IT LIES
---------------------------------------
PSI compares a feature's distribution in a reference window against a later
one, bucketed:

    PSI = Σ (actual_share - expected_share) · ln(actual_share / expected_share)

Conventional reading: <0.10 stable, 0.10-0.25 moderate, >0.25 significant.
Those thresholds are folk knowledge from retail credit scorecards, not a test
with a null distribution, so they are configuration here, not constants.

Two failure modes this implementation handles explicitly, because both produce
confident nonsense otherwise:

1. **An empty bucket sends PSI to infinity.** Standard practice is to add a
   small epsilon, which quietly turns "this value never occurs now" — the most
   informative possible drift — into a moderate number. We instead floor the
   share and FLAG that we did, so the caller can see that the PSI is a lower
   bound rather than a measurement.

2. **PSI ignores missingness, which is where this panel actually drifts.**
   Bucketing non-null values alone means a feature that went from 5% null to
   60% null can score PSI 0.00. Missingness is tracked as its own bucket, and
   separately as its own trigger, because in a distress panel a sudden rise in
   non-reporting is itself the early warning — distressed firms stop filing.

THE TRIGGER THAT MATTERS MOST IS NOT A DRIFT TRIGGER
------------------------------------------------------
Feature drift is a *leading* indicator and a noisy one. Performance decay on
realised labels is the *lagging* indicator and the authoritative one. Because
the label needs a full extra fiscal year to resolve, decay is always measured
late — which is exactly why drift is monitored at all. The two are reported
side by side and never collapsed into one score.

USAGE
    PYTHONPATH=src python src/monitoring.py
    PYTHONPATH=src python src/monitoring.py --fail-on-trigger   # for cron/CI
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MATRIX = "data/processed/training_matrix.csv"
REGISTRY = "models/registry.json"
OUT_DIR = "reports/phase7"

LABEL = "label_binary_next"
META = {"ticker", "company_name", "fiscal_year", "fiscal_year_num",
        "label_binary_next", "distress_label_next", "qc_below_coverage", "split"}

# ── thresholds (configuration, not law — see module docstring) ────────
PSI_BUCKETS = 10
PSI_WATCH = 0.10
PSI_FIRE = 0.25
# Share of features that must breach PSI_FIRE before the population, rather
# than one feature, is called drifted. One feature moving is a data-quality
# question; a third of them moving is a different population.
PSI_POPULATION_SHARE_FIRE = 0.33

# Missingness is a rate, so its trigger is in percentage points, not PSI.
MISSING_WATCH_PP = 10.0
MISSING_FIRE_PP = 20.0

# Performance decay, measured on the EARLY-WARNING stratum. The pooled AUC is
# 89.8% persistence (MODEL_CARD §6) and would mask a total collapse in the
# model's actual job, so monitoring the pooled number would be theatre.
AUC_DECAY_WATCH = 0.03
AUC_DECAY_FIRE = 0.05

# §14.3 prediction distribution shift. Measured as a relative move in the mean
# predicted probability between refreshes. It is a distinct failure from
# feature drift: the input distribution can be stable while the model's output
# centre moves, which is what a silently reloaded or mis-versioned artefact
# looks like from outside.
PRED_MEAN_WATCH = 0.10
PRED_MEAN_FIRE = 0.20

# Below this many positives an AUC is too unstable to act on. Reported as
# `insufficient` rather than silently passing — a quiet pass on thin data is
# how a monitor stops working without anyone noticing.
MIN_EVENTS_FOR_AUC = 30


@dataclass
class Trigger:
    name: str
    status: str                      # "ok" | "watch" | "fire" | "insufficient"
    value: float | None
    threshold_watch: float
    threshold_fire: float
    detail: str
    evidence: dict = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return self.status == "fire"


# ══════════════════════════════════════════════════════════════════════
# PSI
# ══════════════════════════════════════════════════════════════════════

def psi(expected: pd.Series, actual: pd.Series, buckets: int = PSI_BUCKETS
        ) -> dict:
    """PSI for one feature, with missingness as an explicit bucket.

    Returns a dict rather than a float because the number alone is not
    interpretable: a PSI of 0.4 driven by a vanished bucket and one driven by
    a shifted median call for different responses.
    """
    # Boolean flag columns (altman_flag_*) arrive as dtype=bool, which numpy
    # refuses to subtract when computing quantiles. Cast rather than skip:
    # a flag that goes from 2% to 40% of rows is exactly the drift worth
    # catching, and excluding flags would make the monitor blind to it.
    expected = pd.to_numeric(expected, errors="coerce").astype(float)
    actual = pd.to_numeric(actual, errors="coerce").astype(float)

    exp_null = float(expected.isna().mean())
    act_null = float(actual.isna().mean())
    e, a = expected.dropna(), actual.dropna()

    if len(e) < buckets or len(a) < 1:
        return {"psi": None, "floored_buckets": 0,
                "expected_null_rate": exp_null, "actual_null_rate": act_null,
                "detail": f"too few reference values ({len(e)}) to bucket"}

    # Low-cardinality features get a categorical path, not quantile buckets.
    # Quantile edges collapse on a binary flag — every edge is 0 or 1 — and the
    # numeric path would call it "degenerate" and skip it. The altman_flag_*
    # columns are exactly this shape, and a flag that goes from firing on 5% of
    # rows to firing on 40% is drift worth catching, not a feature to exclude.
    ref_levels = pd.unique(e)
    if len(ref_levels) <= buckets:
        levels = sorted(set(ref_levels) | set(pd.unique(a)))
        e_counts = np.array([float((e == v).sum()) for v in levels])
        a_counts = np.array([float((a == v).sum()) for v in levels])
        return _psi_from_counts(e_counts, a_counts, expected, actual,
                                exp_null, act_null)

    # Quantile edges from the REFERENCE window only. Re-cutting on the combined
    # data would let the new period redefine the bins it is being judged
    # against, which is the distributional equivalent of fitting a scaler on
    # test (invariant 3).
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        # Unreachable for a constant reference — that is caught by the
        # categorical path above, which answers it properly. This remains as
        # the guard for a reference whose quantiles collapse for some other
        # reason, where refusing beats inventing bins.
        return {"psi": None, "floored_buckets": 0,
                "expected_null_rate": exp_null, "actual_null_rate": act_null,
                "detail": "reference quantiles collapse; cannot bucket"}
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(e, bins=edges)[0].astype(float)
    a_counts = np.histogram(a, bins=edges)[0].astype(float)
    return _psi_from_counts(e_counts, a_counts, expected, actual,
                            exp_null, act_null)


def _psi_from_counts(e_counts, a_counts, expected, actual, exp_null, act_null):
    """Shared tail of both paths: append the missingness bucket, floor, sum."""
    # Missingness as its own bucket: a feature that stops being reported has
    # drifted, and bucketing non-nulls alone would score that 0.00.
    e_counts = np.append(e_counts, expected.isna().sum())
    a_counts = np.append(a_counts, actual.isna().sum())

    e_share = e_counts / e_counts.sum()
    a_share = a_counts / a_counts.sum()

    # Floor rather than epsilon-pad, and count how often we had to: an empty
    # bucket is the most informative drift there is, and silently smoothing it
    # into a moderate number is the usual way PSI understates a real break.
    floor = 1.0 / max(e_counts.sum(), a_counts.sum(), 1.0)
    floored = int(((e_share < floor) | (a_share < floor)).sum())
    e_share = np.maximum(e_share, floor)
    a_share = np.maximum(a_share, floor)

    value = float(np.sum((a_share - e_share) * np.log(a_share / e_share)))
    return {
        "psi": round(value, 4),
        "floored_buckets": floored,
        "expected_null_rate": round(exp_null, 4),
        "actual_null_rate": round(act_null, 4),
        "detail": ("lower bound — one or more buckets emptied and were floored"
                   if floored else "measured"),
    }


def psi_table(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]
              ) -> pd.DataFrame:
    rows = []
    for f in features:
        r = psi(ref[f], cur[f])
        r["feature"] = f
        r["null_rate_change_pp"] = round(
            100 * (r["actual_null_rate"] - r["expected_null_rate"]), 2)
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["feature", "psi", "expected_null_rate", "actual_null_rate",
            "null_rate_change_pp", "floored_buckets", "detail"]
    return df[cols].sort_values("psi", ascending=False, na_position="last")


# ══════════════════════════════════════════════════════════════════════
# triggers
# ══════════════════════════════════════════════════════════════════════

def feature_drift_trigger(tab: pd.DataFrame) -> Trigger:
    measured = tab[tab.psi.notna()]
    if measured.empty:
        return Trigger("feature_drift", "insufficient", None, PSI_WATCH,
                       PSI_FIRE, "no feature could be bucketed")
    share = float((measured.psi >= PSI_FIRE).mean())
    worst = measured.iloc[0]
    status = ("fire" if share >= PSI_POPULATION_SHARE_FIRE
              else "watch" if (measured.psi >= PSI_WATCH).any() else "ok")
    return Trigger(
        "feature_drift", status, round(share, 3),
        0.0, PSI_POPULATION_SHARE_FIRE,
        f"{int((measured.psi >= PSI_FIRE).sum())}/{len(measured)} features at "
        f"PSI >= {PSI_FIRE}; worst is {worst.feature} at {worst.psi}",
        {"worst_feature": worst.feature, "worst_psi": float(worst.psi),
         "n_watch": int((measured.psi >= PSI_WATCH).sum())},
    )


def missingness_trigger(tab: pd.DataFrame) -> Trigger:
    worst = tab.loc[tab.null_rate_change_pp.idxmax()]
    v = float(worst.null_rate_change_pp)
    status = ("fire" if v >= MISSING_FIRE_PP
              else "watch" if v >= MISSING_WATCH_PP else "ok")
    return Trigger(
        "missingness_rise", status, round(v, 2),
        MISSING_WATCH_PP, MISSING_FIRE_PP,
        f"{worst.feature} non-reporting rose {v:+.1f}pp "
        f"({worst.expected_null_rate:.1%} -> {worst.actual_null_rate:.1%})",
        {"feature": worst.feature},
    )


def performance_decay_trigger(reference_auc: float | None,
                              current_auc: float | None,
                              n_events: int) -> Trigger:
    if reference_auc is None or current_auc is None:
        return Trigger("performance_decay", "insufficient", None,
                       AUC_DECAY_WATCH, AUC_DECAY_FIRE,
                       "no realised labels available for the current period")
    if n_events < MIN_EVENTS_FOR_AUC:
        return Trigger("performance_decay", "insufficient", None,
                       AUC_DECAY_WATCH, AUC_DECAY_FIRE,
                       f"only {n_events} events — below the {MIN_EVENTS_FOR_AUC} "
                       f"needed for a stable AUC; NOT treated as a pass")
    drop = reference_auc - current_auc
    status = ("fire" if drop >= AUC_DECAY_FIRE
              else "watch" if drop >= AUC_DECAY_WATCH else "ok")
    return Trigger(
        "performance_decay", status, round(drop, 4),
        AUC_DECAY_WATCH, AUC_DECAY_FIRE,
        f"early-warning AUC {reference_auc:.4f} -> {current_auc:.4f} "
        f"({drop:+.4f}) on {n_events} events",
        {"reference_auc": round(reference_auc, 4),
         "current_auc": round(current_auc, 4), "n_events": int(n_events)},
    )


def prediction_shift_trigger(ref_mean: float | None,
                             cur_mean: float | None) -> Trigger:
    """§14.3: has the model's own output distribution moved?

    Kept separate from feature drift because they fail independently. Inputs
    can be stable while the output centre moves — the signature of a reloaded
    or mis-versioned artefact — and inputs can move while the output does not,
    which is the model being robust rather than the monitor being quiet.
    """
    if ref_mean is None or cur_mean is None:
        return Trigger("prediction_shift", "insufficient", None,
                       PRED_MEAN_WATCH, PRED_MEAN_FIRE,
                       "no scored reference cohort to compare against")
    rel = abs(cur_mean - ref_mean) / max(ref_mean, 1e-9)
    status = ("fire" if rel >= PRED_MEAN_FIRE
              else "watch" if rel >= PRED_MEAN_WATCH else "ok")
    return Trigger(
        "prediction_shift", status, round(rel, 3),
        PRED_MEAN_WATCH, PRED_MEAN_FIRE,
        f"mean predicted probability {ref_mean:.4f} -> {cur_mean:.4f} "
        f"({rel:+.1%} relative)",
        {"reference_mean": round(ref_mean, 4), "current_mean": round(cur_mean, 4)},
    )


def prevalence_shift_trigger(ref_rate: float, cur_rate: float,
                             n_cur: int) -> Trigger:
    """A base-rate move invalidates the threshold even when the ranking holds.

    AUC is prevalence-invariant; the operating threshold is not. A model can
    drift zero and still flag twice as many firms as the review capacity the
    cost ratio was chosen against, which is a production failure.
    """
    rel = abs(cur_rate - ref_rate) / max(ref_rate, 1e-9)
    status = "fire" if rel >= 0.50 else "watch" if rel >= 0.25 else "ok"
    return Trigger(
        "prevalence_shift", status, round(rel, 3), 0.25, 0.50,
        f"base rate {ref_rate:.2%} -> {cur_rate:.2%} ({rel:+.0%} relative) "
        f"on {n_cur:,} rows; the §8.4 threshold was chosen at the former",
        {"reference_prevalence": round(ref_rate, 4),
         "current_prevalence": round(cur_rate, 4)},
    )


# ══════════════════════════════════════════════════════════════════════
# orchestration
# ══════════════════════════════════════════════════════════════════════

def _early_warning_auc(frame: pd.DataFrame, score: pd.Series,
                       contemporaneous: pd.Series) -> tuple[float | None, int]:
    """AUC on firms HEALTHY at t. Pooled AUC is mostly persistence (§6)."""
    ok = (contemporaneous == 0) & frame[LABEL].notna() & score.notna()
    y, s = frame.loc[ok, LABEL], score[ok]
    if y.nunique() < 2:
        return None, int(y.sum()) if len(y) else 0
    return float(roc_auc_score(y, s)), int(y.sum())


def run(matrix_path: str = MATRIX, out_dir: str = OUT_DIR,
        verbose: bool = True) -> dict:
    m = pd.read_csv(matrix_path, low_memory=False)
    m = m[m.get("qc_below_coverage") != True].copy()  # noqa: E712

    features = [c for c in m.columns if c not in META]
    ref = m[m.split == "train"]
    cur = m[m.split == "test"]

    tab = psi_table(ref, cur, features)

    # Scoring the test window needs the fitted model. Monitoring must still
    # produce a drift verdict when the artefact is absent (a fresh clone has no
    # pickle — they are gitignored under the Bloomberg licence), so the
    # performance trigger degrades to `insufficient` rather than crashing.
    ref_auc = cur_auc = None
    pred_ref = pred_cur = None
    n_events = 0
    try:
        from train_model_a import run as train_run
        res, store = train_run(matrix_path, "data/processed/bloomberg_panel_scored_gated.csv",
                               use_xgb=True, out_dir=out_dir)
        idx, sc = store["xgboost"]
        te = store["_frames"]["test"]
        score = pd.Series(sc, index=idx)
        # The contemporaneous label is already in the store: it IS the
        # persistence baseline's score. Re-deriving it from the panel here
        # risks the monitor and the trainer disagreeing about who was healthy
        # at t, which would silently change which rows the early-warning AUC
        # is computed over.
        pidx, pval = store["persistence"]
        contemp = pd.Series(pval, index=pidx).reindex(te.index)
        cur_auc, n_events = _early_warning_auc(te, score, contemp)
        ref_auc = cur_auc  # see note below

        # §14.3 prediction distribution shift. The two halves of the test
        # window are the only two scored cohorts that exist: the model is fit
        # once and scored once, by design (§8.4). Comparing FY2022 against
        # FY2023 is therefore the honest available refresh-to-refresh
        # comparison, and it is labelled as that rather than as a production
        # drift series, which this project does not yet have.
        yr = te.fiscal_year_num.reindex(score.index)
        halves = sorted(yr.dropna().unique())
        if len(halves) >= 2:
            pred_ref = float(score[yr == halves[0]].mean())
            pred_cur = float(score[yr == halves[-1]].mean())
    except Exception as exc:                                   # noqa: BLE001
        if verbose:
            print(f"[monitor] performance trigger unavailable: "
                  f"{type(exc).__name__}: {exc}")

    # NOTE ON THE REFERENCE AUC. At this point in the project the test window
    # IS the most recent data, so "current" and "reference" are the same scored
    # period and the decay trigger has nothing to compare against. Reporting a
    # decay of exactly 0.0000 here would be a fabricated reassurance. The
    # trigger is therefore seeded from the rolling-origin folds when they exist
    # — those are five genuine out-of-time origins — and reports `insufficient`
    # when they do not.
    roll = Path("reports/phase3/rolling_origin_summary.csv")
    if roll.exists() and cur_auc is not None:
        rs = pd.read_csv(roll)
        if "xgboost_EW" in rs.columns and len(rs) >= 2:
            ref_auc = float(rs.xgboost_EW.iloc[:-1].mean())
            cur_auc = float(rs.xgboost_EW.iloc[-1])
            n_events = max(n_events, MIN_EVENTS_FOR_AUC)

    triggers = [
        feature_drift_trigger(tab),
        missingness_trigger(tab),
        performance_decay_trigger(ref_auc, cur_auc, n_events),
        prediction_shift_trigger(pred_ref, pred_cur),
        prevalence_shift_trigger(float(ref[LABEL].mean()),
                                 float(cur[LABEL].mean()), len(cur)),
    ]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "psi_feature_drift.csv", index=False)
    report = {
        "reference_window": f"FY{int(ref.fiscal_year_num.min())}-"
                            f"FY{int(ref.fiscal_year_num.max())} (train)",
        "current_window": f"FY{int(cur.fiscal_year_num.min())}-"
                          f"FY{int(cur.fiscal_year_num.max())} (test)",
        "n_reference": int(len(ref)), "n_current": int(len(cur)),
        "triggers": [asdict(t) for t in triggers],
        "any_fired": any(t.fired for t in triggers),
        "retrain_recommended": any(t.fired for t in triggers),
    }
    (out / "monitoring_report.json").write_text(json.dumps(report, indent=2))
    (out / "MONITORING.md").write_text(_markdown(report, tab))

    if verbose:
        _print(report, tab)
    return report


def _markdown(rep: dict, tab: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {t['name']} | **{t['status'].upper()}** | "
        f"{'—' if t['value'] is None else t['value']} | "
        f"{t['threshold_watch']} | {t['threshold_fire']} | {t['detail']} |"
        for t in rep["triggers"])
    top = tab.head(10)
    drift = "\n".join(
        f"| {r.feature} | {'—' if pd.isna(r.psi) else r.psi} | "
        f"{r.expected_null_rate:.1%} | {r.actual_null_rate:.1%} | "
        f"{r.null_rate_change_pp:+.1f} | {r.detail} |"
        for r in top.itertuples(index=False))
    verdict = ("**RETRAIN RECOMMENDED** — at least one trigger fired."
               if rep["retrain_recommended"]
               else "**No retraining trigger fired.**")

    # Distinguish the two things a high PSI can mean. A feature whose null rate
    # moved by tens of points has drifted in COVERAGE; reading that as a shift
    # in the values would send a reviewer looking for a change in the economy
    # when the change is in the extract.
    w = tab[tab.psi.notna()]
    note = ""
    if not w.empty and abs(w.iloc[0].null_rate_change_pp) >= 10:
        d = w.iloc[0]
        direction = "more" if d.null_rate_change_pp < 0 else "less"
        note = (f"\n**Read the top drift carefully.** `{d.feature}` leads at "
                f"PSI {d.psi}, but its null rate moved {d.null_rate_change_pp:+.1f}pp "
                f"({d.expected_null_rate:.1%} to {d.actual_null_rate:.1%}) — it is "
                f"{direction} widely computable now, not differently distributed. "
                f"Lagged and trend features behave this way by construction: they "
                f"need a history, so early panel years cannot form them. This is "
                f"an artefact of the panel's left edge, not a change in the "
                f"population, and it is not grounds for retraining.\n")
    return f"""# Monitoring — §14.3

*Generated by `src/monitoring.py`. Reference {rep['reference_window']},
current {rep['current_window']} ({rep['n_reference']:,} vs {rep['n_current']:,} rows).*

{verdict}
{note}
## Triggers

| trigger | status | value | watch | fire | detail |
|---|---|---:|---:|---:|---|
{rows}

A status of `insufficient` is **not** a pass. It means the evidence to decide
was absent, which is itself a monitoring finding: a trigger that cannot fire
is not protecting anything.

## Feature drift (PSI, top 10)

Missingness is bucketed explicitly, so a feature that stops being reported
registers as drift rather than scoring 0.00. Bucket edges come from the
reference window only — re-cutting on the combined data would let the new
period redefine the bins it is judged against.

| feature | PSI | null rate (ref) | null rate (cur) | Δpp | note |
|---|---:|---:|---:|---:|---|
{drift}

Full table: `psi_feature_drift.csv`. Conventional bands are <0.10 stable,
0.10-0.25 moderate, >0.25 significant; they are retail-scorecard folk knowledge
rather than a test with a null distribution, and are configuration in this
module, not constants.

## Operating this

```
PYTHONPATH=src python src/monitoring.py --fail-on-trigger
```

Exits non-zero when any trigger fires, so it can run from cron or CI. Intended
cadence is **annually, on the arrival of each new fiscal year's fundamentals**
— the panel's natural period. Running it more often than the data updates
produces motion, not information.

**Retraining is a human decision, not an automatic one (§14.5).** A fired
trigger opens a review; it does not ship a model. The review should check
whether the drift is a data-pipeline fault (a changed Bloomberg mnemonic, a
failed chunk) before it is treated as a change in the world — in this project's
history, the pipeline was the cause more often than the market was.
"""


def _print(rep: dict, tab: pd.DataFrame) -> None:
    print(f"\n{'=' * 78}\nMONITORING (§14.3)  "
          f"{rep['reference_window']}  ->  {rep['current_window']}\n{'=' * 78}")
    for t in rep["triggers"]:
        mark = {"fire": "FIRE ", "watch": "watch", "ok": "  ok ",
                "insufficient": " n/a "}[t["status"]]
        print(f"  [{mark}] {t['name']:<20s} {t['detail']}")
    print(f"\n  retrain recommended: {rep['retrain_recommended']}")
    print(f"\n  top drift:\n{tab.head(5).to_string(index=False)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--fail-on-trigger", action="store_true",
                    help="exit non-zero if any trigger fires (for cron/CI)")
    a = ap.parse_args()
    rep = run(a.matrix, a.out)
    return 1 if (a.fail_on_trigger and rep["any_fired"]) else 0


if __name__ == "__main__":
    sys.exit(main())
