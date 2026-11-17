"""
outcomes_analysis.py — Phase 7, §14.4
======================================
Backtest loop: predictions against realised outcomes, and an individual review
of **every** false negative.

§14.4 says every false negative is reviewed individually, asking one question:
*was the signal present in the data but unused, or genuinely absent?* That is
the distinction that decides what to do about it. A missed firm whose features
all looked healthy is a limit of the data. A missed firm whose features were
screaming is a limit of the model, and a much more serious finding.

HOW "PRESENT BUT UNUSED" IS DECIDED, AND WHY IT IS NOT A JUDGEMENT CALL
------------------------------------------------------------------------
For each false negative the module checks, against the TRAIN-window
distribution of each feature:

    available   the feature is not null
    adverse     it sits in the worse tail, past the train-window quantile at
                which distress prevalence is materially elevated

"Adverse" is calibrated per feature from the training data, not asserted. For
each feature the train window is split at its own 20th/80th percentile
(whichever tail carries higher distress prevalence), and that direction is
recorded. So "adverse" means "in the tail that historically preceded
distress", which is checkable, rather than "looks bad to me".

The verdict for a case is then:

    SIGNAL ABSENT     no feature was both available and adverse
    SIGNAL UNUSED     one or more were, and the model still scored it low
    DATA MISSING      most features were null — the firm was unscoreable in
                      substance, and a low score here is the model declining
                      to guess rather than the model being wrong

The third category matters. Lumping it into "signal absent" would credit the
data with an absence it never had the chance to show.

USAGE
    PYTHONPATH=src python src/outcomes_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_model_a import run as train_run, META, LABEL

MATRIX = "data/processed/training_matrix.csv"
PANEL = "data/processed/bloomberg_panel_scored_gated.csv"
OUT_DIR = "reports/phase7"

TAIL_Q = 0.20                 # the tail each feature's "adverse" side is cut at
MIN_PREVALENCE_LIFT = 1.25    # a tail must carry >=1.25x base distress to count
DATA_MISSING_SHARE = 0.50     # over half the features null -> unscoreable


def adverse_profile(train: pd.DataFrame, features: list[str]) -> dict:
    """Per feature: which tail historically preceded distress, and its cut.

    Calibrated on the TRAIN window only (invariant 3). Using the test window
    to decide what counts as a bad value would let the outcome define the
    evidence, which is the whole failure this project exists to avoid.
    """
    base = float(train[LABEL].mean())
    prof = {}
    for f in features:
        # Boolean flag columns arrive as dtype=bool and numpy will not take
        # their quantiles. Cast rather than skip: a flag firing IS a signal,
        # and skipping flags would make "signal absent" partly an artefact of
        # which columns this function could read.
        s = pd.to_numeric(train[f], errors="coerce").astype(float)
        if s.notna().sum() < 100:
            continue
        lo, hi = s.quantile(TAIL_Q), s.quantile(1 - TAIL_Q)
        p_lo = float(train.loc[s <= lo, LABEL].mean())
        p_hi = float(train.loc[s >= hi, LABEL].mean())
        if max(p_lo, p_hi) < base * MIN_PREVALENCE_LIFT:
            continue                     # neither tail is informative
        if p_lo >= p_hi:
            prof[f] = {"direction": "low", "cut": float(lo),
                       "tail_prevalence": round(p_lo, 4)}
        else:
            prof[f] = {"direction": "high", "cut": float(hi),
                       "tail_prevalence": round(p_hi, 4)}
    prof["_base_rate"] = round(base, 4)
    return prof


def _is_adverse(value, spec) -> bool:
    if value is None or pd.isna(value):
        return False
    value = float(value)
    return (value <= spec["cut"] if spec["direction"] == "low"
            else value >= spec["cut"])


def review_false_negatives(te: pd.DataFrame, score: pd.Series,
                           threshold: float, prof: dict,
                           features: list[str]) -> pd.DataFrame:
    fn = te[(te[LABEL] == 1) & (score.reindex(te.index) < threshold)]
    specs = {k: v for k, v in prof.items() if not k.startswith("_")}
    rows = []
    for idx, r in fn.iterrows():
        null_share = float(r[features].isna().mean())
        adverse = [f for f, sp in specs.items() if _is_adverse(r.get(f), sp)]
        available_informative = [f for f in specs if pd.notna(r.get(f))]
        if null_share >= DATA_MISSING_SHARE:
            verdict = "DATA MISSING"
        elif adverse:
            verdict = "SIGNAL UNUSED"
        else:
            verdict = "SIGNAL ABSENT"
        rows.append({
            "ticker": r.ticker, "company_name": r.get("company_name", ""),
            "fiscal_year": int(r.fiscal_year_num),
            "probability": round(float(score.loc[idx]), 4),
            "threshold": threshold,
            "null_feature_share": round(null_share, 3),
            "n_informative_available": len(available_informative),
            "n_adverse": len(adverse),
            "adverse_features": "; ".join(sorted(adverse)),
            "verdict": verdict,
        })
    return (pd.DataFrame(rows).sort_values(["verdict", "probability"],
                                           ascending=[True, False])
            if rows else pd.DataFrame(columns=["ticker", "verdict"]))


def decile_calibration(te: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """Does a predicted 0.3 behave like a 0.3? Ranking can be perfect while
    the probability is uncalibrated, and the §8.4 cost threshold is applied to
    the probability, not to the rank."""
    d = pd.DataFrame({"p": score.reindex(te.index), "y": te[LABEL]}).dropna()
    d["decile"] = pd.qcut(d.p.rank(method="first"), 10,
                          labels=range(1, 11)).astype(int)
    g = d.groupby("decile").agg(n=("y", "size"), predicted=("p", "mean"),
                                observed=("y", "mean"))
    g["gap"] = (g.observed - g.predicted).round(4)
    return g.round(4).reset_index()


def run(matrix_path: str = MATRIX, panel_path: str = PANEL,
        out_dir: str = OUT_DIR, verbose: bool = True) -> dict:
    res, store = train_run(matrix_path, panel_path, use_xgb=True, out_dir=out_dir)
    idx, sc = store["xgboost"]
    score = pd.Series(sc, index=idx)
    frames = store["_frames"]
    te, tr = frames["test"], frames["train"]
    features = store["_features"]
    threshold = float(res.loc[res.model == "XGBoost", "threshold"].iloc[0])

    # company_name is blank for chunk-sourced rows; join it from the universe
    # so a reviewer reading the false-negative list can recognise the firms
    # without looking each ticker up.
    uni = pd.read_csv("data/processed/universe_800.csv",
                      usecols=["ticker", "company_name"]).drop_duplicates("ticker")
    names = uni.set_index("ticker").company_name
    te = te.copy()
    te["company_name"] = (te["company_name"].replace("", np.nan)
                          .fillna(te.ticker.map(names)))

    prof = adverse_profile(tr, features)
    fn = review_false_negatives(te, score, threshold, prof, features)
    cal = decile_calibration(te, score)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fn.to_csv(out / "false_negative_review.csv", index=False)
    cal.to_csv(out / "decile_calibration.csv", index=False)
    (out / "adverse_profile.json").write_text(json.dumps(prof, indent=2))

    counts = fn.verdict.value_counts().to_dict() if len(fn) else {}
    summary = {
        "threshold": threshold,
        "n_test": int(len(te)),
        "n_positives": int(te[LABEL].sum()),
        "n_false_negatives": int(len(fn)),
        "recall": round(1 - len(fn) / max(int(te[LABEL].sum()), 1), 4),
        "verdicts": counts,
        "informative_features": len(prof) - 1,
    }
    (out / "outcomes_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "OUTCOMES_ANALYSIS.md").write_text(_markdown(summary, fn, cal, prof))

    if verbose:
        print(f"\n{'=' * 78}\nOUTCOMES ANALYSIS (§14.4)\n{'=' * 78}")
        print(f"  {summary['n_false_negatives']} false negatives of "
              f"{summary['n_positives']} realised events at threshold {threshold}")
        for k, v in counts.items():
            print(f"    {v:>4}  {k}")
        print(f"\n  wrote {out}/false_negative_review.csv, "
              f"decile_calibration.csv, OUTCOMES_ANALYSIS.md")
    return summary


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = "\n".join(
        "| " + " | ".join("—" if pd.isna(v) else (f"{v:g}" if isinstance(v, float)
                                                  else str(v))
                          for v in row) + " |"
        for row in df.itertuples(index=False))
    return "\n".join([head, rule, body])


def _markdown(s: dict, fn: pd.DataFrame, cal: pd.DataFrame, prof: dict) -> str:
    v = s["verdicts"]
    unused = v.get("SIGNAL UNUSED", 0)
    absent = v.get("SIGNAL ABSENT", 0)
    missing = v.get("DATA MISSING", 0)
    worst = fn[fn.verdict == "SIGNAL UNUSED"].head(10) if len(fn) else fn

    # State the calibration verdict rather than leaving it in the table for a
    # reader to compute. The sign is the whole point: an over-confident model
    # that is correctly RANKED still mis-states risk to anyone who reads the
    # number instead of the order, and the dashboard shows the number.
    upper = cal[cal.decile >= 7]
    mean_gap = float(upper.gap.mean())
    worst_dec = upper.loc[upper.gap.abs().idxmax()]
    if abs(mean_gap) < 0.05:
        cal_verdict = (
            f"**Calibration is acceptable.** Mean gap across the top four "
            f"deciles is {mean_gap:+.3f}, so a stated probability in the range "
            f"the threshold operates on is close to the realised rate.")
    else:
        word = "OVER-confident" if mean_gap < 0 else "UNDER-confident"
        cal_verdict = (
            f"**The model is systematically {word} in the upper deciles, and "
            f"this must be stated wherever a probability is shown.** Mean gap "
            f"across deciles 7-10 is {mean_gap:+.3f}; the worst is decile "
            f"{int(worst_dec.decile)}, which predicts {worst_dec.predicted:.1%} against a "
            f"realised {worst_dec.observed:.1%}.\n\n"
            f"The cause is not a defect but a choice: `scale_pos_weight` is "
            f"applied on the training window to handle imbalance (§8.4), and "
            f"re-weighting the positive class inflates predicted probabilities "
            f"by construction. **Ranking is unaffected** — AUC, KS and the "
            f"ordering the product actually uses are all rank statistics, and "
            f"the operating threshold was chosen on validation under the same "
            f"distortion, so the flag/no-flag decision is sound.\n\n"
            f"What is *not* sound is reading the number as a probability. A "
            f"firm shown at 70% has a realised rate nearer "
            f"{float(cal[cal.decile == 8].observed.iloc[0]):.0%}. Either the "
            f"dashboard presents the score as a risk *rank* rather than a "
            f"likelihood, or a calibration layer (Platt or isotonic, fitted on "
            f"the validation window only) is fitted before the number is "
            f"shown. The first is cheaper and is what §12.3's risk bands "
            f"already do; the second is the right fix if a probability is ever "
            f"quoted to a user.")

    unused_block = (
        _md_table(worst[["ticker", "company_name", "fiscal_year", "probability",
                         "n_adverse", "adverse_features"]])
        if len(worst) else
        "_None. No false negative had an available, historically adverse "
        "feature that the model scored through._")

    return f"""# Outcomes analysis — §14.4

*Generated by `src/outcomes_analysis.py`. Out-of-time test window, scored once.*

## 1. The backtest

| | |
|---|---:|
| test rows | {s['n_test']:,} |
| realised distress events | {s['n_positives']:,} |
| operating threshold (§8.4, chosen on validation) | {s['threshold']} |
| **false negatives** | **{s['n_false_negatives']}** |
| recall at threshold | {s['recall']:.1%} |

## 2. Every false negative, reviewed

§14.4 asks one question of each miss: *was the signal present in the data but
unused, or genuinely absent?* The answer is decided against the **training
window's** own distribution, never the test window's — letting the outcome
define what counts as a warning sign would be circular.

For each of the {s['informative_features']} features whose tail carries at least
{MIN_PREVALENCE_LIFT}x the base distress rate in training, the adverse direction and its
{TAIL_Q:.0%} cut are recorded in `adverse_profile.json`. A miss is then:

| verdict | n | meaning |
|---|---:|---|
| SIGNAL ABSENT | {absent} | nothing available was in a historically adverse tail — a limit of the data |
| SIGNAL UNUSED | {unused} | something was, and the model scored through it — a limit of the model |
| DATA MISSING | {missing} | over half the features null; the firm was unscoreable in substance |

`DATA MISSING` is kept separate deliberately. Folding it into `SIGNAL ABSENT`
would credit the data with an absence it never had the chance to demonstrate,
and would quietly convert the model's refusal to guess into the model being
wrong.

### The SIGNAL UNUSED cases — the ones that are the model's fault

{unused_block}

Full review: `false_negative_review.csv`.

## 3. Calibration by decile

Ranking can be excellent while the probability is not a probability, and the
§8.4 cost-optimal threshold is applied to the **probability**, not to the rank.
A systematic gap here would mean the threshold is cutting at a different real
risk level than it claims to.

{_md_table(cal)}

`gap` is observed minus predicted; positive means the model is under-confident
in that decile.

{cal_verdict}

## 4. What this does not do

This is a backtest on a held-out window, not a production outcomes loop. The
test window is the most recent data in the panel, so there is no later cohort
to score and then wait on. A real §14.4 loop runs annually as each fiscal
year's outcomes resolve; this is the first iteration of it, and the harness it
will run in.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    run(a.matrix, a.panel, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
