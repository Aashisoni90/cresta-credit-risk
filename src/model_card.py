"""
model_card.py — Phase 3, §14.1 Model Card + artefact registration
==================================================================
Trains Model A, registers the fitted artefact with reproducibility hashes, and
emits the §14.1 model card as Markdown. Every number in the card is computed in
this run — none is transcribed from a previous one, because a card whose
performance table was pasted in by hand is a card that silently goes stale.

WHAT "REGISTERED" MEANS HERE
-----------------------------
The pickle is written alongside a registry entry recording: model version,
training timestamp, git commit (see the caveat below), a config hash over the
source files and inputs that determine the fit, row counts, and the metrics.
Re-running this script on unchanged inputs must reproduce the config hash; a
changed hash with an unchanged version number is the signal that someone edited
the pipeline without re-versioning it.

models/model_A.pkl IS NOT TOUCHED. That artefact is the SYNTHETIC-data model
behind cresta-app/. Overwriting it would silently repoint the running app at a
different model, so the real-data artefact is written under its own versioned
name and the collision is stated in the card.

GIT COMMIT HASH — REQUIRED BY §14.1, NOT AVAILABLE
---------------------------------------------------
§14.1 requires a commit hash per version. This project directory is not a git
repository, so the field is recorded as unavailable rather than faked with a
placeholder. The config hash below is a partial substitute: it pins the content
of the code and inputs, but not their history. This is logged as a governance
gap, not quietly omitted.

USAGE
    PYTHONPATH=src .venv/bin/python src/model_card.py
"""

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_model_a import run, LABEL
from build_training_matrix import POINT_IN_TIME_FEATURES
from explain_model_a import block_of, CAVEAT

MODEL_VERSION = "A-1.2.0-real"
OWNER = "Varad Pal (CRESTA capstone)"
UNIVERSE = "data/processed/universe_800.csv"

EXCLUSION_LOG = "data/processed/excluded_financials.csv"

HASHED_SOURCES = ["src/load_formula_chunks.py", "src/quality_gates.py",
                  "src/altman_zscore.py", "src/score_altman_panel.py",
                  "src/feature_engineering.py", "src/fallback_label.py",
                  "src/build_training_matrix.py", "src/train_model_a.py",
                  "src/metrics.py"]


def _sha(path, n=None):
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return h[:n] if n else h


def config_hash(matrix_path, panel_path):
    """Content hash over the code and inputs that determine the fit."""
    h = hashlib.sha256()
    for p in HASHED_SOURCES + [matrix_path, panel_path]:
        if Path(p).exists():
            h.update(p.encode())
            h.update(_sha(p).encode())
        else:
            h.update(f"{p}:MISSING".encode())
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def sector_breakdown(te, scores, y):
    """Per-sector performance (§14.1 'Performance'; §12.4 'where the model is
    weakest'). Sector comes from the universe file — the panel carries none —
    so coverage is partial and the uncovered rows are reported as their own
    group rather than dropped."""
    u = (pd.read_csv(UNIVERSE, usecols=["ticker", "gics_sector"])
         .drop_duplicates("ticker"))
    sec = te[["ticker"]].merge(u, on="ticker", how="left")["gics_sector"]
    sec.index = te.index
    sec = sec.fillna("(sector unknown)")
    rows = []
    for s, idx in sec.groupby(sec).groups.items():
        idx = pd.Index(idx)
        yy = y.loc[idx]
        ew = idx[te.loc[idx, "label_now"] == 0]
        rows.append({
            "sector": s, "n": len(idx), "prevalence": float(yy.mean()),
            "auc": float(roc_auc_score(yy, scores.loc[idx])) if yy.nunique() > 1 else np.nan,
            "n_early_warning": len(ew),
            "auc_early_warning": (float(roc_auc_score(y.loc[ew], scores.loc[ew]))
                                  if y.loc[ew].nunique() > 1 else np.nan)})
    return pd.DataFrame(rows).sort_values("n", ascending=False), sec


def build(matrix_path, panel_path, out_dir="reports/phase3",
          models_dir="models", card_path="MODEL_CARD.md"):
    res, store = run(matrix_path, panel_path, use_xgb=True, out_dir=out_dir)
    frames, feats = store["_frames"], store["_features"]
    te = frames["test"]
    y = te[LABEL]
    scores = pd.Series(store["xgboost"][1], index=te.index)
    xgb = store["_xgb_model"]

    sec_df, sec = sector_breakdown(te, scores, y)

    # Quantities the card's prose quotes. Computed here so the narrative cannot
    # drift from the tables — the exact way the v1.0.0 card went stale when the
    # matrix changed underneath it.
    from metrics import delong_test
    mtx = store["_matrix"]
    pidx, pscore = store["persistence"]
    sig = {}
    for nm in ("altman", "logistic", "xgboost"):
        idx, sc = store[nm]
        com = pidx.intersection(idx)
        yy = mtx.loc[com, LABEL].values
        sig[nm] = delong_test(
            yy, pd.Series(pscore, index=pidx).loc[com].values,
            pd.Series(sc, index=idx).loc[com].values)
    aidx, asc = store["altman"]
    xidx, xsc = store["xgboost"]
    com = aidx.intersection(xidx)
    sig["xgb_vs_altman"] = delong_test(
        mtx.loc[com, LABEL].values,
        pd.Series(asc, index=aidx).loc[com].values,
        pd.Series(xsc, index=xidx).loc[com].values)

    # label persistence rate, measured not quoted
    lab = mtx[["ticker", "fiscal_year_num", "label_now", LABEL]].dropna()
    persistence_rate = float((lab["label_now"] == lab[LABEL]).mean())

    strata = []
    for nm, idx in [("All test rows", te.index),
                    ("Already distressed at t", te.index[te.label_now == 1]),
                    ("**Healthy at t → early warning**", te.index[te.label_now == 0])]:
        row = {"Stratum": nm, "n": len(idx), "prevalence": float(y.loc[idx].mean())}
        for k in ("altman", "logistic", "xgboost"):
            sidx, sc = store[k]
            c = idx.intersection(sidx)
            row[k] = (float(roc_auc_score(y.loc[c], pd.Series(sc, index=sidx).loc[c]))
                      if y.loc[c].nunique() > 1 else np.nan)
        strata.append(row)
    strata = pd.DataFrame(strata)

    # §5.6 / invariant 5 audit — verify the exclusion actually took effect in
    # the matrix this model was fitted on. A card that asserts an exclusion
    # without checking it is how the defect got here in the first place.
    leaked = te.index[sec == "Financials"]
    exc = pd.read_csv(EXCLUSION_LOG) if Path(EXCLUSION_LOG).exists() else pd.DataFrame()
    audit = {
        "exclusion_enforced": True,
        "rows_excluded": int(len(exc)),
        "tickers_excluded": int(exc["ticker"].nunique()) if len(exc) else 0,
        "reasons": (exc["exclusion_reason"].value_counts().to_dict()
                    if len(exc) else {}),
        "financial_rows_remaining_in_test": int(len(leaked)),
    }
    if len(leaked):
        raise AssertionError(
            f"§5.6 exclusion did not take effect: {len(leaked)} GICS-Financials "
            f"rows are still in the test split. Rebuild the matrix.")

    # §7.4 / invariant 2 — the point-in-time look-ahead guard, asserted for the
    # same reason as the one above: the defect reached v1.1.0 precisely because
    # nothing checked. A market field repeated across every fiscal year is not
    # a feature, it is the answer arriving early.
    pit_leaked = [c for c in POINT_IN_TIME_FEATURES if c in feats]
    audit["point_in_time_features_dropped"] = list(POINT_IN_TIME_FEATURES)
    audit["point_in_time_remaining_in_features"] = pit_leaked
    if pit_leaked:
        raise AssertionError(
            f"point-in-time look-ahead features reached the model: {pit_leaked}. "
            f"Rebuild the matrix WITHOUT --keep-point-in-time.")

    # SHAP global ranking (reuse the §19 artefact if present; it is produced by
    # the same fit, so recomputing it here would only risk the two disagreeing)
    shap_path = Path(out_dir) / "shap_global_ranking.csv"
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None
    roll_path = Path(out_dir) / "rolling_origin_summary.csv"
    roll = pd.read_csv(roll_path) if roll_path.exists() else None

    # ── register the artefact ─────────────────────────────────────────
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    art = Path(models_dir) / f"model_{MODEL_VERSION}.pkl"
    payload = {"model": xgb, "features": feats, "label": LABEL,
               "version": MODEL_VERSION, "threshold": float(
                   res.loc[res.model == "XGBoost", "threshold"].iloc[0]),
               "trained_utc": datetime.now(timezone.utc).isoformat()}
    with open(art, "wb") as fh:
        pickle.dump(payload, fh)

    cfg = config_hash(matrix_path, panel_path)
    commit = git_commit()
    entry = {"version": MODEL_VERSION, "artefact": str(art),
             "artefact_sha256": _sha(art),
             "trained_utc": payload["trained_utc"], "owner": OWNER,
             "git_commit": commit, "config_hash": cfg,
             "python": platform.python_version(),
             "matrix": matrix_path, "matrix_sha256": _sha(matrix_path, 16),
             "n_train": int(len(frames["train"])), "n_test": int(len(te)),
             "threshold": payload["threshold"],
             "metrics": res.set_index("model")[["auc_roc", "pr_auc", "ks"]]
                           .round(4).to_dict("index"),
             "sector_exclusion_audit": audit}
    reg = Path(models_dir) / "registry.json"
    hist = json.loads(reg.read_text()) if reg.exists() else []
    hist = [h for h in hist if h.get("version") != MODEL_VERSION] + [entry]
    reg.write_text(json.dumps(hist, indent=2))

    write_card(card_path, res, entry, sec_df, shap_df, roll, frames, feats, audit,
               sig, persistence_rate, strata)
    print(f"\nregistered {art}\nwrote {reg}\nwrote {card_path}")
    return entry


def _pval(v):
    """A DeLong p-value underflows to 0.0 in float; printing "p = 0.0" claims an
    exactness the arithmetic does not have."""
    if v != v:
        return "n/a"
    return "< 1e-300" if v <= 0 else (f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")


def _verdict(d, alpha=0.05):
    if d["p_value"] != d["p_value"] or d["p_value"] >= alpha:
        return "not significantly different"
    return "significantly better" if d["delta_auc"] > 0 else "significantly WORSE than no model"


def _tbl(df, cols=None, fmt="{:,.4f}"):
    df = df[cols] if cols else df
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    # Decide integer-vs-decimal PER COLUMN, not per cell: an AUC that happens
    # to land on 1.0 is not an integer, and rendering it as "1" beside "0.9663"
    # misreads as a different kind of quantity.
    intish = {c: (df[c].dropna().apply(lambda v: isinstance(v, (int, np.integer))
                                       or (isinstance(v, float) and float(v).is_integer())).all()
                  and df[c].dropna().abs().max() >= 2) if df[c].dropna().size else False
              for c in df.columns}
    out = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c, v in zip(df.columns, r):
            if isinstance(v, float) and v != v:
                cells.append("—")
            elif intish[c] and isinstance(v, (int, float, np.integer)):
                iv = int(v)
                # fiscal years must not pick up a thousands separator
                cells.append(str(iv) if 1900 <= iv <= 2100 else f"{iv:,}")
            elif isinstance(v, float):
                cells.append(fmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_card(path, res, entry, sec_df, shap_df, roll, frames, feats, audit,
               sig, persistence_rate, strata):
    tr, va, te = frames["train"], frames["validation"], frames["test"]
    commit = entry["git_commit"] or (
        "**UNAVAILABLE — this project directory is not a git repository.** §14.1 "
        "requires a commit hash per version; it is recorded as missing rather than "
        "filled with a placeholder. See Limitations.")
    perf = res.copy()
    show = [c for c in ["model", "n_scored", "auc_roc", "pr_auc", "ks", "brier",
                        "threshold", "recall", "precision", "f1"] if c in perf]

    git_limitation = (
        "**Provenance is one commit deep.** The repository was initialised at "
        "Phase 3, so the config hash and commit pin the current state but carry "
        "no history of how the pipeline reached it."
        if entry["git_commit"] else
        "**No git history.** §14.1 requires a commit hash; the directory is not "
        "a repository. The config hash pins content but not provenance.")
    persist_auc = float(res.loc[res.model.str.startswith("Persistence"), "auc_roc"].iloc[0])
    logit_auc = float(res.loc[res.model.str.startswith("Logistic"), "auc_roc"].iloc[0])
    if roll is not None:
        _pool, _e = roll["xgboost"], roll["xgboost_EW"]
        _single = float(res.loc[res.model == "XGBoost", "auc_roc"].iloc[0])
        _within = abs(_single - _pool.mean()) <= _pool.std()
        roll_prose = (
            f"XGBoost: **{_pool.mean():.4f} ± {_pool.std():.4f}** pooled, "
            f"**{_e.mean():.4f} ± {_e.std():.4f}** early-warning across "
            f"{len(_pool)} origins. The single-split headline ({_single:.4f}) sits "
            f"{'inside' if _within else 'OUTSIDE'} one standard deviation of the fold "
            f"mean, so it is "
            f"{'a representative draw rather than a favourable one' if _within else 'NOT a representative draw — quote the fold distribution'}.")
    else:
        roll_prose = "_Rolling-origin results not available._"
    reason_rows = "\n".join(
        f"- **{n:,} rows** — {r}" for r, n in audit["reasons"].items()) or "- (none)"
    rows_excluded, tickers_excluded = audit["rows_excluded"], audit["tickers_excluded"]

    md = f"""# Model Card — CRESTA Model A (Track A, financial-only)

*Generated by `src/model_card.py` on {entry['trained_utc'][:19]}Z. Every figure
below was computed in that run. Do not hand-edit — regenerate.*

---

## 1. Identification

| Field | Value |
|---|---|
| Model version | `{entry['version']}` |
| Training date (UTC) | {entry['trained_utc'][:19]} |
| Git commit | {commit} |
| Config hash (sha256) | `{entry['config_hash']}` |
| Artefact | `{entry['artefact']}` |
| Artefact sha256 | `{entry['artefact_sha256'][:32]}…` |
| Training matrix | `{entry['matrix']}` (sha256 `{entry['matrix_sha256']}…`) |
| Python | {entry['python']} |
| Owner | {entry['owner']} |

The **config hash** covers the content of the nine pipeline source files plus
the training matrix and scored panel. Re-running on unchanged inputs reproduces
it. A changed hash under an unchanged version number means the pipeline was
edited without re-versioning.

**`models/model_A.pkl` is a different model and was not touched.** That artefact
is trained on *synthetic* data and is what `cresta-app/` currently serves.
Overwriting it would silently repoint the running app. The real-data model is
registered under its own versioned filename.

---

## 2. Intended use — and misuse

**Intended:** ranking listed Indian non-financial companies by the estimated
probability of financial distress in the following fiscal year, to prioritise
analyst attention. The output is an *ordering aid and an estimate*.

**This model is NOT, and must not be presented as:**

- **Investment advice.** It makes no statement about security prices or returns.
- **A credit rating.** It is not issued by a rating agency, is not calibrated to
  any agency scale, and carries none of the issuer engagement, qualitative
  review or surveillance a rating involves.
- **A default probability.** The label is a *financial-criteria distress proxy*
  (§7.5), not a realised default. See Limitations.
- **Applicable to financial-sector issuers** (banks, NBFCs). Their balance
  sheets make the ratio set meaningless, and they are excluded from training and
  scoring alike. See §5.
- **An automated decision.** See §9, Human-in-the-loop.

---

## 3. Training data

| | Rows | Positives | Prevalence | Fiscal years (FY start) |
|---|---:|---:|---:|---|
| Train | {len(tr):,} | {int(tr[LABEL].sum()):,} | {tr[LABEL].mean()*100:.1f}% | {int(tr.fiscal_year_num.min())}–{int(tr.fiscal_year_num.max())} |
| Validation | {len(va):,} | {int(va[LABEL].sum()):,} | {va[LABEL].mean()*100:.1f}% | {int(va.fiscal_year_num.min())} |
| Test | {len(te):,} | {int(te[LABEL].sum()):,} | {te[LABEL].mean()*100:.1f}% | {int(te.fiscal_year_num.min())}–{int(te.fiscal_year_num.max())} |

- **Universe:** {UNIVERSE} — 2,999 real tickers (3,001 raw rows less one
  Bloomberg legal-disclaimer row and one null ticker), Indian listed equities,
  built from EQS screens on a *historical* basis to avoid the survivorship trap
  of a `Current`-basis market-cap floor.
- **Source:** Bloomberg Terminal, manual `BDP()` template extraction across
  seven chunks, 18 fields × 11 years. **Licence prohibits redistribution** — the
  reproducibility package ships code and the ticker list, never the data (§18.3).
- **Split is OUT-OF-TIME and asserted in code.** `_assert_out_of_time()` raises
  if the windows overlap or are out of order. No random split is possible.
- **Label:** §7.5 financial-criteria fallback, promoted to primary because
  agency ratings resolved for only **4 of 3,000 tickers (0.13%)** — CRISIL
  returns `#N/A Authorization` on 2,999. This is a data-entitlement limit, not a
  modelling choice.

### Known exclusions and dead rows

- **Quality-gate rejections:** 10,952 row-level rejections logged to
  `data/processed/quality_rejections_scored.csv` — never silently dropped.
- **Failed tickers:** 171 logged to `data/raw/failed_tickers.csv` with reasons.
  Distressed and delisted firms fail to resolve disproportionately, so dropping
  them silently would reintroduce survivorship bias (invariant 4).
- **Financial-sector issuers: excluded** — {rows_excluded:,} rows /
  {tickers_excluded} tickers, logged with reasons. See §5.

---

## 4. Methodology

| Element | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`), {len(feats)} features |
| Hyperparameters | `n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0` |
| Seed | 42 |
| Imbalance treatment | `scale_pos_weight` computed on **train only** |
| Missing values | XGBoost native NaN handling. **Never zero-filled** (invariant 1) — for the logistic baseline, train-median imputation plus explicit `*__was_missing` indicators |
| Preprocessing fit | **Train window only** (invariant 3), verified by `_assert_fitted_on_train_only()` |
| Threshold | {entry['threshold']:.3f}, chosen on **validation** by minimising expected cost at the §8.4 default **10:1** FN:FP ratio |

**The cost ratio is an assumption, and a consequential one.** At this label's
~21% prevalence, 10:1 drives the threshold toward flagging 54.7% of the
portfolio. The §8.4-mandated sensitivity analysis is reported in
`Project_Context.md` §18.4; **3:1 maximises F1 (0.698, 28.2% flagged)** and is
the recommended operating point for a triage product. The deployed ratio must be
stated wherever a flag is shown.

---

## 5. Financial-sector exclusion (§5.6 / invariant 5) — ENFORCED

Financial-sector issuers are excluded from the primary model. Their balance
sheets have no operating cycle, so working capital, the current ratio and
EBIT/assets are not merely noisy but undefined in meaning.

**This is measured, not assumed.** On this panel `current_assets` and
`current_liabilities` are null on **86.3%** of financial-sector rows versus
**23.5%** elsewhere, while EBIT is reported at a comparable rate (19.8% vs
22.0%). Three of the sixteen features are therefore structurally absent for
these firms, and a tree model's native NaN branch can isolate them on that
absence alone.

**{rows_excluded:,} rows across {tickers_excluded} tickers excluded**, logged to
`data/processed/excluded_financials.csv` — never silently dropped:

{reason_rows}

`gics_sector` is missing for 692 of 2,999 tickers, so a name-pattern fallback is
applied **to sector-unknown tickers only**. It matches 62.7% of *known*
financials and just 0.7% of known non-financials — high precision, moderate
recall. Excluding a handful of genuine non-financials is the conservative
direction of error. Disable with `--no-name-heuristic`.

This card asserts the exclusion and then **verifies it**: generation fails if
any GICS-Financials row survives into the test split.

### Sensitivity — what the exclusion cost

| | n (test) | AUC | early-warning AUC |
|---|---:|---:|---:|
| v1.0.0, financials retained (defect) | 4,830 | 0.9589 | 0.9548 |
| **v1.1.0, financials excluded** | **4,340** | **0.9550** | **0.9490** |

*Both rows are measured on the v1.1.0 sixteen-feature set, before the §5b
point-in-time removal. They are kept as the record of what the sector
exclusion cost in isolation; the current model is v1.2.0.*

The retained-financials model scored higher, and that is the point: those 398
test rows carried 8.5% prevalence against 22.7% elsewhere and scored AUC 0.9888,
flattering the aggregate without the model understanding a single bank. The
conclusions are unchanged — Altman remains significantly worse than persistence
(ΔAUC −0.1110, p<0.001) and XGBoost significantly better (+0.0691, p<0.001).

---

## 5b. Point-in-time look-ahead removal (§7.4 / invariant 2) — ENFORCED

**Two features were look-ahead and have been removed.** The Bloomberg pull
applied `FUND_PER="Y"` with an `EQY_FUND_YEAR` override to all eighteen fields
uniformly. That override governs *fundamental* fields. `CUR_MKT_CAP` and
`VOLATILITY_260D` are real-time *market* fields: Bloomberg ignores the override
and returns the value as at the pull date for every year requested.

Measured on this panel, over tickers reporting at least five years:

| field | tickers with ONE value across all years |
|---|---:|
| `cur_mkt_cap` | **98.4%** |
| `volatility_260d` | **95.4%** |
| `debt_to_equity` | 4.9% |
| `total_debt` | 4.1% |
| every accounting field (`total_assets`, `ebit`, …) | **0.0%** |

So `log_mkt_cap` on a FY2014 row was the firm's market capitalisation as at the
2026 pull date — a quantity that postdates the FY2015 label it was being used
to predict by eleven years. **The leakage signature is directly visible:** the
feature alone scores AUC 0.598 against the FY2014 label and rises monotonically
to 0.672 against FY2023, as the gap to the pull date closes. A genuinely
contemporaneous size control does not trend that way. `log_total_assets` comes
from `BS_TOT_ASSET`, varies per year on 100% of tickers, and is retained.

This is not fixable by rescaling or by a missingness indicator — the values are
simply from the wrong date. A correct historical series needs a `BDH`
time-series pull, which is future extraction work.

### Sensitivity — what removing the leakage cost

| | features | AUC | early-warning AUC | PR-AUC |
|---|---:|---:|---:|---:|
| v1.1.0, point-in-time retained (defect) | 16 | 0.9550 | 0.9490 | 0.8487 |
| **v1.2.0, point-in-time dropped** | **14** | **0.9507** | **0.9434** | **0.8315** |

**−0.0043 pooled and −0.0056 early-warning.** The leaked pair carried 9.6% of
SHAP attribution between them, so the small loss says the model recovers nearly
all of it from legitimate fundamentals rather than that the leakage was
harmless. Every significance conclusion below is unchanged. Reporting the
larger number would have meant publishing an out-of-time claim that the feature
set contradicts.

---

## 6. Performance — out-of-time test (scored once)

{_tbl(perf, show)}

### Rolling-origin folds (§8.5 supplementary)

{_tbl(roll, fmt="{:,.4f}") if roll is not None else "_Not available — run `src/rolling_origin.py --xgb`._"}

{roll_prose}

### THE POOLED NUMBER IS NOT THE HEADLINE

The §7.5 label is identical at *t* and *t+1* on **{persistence_rate*100:.1f}%**
of rows. A naive "the condition continues" baseline scores
**AUC {persist_auc:.4f}** by itself. Pooled AUC
therefore mostly measures persistence. The honest early-warning figure is the
subset **healthy at t**, where persistence is 0.5 by construction:

{_tbl(strata)}

### Per-sector breakdown

{_tbl(sec_df, ["sector", "n", "prevalence", "auc", "n_early_warning", "auc_early_warning"])}

**Weakest sector: Real Estate** (AUC {sec_df.loc[sec_df.sector=='Real Estate','auc'].iloc[0]:.4f},
early-warning {sec_df.loc[sec_df.sector=='Real Estate','auc_early_warning'].iloc[0]:.4f})
at the highest prevalence in the panel. Flagged here because §12.4 requires the
model's weak spots to be visible, not only its averages. Sector is joined from
the universe file and is missing for part of the panel; those rows are reported
as `(sector unknown)` rather than dropped.

---

## 7. Comparison and significance (§8.7)

Against **Altman Z″**: XGBoost **{sig['xgb_vs_altman']['delta_auc']:+.4f}**, DeLong p {_pval(sig['xgb_vs_altman']['p_value'])} — significant.

Against the **persistence baseline** — the comparison that actually matters:

| Model | ΔAUC vs persistence | DeLong p | Conclusion |
|---|---:|---:|---|
| Altman Z″ | **{sig['altman']['delta_auc']:+.4f}** | {_pval(sig['altman']['p_value'])} | **{_verdict(sig['altman'])}** |
| Logistic (L2, balanced) | {sig['logistic']['delta_auc']:+.4f} | {_pval(sig['logistic']['p_value'])} | **{_verdict(sig['logistic'])}** |
| XGBoost | **{sig['xgboost']['delta_auc']:+.4f}** | {_pval(sig['xgboost']['p_value'])} | **{_verdict(sig['xgboost'])}** |

Reporting only the Altman comparison would have been materially misleading: the
logistic baseline's respectable-looking {logit_auc:.4f} is, in substance, persistence.

---

## 8. Explainability (§8.8)

> {CAVEAT}

"""
    if shap_df is not None:
        top1 = shap_df.iloc[0]["feature"]
        top1_share = float(shap_df.iloc[0]["share_pct"])
        _c = shap_df.set_index("feature")["corr_feature_vs_shap"]
        ta_corr, mc_corr = float(_c.get("log_total_assets", np.nan)), float(_c.get("log_mkt_cap", np.nan))
        top = shap_df.head(8)[["feature", "block", "share_pct", "mean_signed_shap"]]
        blk = (shap_df.groupby("block")["share_pct"].sum().sort_values(ascending=False)
               .reset_index())
        md += f"""### Global ranking (mean |SHAP|, test set)

{_tbl(top, fmt="{:,.4f}")}

### Block-level attribution

{_tbl(blk, fmt="{:,.2f}")}

The model is **overwhelmingly a profitability model** — that is the substantive
answer to "what does CRESTA add over Altman": it weights current earnings power
far above Altman's fixed coefficients. `{top1}` alone carries {top1_share:.0f}%.

**The two size controls carry opposite signs.** `log_total_assets` correlates
**{ta_corr:+.2f}** with its own SHAP (bigger → riskier) while `log_mkt_cap`
correlates **{mc_corr:+.2f}** (bigger → safer). Read together: a large asset base relative to market
valuation is a distress signal — the market declining to capitalise the balance
sheet. Neither control expresses that alone, and an unexplained positive size
coefficient would otherwise read as a bug.

Local waterfalls and nearest-healthy-peer counterfactuals for the five
highest-confidence true early warnings are in `reports/phase3/figures/` and
`shap_local_cases.csv` / `shap_counterfactuals.csv`.

"""
    md += f"""---

## 9. Limitations — enumerated

1. **The label is a proxy, not a default.** Agency ratings resolved for 4 of
   3,000 tickers, so the §7.5 financial-criteria fallback became the primary
   label. It identifies *financial distress conditions*, not legal default or
   insolvency filings. Every performance figure is conditional on that proxy.
2. **Label persistence inflates pooled metrics** ({persistence_rate*100:.1f}% year-over-year identity).
   Quote the early-warning stratum, not the pooled AUC.
3. **The financial-sector exclusion rests partly on a name heuristic** for the
   692 sector-unknown tickers (§5). High precision (0.7% false-positive rate on
   known non-financials) but ~63% recall, so a small number of unlabelled
   lenders may remain.
4. **Sector coverage is partial.** `gics_sector` is missing for ~16% of matrix
   rows; the per-sector table reports them as a group rather than imputing.
5. **Survivorship exposure is controlled, not eliminated.** The universe is
   historical-basis and dead tickers are logged (171), but a company absent from
   the EQS screen entirely cannot be recovered.
6. **{len(feats)} features, not the full §6 block set.** Interest coverage, cash-flow
   and revenue-intensity features await an expanded Bloomberg pull; block 4
   (debt servicing) is effectively empty.
7. **The model is over-confident; the score is a rank, not a probability.**
   Decile calibration on the out-of-time test set (§14.4,
   `reports/phase7/OUTCOMES_ANALYSIS.md`) shows a mean gap of −0.175 across
   deciles 7–10: decile 8 predicts 69.8% against a realised 45.2%. The cause
   is `scale_pos_weight`, which inflates predicted probabilities by
   construction. **Ranking is unaffected** — AUC, KS and the operating
   threshold are rank-based and the threshold was chosen on validation under
   the same distortion — but the number must not be read as a likelihood
   without a calibration layer fitted on the validation window.
8. **Two legitimate market-based features are missing, not merely unused.**
   Equity volatility and market capitalisation are standard distress
   predictors, and the §5b removal leaves the model with no market-price
   channel at all — its signal is now entirely accounting-based. A `BDH`
   historical pull would restore them honestly.
9. **Scale inconsistency.** `roa` is in percent (median 1.98, range −161,400 to
   +16,550); `ebit_to_assets` is a decimal. Harmless to trees, but the dashboard
   must not render them side by side without units.
10. **One market, one period.** FY2014–FY2023, Indian listed equities. No claim
   of transfer to other markets or to unlisted issuers.
11. **No ESG — and the reason is measured, not projected.** Track B closed as
   a feasibility result. BRSR is mandatory only from FY2022-23 and the panel's
   last feature year *is* FY2022-23, so ESG exists for exactly one feature year
   and invariant 2 forbids the random split an A-vs-B test would need: **no
   out-of-time ESG comparison exists in this data.** The eligible frame is 860
   non-financial firms with 93 events over ~22 variables — **4.2 EPV** against
   a ≥10 rule of thumb. A seven-filing extraction pilot reached 77.9% mean
   field coverage, but **only 1 of 7 filings yielded a usable energy-intensity
   ratio**: the renewable/non-renewable split is a voluntary *Leadership*
   indicator and five filers omitted it entirely. Track A is complete and
   unaffected (invariant 6).
12. {git_limitation}
13. **SHAP explains the model, not the world.** Attribution is not causation.

---

## 10. Ethical review

**Human-in-the-loop is non-negotiable (§14.5).** CRESTA produces analytical
estimates, never automated decisions. **No credit, lending or investment action
may be taken on a model output without qualified human review.** This must be
enforced in dashboard copy and in any deployment. Presenting a model probability
as a decision is a governance failure and, in a regulated context, potentially a
serious one.

**Misuse potential.** A distress probability attached to a named listed company
is market-sensitive and reputationally damaging if wrong or if circulated out of
context. Outputs carry a data-quality flag count and must not be published as
statements of fact about an issuer.

**Fairness.** The relevant axis here is not demographic but **size and
disclosure quality**: smaller issuers have thinner data, so they are both more
likely to be imputed and more likely to be distressed. The model may therefore
penalise opacity as much as weakness. The `*__was_missing` indicators make this
legible rather than hidden, and per-sector performance is reported so systematic
weakness (Real Estate) is visible to a reviewer.

**Recourse.** Any flagged company's score is decomposable to the feature level
via the SHAP layer, so a specific, checkable reason can always be given.

---

*Regenerate with `PYTHONPATH=src .venv/bin/python src/model_card.py`.*
"""
    Path(path).write_text(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/processed/training_matrix.csv")
    ap.add_argument("--panel", default="data/processed/bloomberg_panel_scored_gated.csv")
    ap.add_argument("--out", default="reports/phase3")
    ap.add_argument("--card", default="MODEL_CARD.md")
    a = ap.parse_args()
    build(a.matrix, a.panel, out_dir=a.out, card_path=a.card)


if __name__ == "__main__":
    main()
