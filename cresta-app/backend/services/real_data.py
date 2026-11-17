"""
real_data.py — single source of truth for the real (Bloomberg-derived) artefacts
=================================================================================
Everything the API serves about companies, features and scores comes through
here. Loaded once at startup and cached in module state.

DESIGN NOTES
------------
* SCORED ONCE AT STARTUP. The matrix is ~20k rows x 16 features; scoring it all
  up front costs well under a second and makes the portfolio view O(1) instead
  of re-running the model per request. §12.6 requires the portfolio view
  interactive in under 2s.

* THE MODEL IS LOADED FROM THE REGISTRY, NOT A BARE PICKLE. The artefact is a
  payload dict carrying the model, its exact feature order, its validated
  threshold and its version — so the API can never silently serve a model whose
  feature order has drifted from what it was fitted on.

* FINANCIAL-SECTOR ISSUERS ARE A DISTINCT STATE, NOT A MISSING ROW. They were
  excluded from training (§5.6), so "no data" and "out of scope" are different
  answers and the API must not conflate them. `sector_unsupported()` answers
  from the exclusion log written at matrix-build time.

* COMPANY NAMES COME FROM THE UNIVERSE FILE. The panel carries no
  `company_name` column at all, so every name is joined from universe_800.csv
  and may be absent — the API returns the ticker rather than inventing a name.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

import config

_state: dict = {}


# ── loading ───────────────────────────────────────────────────────────
def _load_universe() -> pd.DataFrame:
    u = pd.read_csv(config.UNIVERSE_PATH)
    keep = [c for c in ["ticker", "company_name", "gics_sector", "mkt_cap"]
            if c in u.columns]
    return u[keep].dropna(subset=["ticker"]).drop_duplicates("ticker")


def _load_excluded() -> pd.DataFrame:
    if not config.EXCLUDED_FINANCIALS_PATH.exists():
        return pd.DataFrame(columns=["ticker", "exclusion_reason"])
    return pd.read_csv(config.EXCLUDED_FINANCIALS_PATH)


QC_FLAG_LABELS = [
    ("qc_identity_fail", "balance-sheet identity off by >2%"),
    ("qc_sign_fail", "sign check failed"),
    ("qc_range_outlier", "value outside the expected range"),
    ("qc_below_coverage", "below the field-coverage threshold"),
    ("qc_scale_suspect", "possible units/scaling error"),
    ("altman_denominator_degenerate", "Altman denominator near zero"),
    ("altman_flag_negative_equity", "negative book equity"),
]


def _build_flag_index(panel: pd.DataFrame) -> dict:
    """{(ticker, year): [flag, ...]} for the rows that have any flag.

    Only flagged rows are stored — the overwhelming majority are clean, and a
    dict holding 33k empty lists costs memory to say nothing.
    """
    present = [(c, label) for c, label in QC_FLAG_LABELS if c in panel.columns]
    if not present:
        return {}
    cols = [c for c, _ in present]
    mask = panel[cols].fillna(False).astype(bool)
    any_flag = mask.any(axis=1)
    if not any_flag.any():
        return {}
    sub = panel.loc[any_flag, ["ticker", "fiscal_year_num"]]
    msub = mask.loc[any_flag]
    labels = [label for _, label in present]
    return {
        (t, int(y)): [labels[i] for i in range(len(labels)) if row[i]]
        for t, y, row in zip(sub["ticker"], sub["fiscal_year_num"],
                             msub.to_numpy())
    }


def warm_explainer():
    """Build TreeSHAP at startup. Measured at 2.46s on first use otherwise,
    which an analyst would pay on their first click into a company."""
    import shap
    s = state()
    s["explainer"] = shap.TreeExplainer(s["model"])
    return s["explainer"]


def load_all(verbose: bool = True) -> dict:
    """Load panel, matrix, universe, model; score every row. Idempotent."""
    if _state:
        return _state

    import pickle
    with open(config.MODEL_A_PATH, "rb") as fh:
        payload = pickle.load(fh)
    model, feats = payload["model"], payload["features"]

    matrix = pd.read_csv(config.MATRIX_PATH, low_memory=False)
    panel = pd.read_csv(config.PANEL_PATH, low_memory=False)
    universe = _load_universe()
    excluded = _load_excluded()

    missing = [f for f in feats if f not in matrix.columns]
    if missing:
        raise RuntimeError(
            f"matrix is missing {len(missing)} feature(s) the model was fitted "
            f"on: {missing}. Refusing to serve — a reindexed frame would be "
            f"scored against the wrong columns.")

    # Score every row in the model's exact training feature order.
    matrix["probability"] = model.predict_proba(matrix[feats])[:, 1]

    # THE EVALUATION POPULATION IS NARROWER THAN THE SERVED POPULATION.
    # Training applies the coverage gate (`qc_below_coverage`); scoring does
    # not. Both are deliberate: a thin-coverage company still gets an estimate,
    # marked, rather than silently vanishing from the portfolio (invariant 4) —
    # but it must NOT enter a reported metric, or the dashboard will disagree
    # with the model card. 4,525 test rows are served; 4,340 are evaluated.
    # Endpoints that report performance filter on `is_evaluation_row`.
    matrix["is_evaluation_row"] = matrix.get("qc_below_coverage") != True

    # `label_now` — the §7.5 label at FY(t), used to split the early-warning
    # stratum. Derived with the SAME function the training pipeline uses, not
    # re-derived by shifting the forward label: the two disagree at year
    # boundaries, and a dashboard that reports a stratum one row different
    # from the model card is a dashboard nobody can reconcile.
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _src = _Path(__file__).resolve().parents[3] / "src"
        if str(_src) not in _sys.path:
            _sys.path.insert(0, str(_src))
        from fallback_label import build_fallback_labels
        _p = panel.copy()
        if "company_name" not in _p.columns:
            _p["company_name"] = ""
        _lab = build_fallback_labels(_p, verbose=False)[
            ["ticker", "fiscal_year_num", "label_binary"]]
        matrix = matrix.merge(_lab.rename(columns={"label_binary": "label_now"}),
                              on=["ticker", "fiscal_year_num"], how="left")
    except Exception as exc:                      # pragma: no cover
        print(f"[real_data] WARNING: could not derive label_now ({exc}). "
              f"Stratified endpoints will fall back to shifting the forward "
              f"label, which may differ from the model card by a row or two.")

    matrix = matrix.merge(universe, on="ticker", how="left",
                          suffixes=("", "_universe"))
    if "company_name_universe" in matrix.columns:
        matrix["company_name"] = matrix["company_name"].fillna(
            matrix.pop("company_name_universe"))

    # ── O(1) lookup indexes ───────────────────────────────────────────
    # Built once at load because the per-row alternative is quadratic: the
    # portfolio view asks for flags on up to 800 companies, and each lookup
    # was scanning all 32,973 panel rows with a STRING comparison on ticker.
    # Measured at 2,387 ms for 800 rows, over the §12.6 two-second budget.
    name_by_ticker, sector_by_ticker = {}, {}
    if "company_name" in universe.columns:
        name_by_ticker = {t: n for t, n in
                          zip(universe["ticker"], universe["company_name"])
                          if pd.notna(n)}
    if "gics_sector" in universe.columns:
        sector_by_ticker = {t: g for t, g in
                            zip(universe["ticker"], universe["gics_sector"])
                            if pd.notna(g)}

    _state.update({
        "name_by_ticker": name_by_ticker,
        "sector_by_ticker": sector_by_ticker,
        "flags_by_key": _build_flag_index(panel),
        "exclusion_reason_by_ticker": (
            dict(zip(excluded["ticker"], excluded["exclusion_reason"]))
            if len(excluded) else {}),
        "model": model, "features": feats, "payload": payload,
        "version": payload.get("version", config.MODEL_A_VERSION),
        "threshold": float(payload.get("threshold", 0.5)),
        "matrix": matrix, "panel": panel, "universe": universe,
        "excluded_tickers": set(excluded["ticker"].unique()),
        "excluded": excluded,
    })

    if verbose:
        print(f"[real_data] model {_state['version']}, {len(feats)} features, "
              f"operating threshold {_state['threshold']:.3f}")
        n_eval = int(matrix["is_evaluation_row"].sum())
        print(f"[real_data] matrix {len(matrix):,} rows / "
              f"{matrix.ticker.nunique():,} tickers scored "
              f"({n_eval:,} in the evaluation population, "
              f"{len(matrix) - n_eval:,} below the coverage gate)")
        print(f"[real_data] {len(_state['excluded_tickers']):,} financial-sector "
              f"tickers marked SECTOR_UNSUPPORTED")
        print(f"[real_data] quality-flag index: "
              f"{len(_state['flags_by_key']):,} flagged company-years")
    return _state


def state() -> dict:
    return _state or load_all()


# ── scope ─────────────────────────────────────────────────────────────
def sector_unsupported(ticker: str) -> bool:
    """True when the ticker was excluded from the model under §5.6."""
    return ticker in state()["excluded_tickers"]


def exclusion_reason(ticker: str) -> str | None:
    r = state()["exclusion_reason_by_ticker"].get(ticker)
    return str(r) if r is not None else None


# ── risk banding, anchored to the model's own threshold ───────────────
def risk_bands() -> dict:
    """Band edges derived from the operating threshold, not round numbers.

    A fixed 0.30/0.50/0.70 scheme would call a company "low risk" at p=0.29
    while the model, whose cost-optimal cut is 0.200, flags it. Bands that
    contradict the flag shown beside them are a UI that argues with itself.
    """
    t = state()["threshold"]
    return {"low": t / 2, "moderate": t, "high": t + (1 - t) / 2}


def classify_risk(p: float) -> str:
    b = risk_bands()
    if p < b["low"]:
        return "low"
    if p < b["moderate"]:
        return "moderate"
    if p < b["high"]:
        return "high"
    return "critical"


# ── lookups ───────────────────────────────────────────────────────────
def company_row(ticker: str, year: int | None = None):
    m = state()["matrix"]
    rows = m[m["ticker"] == ticker]
    if rows.empty:
        return None
    if year is not None:
        rows = rows[rows["fiscal_year_num"] == year]
        if rows.empty:
            return None
        return rows.iloc[0]
    return rows.sort_values("fiscal_year_num").iloc[-1]


def company_history(ticker: str) -> pd.DataFrame:
    m = state()["matrix"]
    return m[m["ticker"] == ticker].sort_values("fiscal_year_num")


def display_name(ticker: str) -> str:
    # Falls back to the ticker — never invents a name.
    return state()["name_by_ticker"].get(ticker, ticker)


def sector_of(ticker: str) -> str | None:
    return state()["sector_by_ticker"].get(ticker)


def predict_row(row) -> float:
    s = state()
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in s["features"]}],
                     columns=s["features"]).apply(pd.to_numeric, errors="coerce")
    return float(s["model"].predict_proba(X)[:, 1][0])


def data_quality_flags(ticker: str, year: int) -> list[str]:
    """Flags §12.3 requires beside every prediction. Absence of a ratio is
    information, not a blank — invariant 1. O(1) via the startup index."""
    return state()["flags_by_key"].get((ticker, int(year)), [])


@lru_cache(maxsize=1)
def registry() -> list:
    if config.REGISTRY_PATH.exists():
        return json.loads(config.REGISTRY_PATH.read_text())
    return []
