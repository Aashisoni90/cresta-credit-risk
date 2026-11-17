"""
company_service.py — company lookups over the REAL panel (Phase 4)
====================================================================
Replaces the synthetic-panel implementation. Same function names so the
existing routers keep working; the data underneath is now the real
Bloomberg-derived matrix, scored by Model A v1.2.0.

WHAT IS DELIBERATELY ABSENT
----------------------------
`e_score` / `s_score` / `g_score` / `carbon_intensity` do not exist in the real
panel and are not synthesised. Every ESG field returns None with Track B's
unavailability stated, rather than a plausible-looking number. Invariant 6:
Track A works with zero ESG data present, always.

`is_distress_event` likewise does not exist. The real label is the §7.5
forward-looking financial-criteria proxy (`label_binary_next`), which is a
different quantity and is named as such.
"""
import numpy as np
import pandas as pd

import config
from services import real_data

LABEL = "label_binary_next"


def _matrix() -> pd.DataFrame:
    return real_data.state()["matrix"]


def _safe_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), 4)


def get_company_list(search: str = None, sector: str = None, page: int = 1,
                     per_page: int = 25):
    m = _matrix()
    latest = (m.sort_values("fiscal_year_num").groupby("ticker").last()
              .reset_index())

    if search:
        s = str(search)
        mask = latest["ticker"].str.contains(s, case=False, na=False)
        if "company_name" in latest.columns:
            mask |= latest["company_name"].astype(str).str.contains(s, case=False, na=False)
        latest = latest[mask]
    if sector and "gics_sector" in latest.columns:
        latest = latest[latest["gics_sector"] == sector]

    latest = latest.sort_values("probability", ascending=False)
    total = len(latest)
    start = (page - 1) * per_page
    page_data = latest.iloc[start:start + per_page]
    thr = real_data.state()["threshold"]

    companies = [{
        "ticker": r["ticker"],
        "name": real_data.display_name(r["ticker"]),
        "sector": r.get("gics_sector") if pd.notna(r.get("gics_sector")) else None,
        "latest_year": int(r["fiscal_year_num"]),
        "probability": _safe_float(r["probability"]),
        "risk_category": real_data.classify_risk(float(r["probability"])),
        "flagged": bool(r["probability"] >= thr),
        "total_assets": _safe_float(r.get("log_total_assets")),
        # ESG is absent, not zero. See module docstring.
        "esg_average": None,
    } for _, r in page_data.iterrows()]

    return {
        "companies": companies, "total": total, "page": page,
        "per_page": per_page, "pages": max(1, (total + per_page - 1) // per_page),
        "esg_available": False,
    }


SNAPSHOT_FIELDS = [
    ("total_assets", "Total assets", "INR mn"),
    ("cur_mkt_cap", "Market cap", "INR mn"),
    ("total_revenue", "Revenue", "INR mn"),
    ("roa", "Return on assets", "%"),
    ("debt_to_equity", "Debt / equity", "x"),
    ("cur_ratio", "Current ratio", "x"),
    ("interest_coverage", "Interest coverage", "x"),
    ("altman_z_double_prime", "Altman Z\u2033", ""),
]


def _snapshot(ticker: str, year: int) -> list[dict]:
    panel = real_data.state()["panel"]
    hit = panel[(panel["ticker"] == ticker) & (panel["fiscal_year_num"] == year)]
    if hit.empty:
        return []
    r = hit.iloc[0]
    return [{"key": k, "label": label, "unit": unit,
             "value": _safe_float(r.get(k))}
            for k, label, unit in SNAPSHOT_FIELDS if k in r.index]


def get_company_detail(ticker: str):
    if real_data.sector_unsupported(ticker):
        return {
            "ticker": ticker,
            "name": real_data.display_name(ticker),
            "sector": real_data.sector_of(ticker),
            "supported": False,
            "code": config.SECTOR_UNSUPPORTED_CODE,
            "message": config.SECTOR_UNSUPPORTED_MESSAGE,
        }
    row = real_data.company_row(ticker)
    if row is None:
        return None
    hist = real_data.company_history(ticker)
    year = int(row["fiscal_year_num"])
    return {
        "ticker": ticker,
        "name": real_data.display_name(ticker),
        "sector": real_data.sector_of(ticker),
        "supported": True,
        "latest_year": year,
        "years_covered": [int(y) for y in hist["fiscal_year_num"]],
        "probability": _safe_float(row["probability"]),
        "risk_category": real_data.classify_risk(float(row["probability"])),
        "label_next": (None if pd.isna(row.get(LABEL))
                       else int(row[LABEL])),
        "label_definition": ("§7.5 forward-looking financial-criteria distress "
                             "proxy at FY(t+1) — not a realised default"),
        "data_quality_flags": real_data.data_quality_flags(ticker, year),
        # Headline financials for the §12.3 snapshot, read from the panel.
        # A value that is genuinely absent stays null — never zero-filled
        # (invariant 1), because a missing ratio is information.
        "snapshot": _snapshot(ticker, year),
        "esg": None,
        "esg_status": "Track B (ESG) unavailable — see /api/model/info",
    }


def get_company_history(ticker: str):
    hist = real_data.company_history(ticker)
    if hist.empty:
        return None
    thr = real_data.state()["threshold"]
    return {
        "ticker": ticker,
        "name": real_data.display_name(ticker),
        "history": [{
            "fiscal_year": int(r["fiscal_year_num"]),
            "fiscal_year_label": f"FY{int(r['fiscal_year_num'])}-"
                                 f"{str(int(r['fiscal_year_num']) + 1)[-2:]}",
            "probability": _safe_float(r["probability"]),
            "risk_category": real_data.classify_risk(float(r["probability"])),
            "flagged": bool(r["probability"] >= thr),
            "label_next": None if pd.isna(r.get(LABEL)) else int(r[LABEL]),
            "altman_z_ex_x4": _safe_float(r.get("altman_z_ex_x4")),
        } for _, r in hist.iterrows()],
    }


def get_company_features(ticker: str, year: int):
    row = real_data.company_row(ticker, year)
    if row is None:
        return None
    return {f: _safe_float(row.get(f)) for f in real_data.state()["features"]}


def get_sectors():
    m = _matrix()
    if "gics_sector" not in m.columns:
        return []
    return sorted(m["gics_sector"].dropna().unique().tolist())


def get_dashboard_summary():
    m = _matrix()
    s = real_data.state()
    thr = s["threshold"]
    latest_year = int(m["fiscal_year_num"].max())
    cur = m[m["fiscal_year_num"] == latest_year]

    # Vectorised banding: iterrows over the current year was the whole cost of
    # the landing screen. np.searchsorted assigns every row's band in one pass.
    edges = real_data.risk_bands()
    cuts = np.array([edges["low"], edges["moderate"], edges["high"]])
    idx = np.searchsorted(cuts, cur["probability"].to_numpy(), side="right")
    labels = np.array(config.RISK_BAND_LABELS)
    vc = pd.Series(labels[idx]).value_counts()
    bands = {k: int(v) for k, v in vc.items()}

    # §12.5 "universe composition: sector mix". Counted over distinct tickers,
    # not company-years, so a long-lived firm does not outweigh a short-lived
    # one in what is meant to describe the universe.
    sector_distribution = {}
    if "gics_sector" in m.columns:
        per_ticker = m.drop_duplicates("ticker")
        sector_distribution = {
            str(k): int(v) for k, v in
            per_ticker["gics_sector"].fillna("(unknown)").value_counts().items()}

    return {
        "model_version": s["version"],
        "sector_distribution": sector_distribution,
        "universe": {
            "n_tickers": int(m["ticker"].nunique()),
            "n_company_years": int(len(m)),
            "fiscal_years": [int(m["fiscal_year_num"].min()), latest_year],
            "n_sector_unsupported_tickers": len(s["excluded_tickers"]),
        },
        "latest_year": latest_year,
        "prediction_horizon": f"FY{latest_year + 1}-{str(latest_year + 2)[-2:]}",
        "threshold": round(thr, 4),
        "n_flagged": int((cur["probability"] >= thr).sum()),
        "n_with_distress_label": int(m.groupby("ticker")[LABEL].max().fillna(0).sum()),
        "pct_flagged": round(float((cur["probability"] >= thr).mean()) * 100, 1),
        "risk_bands": bands,
        "esg_available": False,
        "disclaimer": "Analytical estimates requiring human review (PRD §14.5).",
    }
