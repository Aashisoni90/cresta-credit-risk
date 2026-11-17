"""
model_manager.py — real-model access (Phase 4)
===============================================
Loads the registered Model A artefact once at startup.

THERE IS NO MODEL B, AND THAT IS NOT AN ERROR
----------------------------------------------
Track B (ESG) is deferred. `get_model_b()` raises `TrackBUnavailable` rather
than falling back to `models/model_B.pkl`, which is fitted on SYNTHETIC data
with an ESG effect injected by construction. A silent fallback there would ship
fabricated ESG findings about named companies under a real-data banner — the
single worst failure this service could have. Callers must handle the
exception and report Track B as unavailable.
"""
from services import real_data


class TrackBUnavailable(RuntimeError):
    """Raised whenever Track B (ESG) output is requested. See module docstring."""

    CODE = "TRACK_B_UNAVAILABLE"
    MESSAGE = (
        "Track B (ESG-augmented) is not available. ESG extraction is deferred: "
        "the BRSR-mandated subset yields roughly 5.3 distress events per "
        "variable against a >=10 rule of thumb, which cannot support a credible "
        "incremental-power claim. Track A operates with zero ESG data by design "
        "(invariant 6). No ESG-based estimate is produced."
    )


def load_models():
    real_data.load_all()


def load_test_data():
    """Retained for the startup hook; the real matrix carries its own split."""
    return real_data.state()["matrix"]


def get_model_a():
    return real_data.state()["model"]


def get_model_b():
    raise TrackBUnavailable(TrackBUnavailable.MESSAGE)


def model_info() -> dict:
    s = real_data.state()
    return {
        "version": s["version"],
        "n_features": len(s["features"]),
        "features": list(s["features"]),
        "operating_threshold": s["threshold"],
        "trained_utc": s["payload"].get("trained_utc"),
    }


def predict_single(features: dict, model_type: str = "financial_only") -> float:
    if model_type != "financial_only":
        raise TrackBUnavailable(TrackBUnavailable.MESSAGE)
    import numpy as np
    import pandas as pd
    s = real_data.state()
    names = s["features"]
    row = {f: features.get(f, np.nan) for f in names}
    # Values arriving as None make the column dtype `object`, which XGBoost
    # rejects; coerce to float64 so missing stays NaN (never zero — invariant 1).
    X = (pd.DataFrame([row], columns=names)
         .apply(pd.to_numeric, errors="coerce").astype("float64"))
    return float(s["model"].predict_proba(X)[:, 1][0])
