"""Feature catalogue endpoint — what the model actually sees, and what it does not."""
import pandas as pd
from fastapi import APIRouter

import config
from services import real_data

router = APIRouter()


def _circular_features() -> list[dict]:
    """The features the matrix builder refuses to admit, with the label
    criterion each one would reproduce. Imported from the guard itself rather
    than restated here — a hand-copied list is a list that goes stale."""
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from fallback_label import CIRCULAR_FEATURE_MAP, PARTIALLY_CIRCULAR
    except Exception:
        return []
    out = []
    for criterion, feats in CIRCULAR_FEATURE_MAP.items():
        for f in feats:
            out.append({"name": f, "criterion": criterion, "partial": False,
                        "reason": f"built from the same quantity as label "
                                  f"criterion '{criterion}'"})
    for criterion, feats in PARTIALLY_CIRCULAR.items():
        for f in feats:
            out.append({"name": f, "criterion": criterion, "partial": True,
                        "reason": f"partially circular with '{criterion}' "
                                  f"(Altman-derived)"})
    return out


@router.get("/features")
async def features():
    s = real_data.state()
    feats = s["features"]
    shap_share = {}
    if config.SHAP_GLOBAL_PATH.exists():
        g = pd.read_csv(config.SHAP_GLOBAL_PATH)
        shap_share = dict(zip(g["feature"], g["share_pct"]))
        blocks = dict(zip(g["feature"], g["block"]))
    else:
        blocks = {}

    return {
        "model_version": s["version"],
        "n_features": len(feats),
        "features": [{
            "name": f,
            "block": blocks.get(f),
            "shap_share_pct": (round(float(shap_share[f]), 2)
                               if f in shap_share else None),
            "null_rate_pct": round(float(s["matrix"][f].isna().mean()) * 100, 2),
        } for f in feats],
        # Grouped by §6 block so the screen can show what each block
        # contributes rather than one flat list of 16 names.
        "blocks": [
            {"block": b,
             "name": b.split("_", 1)[-1].replace("_", " "),
             "features": [{"name": f,
                           "shap_share_pct": (round(float(shap_share[f]), 2)
                                              if f in shap_share else None),
                           "null_rate_pct": round(
                               float(s["matrix"][f].isna().mean()) * 100, 2)}
                          for f in feats if blocks.get(f) == b]}
            for b in sorted({blocks.get(f) for f in feats if blocks.get(f)})
        ],
        # §7.5's circularity guard, read from the module that enforces it so
        # this screen cannot drift from what the pipeline actually blocks.
        "circular_features": _circular_features(),
        "absent_blocks": {
            "block4_debt_service": (
                "Interest-coverage features are not in this matrix — they await "
                "an expanded Bloomberg pull. The block is effectively empty."),
            "block8_esg": (
                "Track B deferred. Track A operates with zero ESG features by "
                "design (invariant 6)."),
        },
        "excluded_as_circular": (
            "Features built from the same quantity as a label criterion are "
            "blocked at matrix assembly (§7.5 circularity guard) — including "
            "interest coverage, negative-equity flags and canonical Altman Z″, "
            "whose X4 term reproduces label criterion 2 on 100% of rows."),
    }
