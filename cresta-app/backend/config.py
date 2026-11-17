"""
CRESTA Backend — Configuration (REAL DATA, Phase 4)
====================================================
Repointed from the synthetic panel to the real Bloomberg-derived artefacts
produced in Phases 1-3.

WHAT CHANGED AND WHY IT MATTERS
--------------------------------
The previous configuration pointed at `synthetic_panel.csv` and
`models/model_A.pkl` / `model_B.pkl`, which were fitted on generated data with
an ESG effect injected by construction (gamma = -4.0). Serving those alongside
real company names would present fabricated risk estimates about identifiable
issuers. The synthetic artefacts are left on disk for reference but are no
longer wired to the API.

THERE IS NO REAL MODEL B
-------------------------
Track B (ESG) is deferred — the BRSR-mandated subset yields ~5.3 events per
variable against a >=10 rule of thumb (Project_Context.md, ESG feasibility).
Invariant 6 requires Track A to work with zero ESG data present, always. So
MODEL_B_PATH is deliberately absent rather than pointed at the synthetic
artefact: the API reports Track B as unavailable instead of returning a number
that would silently be synthetic.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── real data ─────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PANEL_PATH = DATA_DIR / "bloomberg_panel_scored_gated.csv"
MATRIX_PATH = DATA_DIR / "training_matrix.csv"
UNIVERSE_PATH = DATA_DIR / "universe_800.csv"
EXCLUDED_FINANCIALS_PATH = DATA_DIR / "excluded_financials.csv"
QUALITY_REJECTIONS_PATH = DATA_DIR / "quality_rejections_scored.csv"
FAILED_TICKERS_PATH = PROJECT_ROOT / "data" / "raw" / "failed_tickers.csv"

# ── model registry ────────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODEL_DIR / "registry.json"
MODEL_A_VERSION = "A-1.2.0-real"
MODEL_A_PATH = MODEL_DIR / f"model_{MODEL_A_VERSION}.pkl"
MODEL_B_PATH = None          # Track B deferred — see module docstring

# ── Phase 3 reports (served to the model-performance screen) ──────────
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase3"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
ROLLING_ORIGIN_PATH = REPORTS_DIR / "rolling_origin_summary.csv"
SHAP_GLOBAL_PATH = REPORTS_DIR / "shap_global_ranking.csv"
SHAP_BLOCK_PATH = REPORTS_DIR / "shap_block_ranking.csv"
SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary.json"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_CARD_PATH = PROJECT_ROOT / "MODEL_CARD.md"

# ── ML config ─────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

# §8.4 cost assumption. The deployed ratio is a STATED ASSUMPTION, not a
# default to be left unexamined: at ~22% prevalence a 10:1 ratio flags over half
# the portfolio, which is useless for triage. Surfaced through /api/threshold so
# the UI can state it wherever a flag is shown.
COST_RATIO_FN_FP = 10.0

# Track A operates with zero ESG features. Kept only so request payloads
# carrying ESG keys are stripped rather than silently passed to the model.
ESG_FEATURES = ["e_score", "s_score", "g_score", "carbon_intensity"]

# §5.6 / invariant 5 — financial-sector issuers are out of scope for the
# primary model, not merely missing from it. The API answers SECTOR_UNSUPPORTED
# rather than scoring them on ratios that have no meaning for a bank.
SECTOR_UNSUPPORTED_CODE = "SECTOR_UNSUPPORTED"
SECTOR_UNSUPPORTED_MESSAGE = (
    "This issuer is in the financial sector (bank / NBFC). CRESTA's primary "
    "model excludes financial-sector issuers: their balance sheets have no "
    "operating cycle, so working capital, the current ratio and EBIT/assets "
    "are undefined in meaning rather than merely noisy (PRD §5.6). No distress "
    "estimate is produced for these companies."
)

# Risk bands are anchored to the MODEL'S OWN operating threshold, not to round
# numbers. A band scheme of 0.30/0.50/0.70 bears no relation to a model whose
# cost-optimal cut sits at 0.200, and would label companies "low risk" that the
# model flags. See risk_bands() in services/real_data.py.
RISK_BAND_LABELS = ["low", "moderate", "high", "critical"]

DB_PATH = Path(__file__).resolve().parent / "cresta.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
