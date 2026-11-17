"""
score_altman_panel.py
======================
Applies Altman Z'' (PRD §6.5) to the real merged panel and adds the two
companion columns the real data turned out to require.

WHY A SEPARATE SCRIPT
---------------------
`altman_zscore.py` computes Z'' correctly and is covered by the Phase 1 test
suite; it is deliberately NOT modified here. This script is the panel-level
application plus the degenerate-denominator handling that only became visible
once Z'' was run on 2,999 real companies rather than 37.

THE PROBLEM THIS HANDLES (found 2026-10-27, Project_Context.md §15)
--------------------------------------------------------------------
On the real panel 1,265 rows (6.1%) score |Z''| > 50. The cause is NOT a units
error — the arithmetic was hand-verified exact against four companies. It is
TWO degenerate denominators:

  (a) total_liabilities ~ 0  ->  X4 = equity/liabilities explodes POSITIVE.
      Debt-free holding shells where assets ~ equity. 61.7% of extremes.
      Worst case MIHI IN Equity FY2016-17: TL = 0.014 INR mn against TA = 247.4,
      giving X4 = 17,668 and Z'' = 18,558 — and the row is classified "safe"
      despite EBIT of -13.7. A loss-making shell scored as healthy.

  (b) total_assets ~ 0  ->  X1, X2 and X3 all explode, usually NEGATIVE.
      Collapsed/defunct shells. Worst case NBF IN Equity FY2024-25:
      TA = 0.001 INR mn (about Rs 1,000) against 15.9 mn of liabilities,
      giving Z'' = -642,105.

Both are mathematically correct and economically meaningless. Left raw they
would dominate any model that uses Z'' as a feature (§6.5) and would make the
SHAP attribution unreadable.

TREATMENT — and why this one
-----------------------------
1. `altman_denominator_degenerate` — a deterministic row-level flag. Purely a
   function of the row, so it introduces no fitted statistic and therefore no
   train/test leak. Flagged, never dropped: per cross-phase invariant #4 these
   rows stay in the panel and stay countable.

2. `altman_z_double_prime_capped` — winsorised to FIXED bounds [-50, +50].
   Fixed, not percentile-derived, for two reasons:
     * PRD §5.4's scale_sanity gate already declares |Z''| > 50 implausible, so
       the bound is the PRD's own stated threshold rather than a free parameter.
     * A 1st/99th-percentile winsorisation is a statistic fitted to the data.
       Fitting it on the full panel before the out-of-time split would leak test
       information into training (§7.4, invariant #3). `feature_engineering.py`
       already sets this precedent: `interest_coverage_capped` uses fixed
       [-10, 50] rather than the percentiles PRD §6.4 nominally specifies.
   The RAW column is preserved unchanged alongside it. Nothing is overwritten.

Neither column decides anything. Which of raw / capped / degenerate-excluded is
used for modelling is a methodology choice recorded at feature-assembly time.

USAGE
-----
    python src/score_altman_panel.py \
        --panel data/processed/bloomberg_panel_full_gated.csv \
        --out   data/processed/bloomberg_panel_scored_full.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from altman_zscore import compute_altman_zscore_components

# A liabilities base below this fraction of total assets makes X4 degenerate.
# Calibrated 2026-10-27 against the scale_sanity gate: 5% captures 98.0% of the
# rows the gate flags as |Z''| > 50, at 7.0% of scored rows. 1% captured only
# 59.9%; 10% adds just 0.5pp more for half again as many flags. The threshold is
# also principled rather than fitted — below TL = 5% of TA the 1.05*X4 term alone
# exceeds 21, i.e. 8x the entire "safe" cutoff of 2.6, so the score is decided by
# the absence of debt rather than by any distress signal.
MIN_LIABILITIES_FRACTION = 0.05          # 5% of total assets
# A balance sheet smaller than this is not a going concern worth scoring.
MIN_TOTAL_ASSETS_INR_MN = 1.0            # Rs 10 lakh
# PRD §5.4 scale_sanity already calls |Z''| > 50 implausible.
Z_CAP = 50.0


def add_degeneracy_columns(z: pd.DataFrame) -> pd.DataFrame:
    out = z.copy()
    ta, tl = out["total_assets"], out["total_liabilities"]

    thin_liabilities = tl.notna() & ta.notna() & (tl < MIN_LIABILITIES_FRACTION * ta)
    thin_assets = ta.notna() & (ta < MIN_TOTAL_ASSETS_INR_MN)

    out["altman_flag_thin_liabilities"] = thin_liabilities
    out["altman_flag_thin_assets"] = thin_assets
    out["altman_denominator_degenerate"] = thin_liabilities | thin_assets

    out["altman_z_double_prime_capped"] = out["altman_z_double_prime"].clip(-Z_CAP, Z_CAP)

    # ── Non-circular Z'' variant, for use as a FEATURE ──────────────────
    # Under the PRD §7.5 financial-criteria label (promoted to primary 2026-10-27,
    # Project_Context.md §16.1), the X4 = equity/liabilities term is CIRCULAR:
    # sign(X4) == (book equity < 0) on 100.00% of rows, which is exactly label
    # criterion 2. Canonical Z'' therefore cannot be a feature without leaking.
    #
    # Dropping the term costs nothing — it HELPS:
    #     Z'' full (4 terms)      AUC 0.7661
    #     Z'' ex-X4 (3 terms)     AUC 0.8069   <- better
    #     X4 alone                AUC 0.7209   (the circular part)
    #     corr(full, ex-X4) = 0.9943
    # X4 is unbounded and carries the thin-liabilities degeneracy, so removing it
    # strips noise as well as leakage.
    #
    # Canonical `altman_z_double_prime` is PRESERVED unchanged and remains the
    # §8.1 Tier-1 BENCHMARK. Only the feature-side use switches to this variant.
    out["altman_z_ex_x4"] = (
        6.56 * out["X1_working_cap_to_assets"]
        + 3.26 * out["X2_retained_earnings_to_assets"]
        + 6.72 * out["X3_ebit_to_assets"]
    )
    out["altman_z_ex_x4_capped"] = out["altman_z_ex_x4"].clip(-Z_CAP, Z_CAP)

    # Temporal derivatives rebuilt from the NON-circular variant, so Block 6
    # does not reintroduce the leak through altman_z_yoy_change / _3yr_trend.
    out = out.sort_values(["ticker", "fiscal_year_num"])
    out["altman_z_ex_x4_yoy_change"] = out.groupby("ticker")["altman_z_ex_x4"].diff()

    def _slope(s):
        v = s.dropna()
        if len(v) < 2:
            return np.nan
        return np.polyfit(range(len(v)), v.values, 1)[0]

    out["altman_z_ex_x4_3yr_trend"] = (
        out.groupby("ticker")["altman_z_ex_x4"]
           .rolling(3, min_periods=2).apply(_slope, raw=False)
           .reset_index(level=0, drop=True)
    )
    return out


def report(z: pd.DataFrame) -> None:
    scored = z["altman_z_double_prime"].notna()
    n = int(scored.sum())
    deg = z["altman_denominator_degenerate"] & scored
    clean = scored & ~z["altman_denominator_degenerate"]

    print(f"\nscored rows: {n}/{len(z)} ({n/len(z)*100:.1f}%)")
    print(f"  degenerate denominator : {int(deg.sum()):6d} ({deg.sum()/n*100:5.1f}% of scored)")
    print(f"      thin liabilities   : {int((z['altman_flag_thin_liabilities'] & scored).sum()):6d}")
    print(f"      thin assets        : {int((z['altman_flag_thin_assets'] & scored).sum()):6d}")
    print(f"  clean                  : {int(clean.sum()):6d} ({clean.sum()/n*100:5.1f}% of scored)")

    print("\n--- Z'' distribution ---")
    cmp = pd.DataFrame({
        "raw (all scored)": z.loc[scored, "altman_z_double_prime"].describe(
            percentiles=[.01, .25, .5, .75, .99]),
        "raw (clean only)": z.loc[clean, "altman_z_double_prime"].describe(
            percentiles=[.01, .25, .5, .75, .99]),
        "capped (all)": z.loc[scored, "altman_z_double_prime_capped"].describe(
            percentiles=[.01, .25, .5, .75, .99]),
    })
    print(cmp.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n--- zone distribution (cutoffs 1.1 / 2.6) ---")
    for label, mask in [("all scored", scored), ("clean only", clean)]:
        vc = z.loc[mask, "altman_zone"].value_counts()
        tot = int(vc.sum())
        line = "  ".join(f"{k}={v} ({v/tot*100:.1f}%)" for k, v in vc.items())
        print(f"  {label:<12s} n={tot:6d}   {line}")

    mis = z[clean & (z["altman_zone"] == "safe") & (z["ebit"] < 0)]
    print(f"\n  sanity: clean rows classified 'safe' despite negative EBIT: {len(mis)}")
    degmis = z[deg & (z["altman_zone"] == "safe") & (z["ebit"] < 0)]
    print(f"          degenerate rows ditto (the pathology being excluded): {len(degmis)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/bloomberg_panel_full_gated.csv")
    ap.add_argument("--out", default="data/processed/bloomberg_panel_scored_full.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.panel)
    print(f"[score_altman] loaded {len(df)} rows from {a.panel}")
    z = compute_altman_zscore_components(df)
    z = add_degeneracy_columns(z)
    report(z)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out, index=False)
    print(f"\n[score_altman] wrote {len(z)} rows -> {a.out}")


if __name__ == "__main__":
    main()
