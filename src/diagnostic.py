import pandas as pd
import numpy as np
import statsmodels.api as sm

def test_signal(df, x_col, y_col, stage_name):
    if x_col not in df.columns or y_col not in df.columns:
        print(f"[{stage_name}] SKIPPED (missing {x_col} or {y_col})")
        return
    
    # Drop NaNs
    valid = df[[x_col, y_col]].dropna()
    X = valid[x_col]
    X = sm.add_constant(X)
    y = valid[y_col]
    
    if len(y.unique()) < 2:
        print(f"[{stage_name}] SKIPPED (only 1 class present)")
        return
        
    try:
        model = sm.Logit(y, X).fit(disp=0)
        coef = model.params[x_col]
        pval = model.pvalues[x_col]
        print(f"[{stage_name}] Rows: {len(valid)} | Positives: {y.sum()} | Coef: {coef:.4f} | P-value: {pval:.4e}")
    except Exception as e:
        print(f"[{stage_name}] ERROR: {e}")

def main():
    print("--- ISOLATED SIGNAL DIAGNOSTIC ---")
    
    # Stage 1: Raw generated panel
    df_raw = pd.read_csv("data/processed/synthetic_panel.csv")
    test_signal(df_raw, "e_score", "is_distress_event", "1. Raw Panel")
    
    # Stage 2: Quality Gates
    df_gated = pd.read_csv("data/processed/synthetic_panel_gated.csv")
    test_signal(df_gated, "e_score", "is_distress_event", "2. Post-Quality Gates")
    
    # Stage 3: Feature Engineering
    df_feat = pd.read_csv("data/processed/synthetic_feature_matrix.csv")
    # Base_esg is mapped to e_score, s_score, g_score. Let's test e_score.
    # We still need the label, but feature matrix doesn't contain is_distress_event, it needs to be merged back
    df_gated_label = df_gated[["ticker", "fiscal_year_num", "is_distress_event"]]
    df_feat_merged = pd.merge(df_feat, df_gated_label, on=["ticker", "fiscal_year_num"])
    test_signal(df_feat_merged, "e_score", "is_distress_event", "3. Post-Feature Eng (Concurrent Label)")
    
    # Stage 4: Forward-Label Shift (Target creation)
    raw_panel = df_gated.sort_values(["ticker", "fiscal_year_num"]).copy()
    raw_panel["is_distress_next"] = raw_panel.groupby("ticker")["is_distress_event"].shift(-1)
    df_feat_shifted = pd.merge(df_feat, raw_panel[["ticker", "fiscal_year_num", "is_distress_next"]], 
                  on=["ticker", "fiscal_year_num"], how="inner")
    df_feat_shifted["target"] = df_feat_shifted["is_distress_next"].fillna(0).astype(int)
    test_signal(df_feat_shifted, "e_score", "target", "4. Post-Forward Shift (Label=Target)")
    
    # Stage 5: Train/Test Split (Out-of-time test set >= 2023)
    df_test = df_feat_shifted[df_feat_shifted["fiscal_year_num"] >= 2023]
    test_signal(df_test, "e_score", "target", "5. Test Set Only (>= 2023)")

if __name__ == "__main__":
    main()
