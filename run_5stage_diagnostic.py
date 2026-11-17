import pandas as pd
import statsmodels.api as sm
import numpy as np

raw = pd.read_csv('data/processed/synthetic_panel.csv')
gated = pd.read_csv('data/processed/synthetic_panel_gated.csv')
feat = pd.read_csv('data/processed/synthetic_feature_matrix.csv')

feat = pd.merge(feat, gated[["ticker", "fiscal_year_num", "is_distress_event"]], on=["ticker", "fiscal_year_num"], how="left")
feat["is_distress_next"] = feat.groupby("ticker")["is_distress_event"].shift(-1)

train = feat[(feat['fiscal_year_num'] <= 2022) & feat['is_distress_next'].notna()]
test = feat[(feat['fiscal_year_num'] >= 2023) & feat['is_distress_next'].notna()]

stages = [
    ("1. Raw (synthetic_panel)", raw, 'e_score', 'is_distress_event'),
    ("2. Gated (synthetic_panel_gated)", gated, 'e_score', 'is_distress_event'),
    ("3. Engineered (synthetic_feature_matrix)", feat, 'e_score', 'is_distress_event'),
    ("4. Forward Shifted (Train split)", train, 'e_score', 'is_distress_next'),
    ("5. Out-of-time (Test split)", test, 'e_score', 'is_distress_next')
]

for name, df, esg_col, target_col in stages:
    df_clean = df.dropna(subset=[esg_col, target_col]).copy()
    y = df_clean[target_col]
    X = sm.add_constant(df_clean[esg_col])
    
    try:
        model = sm.Logit(y, X, missing='drop')
        result = model.fit(disp=0)
        coef = result.params[esg_col]
        pval = result.pvalues[esg_col]
        print(f"{name}: N={len(df_clean)}, Positives={int(y.sum())} | Coef: {coef:.5f}, p-val: {pval:.2e}")
    except Exception as e:
        print(f"{name}: Failed to run logistic regression - {e}")
