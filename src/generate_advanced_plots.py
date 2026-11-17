import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
import os

def main():
    print("Loading synthetic data for advanced plots...")
    features = pd.read_csv("data/processed/synthetic_feature_matrix.csv")
    raw_panel = pd.read_csv("data/processed/synthetic_panel.csv")
    
    # Create target (forward looking)
    raw_panel = raw_panel.sort_values(["ticker", "fiscal_year_num"])
    raw_panel["is_distress_next"] = raw_panel.groupby("ticker")["is_distress_event"].shift(-1)
    raw_panel["is_distress_next"] = raw_panel["is_distress_next"].fillna(0).astype(int)
    
    df = pd.merge(features, raw_panel[["ticker", "fiscal_year_num", "is_distress_next"]],
                  on=["ticker", "fiscal_year_num"], how="inner")
                  
    drop_cols = ["ticker", "company_name", "fiscal_year", "fiscal_year_num", 
                 "period_end_date", "is_distress_event", "is_distress_next", "target", "gics_sector"]
    all_features = [c for c in df.columns if c not in drop_cols]
    
    X = df[all_features]
    y = df["is_distress_next"]
    
    imputer = SimpleImputer(strategy="median")
    imputed_data = imputer.fit_transform(X)
    valid_cols = imputer.get_feature_names_out(all_features)
    X_imp = pd.DataFrame(imputed_data, columns=valid_cols)
    
    os.makedirs("reports/figures", exist_ok=True)
    
    # 1. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    # Select a subset of interesting features so the heatmap is readable
    subset_cols = ["e_score", "s_score", "g_score", "roa", "ebit_margin", "debt_to_equity", "current_ratio", "volatility_260d"]
    # Filter only columns that exist
    subset_cols = [c for c in subset_cols if c in X_imp.columns]
    
    plt.figure(figsize=(10, 8))
    corr = X_imp[subset_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Collinearity Matrix (Pearson)")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png", dpi=300)
    plt.close()
    
    # 2. Class Distribution Plot
    print("Generating Class Distribution...")
    plt.figure(figsize=(8, 6))
    if "roa" in df.columns:
        sns.kdeplot(data=df[df['is_distress_next']==0], x="roa", label="Healthy (0)", fill=True, color="blue", alpha=0.3)
        sns.kdeplot(data=df[df['is_distress_next']==1], x="roa", label="Distress (1)", fill=True, color="red", alpha=0.3)
        plt.xlim(-0.5, 0.5)
        plt.title("Distribution of Return on Assets (ROA) by Class")
        plt.xlabel("ROA")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/figures/class_distribution.png", dpi=300)
        plt.close()
        
    # 3. Train XGBoost Model
    print("Training XGBoost...")
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    model.fit(X_imp, y)
    
    # 4. Feature Importance (Weight/Gain)
    print("Generating Global Feature Importance...")
    plt.figure(figsize=(10, 8))
    # XGBoost's native plot_importance
    ax = xgb.plot_importance(model, max_num_features=15, importance_type="gain", title="XGBoost Feature Importance (Gain)", show_values=False)
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Partial Dependence Plot (PDP)
    print("Generating Partial Dependence Plots...")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        # PartialDependenceDisplay currently works better with sklearn models or requires specific wrappers
        # We will train a quick HistGradientBoostingClassifier just for the PDP plot 
        hgb = HistGradientBoostingClassifier(random_state=42, max_iter=50)
        hgb.fit(X_imp, y)
        
        features_to_plot = []
        if "e_score" in X_imp.columns: features_to_plot.append("e_score")
        if "debt_to_equity" in X_imp.columns: features_to_plot.append("debt_to_equity")
        
        if features_to_plot:
            fig, ax = plt.subplots(figsize=(12, 5))
            PartialDependenceDisplay.from_estimator(hgb, X_imp, features_to_plot, ax=ax, grid_resolution=50)
            fig.suptitle("Partial Dependence Plots: Marginal Effect on Distress Probability")
            plt.tight_layout()
            plt.savefig("reports/figures/pdp_escore.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Skipping PDP due to error: {e}")

    print("All advanced plots generated successfully.")

if __name__ == "__main__":
    main()
