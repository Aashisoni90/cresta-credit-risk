"""
train_model.py
===============
Trains and compares TWO models on the same distress-prediction task:

    Model A (baseline)  = financial/market features only
    Model B (augmented) = financial/market features + ESG features

...then tests whether B's improvement over A is statistically significant,
not just numerically larger. This is the actual research design from the
Foundations Guide — everything else in this project exists to feed this
script clean data.

Implements:
  - Out-of-time train/test split (NEVER random split on time-series financial
    data — see Foundations Guide §6.3)
  - Class imbalance handling via class_weight (SMOTE as a documented
    alternative — see notes below)
  - XGBoost as the model (swap for LightGBM/RandomForest easily — same
    sklearn-style API)
  - SHAP for both global and local explanation
  - Bootstrapped confidence interval on the AUC difference between A and B
    (a practical alternative to DeLong's test — see notes below) to answer
    "is the improvement real or noise"

USAGE:
    Import build_and_compare_models(df, feature_cols_a, feature_cols_b,
    label_col, time_col) with your actual merged dataset once it exists.
    Until then, running this file directly demonstrates the full pipeline
    end-to-end on synthetic data so you can see the shape it produces.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap


# ── Out-of-time split ───────────────────────────────────────────────────
def out_of_time_split(df: pd.DataFrame, time_col: str, split_point):
    """Train = strictly before split_point. Test = at or after split_point.
    NEVER use train_test_split() with shuffle=True on this kind of data —
    that leaks future information into training (Foundations Guide §6.3)."""
    train = df[df[time_col] < split_point].copy()
    test = df[df[time_col] >= split_point].copy()
    print(f"Out-of-time split at {split_point}: train={len(train)} rows, test={len(test)} rows")
    if len(test) == 0 or len(train) == 0:
        raise ValueError("Split produced an empty train or test set — check split_point and time_col values.")
    return train, test


# ── Model training ──────────────────────────────────────────────────────
def train_xgb_classifier(X_train, y_train, use_class_weight=True):
    """Binary distress classifier. For the graded 0/1/2 label, collapse to
    binary (0 vs {1,2}) here for the core comparison — keep the graded
    version for the richer SHAP narrative in a separate analysis."""
    scale_pos_weight = 1.0
    if use_class_weight:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if n_pos > 0:
            scale_pos_weight = n_neg / n_pos
        print(f"  class balance -> negatives={n_neg}, positives={n_pos}, "
              f"scale_pos_weight={scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


from sklearn.impute import SimpleImputer
def train_logistic_baseline(X_train, y_train):
    """The classical statistical benchmark — reviewers expect this in the
    results table alongside XGBoost, not just XGBoost alone."""
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_imputed, y_train)
    
    # Store the imputer on the model so we can use it during evaluation
    model.imputer = imputer
    return model


# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, label=""):
    if hasattr(model, "imputer"):
        X_test = model.imputer.transform(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = sk_auc(recall, precision)
    print(f"[{label}] AUC-ROC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}")
    return {"auc_roc": auc_roc, "pr_auc": pr_auc, "proba": proba}


def bootstrap_auc_difference_ci(y_test, proba_a, proba_b, n_boot=2000, seed=42):
    """Bootstrapped confidence interval on (AUC_B - AUC_A). If the interval
    excludes zero, the improvement is statistically significant at that
    confidence level. This is a practical, easy-to-implement alternative to
    DeLong's test and is standard practice for this kind of comparison.

    NOTE: for a more formally standard approach, look up `delong_roc_test`
    implementations (not in sklearn directly) — bootstrap is a legitimate
    and simpler substitute, cite it as such in your methodology.
    """
    rng = np.random.default_rng(seed)
    y_test = np.asarray(y_test)
    n = len(y_test)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue  # skip degenerate resamples with only one class
        auc_a = roc_auc_score(y_b, proba_a[idx])
        auc_b = roc_auc_score(y_b, proba_b[idx])
        diffs.append(auc_b - auc_a)
    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    mean_diff = diffs.mean()
    significant = not (ci_low <= 0 <= ci_high)
    print(f"\nBootstrap AUC(B) - AUC(A): mean={mean_diff:+.4f}, "
          f"95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]  "
          f"{'SIGNIFICANT' if significant else 'NOT significant'} at 95%")
    return {"mean_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high, "significant": significant}


# ── SHAP explanation ─────────────────────────────────────────────────────
def explain_with_shap(model, X_test, top_n=10):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
    print(f"\nTop {top_n} features by mean |SHAP value| (global importance):")
    print(importance.head(top_n).to_string())
    return shap_values, importance


# ── The full comparison pipeline ────────────────────────────────────────
def build_and_compare_models(
    df: pd.DataFrame,
    feature_cols_a: list,
    feature_cols_b: list,
    label_col: str,
    time_col: str,
    split_point,
):
    """
    df must already be: cleaned, joined (Bloomberg + BRSR), with a binary
    label_col (0/1) and a time_col usable for chronological ordering.

    feature_cols_a: financial/market features only (Model A)
    feature_cols_b: feature_cols_a + ESG features (Model B)
    """
    train, test = out_of_time_split(df, time_col, split_point)

    X_train_a, X_test_a = train[feature_cols_a], test[feature_cols_a]
    X_train_b, X_test_b = train[feature_cols_b], test[feature_cols_b]
    y_train, y_test = train[label_col], test[label_col]

    print("\n=== Model A: financial/market features only (baseline) ===")
    model_a_lr = train_logistic_baseline(X_train_a, y_train)
    eval_a_lr = evaluate_model(model_a_lr, X_test_a, y_test, "A - Logistic Regression")

    model_a_xgb = train_xgb_classifier(X_train_a, y_train)
    eval_a_xgb = evaluate_model(model_a_xgb, X_test_a, y_test, "A - XGBoost")

    print("\n=== Model B: financial/market + ESG features (augmented) ===")
    model_b_xgb = train_xgb_classifier(X_train_b, y_train)
    eval_b_xgb = evaluate_model(model_b_xgb, X_test_b, y_test, "B - XGBoost (+ESG)")

    print("\n=== Is the ESG improvement statistically significant? ===")
    sig_result = bootstrap_auc_difference_ci(
        y_test, eval_a_xgb["proba"], eval_b_xgb["proba"]
    )

    print("\n=== SHAP explanation for Model B (which features drive it) ===")
    shap_values, importance = explain_with_shap(model_b_xgb, X_test_b)

    return {
        "model_a_logistic": model_a_lr,
        "model_a_xgb": model_a_xgb,
        "model_b_xgb": model_b_xgb,
        "eval_a_logistic": eval_a_lr,
        "eval_a_xgb": eval_a_xgb,
        "eval_b_xgb": eval_b_xgb,
        "significance": sig_result,
        "shap_importance": importance,
    }


if __name__ == "__main__":
    import json
    import joblib
    from pathlib import Path
    
    # Load actual synthetic dataset generated by our pipeline
    features = pd.read_csv("data/processed/synthetic_feature_matrix.csv")
    raw_panel = pd.read_csv("data/processed/synthetic_panel.csv")
    
    # Create shifted target: predict NEXT year's distress using THIS year's features
    raw_panel = raw_panel.sort_values(["ticker", "fiscal_year_num"])
    raw_panel["is_distress_next"] = raw_panel.groupby("ticker")["is_distress_event"].shift(-1)
    
    # Use the TRUE latent synthetic target, bypassing the circular fallback label!
    df = pd.merge(features, raw_panel[["ticker", "fiscal_year_num", "is_distress_next"]], 
                  on=["ticker", "fiscal_year_num"], how="inner")
                  
    # Drop strictly circular features (as warned by fallback_label.py)
    circular = [
        "X4_equity_to_liabilities", "equity_to_liabilities", "interest_coverage", 
        "interest_coverage_below_1_flag", "interest_coverage_capped", 
        "interest_coverage_yoy_change", "liabilities_to_assets", 
        "negative_equity_flag", "total_equity_book"
    ]
    df = df.drop(columns=[c for c in circular if c in df.columns])
    
    # Target definition: fill NaNs from random missingness injector or the shift
    df["target"] = df["is_distress_next"].fillna(0).astype(int)
    
    print(f"Dataset loaded: {len(df)} rows, {df['target'].mean()*100:.1f}% positive class\n")
    
    drop_cols = ["ticker", "company_name", "fiscal_year", "fiscal_year_num", 
                 "period_end_date", "is_distress_event", "is_distress_next", "target", "gics_sector"]
    all_features = [c for c in df.columns if c not in drop_cols]
    
    esg_features_only = ["e_score", "s_score", "g_score", "carbon_intensity"]
    financial_features = [c for c in all_features if c not in esg_features_only]
    esg_features = all_features.copy()
    
    results = build_and_compare_models(
        df=df,
        feature_cols_a=financial_features,
        feature_cols_b=esg_features,
        label_col="target",
        time_col="fiscal_year_num",
        split_point=2023,  # train on <=2022, test on >=2023
    )
    
    # Save the test set and predictions for the diagrams script
    Path("data/processed/test_set").mkdir(exist_ok=True, parents=True)
    # The split point was 2023
    test_mask = df["fiscal_year_num"] >= 2023
    X_test_b = df.loc[test_mask, esg_features]
    y_test = df.loc[test_mask, "target"]
    
    X_test_b.to_csv("data/processed/test_set/X_test.csv", index=False)
    y_test.to_csv("data/processed/test_set/y_test.csv", index=False)
    pd.DataFrame({
        "prob_A": results["eval_a_xgb"]["proba"], 
        "prob_B": results["eval_b_xgb"]["proba"]
    }).to_csv("data/processed/test_set/y_probs.csv", index=False)
    
    # Save the models
    Path("models").mkdir(exist_ok=True)
    joblib.dump(results["model_a_xgb"], "models/deprecated_synthetic/model_A.pkl")
    joblib.dump(results["model_b_xgb"], "models/deprecated_synthetic/model_B.pkl")
    
    # Save metrics for tables
    Path("reports/logs").mkdir(exist_ok=True, parents=True)
    with open("reports/logs/training_metrics.json", "w") as f:
        metrics_out = {
            "Model_A_AUC": results["eval_a_xgb"]["auc_roc"],
            "Model_B_AUC": results["eval_b_xgb"]["auc_roc"],
            "Significance_p95": results["significance"]["significant"]
        }
        json.dump(metrics_out, f, indent=4)
    print("\n✅ Training complete. Models and metrics saved.")
