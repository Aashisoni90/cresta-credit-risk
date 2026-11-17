import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import time
import os

def out_of_time_split(df, time_col, split_point):
    train = df[df[time_col] < split_point].copy()
    test = df[df[time_col] >= split_point].copy()
    return train, test

def main():
    print("Loading synthetic data for expanded benchmark...")
    features = pd.read_csv("data/processed/synthetic_feature_matrix.csv")
    raw_panel = pd.read_csv("data/processed/synthetic_panel.csv")
    
    # Create target (forward looking)
    raw_panel = raw_panel.sort_values(["ticker", "fiscal_year_num"])
    raw_panel["is_distress_next"] = raw_panel.groupby("ticker")["is_distress_event"].shift(-1)
    raw_panel["is_distress_next"] = raw_panel["is_distress_next"].fillna(0).astype(int)
    
    df = pd.merge(features, raw_panel[["ticker", "fiscal_year_num", "is_distress_next"]],
                  on=["ticker", "fiscal_year_num"], how="inner")
                  
    # Clean features (Model B - including ESG)
    drop_cols = ["ticker", "company_name", "fiscal_year", "fiscal_year_num", 
                 "period_end_date", "is_distress_event", "is_distress_next", "target", "gics_sector"]
    all_features = [c for c in df.columns if c not in drop_cols]
    
    # Split
    train_df, test_df = out_of_time_split(df, "fiscal_year_num", 2023)
    X_train = train_df[all_features]
    y_train = train_df["is_distress_next"]
    X_test = test_df[all_features]
    y_test = test_df["is_distress_next"]
    
    # Impute and Scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_imp = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_imp = scaler.transform(imputer.transform(X_test))
    
    # Class weights for highly imbalanced
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    classifiers = {
        "Logistic Regression (Linear)": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Naive Bayes (Probabilistic)": GaussianNB(),
        "Decision Tree (Single Tree)": DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=42),
        "Random Forest (Bagging)": RandomForestClassifier(class_weight="balanced", n_estimators=200, max_depth=15, random_state=42),
        "XGBoost (Level-wise Boosting)": xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1, 
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric="auc"
        )
    }
    
    results = []
    
    print("\nTraining models and evaluating...")
    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train_imp, y_train)
        y_pred = clf.predict(X_test_imp)
        latency = (time.time() - start_time) * 1000  # simple training latency approx for demo
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            "Classifier": name,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision (Macro)": f"{prec:.4f}",
            "Recall (Macro)": f"{rec:.4f}",
            "F1-Score (Macro)": f"{f1:.4f}",
            "Latency (ms)": f"~{latency:.1f} ms"
        })
        
        # Save confusion matrix for XGBoost
        if "XGBoost" in name:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Healthy (0)', 'Distress (1)'], 
                        yticklabels=['Healthy (0)', 'Distress (1)'])
            plt.title('XGBoost Confusion Matrix (Test Set)')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            os.makedirs("reports/figures", exist_ok=True)
            plt.savefig("reports/figures/confusion_matrix.png", dpi=300)
            plt.close()
            print("Saved confusion matrix to reports/figures/confusion_matrix.png")
            
    res_df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(res_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
