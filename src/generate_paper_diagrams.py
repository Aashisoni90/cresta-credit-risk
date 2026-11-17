import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path
import warnings

# Suppress warnings for clean plots
warnings.filterwarnings("ignore")

def generate_shap_plot():
    print("Generating SHAP summary plot...")
    model_b = joblib.load("models/deprecated_synthetic/model_B.pkl")
    X_test = pd.read_csv("data/processed/test_set/X_test.csv")
    
    explainer = shap.TreeExplainer(model_b)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(10, 8))
    # We want a beautiful summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
    plt.title("Model B (Fin+ESG): SHAP Feature Importance (Synthetic Benchmark)")
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def generate_roc_curve():
    print("Generating ROC Curve...")
    y_test = pd.read_csv("data/processed/test_set/y_test.csv")["target"]
    probs = pd.read_csv("data/processed/test_set/y_probs.csv")
    
    fpr_a, tpr_a, _ = roc_curve(y_test, probs["prob_A"])
    roc_auc_a = auc(fpr_a, tpr_a)
    
    fpr_b, tpr_b, _ = roc_curve(y_test, probs["prob_B"])
    roc_auc_b = auc(fpr_b, tpr_b)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_a, tpr_a, color='darkorange', lw=2, label=f'Model A (Fin) - AUC = {roc_auc_a:.3f}')
    plt.plot(fpr_b, tpr_b, color='navy', lw=2, label=f'Model B (Fin+ESG) - AUC = {roc_auc_b:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Synthetic Data')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig("reports/figures/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def generate_pr_curve():
    print("Generating Precision-Recall Curve...")
    y_test = pd.read_csv("data/processed/test_set/y_test.csv")["target"]
    probs = pd.read_csv("data/processed/test_set/y_probs.csv")
    
    precision_a, recall_a, _ = precision_recall_curve(y_test, probs["prob_A"])
    pr_auc_a = auc(recall_a, precision_a)
    
    precision_b, recall_b, _ = precision_recall_curve(y_test, probs["prob_B"])
    pr_auc_b = auc(recall_b, precision_b)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_a, precision_a, color='darkorange', lw=2, label=f'Model A (Fin) - PR-AUC = {pr_auc_a:.3f}')
    plt.plot(recall_b, precision_b, color='navy', lw=2, label=f'Model B (Fin+ESG) - PR-AUC = {pr_auc_b:.3f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Synthetic Data')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    plt.savefig("reports/figures/pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    Path("reports/figures").mkdir(exist_ok=True, parents=True)
    generate_shap_plot()
    generate_roc_curve()
    generate_pr_curve()
    print("✅ All plots generated and saved to reports/figures/")

if __name__ == "__main__":
    main()
