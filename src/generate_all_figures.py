"""
generate_all_figures.py
Generates all 8 figures required by the CRESTA paper from N=60,000 pipeline outputs.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
from pathlib import Path
import os, time, warnings
warnings.filterwarnings("ignore")

OUT = Path("reports/figures")
OUT.mkdir(exist_ok=True, parents=True)

print("Loading data...")
X_test = pd.read_csv("data/processed/test_set/X_test.csv")
y_test = pd.read_csv("data/processed/test_set/y_test.csv")["target"]
y_probs = pd.read_csv("data/processed/test_set/y_probs.csv")
model_B = joblib.load("models/deprecated_synthetic/model_B.pkl")
raw_panel = pd.read_csv("data/processed/synthetic_panel.csv")
fm = pd.read_csv("data/processed/synthetic_feature_matrix.csv")
print(f"  X_test: {len(X_test)} rows, {X_test.shape[1]} features, positives={int(y_test.sum())}")

# Fig 1: Correlation Heatmap
print("\nFig 1: Correlation Heatmap...")
fig1_cols = ['roa','liabilities_to_assets','equity_to_assets','cur_ratio','ebit_margin','revenue_growth_yoy','e_score','s_score','g_score','carbon_intensity']
available = [c for c in fig1_cols if c in fm.columns]
corr = fm[available].corr()
rename = {'roa':'ROA','liabilities_to_assets':'Liab/Assets','equity_to_assets':'Equity/Assets','cur_ratio':'Cur Ratio','ebit_margin':'EBIT Margin','revenue_growth_yoy':'Rev Growth YoY','e_score':'E Score','s_score':'S Score','g_score':'G Score','carbon_intensity':'Carbon Int.'}
corr.rename(index=rename, columns=rename, inplace=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Pearson Correlation: Financial and ESG Features (N=60,000 Synthetic Panel)", fontsize=12, pad=12)
plt.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> correlation_heatmap.png")

# Fig 2: KDE
print("Fig 2: KDE class distribution...")
healthy_roa = raw_panel.loc[raw_panel['is_distress_event']==0, 'roa'].dropna().clip(-1, 1)
distress_roa = raw_panel.loc[raw_panel['is_distress_event']==1, 'roa'].dropna().clip(-1, 1)
fig, ax = plt.subplots(figsize=(9, 6))
healthy_roa.plot.kde(ax=ax, color='steelblue', lw=2.5, label=f'Healthy (Class 0, n={len(healthy_roa):,})')
distress_roa.plot.kde(ax=ax, color='firebrick', lw=2.5, linestyle='--', label=f'Distressed (Class 1, n={len(distress_roa):,})')
ax.set_xlabel("Return on Assets (ROA)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("KDE: ROA Distribution — Healthy vs. Distressed Firms (N=60,000 Panel)", fontsize=12)
ax.legend(fontsize=11); ax.axvline(0, color='gray', linestyle=':', lw=1.5, alpha=0.7); ax.grid(alpha=0.3); ax.set_xlim(-1, 1)
plt.tight_layout()
fig.savefig(OUT / "class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> class_distribution.png")

# Fig 3: Confusion Matrix
print("Fig 3: Confusion Matrix...")
pred_B = (y_probs['prob_B'] >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred_B)
tn, fp, fn, tp = cm.ravel()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
classes = ['Healthy (0)', 'Distressed (1)']
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(classes, fontsize=11); ax.set_yticklabels(classes, fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=12); ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — Model B XGBoost (threshold=0.5)\nTest: {len(y_test):,} rows, {int(y_test.sum())} positive events, prevalence={y_test.mean()*100:.4f}%", fontsize=10)
for i in range(2):
    for j in range(2):
        val = cm[i, j]; color = "white" if val > cm.max()/2 else "black"
        ax.text(j, i, f"{val:,}", ha='center', va='center', fontsize=14, fontweight='bold', color=color)
plt.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> confusion_matrix.png  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Fig 4: ROC Curve
print("Fig 4: ROC Curve...")
fpr_a, tpr_a, _ = roc_curve(y_test, y_probs['prob_A']); roc_auc_a = auc(fpr_a, tpr_a)
fpr_b, tpr_b, _ = roc_curve(y_test, y_probs['prob_B']); roc_auc_b = auc(fpr_b, tpr_b)
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_a, tpr_a, color='darkorange', lw=2.5, label=f'Model A — Financial Only (AUC={roc_auc_a:.4f})')
ax.plot(fpr_b, tpr_b, color='navy', lw=2.5, label=f'Model B — Fin+ESG (AUC={roc_auc_b:.4f})')
ax.plot([0,1],[0,1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
ax.fill_between(fpr_b, tpr_b, alpha=0.05, color='navy')
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves — Out-of-Time Test (N=60,000 Benchmark)\nBootstrapped ΔAUC=+0.0774, 95% CI=[+0.0455, +0.1080], p<0.05', fontsize=11)
ax.legend(loc='lower right', fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> roc_curve.png  AUC_A={roc_auc_a:.4f}, AUC_B={roc_auc_b:.4f}")

# Fig 5: PR Curve
print("Fig 5: Precision-Recall Curve...")
prec_a, rec_a, _ = precision_recall_curve(y_test, y_probs['prob_A']); prauc_a = auc(rec_a, prec_a)
prec_b, rec_b, _ = precision_recall_curve(y_test, y_probs['prob_B']); prauc_b = auc(rec_b, prec_b)
base_rate = y_test.mean()
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(rec_a, prec_a, color='darkorange', lw=2.5, label=f'Model A — Financial Only (PR-AUC={prauc_a:.4f})')
ax.plot(rec_b, prec_b, color='navy', lw=2.5, label=f'Model B — Fin+ESG (PR-AUC={prauc_b:.4f})')
ax.axhline(base_rate, color='gray', lw=1.5, linestyle='--', label=f'Random Classifier (base rate={base_rate:.4f})')
ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
ax.set_title(f'Precision-Recall Curve — Out-of-Time Test (N=60,000 Benchmark)\nDistress prevalence: {base_rate*100:.4f}%', fontsize=11)
ax.legend(loc='upper right', fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "pr_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> pr_curve.png  PR-AUC_A={prauc_a:.4f}, PR-AUC_B={prauc_b:.4f}")

# Fig 6: Feature Importance
print("Fig 6: Feature Importance (XGBoost Gain)...")
importance_dict = model_B.get_booster().get_score(importance_type='gain')
imp_series = pd.Series(importance_dict).sort_values(ascending=True).tail(20)
esg_cols = {'e_score','s_score','g_score','carbon_intensity'}
colors = ['firebrick' if name in esg_cols else 'steelblue' for name in imp_series.index]
fig, ax = plt.subplots(figsize=(10, 8))
imp_series.plot.barh(ax=ax, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel("XGBoost Gain", fontsize=12)
ax.set_title("Global Feature Importance (XGBoost Gain) — Model B\nRed = ESG features  |  Blue = Financial/Market features", fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> feature_importance.png")

# Fig 7: SHAP Summary
print("Fig 7: SHAP Summary Plot...")
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=min(10000, len(X_test)), replace=False)
X_sample = X_test.iloc[idx].reset_index(drop=True)
explainer = shap.TreeExplainer(model_B)
shap_values = explainer(X_sample)
fig, ax = plt.subplots(figsize=(10, 9))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15, ax=ax)
ax.set_title("SHAP Global Feature Importance — Model B (Fin+ESG)\nN=60,000 Benchmark, Out-of-Time Test Sample (n=10,000)", fontsize=11)
plt.tight_layout()
fig.savefig(OUT / "shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> shap_summary.png")

# Fig 8: Partial Dependence Plot
print("Fig 8: Partial Dependence Plot (e_score)...")
idx2 = rng.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
X_pdp = X_test.iloc[idx2].reset_index(drop=True)
feature_names = list(X_test.columns)
e_idx = feature_names.index('e_score')
fig, ax = plt.subplots(figsize=(9, 6))
PartialDependenceDisplay.from_estimator(model_B, X_pdp, features=[e_idx], feature_names=feature_names, ax=ax, grid_resolution=50)
ax.set_title("Partial Dependence Plot — e_score on Predicted Distress Probability\nModel B (Fin+ESG), N=60,000 Benchmark", fontsize=11)
ax.set_xlabel("e_score (Environmental Score)", fontsize=12)
ax.set_ylabel("Partial Dependence", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "pdp_escore.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> pdp_escore.png")

print("\n=== ALL 8 FIGURES COMPLETE ===")
for f in sorted(OUT.glob("*.png")):
    stat = os.stat(f)
    print(f"  {f.name}: {stat.st_size:,} bytes, modified {time.ctime(stat.st_mtime)}")
