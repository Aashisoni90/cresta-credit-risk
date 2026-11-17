import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

def main():
    print("Loading test set and probabilities...")
    y_test = pd.read_csv("data/processed/test_set/y_test.csv")["target"].values
    y_probs = pd.read_csv("data/processed/test_set/y_probs.csv")
    
    prob_b = y_probs["prob_B"].values
    
    # 1. Base rate calculation
    n_pos = np.sum(y_test == 1)
    n_total = len(y_test)
    base_rate = n_pos / n_total
    print(f"\n--- Base Rate Analysis ---")
    print(f"Total rows in test set: {n_total}")
    print(f"Total positive (distress) events: {n_pos}")
    print(f"Base rate (theoretical PR-AUC baseline): {base_rate:.6f} ({base_rate*100:.3f}%)")
    
    # 2. Threshold Sweeping
    print("\n--- Threshold Sweep for Model B ---")
    # Using specific candidate thresholds
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # We can also compute exact precision/recall curve
    precision, recall, th = precision_recall_curve(y_test, prob_b)
    pr_auc = auc(recall, precision)
    print(f"Model B PR-AUC: {pr_auc:.6f}")
    
    for t in thresholds_to_test:
        preds = (prob_b >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        
        # If model predicts 0 everywhere, cm might be 1x1 if y_test had only 0s, but we know it has some 1s.
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            print(f"Threshold {t:.1f} | TP: {tp:3d} | FP: {fp:4d} | FN: {fn:3d} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        else:
            print(f"Threshold {t:.1f} | cm shape: {cm.shape} (likely 0 predictions of 1)")

if __name__ == "__main__":
    main()
