import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

cm = np.array([[77461, 38148],
               [128, 136]])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy (0)", "Distress (1)"],
            yticklabels=["Healthy (0)", "Distress (1)"],
            cbar=False,
            annot_kws={"size": 16})

plt.ylabel("True Class", fontsize=14)
plt.xlabel("Predicted Class", fontsize=14)
plt.title("XGBoost (Fin+ESG) Confusion Matrix (Threshold=0.5)", fontsize=16)
plt.tight_layout()

out_path = "reports/figures/confusion_matrix.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
