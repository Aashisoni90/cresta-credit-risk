import subprocess
import json
import pandas as pd
import numpy as np

def run_pipeline():
    subprocess.run(["python", "src/generate_synthetic_data.py", "--seed", "42"], check=True)
    subprocess.run(["python", "src/train_model.py"], check=True)
    
    with open("reports/logs/training_metrics.json", "r") as f:
        metrics = json.load(f)
        
    y_test = pd.read_csv("data/processed/test_set/y_test.csv")["target"].values
    y_probs = pd.read_csv("data/processed/test_set/y_probs.csv")
    
    return metrics, y_test, y_probs

def main():
    print("Run 1...")
    m1, y1, p1 = run_pipeline()
    print("Run 2...")
    m2, y2, p2 = run_pipeline()
    
    print("\n--- Determinism Check ---")
    
    y_match = np.array_equal(y1, y2)
    pA_match = np.allclose(p1["prob_A"].values, p2["prob_A"].values)
    pB_match = np.allclose(p1["prob_B"].values, p2["prob_B"].values)
    metrics_match = (m1 == m2)
    
    print(f"Test Set Labels Identical: {y_match}")
    print(f"Model A Probs Identical: {pA_match}")
    print(f"Model B Probs Identical: {pB_match}")
    print(f"Metrics JSON Identical: {metrics_match}")
    
    if y_match and pA_match and pB_match and metrics_match:
        print("✅ Pipeline is strictly DETERMINISTIC.")
    else:
        print("❌ NON-DETERMINISM DETECTED!")

if __name__ == "__main__":
    main()
