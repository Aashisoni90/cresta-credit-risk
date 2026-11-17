import json
import pandas as pd
from pathlib import Path

def generate_table_1_metrics():
    print("Generating Table 1 (Performance Metrics)...")
    
    with open("reports/logs/training_metrics.json", "r") as f:
        metrics = json.load(f)
        
    # Table 1 structure
    # Model | ROC-AUC | Significant?
    table_data = [
        {"Model": "Model A (Financial + Market)", "ROC-AUC": f"{metrics['Model_A_AUC']:.3f}", "Significant vs Baseline": "-"},
        {"Model": "Model B (Financial + ESG)", "ROC-AUC": f"{metrics['Model_B_AUC']:.3f}", "Significant vs Baseline": "Yes" if metrics["Significance_p95"] else "No"}
    ]
    
    df = pd.DataFrame(table_data)
    df.to_csv("reports/tables/table_1_performance.csv", index=False)
    
    # Also write a markdown version for the appendix
    with open("reports/tables/table_1_performance.md", "w") as f:
        f.write("# Table 1: Out-of-Time Predictive Performance (Synthetic Benchmark)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Note: Significance assessed via bootstrapped 95% confidence intervals on $\\Delta$AUC.*\n")

def generate_appendix_synthetic_parameters():
    print("Formatting Synthetic Generation Parameters...")
    with open("reports/tables/synthetic_parameters.json", "r") as f:
        params = json.load(f)
        
    with open("reports/tables/appendix_a_synthetic_params.md", "w") as f:
        f.write("# Appendix A: Synthetic Data Generation Parameters\n\n")
        f.write("To ensure reproducibility and transparency, the synthetic baseline was generated with the following parameters:\n\n")
        
        f.write("| Parameter | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| Base Distress Rate | {params['distress_base_rate']} |\n")
        f.write(f"| ESG-Distress Log-Odds Uplift | {params['esg_distress_uplift']} |\n")
        f.write(f"| ESG-Size Correlation | {params['esg_size_correlation']} |\n")
        f.write(f"| Mean Log Assets | {params['mean_log_assets']} |\n")
        f.write(f"| Std Log Assets | {params['std_log_assets']} |\n")
        f.write(f"| Missingness Rate (Simulation) | 0.05 |\n")
        
def main():
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    generate_table_1_metrics()
    generate_appendix_synthetic_parameters()
    print("✅ Tables generated and saved to reports/tables/")

if __name__ == "__main__":
    main()
