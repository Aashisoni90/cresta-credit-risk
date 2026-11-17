import pandas as pd
from pathlib import Path
import math

def generate_templates(
    universe_path="data/processed/universe_800.csv",
    out_dir="data/raw/templates",
    chunk_size=500
):
    df = pd.read_csv(universe_path)
    tickers = df["ticker"].unique()
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. Base fields (from Phase 1)
    base_fields = [
        "TOTAL_ASSETS", "CUR_CAPITAL", "CUR_LIAB", "RETAIN_EARN", 
        "EBIT", "TOT_EQUITY", "TOT_LIAB_AND_EQY", "RETURN_ON_ASSET", 
        "TOT_DEBT_TO_TOT_EQY", "INT_COVERAGE_RATIO", "CUR_MKT_CAP", "CUR_RATIO"
    ]
    
    # 2. Expanded fields (for Phase 2 feature blocks)
    expanded_fields = [
        "NET_INCOME",                 # Block 3
        "SALES_REV_TURN",             # Block 2 & 3
        "VOLATILITY_260D",            # Block 7 (Merton proxy)
        "SHORT_AND_LONG_TERM_DEBT",   # Block 1
        "CF_CASH_FROM_OPER",          # Block 3
        "IS_INT_EXPENSE"              # Block 4
    ]
    
    all_fields = base_fields + expanded_fields
    
    n_chunks = math.ceil(len(tickers) / chunk_size)
    print(f"Generating {n_chunks} templates of ~{chunk_size} companies each...")
    
    for i in range(n_chunks):
        chunk_tickers = tickers[i*chunk_size : (i+1)*chunk_size]
        
        # Create an empty DataFrame with Tickers as rows and Fields as columns
        template = pd.DataFrame(index=chunk_tickers, columns=all_fields)
        template.index.name = "Ticker"
        
        out_file = out / f"bloomberg_template_chunk_{i+1}.csv"
        template.to_csv(out_file)
        
    print(f"\n✅ Created {n_chunks} templates in {out_dir}/")
    print(f"Total tickers: {len(tickers)}")
    print(f"Total fields to pull: {len(all_fields)}")
    print("\nInstructions for Bloomberg Terminal:")
    print("1. Open these CSVs in Excel.")
    print("2. Use the Bloomberg add-in or BDH() formula to pull these fields for FY2015-FY2025.")
    print("3. Flatten the results into the standard vertical format and save as CSV.")

if __name__ == "__main__":
    generate_templates()
