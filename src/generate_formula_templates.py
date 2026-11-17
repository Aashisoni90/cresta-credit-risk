import pandas as pd
from pathlib import Path

def create_formula_template(universe_path="data/processed/universe_800.csv", out_dir="data/raw/templates"):
    df = pd.read_csv(universe_path)
    tickers = df["ticker"].unique()
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    fields = [
        "BS_TOT_ASSET", "BS_CUR_ASSET_REPORT", "BS_CUR_LIAB", "BS_RETAIN_EARN",
        "EBIT", "TOTAL_EQUITY", "BS_TOT_LIAB2", "RETURN_ON_ASSET",
        "TOT_DEBT_TO_TOT_EQY", "INTEREST_COVERAGE_RATIO", "CUR_MKT_CAP", "CUR_RATIO",
        "NET_INCOME", "SALES_REV_TURN", "VOLATILITY_260D", "SHORT_AND_LONG_TERM_DEBT",
        "CF_CASH_FROM_OPER", "IS_INT_EXPENSE"
    ]
    # BS_TOT_LIAB2 and INTEREST_COVERAGE_RATIO confirmed correct via live BDP test on
    # RELIANCE IN Equity FY2023 (2026-09-19) — TOTAL_LIABILITIES returned #N/A Invalid
    # Field, INT_COVERAGE_RATIO returned #N/A N/A.
    
    years = list(range(2015, 2026)) # 2015 to 2025
    
    # We will chunk this so Excel doesn't crash. 500 companies per chunk.
    chunk_size = 500
    n_chunks = len(tickers) // chunk_size + 1
    
    for c in range(n_chunks):
        chunk_tickers = tickers[c*chunk_size : (c+1)*chunk_size]
        if len(chunk_tickers) == 0:
            break
            
        rows = []
        for ticker in chunk_tickers:
            for year in years:
                row = {"ticker": ticker, "fiscal_year": year}
                # Excel row index (starts at 2 because row 1 is header)
                idx = len(rows) + 2 
                for f in fields:
                    # e.g., =BDP($A2, "TOTAL_ASSETS", "FUND_PER", "Y", "EQY_FUND_YEAR", $B2)
                    row[f] = f'=BDP($A{idx}, "{f}", "FUND_PER", "Y", "EQY_FUND_YEAR", $B{idx})'
                rows.append(row)
                
        out_df = pd.DataFrame(rows)
        out_file = out / f"bloomberg_formula_chunk_{c+1}.xlsx"
        
        # Write to Excel
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            out_df.to_excel(writer, index=False)
            
        print(f"Created {out_file} ({len(rows)} rows)")

if __name__ == "__main__":
    create_formula_template()
