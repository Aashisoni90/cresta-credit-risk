import pandas as pd
import numpy as np
from pathlib import Path

def clean_numeric(series):
    """Convert Bloomberg #N/A strings to typed np.nan"""
    # Common Bloomberg NA strings
    na_strings = [
        '#N/A Field Not Applicable',
        '#N/A Invalid Field',
        '#N/A N/A',
        '#N/A',
        '#N/A Requesting Data...',
        '#N/A Authorization'
    ]
    
    cleaned = series.astype(str).replace(na_strings, np.nan)
    # Remove commas and parse to float
    cleaned = cleaned.str.replace(',', '', regex=False)
    cleaned = cleaned.replace(['nan', 'None', ''], np.nan)
    return pd.to_numeric(cleaned, errors='coerce')

def main():
    print("Loading p2_export_1...")
    df = pd.read_csv("p2_export_1 - Sheet1.csv")
    
    # Rename columns to match pipeline expectations
    col_mapping = {
        'TOTAL_ASSETS': 'total_assets',
        'CUR_CAPITAL': 'current_assets', # Assumption based on context
        'CUR_LIAB': 'current_liabilities',
        'RETAIN_EARN': 'retained_earnings',
        'EBIT': 'ebit',
        'TOT_EQUITY': 'total_equity_book',
        'RETURN_ON_ASSET': 'return_on_assets',
        'TOT_DEBT_TO_TOT_EQY': 'debt_to_equity',
        'INT_COVERAGE_RATIO': 'interest_coverage',
        'CUR_MKT_CAP': 'market_cap',
        'CUR_RATIO': 'current_ratio',
        'NET_INCOME': 'net_income',
        'SALES_REV_TURN': 'total_revenue',
        'VOLATILITY_260D': 'volatility_260d',
        'SHORT_AND_LONG_TERM_DEBT': 'short_and_long_term_debt',
        'CF_CASH_FROM_OPER': 'cash_from_operations',
        'IS_INT_EXPENSE': 'interest_expense'
    }
    df = df.rename(columns=col_mapping)
    
    # Convert fiscal_year to string FYYYYY-YY format and fiscal_year_num
    # Assuming fiscal_year is ending year, e.g. 2015 -> FY2014-15
    df['fiscal_year_num'] = df['fiscal_year'] - 1
    df['fiscal_year_str'] = 'FY' + df['fiscal_year_num'].astype(str) + '-' + df['fiscal_year'].astype(str).str[-2:]
    df['fiscal_year'] = df['fiscal_year_str']
    df = df.drop(columns=['fiscal_year_str'])
    
    # Apply clean_numeric to all feature columns
    feature_cols = list(col_mapping.values())
    for col in feature_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
            
    # Derive missing CORE_FEATURES for quality gates
    df['total_liabilities'] = df['TOTAL_LIAB_AND_EQY'] if 'TOTAL_LIAB_AND_EQY' in df.columns else df['total_assets'] - df['total_equity_book']
    feature_cols.append('total_liabilities')
    
    # Derive negative_equity_flag (crucial for Phase 1 requirements)
    df['negative_equity_flag'] = (df['total_equity_book'] < 0).astype(int)
    # Never zero-fill debt_to_equity; it is correctly NaN when equity < 0
    df.loc[df['total_equity_book'] < 0, 'debt_to_equity'] = np.nan
    
    # Keep only required columns and drop null tickers
    out_cols = ['ticker', 'fiscal_year', 'fiscal_year_num'] + feature_cols + ['negative_equity_flag']
    df = df[out_cols]
    df = df.dropna(subset=['ticker'])
    
    out_path = Path("data/processed/bloomberg_panel.csv")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
