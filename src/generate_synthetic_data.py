import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, Any

class SyntheticDataGenerator:
    """
    Generates a realistic synthetic financial & ESG panel dataset.
    Uses latent factors to ensure realistic financial covariance and structural biases.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)
        
        # Configuration & Ground Truth Parameters (Exported to Table 1 later)
        self.config: Dict[str, Any] = {
            "n_companies": 60000,
            "years": list(range(2015, 2025)),
            "missingness_rate": 0.05,
            
            # True effect sizes & base rates
            "distress_base_rate": 0.03, # 3% base rate of distress
            "esg_size_correlation": 0.4, # Large caps have higher ESG due to disclosure
            "esg_distress_uplift": -4.0, # Higher ESG score *reduces* log-odds of distress (Injected Signal)
            
            # Latent factor base means
            "mean_log_assets": 12.0,
            "std_log_assets": 1.5,
        }
        
    def _generate_universe(self) -> pd.DataFrame:
        """Generate static company characteristics."""
        n = self.config["n_companies"]
        
        tickers = [f"SYN{i:04d} IN Equity" for i in range(1, n + 1)]
        names = [f"Synthetic Co {i}" for i in range(1, n + 1)]
        
        # Sectors
        sectors = ["Financials", "Industrials", "Consumer Discretionary", "Materials", 
                   "Information Technology", "Health Care", "Consumer Staples", "Energy", 
                   "Real Estate", "Communication Services", "Utilities"]
        # Realistic Indian market distribution roughly
        probs = [0.20, 0.15, 0.12, 0.12, 0.10, 0.08, 0.08, 0.05, 0.05, 0.03, 0.02]
        assigned_sectors = np.random.choice(sectors, size=n, p=probs)
        
        # Generate base latent factors for each company
        # Size (log assets), structural profitability (ROA baseline), structural leverage
        base_size = np.random.normal(self.config["mean_log_assets"], self.config["std_log_assets"], size=n)
        base_profitability = np.random.normal(0.08, 0.04, size=n) # avg 8% ROA
        base_leverage = np.random.normal(0.6, 0.3, size=n) # avg debt/equity = 0.6
        base_leverage = np.clip(base_leverage, 0.1, 5.0)
        
        # Base ESG Factor (Correlated with size)
        # Higher size -> Higher ESG
        esg_noise = np.random.normal(0, 10, size=n)
        base_esg = 40 + (base_size - self.config["mean_log_assets"]) * self.config["esg_size_correlation"] * 10 + esg_noise
        base_esg = np.clip(base_esg, 10, 95)
        
        # Sector ESG penalty (Energy/Materials have structurally lower E scores)
        sector_penalties = {"Energy": -15, "Materials": -10, "Utilities": -5, "Information Technology": 10}
        esg_penalty = np.array([sector_penalties.get(s, 0) for s in assigned_sectors])
        base_esg = np.clip(base_esg + esg_penalty, 5, 95)
        
        universe = pd.DataFrame({
            "ticker": tickers,
            "company_name": names,
            "gics_sector": assigned_sectors,
            "base_size": base_size,
            "base_profitability": base_profitability,
            "base_leverage": base_leverage,
            "base_esg": base_esg
        })
        return universe
        
    def _simulate_time_series(self, universe: pd.DataFrame) -> pd.DataFrame:
        """Simulate the financial panel data over time using random walks from base factors."""
        records = []
        
        for _, co in universe.iterrows():
            # Initial state in 2015
            current_size = co["base_size"]
            current_profit = co["base_profitability"]
            current_lev = co["base_leverage"]
            
            # Will it default during the window?
            # Compute underlying distress risk log-odds
            # High leverage + low profit = higher distress risk
            # Plus the injected ESG signal (higher ESG = lower distress risk)
            z_score = -2.0 * current_profit + 0.5 * current_lev + self.config["esg_distress_uplift"] * (co["base_esg"]/100.0)
            
            # Baseline offset to hit ~3% distress rate
            # If standard normal > 1.88, roughly 3% chance.
            prob_distress = 1 / (1 + np.exp(-(z_score - 2.0))) 
            
            is_distressed = np.random.random() < prob_distress
            distress_year = np.random.choice(self.config["years"][2:]) if is_distressed else 9999
            
            for year in self.config["years"]:
                # If they defaulted previously, no data (survivorship bias simulation)
                if year > distress_year + 1:
                    continue
                    
                # If they default THIS year, tank their financials (but not too much, so ESG still matters)
                if year == distress_year:
                    current_profit -= 0.10 # Moderate loss
                    current_lev += 0.5     # Moderate debt spike
                    
                # Random walk
                current_size += np.random.normal(0.05, 0.1) # 5% avg growth
                current_profit += np.random.normal(0, 0.02)
                current_lev += np.random.normal(0, 0.1)
                
                # Reverse engineer financial fields from latent factors
                total_assets = np.exp(current_size)
                net_income = total_assets * current_profit
                
                # Leverage factor: Lev = Debt / Equity -> Total Assets = Debt + Equity
                # Eq = Assets / (1 + Lev)
                total_equity = total_assets / (1 + max(current_lev, 0.1))
                total_debt = total_assets - total_equity
                
                # Granular fields
                cur_assets = total_assets * np.random.uniform(0.2, 0.6)
                cur_liab = total_debt * np.random.uniform(0.3, 0.8)
                retain_earn = total_equity * np.random.uniform(0.1, 0.9)
                ebit = net_income * np.random.uniform(1.1, 1.5) # Taxes/Interest buffer
                
                # Sales / Cashflow
                sales = total_assets * np.random.uniform(0.5, 2.0)
                cf_oper = net_income + (total_assets * np.random.uniform(0.01, 0.05)) # D&A addback
                
                # ESG metrics (small random walk around base)
                e_score = np.clip(co["base_esg"] + np.random.normal(0, 3), 0, 100)
                s_score = np.clip(co["base_esg"] + np.random.normal(0, 3), 0, 100)
                g_score = np.clip(co["base_esg"] + np.random.normal(0, 3), 0, 100)
                carbon = (total_assets / 1e6) * (100 - e_score) * np.random.uniform(0.8, 1.2)
                
                # Trading Status for universe builder
                trading_status = "N" if (is_distressed and year >= distress_year) else "Y"
                
                record = {
                    "ticker": co["ticker"],
                    "company_name": co["company_name"],
                    "gics_sector": co["gics_sector"],
                    "fiscal_year": str(year),
                    "fiscal_year_num": year,
                    "date_range_used": f"1/1/{year} - 12/31/{year}",
                    "period_end_date": f"12/31/{year}",
                    
                    # Core Financials
                    "total_assets": total_assets,
                    "current_assets": cur_assets,
                    "current_liabilities": cur_liab,
                    "retained_earnings": retain_earn,
                    "ebit": ebit,
                    "total_equity_book": total_equity,
                    "total_liabilities": total_debt,
                    
                    # Derived (Computed by Bloomberg normally)
                    "roa": (net_income / total_assets) * 100,
                    "debt_to_equity": (total_debt / total_equity) if total_equity > 0 else np.nan,
                    "interest_coverage": (ebit / (total_debt * 0.08)) if total_debt > 0 else 999,
                    "cur_mkt_cap": total_equity * np.random.uniform(0.8, 3.0), # P/B ratio
                    "cur_ratio": (cur_assets / cur_liab) if cur_liab > 0 else np.nan,
                    
                    # Expanded Financials
                    "net_income": net_income,
                    "sales_rev_turn": sales,
                    "volatility_260d": np.random.uniform(15, 60),
                    "short_and_long_term_debt": total_debt,
                    "cf_cash_from_oper": cf_oper,
                    "is_int_expense": total_debt * 0.08,
                    
                    # ESG
                    "e_score": e_score,
                    "s_score": s_score,
                    "g_score": g_score,
                    "carbon_intensity": carbon,
                    
                    # Target / Metadata
                    "trading_status": trading_status,
                    "is_distress_event": int(year == distress_year)
                }
                records.append(record)
                
        return pd.DataFrame(records)

    def _inject_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject structured missingness and data errors for quality gates to catch."""
        # 1. Missingness
        mask = np.random.rand(*df.shape) < self.config["missingness_rate"]
        
        # Don't drop key identifier columns
        safe_cols = ["ticker", "company_name", "fiscal_year", "fiscal_year_num", "period_end_date", "gics_sector", "trading_status"]
        for j, col in enumerate(df.columns):
            if col not in safe_cols:
                df.iloc[mask[:, j], j] = np.nan
                
        # 2. Add some deliberate negative assets (Gate 4 failure)
        idx = np.random.choice(df.index, size=int(len(df)*0.01), replace=False)
        df.loc[idx, "total_assets"] = -5000
        
        return df

    def generate(self, out_dir="data/processed"):
        print(f"Generating synthetic panel with seed {self.seed}...")
        universe = self._generate_universe()
        panel = self._simulate_time_series(universe)
        
        # Inject noise
        panel = self._inject_noise(panel)
        
        # Export the table of assumptions for the paper
        Path("reports/tables").mkdir(parents=True, exist_ok=True)
        with open("reports/tables/synthetic_parameters.json", "w") as f:
            json.dump(self.config, f, indent=4)
            
        # Export the flat panel (mimics load_filled_template output)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        panel_path = Path(out_dir) / "synthetic_panel.csv"
        panel.to_csv(panel_path, index=False)
        print(f"✅ Generated {len(panel)} rows across {universe.shape[0]} companies -> {panel_path}")
        print(f"✅ Exported synthetic parameters -> reports/tables/synthetic_parameters.json")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    gen = SyntheticDataGenerator(seed=args.seed)
    gen.generate()
