import pandas as pd
import statsmodels.api as sm

print("Loading N=60,000 panel...")
df = pd.read_csv('data/processed/synthetic_panel.csv')
print(f"Loaded {len(df)} rows.")

print("Computing in-sample correlation...")
corr = df['base_esg'].corr(df['distress_event'])
print(f"Pearson Correlation (distress_event vs base_esg): {corr:.5f}")

print("\nLogistic Regression of distress_event on base_esg:")
X = sm.add_constant(df['base_esg'])
y = df['distress_event']
model = sm.Logit(y, X, missing='drop')
result = model.fit(disp=0)
print(result.summary())
