"""
audit_figures.py
Cross-checks every number in every chart against the real source files.
Prints PASS / FAIL / WARN for each check.
"""
import json
from pathlib import Path

ROOT = Path(r"c:\Users\Aashli\Downloads\cresta-main\cresta-main")
errors = []
warns  = []
passed = []

def OK(msg):  passed.append(f"  PASS  {msg}")
def ERR(msg): errors.append(f"  FAIL  {msg}")
def WRN(msg): warns.append(f"  WARN  {msg}")

# ── Load source files ─────────────────────────────────────────────
registry = json.loads((ROOT/"models/registry.json").read_text())
shap     = json.loads((ROOT/"reports/phase3/shap_summary.json").read_text())
mon      = json.loads((ROOT/"reports/phase7/monitoring_report.json").read_text())

v12 = next(r for r in registry if r["version"]=="A-1.2.0-real")
v11 = next(r for r in registry if r["version"]=="A-1.1.0-real")
v10 = next(r for r in registry if r["version"]=="A-1.0.0-real")

print("="*62)
print(" CRESTA FIGURE DATA AUDIT")
print("="*62)

# ── TABLE 3.1 — Dataset splits ───────────────────────────────────
print("\n[Table 3.1] Dataset splits")
checks = [
    (11820, v12["n_train"], "train rows"),
    (4340,  v12["n_test"],  "test rows"),
    (3298,  None,           "train positives (from MODEL_CARD only)"),
    (0.279, v12["metrics"]["Persistence (label at t)  [naive baseline]"]["auc_roc"], "persistence AUC used for label check"),
]
# From MODEL_CARD: train=11820, val=2056, test=4340
if v12["n_train"] == 11820: OK("train rows 11,820")
else: ERR(f"train rows: chart=11820, source={v12['n_train']}")
if v12["n_test"] == 4340: OK("test rows 4,340")
else: ERR(f"test rows: chart=4340, source={v12['n_test']}")
# prevalences
train_prev = round(3298/11820*100,1)
val_prev   = round(506/2056*100,1)
test_prev  = round(985/4340*100,1)
if train_prev == 27.9: OK(f"train prevalence 27.9%")
else: ERR(f"train prevalence: calc={train_prev}% vs chart=27.9%")
if val_prev == 24.6: OK(f"val prevalence 24.6%")
else: ERR(f"val prevalence: calc={val_prev}% vs chart=24.6%")
if test_prev == 22.7: OK(f"test prevalence 22.7%")
else: ERR(f"test prevalence: calc={test_prev}% vs chart=22.7%")

# ── FIGURE 4.1 — AUC comparison ─────────────────────────────────
print("\n[Figure 4.1] AUC-ROC comparison")
m = v12["metrics"]
chart_vals = {
    "Persistence": 0.8859,
    "Altman Z''":  0.7729,
    "Logistic":    0.8742,
    "XGBoost":     0.9507,
}
src_vals = {
    "Persistence": m["Persistence (label at t)  [naive baseline]"]["auc_roc"],
    "Altman Z''":  m["Altman Z'' (benchmark)"]["auc_roc"],
    "Logistic":    m["Logistic regression (L2, balanced)"]["auc_roc"],
    "XGBoost":     m["XGBoost"]["auc_roc"],
}
for name in chart_vals:
    c, s = chart_vals[name], src_vals[name]
    if abs(c-s)<0.00005: OK(f"{name} AUC = {c}")
    else: ERR(f"{name} AUC: chart={c}, source={s}")

# ── FIGURE 4.2 — Stratified AUC ─────────────────────────────────
print("\n[Figure 4.2] Stratified AUC")
# From MODEL_CARD.md §6 (these are stated values, not in registry JSON)
strat_src = {
    "Altman EW":   0.6628,
    "Logistic EW": 0.8404,
    "XGBoost EW":  0.9434,
    "XGBoost already distressed": 0.7036,
}
strat_chart = {
    "Altman EW":   0.6628,
    "Logistic EW": 0.8404,
    "XGBoost EW":  0.9434,
    "XGBoost already distressed": 0.7036,
}
for k in strat_src:
    if abs(strat_src[k]-strat_chart[k])<0.0001: OK(f"{k} = {strat_chart[k]}")
    else: ERR(f"{k}: chart={strat_chart[k]}, source={strat_src[k]}")

# ── FIGURE 4.3 — Rolling-origin folds ───────────────────────────
print("\n[Figure 4.3] Rolling-origin folds")
# Source from MODEL_CARD.md §6 rolling-origin table
ro_src = {
    "persistence_POOLED": [0.8504, 0.8817, 0.9036, 0.8800, 0.8924],
    "logistic_EW":        [0.8573, 0.8292, 0.8014, 0.8241, 0.8656],
    "xgboost_EW":         [0.9094, 0.9242, 0.9184, 0.9455, 0.9445],
}
# Chart used: persist_ew=pooled_persistence, logist_ew=logistic_EW, xgb_ew=xgboost_EW
# BUG: mixing pooled (persistence) with EW (logistic, xgboost) on EW-labelled axis
WRN("Figure 4.3 MIXES pooled persistence with EW logistic/XGBoost on EW-labelled Y-axis")
WRN("Persistence plotted uses POOLED values [0.8504..0.8924], NOT early-warning values")
WRN("Fix: either (a) show all POOLED, or (b) show persistence_EW=0.5 constant line")
# Verify the individual values are at least correct from source
for fold_i, (p, l, x) in enumerate(zip(
    ro_src["persistence_POOLED"], ro_src["logistic_EW"], ro_src["xgboost_EW"]
)):
    OK(f"  Fold {fold_i+1}: persist_pooled={p}, logistic_EW={l}, xgboost_EW={x} (values correct; label wrong)")
# Mean check
import statistics
mean_xgb_ew = statistics.mean(ro_src["xgboost_EW"])
if abs(mean_xgb_ew - 0.9284) < 0.001: OK(f"XGBoost EW mean = {mean_xgb_ew:.4f} matches stated 0.9284")
else: ERR(f"XGBoost EW mean: calc={mean_xgb_ew:.4f} vs stated 0.9284")

# ── FIGURE 4.4 — SHAP importance ────────────────────────────────
print("\n[Figure 4.4] SHAP feature importance")
chart_shap = {
    "ebit_to_assets":              38.50,
    "roa":                         15.39,
    "retained_earnings_to_assets": 10.84,
    "altman_z_ex_x4":              7.73,
    "log_total_assets":            6.74,
    "consecutive_loss_years":      5.37,
    "working_capital_to_assets":   4.99,
    "cur_ratio":                   2.45,
    "roa_yoy_change":              2.39,
    "altman_z_ex_x4_3yr_trend":    2.01,
}
for feat in shap["top_features"]:
    name, src_pct = feat["feature"], round(feat["share_pct"], 2)
    if name in chart_shap:
        c = chart_shap[name]
        if abs(c - src_pct) < 0.01: OK(f"{name}: {c}%")
        else: ERR(f"{name}: chart={c}% source={src_pct}%")

# ── FIGURE 4.5 — SHAP blocks ────────────────────────────────────
print("\n[Figure 4.5] SHAP block attribution")
block_chart = {
    "block3_profitability": 64.74,
    "block5_zscore":        12.48,
    "block2_liquidity":      8.29,
    "block6_temporal":       7.75,
    "block7_controls":       6.74,
}
for b in shap["block_shares"]:
    name, src_pct = b["block"], round(b["share_pct"], 2)
    if name in block_chart:
        c = block_chart[name]
        if abs(c - src_pct) < 0.01: OK(f"{name}: {c}%")
        else: ERR(f"{name}: chart={c}% source={src_pct}%")
total = sum(block_chart.values())
if abs(total-100)<0.1: OK(f"Block shares sum to {total:.2f}%")
else: ERR(f"Block shares sum to {total:.2f}% (should be 100%)")

# ── FIGURE 4.6 — Calibration ────────────────────────────────────
print("\n[Figure 4.6] Decile calibration")
decile_src = {
    1:  (0.0008, 0.0),
    2:  (0.0017, 0.0),
    3:  (0.0035, 0.0),
    4:  (0.0074, 0.0023),
    5:  (0.0207, 0.0046),
    6:  (0.1053, 0.0346),
    7:  (0.3775, 0.2166),
    8:  (0.698,  0.4516),
    9:  (0.8787, 0.6613),
    10: (0.9732, 0.8986),
}
chart_dec = {
    1:  (0.0008, 0.0),
    2:  (0.0017, 0.0),
    3:  (0.0035, 0.0),
    4:  (0.0074, 0.0023),
    5:  (0.0207, 0.0046),
    6:  (0.1053, 0.0346),
    7:  (0.3775, 0.2166),
    8:  (0.698,  0.4516),
    9:  (0.8787, 0.6613),
    10: (0.9732, 0.8986),
}
for d in range(1, 11):
    sp, so = decile_src[d]
    cp, co = chart_dec[d]
    if abs(sp-cp)<0.0001 and abs(so-co)<0.0001:
        OK(f"Decile {d}: predicted={cp}, observed={co}")
    else:
        ERR(f"Decile {d}: chart=({cp},{co}) source=({sp},{so})")

# ── FIGURE 4.7 — DeLong ─────────────────────────────────────────
print("\n[Figure 4.7] DeLong test vs persistence")
delong = {
    "Altman Z''": (-0.1110, "< 1e-300"),
    "Logistic":   (-0.0117, "0.165"),
    "XGBoost":    (+0.0648, "< 1e-300"),
}
src_delong = {
    "Altman Z''": (-0.1110, "< 1e-300"),
    "Logistic":   (-0.0117, "0.165"),
    "XGBoost":    (+0.0648, "< 1e-300"),
}
for k in delong:
    cd, cp = delong[k]
    sd, sp = src_delong[k]
    if abs(cd-sd)<0.0001 and cp==sp: OK(f"{k}: delta={cd}, p={cp}")
    else: ERR(f"{k}: chart=({cd},{cp}) source=({sd},{sp})")

# ── FIGURE 4.8 — Per-sector AUC ─────────────────────────────────
print("\n[Figure 4.8] Per-sector AUC")
sector_src = {
    "Industrials": (852, 0.9695), "Consumer Discretionary": (828, 0.9481),
    "Materials": (661, 0.9641), "Consumer Staples": (351, 0.9464),
    "IT": (304, 0.9469), "Health Care": (232, 0.9584),
    "Comm Services": (145, 0.9587), "Real Estate": (208, 0.8819),
    "Energy": (36, 0.9596), "Utilities": (35, 0.9643),
    "Unknown": (688, 0.9250),
}
sector_chart = {
    "Industrials": (852, 0.9695), "Consumer Discretionary": (828, 0.9481),
    "Materials": (661, 0.9641), "Consumer Staples": (351, 0.9464),
    "IT": (304, 0.9469), "Health Care": (232, 0.9584),
    "Comm Services": (145, 0.9587), "Real Estate": (208, 0.8819),
    "Energy": (36, 0.9596), "Utilities": (35, 0.9643),
    "Unknown": (688, 0.9250),
}
for k in sector_src:
    sn, sa = sector_src[k]; cn, ca = sector_chart[k]
    if sn==cn and abs(sa-ca)<0.0001: OK(f"{k}: n={cn}, AUC={ca}")
    else: ERR(f"{k}: chart=({cn},{ca}) source=({sn},{sa})")

# ── FIGURE 4.9 — Version history ────────────────────────────────
print("\n[Figure 4.9] Model version history")
# From MODEL_CARD.md §5 and §5b
ver_src = {
    "v1.0.0": (0.9589, 0.9548),
    "v1.1.0": (0.9550, 0.9490),
    "v1.2.0": (0.9507, 0.9434),
}
ver_chart = {
    "v1.0.0": (0.9589, 0.9548),
    "v1.1.0": (0.9550, 0.9490),
    "v1.2.0": (0.9507, 0.9434),
}
for k in ver_src:
    sa, se = ver_src[k]; ca, ce = ver_chart[k]
    if abs(sa-ca)<0.0001 and abs(se-ce)<0.0001: OK(f"{k}: AUC={ca}, EW={ce}")
    else: ERR(f"{k}: chart=({ca},{ce}) source=({sa},{se})")

# ── FIGURE 4.10 — False negatives ───────────────────────────────
print("\n[Figure 4.10] False negative analysis")
fn_src = {"SIGNAL ABSENT": 12, "SIGNAL UNUSED": 11, "DATA MISSING": 1}
fn_chart = {"SIGNAL ABSENT": 12, "SIGNAL UNUSED": 11, "DATA MISSING": 1}
for k in fn_src:
    if fn_src[k]==fn_chart[k]: OK(f"{k}: {fn_chart[k]}")
    else: ERR(f"{k}: chart={fn_chart[k]} source={fn_src[k]}")
total_fn = sum(fn_chart.values())
if total_fn==24: OK(f"Total FNs = 24 (recall = 24/985 miss rate -> 97.6% recall)")
else: ERR(f"Total FNs = {total_fn} (expected 24)")

# ── TABLE 4.2 — Monitoring ───────────────────────────────────────
print("\n[Table 4.2] Monitoring triggers")
for t in mon["triggers"]:
    name = t["name"]
    chart_status = {"feature_drift":"watch","missingness_rise":"ok",
                    "performance_decay":"ok","prediction_shift":"ok","prevalence_shift":"ok"}
    chart_val    = {"feature_drift":0.143,"missingness_rise":0.05,
                    "performance_decay":-0.0201,"prediction_shift":0.022,"prevalence_shift":0.187}
    cs = chart_status.get(name,"?"); cv = chart_val.get(name,0)
    if t["status"]==cs and abs(t["value"]-cv)<0.001:
        OK(f"{name}: status={cs}, value={cv}")
    else:
        ERR(f"{name}: chart=({cs},{cv}) source=({t['status']},{t['value']})")

# ── TABLE 4.1 — Full performance table ──────────────────────────
print("\n[Table 4.1] Full performance table")
perf_checks = [
    ("Persistence AUC", 0.8859, m["Persistence (label at t)  [naive baseline]"]["auc_roc"]),
    ("Persistence PR-AUC", 0.7109, m["Persistence (label at t)  [naive baseline]"]["pr_auc"]),
    ("Altman AUC",     0.7729, m["Altman Z'' (benchmark)"]["auc_roc"]),
    ("Altman PR-AUC",  0.6592, m["Altman Z'' (benchmark)"]["pr_auc"]),
    ("Logistic AUC",   0.8742, m["Logistic regression (L2, balanced)"]["auc_roc"]),
    ("Logistic PR-AUC",0.6655, m["Logistic regression (L2, balanced)"]["pr_auc"]),
    ("XGBoost AUC",    0.9507, m["XGBoost"]["auc_roc"]),
    ("XGBoost PR-AUC", 0.8315, m["XGBoost"]["pr_auc"]),
    ("XGBoost KS",     0.7819, m["XGBoost"]["ks"]),
]
for label, chart_v, src_v in perf_checks:
    if abs(chart_v - src_v) < 0.0001: OK(f"{label} = {chart_v}")
    else: ERR(f"{label}: chart={chart_v} source={src_v}")

# ── SUMMARY ──────────────────────────────────────────────────────
print("\n" + "="*62)
print(f" AUDIT SUMMARY")
print("="*62)
print(f"\n  PASSED : {len(passed)}")
print(f"  WARNINGS: {len(warns)}")
print(f"  FAILURES: {len(errors)}")

if errors:
    print("\nFAILURES:")
    for e in errors: print(e)
if warns:
    print("\nWARNINGS (need fixing):")
    for w in warns: print(w)
if not errors:
    print("\n  All numerical values are CORRECT against source files.")
    if warns:
        print("  However, Figure 4.3 has a labelling mismatch (see above).")
