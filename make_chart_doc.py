"""
make_chart_doc.py
Creates a single Word document (.docx) with ALL 19 charts/tables
embedded in order, labelled, with notes on where each goes in the report.
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

FIG_DIR = Path(r"c:\Users\Aashli\Downloads\cresta-main\cresta-main\reports\capstone_figures")
OUT     = Path(r"c:\Users\Aashli\Downloads\cresta-main\cresta-main\CRESTA_All_Charts_and_Tables.docx")

doc = Document()

# ── Page size A4, narrow margins ──────────────────────────────────────
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

section = doc.sections[0]
section.page_width  = int(8.27 * 914400)
section.page_height = int(11.69 * 914400)
section.left_margin   = int(1.0 * 914400)
section.right_margin  = int(1.0 * 914400)
section.top_margin    = int(1.0 * 914400)
section.bottom_margin = int(0.8 * 914400)

def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(16 if level == 1 else 13)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    return p

def caption(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.runs[0]
    run.font.size = Pt(10)
    run.italic = True
    run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    doc.add_paragraph()

def note(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.runs[0]
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x70, 0x30, 0xA0)
    return p

def add_figure(doc, filename, cap, report_loc, note_text=None, width=6.2):
    path = FIG_DIR / filename
    if not path.exists():
        p = doc.add_paragraph(f"[FILE NOT FOUND: {filename}]")
        p.runs[0].font.color.rgb = RGBColor(0xFF, 0, 0)
        return
    doc.add_picture(str(path), width=Inches(width))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, cap)
    note(doc, f"  Report location: {report_loc}")
    if note_text:
        note(doc, f"  Note: {note_text}")
    doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────
# Title page
# ─────────────────────────────────────────────────────────────────────
heading(doc, "CRESTA Capstone Project Report")
heading(doc, "All Charts, Diagrams & Tables", level=2)
p = doc.add_paragraph(
    "All figures generated from real project data only.\n"
    "Source: MODEL_CARD.md | VALIDATION_REPORT.md | reports/phase7/OUTCOMES_ANALYSIS.md\n"
    "Model version: A-1.2.0-real  |  Test set: n=4,340 firm-years (FY2022-FY2023)\n\n"
    "19 figures total  |  Generated: 2026-10-28"
)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p.runs:
    run.font.size = Pt(10)
doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────
# CHAPTER 3 FIGURES
# ─────────────────────────────────────────────────────────────────────
heading(doc, "Chapter 3 — Methodology & Implementation", level=2)
doc.add_paragraph()

add_figure(doc,
    "table_3_1_data_splits.png",
    "Table 3.1  Dataset Splits — Out-of-Time Partition (FY2014–FY2023)",
    "Chapter 3, Section 3.3 (Dataset)",
    "Out-of-time split only; asserted in code. Label: financial-criteria distress proxy (§7.5)."
)

add_figure(doc,
    "figure_3_1_class_distribution.png",
    "Figure 3.1  Class Distribution Across Data Splits",
    "Chapter 3, Section 3.3 (Dataset)",
    "Prevalence: Train 27.9%, Val 24.6%, Test 22.7%. Higher than literature norm — property of §7.5 label."
)

add_figure(doc,
    "figure_3_2_architecture.png",
    "Figure 3.2  CRESTA End-to-End System Architecture (v1.2.0-real)",
    "Chapter 3, Section 3.1 (Block Diagram)",
    "Two-track design: Track A (financial distress, operational) and Track B (ESG, closed as infeasible).",
    width=6.5
)

add_figure(doc,
    "figure_3_3_pipeline_flowchart.png",
    "Figure 3.3  CRESTA Data & Model Pipeline Flowchart",
    "Chapter 3, Section 3.3 (Software Description / Flowchart)",
    "Key invariants: never zero-fill; out-of-time split only; preprocessing on train only.",
    width=4.5
)

add_figure(doc,
    "table_3_2_config.png",
    "Table 3.2  XGBoost Model Configuration (v1.2.0-real)",
    "Chapter 3, Section 3.3 (Configuration Table)",
    "14 features used after removing 2 look-ahead: log_mkt_cap, equity_volatility."
)

add_figure(doc,
    "table_3_3_features.png",
    "Table 3.3  14 Model Features — Feature Blocks, Names, and Definitions",
    "Chapter 3, Section 3.3 (Feature Engineering)",
    "Block 3 (Profitability) dominates with 64.7% SHAP attribution.",
    width=6.5
)

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────
# CHAPTER 4 FIGURES
# ─────────────────────────────────────────────────────────────────────
heading(doc, "Chapter 4 — Results & Analysis", level=2)
doc.add_paragraph()

add_figure(doc,
    "table_4_1_performance.png",
    "Table 4.1  Out-of-Time Test Performance — All Models (n=4,340, scored once)",
    "Chapter 4 (main results table)",
    "XGBoost: AUC=0.9507, PR-AUC=0.8315, KS=0.7819, Recall=0.9756 @ threshold=0.245.",
    width=6.5
)

add_figure(doc,
    "figure_4_1_auc_comparison.png",
    "Figure 4.1  AUC-ROC Comparison — All Models vs Baselines",
    "Chapter 4, Section 4.1 (Evaluation Setup / Metrics)",
    "Dashed line = persistence baseline AUC=0.8859. XGBoost exceeds by +0.0648 (DeLong p<1e-300)."
)

add_figure(doc,
    "figure_4_2_stratified_auc.png",
    "Figure 4.2  Stratified Performance — Pooled vs Early-Warning AUC",
    "Chapter 4, Section 4.2 (Results Discussion)",
    "The honest headline is the early-warning stratum (healthy at t): XGBoost AUC=0.9434, n=3,334, 5.1% prevalence."
)

add_figure(doc,
    "figure_4_3_rolling_origin.png",
    "Figure 4.3  Rolling-Origin Walk-Forward Validation (5 Folds)",
    "Chapter 4, Section 4.2 (Stability / Supplementary)",
    "XGBoost: 0.9284 +/- 0.0160 early-warning across 5 origins. Single-split headline (0.9507) within 1 std dev of fold mean."
)

add_figure(doc,
    "figure_4_4_shap_importance.png",
    "Figure 4.4  Global SHAP Feature Importance (Test Set, n=4,340)",
    "Chapter 4, Section 4.3 (Explainability / SHAP)",
    "ebit_to_assets alone: 38.5%. Block 3 (Profitability): 64.7% total. SHAP explains the MODEL, not the world."
)

add_figure(doc,
    "figure_4_5_shap_blocks.png",
    "Figure 4.5  SHAP Attribution by Feature Block",
    "Chapter 4, Section 4.3 (SHAP Block Analysis)",
    "Model is overwhelmingly a profitability model. Block 5 (Z-Score) second at 12.5%."
)

add_figure(doc,
    "figure_4_6_calibration.png",
    "Figure 4.6  Calibration by Decile — Model is Systematically Over-Confident",
    "Chapter 4, Section 4.4 (Error Analysis / Calibration)",
    "Decile 8: predicted 69.8% vs observed 45.2%. Mean gap deciles 7-10: -0.175. Score is a RANK not a probability."
)

add_figure(doc,
    "figure_4_7_delong.png",
    "Figure 4.7  DeLong Test — Delta-AUC vs Naive Persistence Baseline",
    "Chapter 4, Section 4.2 (Comparison with Baselines)",
    "Altman Z'' is SIGNIFICANTLY WORSE than no model (delta=-0.1110, p<1e-300). Only XGBoost beats persistence."
)

add_figure(doc,
    "figure_4_8_sector_auc.png",
    "Figure 4.8  Per-Sector Model Performance (XGBoost v1.2.0-real)",
    "Chapter 4, Section 4.4 (Error Analysis / Sector Breakdown)",
    "Weakest sector: Real Estate (AUC=0.8819, EW=0.8155) at highest distress prevalence (38.0%)."
)

add_figure(doc,
    "figure_4_9_version_history.png",
    "Figure 4.9  Model Version History — Each Correction Lowered the Metric",
    "Chapter 4, Section 4.1 or Chapter 3 (Methodology / Defect Corrections)",
    "v1.0.0->v1.1.0: financials excluded (-0.0039 AUC). v1.1.0->v1.2.0: look-ahead removed (-0.0043 AUC)."
)

add_figure(doc,
    "figure_4_10_fn_analysis.png",
    "Figure 4.10  False Negative Analysis and Threshold-Based Metrics",
    "Chapter 4, Section 4.4 (Outcomes Analysis / Error Analysis)",
    "24 false negatives at threshold=0.245: 12 signal absent, 11 signal unused, 1 data missing. Recall=97.6%."
)

add_figure(doc,
    "table_4_2_monitoring.png",
    "Table 4.2  Monitoring Triggers — Sec 14.3 (Retrain Recommended: No)",
    "Chapter 4, Section 4.4 (Monitoring / Governance)",
    "feature_drift on WATCH (PSI=0.143; consecutive_loss_years PSI=0.4295). All other triggers OK."
)

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────
# CHAPTER 5 FIGURES
# ─────────────────────────────────────────────────────────────────────
heading(doc, "Chapter 5 — Advantages, Limitations & Applications", level=2)
doc.add_paragraph()

add_figure(doc,
    "figure_5_1_track_b.png",
    "Figure 5.1  Track B (ESG/BRSR) Feasibility — Three Independent Constraints",
    "Chapter 5, Section 5.2 (Limitations) or Chapter 3 (Methodology)",
    "Track B closed as a finding: only 1/7 filings yield the energy-intensity ratio. EPV=4.32 vs >=10 needed."
)

# ─────────────────────────────────────────────────────────────────────
# Summary index
# ─────────────────────────────────────────────────────────────────────
doc.add_page_break()
heading(doc, "Figure & Table Index", level=2)
doc.add_paragraph()

index = [
    ("Table 3.1", "table_3_1_data_splits.png",        "Dataset splits (train/val/test)"),
    ("Figure 3.1","figure_3_1_class_distribution.png", "Class distribution bar chart"),
    ("Figure 3.2","figure_3_2_architecture.png",       "CRESTA system architecture diagram"),
    ("Figure 3.3","figure_3_3_pipeline_flowchart.png", "Data & model pipeline flowchart"),
    ("Table 3.2", "table_3_2_config.png",              "XGBoost hyperparameter configuration"),
    ("Table 3.3", "table_3_3_features.png",            "14 model features with definitions"),
    ("Table 4.1", "table_4_1_performance.png",         "Full performance results table"),
    ("Figure 4.1","figure_4_1_auc_comparison.png",     "AUC-ROC comparison bar chart"),
    ("Figure 4.2","figure_4_2_stratified_auc.png",     "Stratified AUC (pooled vs early-warning)"),
    ("Figure 4.3","figure_4_3_rolling_origin.png",     "Rolling-origin walk-forward folds"),
    ("Figure 4.4","figure_4_4_shap_importance.png",    "SHAP global feature importance"),
    ("Figure 4.5","figure_4_5_shap_blocks.png",        "SHAP attribution by block (pie chart)"),
    ("Figure 4.6","figure_4_6_calibration.png",        "Decile calibration plot"),
    ("Figure 4.7","figure_4_7_delong.png",             "DeLong test vs persistence baseline"),
    ("Figure 4.8","figure_4_8_sector_auc.png",         "Per-sector AUC breakdown"),
    ("Figure 4.9","figure_4_9_version_history.png",    "Model version history (3 versions)"),
    ("Figure 4.10","figure_4_10_fn_analysis.png",      "False negative analysis + threshold metrics"),
    ("Table 4.2", "table_4_2_monitoring.png",          "Monitoring triggers (Sec 14.3)"),
    ("Figure 5.1","figure_5_1_track_b.png",            "Track B ESG feasibility constraints"),
]

table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = "Label"
hdr[1].text = "Filename"
hdr[2].text = "Description"
for cell in hdr:
    for run in cell.paragraphs[0].runs:
        run.bold = True
        run.font.size = Pt(9)

for label, fname, desc in index:
    row = table.add_row().cells
    row[0].text = label
    row[1].text = fname
    row[2].text = desc
    for cell in row:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(8.5)

doc.save(str(OUT))
print(f"\nSaved: {OUT}")
print(f"Size:  {OUT.stat().st_size // 1024} KB")
print(f"Contains {len(index)} charts/tables embedded from real project data.")

