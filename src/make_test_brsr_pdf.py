"""
make_test_brsr_pdf.py
======================
Generates a synthetic PDF that mimics the real structure of a BRSR filing
(Section A / B / C, numbered Principles and Essential Indicators), purely so
brsr_extractor.py can be tested end-to-end before real filings are available.

This is NOT a substitute for testing on real BRSR PDFs — company filings
vary in table vs. prose formatting, which this synthetic file won't fully
capture. Real test PDFs from the team are still needed (see the extraction
guide). This script exists so the extraction logic can be proven to at
least work correctly against a known-good, known-structure input first.
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def build_test_pdf(path="data/raw/TESTCO_BRSR_FY23-24.pdf"):
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Business Responsibility and Sustainability Report", styles["Title"]))
    story.append(Paragraph("TESTCO Limited — FY 2023-24", styles["Heading2"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("SECTION A: GENERAL DISCLOSURES", styles["Heading1"]))
    story.append(Paragraph("1. Details of the listed entity", styles["Normal"]))
    story.append(Paragraph("CIN: L12345MH1990PLC000000", styles["Normal"]))
    story.append(Paragraph("Total employees (permanent): 4,582", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("SECTION B: MANAGEMENT AND PROCESS DISCLOSURES", styles["Heading1"]))
    story.append(Paragraph(
        "Details of the Board of Directors: the Board comprises 9 Directors, of whom "
        "4 are Independent Directors (44.4%). 2 of the 9 Directors are women (22.2%).",
        styles["Normal"]))
    story.append(Paragraph("Number of Board meetings held during the year: 6", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("SECTION C: PRINCIPLE-WISE PERFORMANCE DISCLOSURE", styles["Heading1"]))

    story.append(Paragraph(
        "PRINCIPLE 1: Businesses should conduct and govern themselves with integrity, "
        "and in a manner that is Ethical, Transparent and Accountable", styles["Heading2"]))
    story.append(Paragraph("Essential Indicator 1: Details of training programs on anti-corruption policy", styles["Normal"]))
    story.append(Paragraph("Total number of directors trained: 9 (100%)", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "PRINCIPLE 3: Businesses should respect and promote the well-being of all employees", styles["Heading2"]))
    story.append(Paragraph("Essential Indicator 1: Employee well-being measures", styles["Normal"]))
    table_data = [
        ["Category", "FY 2023-24", "FY 2022-23"],
        ["Employee turnover rate (%)", "14.2", "16.8"],
        ["Women employees (%)", "27.3", "24.1"],
        ["Median remuneration - women (INR lakh)", "9.4", "8.7"],
        ["Median remuneration - men (INR lakh)", "10.1", "9.5"],
    ]
    t = Table(table_data, colWidths=[220, 100, 100])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "PRINCIPLE 5: Businesses should respect and promote human rights", styles["Heading2"]))
    story.append(Paragraph("Essential Indicator 3: Details of safety related incidents", styles["Normal"]))
    story.append(Paragraph("Lost Time Injury Frequency Rate (LTIFR) per one million person-hours worked: 0.34", styles["Normal"]))
    story.append(Paragraph("Number of fatalities: 0", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "PRINCIPLE 6: Businesses should respect and make efforts to protect and restore the environment",
        styles["Heading2"]))
    story.append(Paragraph("Essential Indicator 1: Total energy consumption", styles["Normal"]))
    table_data2 = [
        ["Parameter", "FY 2023-24", "Unit"],
        ["Total energy consumption (A+B)", "184,320", "GJ"],
        ["From renewable sources (A)", "41,760", "GJ"],
        ["From non-renewable sources (B)", "142,560", "GJ"],
        ["Energy intensity per rupee of turnover", "0.0021", "GJ/INR"],
    ]
    t2 = Table(table_data2, colWidths=[220, 100, 100])
    t2.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Essential Indicator 4: Greenhouse Gas (GHG) emissions", styles["Normal"]))
    story.append(Paragraph("Total Scope 1 emissions: 12,450 tonnes CO2 equivalent", styles["Normal"]))
    story.append(Paragraph("Total Scope 2 emissions: 8,120 tonnes CO2 equivalent", styles["Normal"]))
    story.append(Paragraph("Total Scope 3 emissions: Not disclosed this reporting period", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "PRINCIPLE 8: Businesses should promote inclusive growth and equitable development",
        styles["Heading2"]))
    story.append(Paragraph("This principle is not material to the Company's operations and is therefore not applicable.", styles["Normal"]))

    doc.build(story)
    print(f"Test BRSR PDF written to {path}")


if __name__ == "__main__":
    build_test_pdf()
