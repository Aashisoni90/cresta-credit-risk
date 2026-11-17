"""
brsr_extractor.py
==================
Extracts a defined set of ESG fields from a BRSR PDF filing, exploiting the
fact that BRSR has a SEBI-mandated, numbered Principle/Indicator structure
that (mostly) repeats across every company's filing.

REALISTIC EXPECTATIONS (say this in your report, don't hide it):
- This will NOT achieve 100% extraction across all companies. Some report in
  tables (easier), some in prose (harder), some skip sections without
  clearly flagging them.
- Target: ~80-90% field coverage on Environmental/Governance for companies
  that use table-based reporting; lower for Social and for prose-heavy
  filings.
- ALWAYS spot-check extracted values against the source PDF for a sample of
  companies before trusting the column. This script gets you 80% of the way
  fast; the remaining 20% needs human eyes on outputs, not more regex.

HOW IT WORKS:
1. Extracts all text from the PDF (page by page, preserving reasonable
   layout) using pdfplumber.
2. Also extracts any tables pdfplumber can detect (BRSR tabular sections).
3. Searches for each target field using a set of regex patterns anchored to
   how BRSR filings typically phrase that indicator.
4. Returns a structured record + a coverage report showing what was and
   wasn't found, so you know what needs manual follow-up.

USAGE:
    python brsr_extractor.py data/raw/COMPANY_BRSR_FY23-24.pdf

    or import and call extract_brsr_fields(pdf_path) from your pipeline.
"""

import re
import sys
import json
from pathlib import Path

import pdfplumber
import pandas as pd


# ── Field patterns ──────────────────────────────────────────────────────
# Each entry: (output_field_name, [list of regex patterns to try in order],
#              value_type)
# Patterns are intentionally permissive (case-insensitive, flexible
# whitespace/punctuation) because real filings phrase things inconsistently.
# EXTEND THIS LIST as you discover real-filing phrasings that don't match —
# treat this as a living file you'll edit after every batch of real PDFs.

FIELD_PATTERNS = [
    ("board_independent_directors_pct", [
        r"independent directors?\D{0,30}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(\d{1,3}(?:\.\d+)?)\s*%\D{0,30}?independent directors?",
    ], "float"),

    ("board_women_pct", [
        r"women\D{0,30}?directors?\D{0,30}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(\d{1,3}(?:\.\d+)?)\s*%\D{0,30}?women\D{0,10}?(?:on\s+)?(?:the\s+)?board",
        r"directors?\s+are\s+women\D{0,10}?(\d{1,3}(?:\.\d+)?)\s*%",
    ], "float"),

    ("board_meetings_held", [
        r"board meetings? held[^\d]{0,20}?(\d{1,3})",
        r"number of board meetings[^\d]{0,20}?(\d{1,3})",
    ], "int"),

    ("employee_turnover_rate_pct", [
        r"employee turnover rate\D{0,20}?(\d{1,3}(?:\.\d+)?)",
    ], "float"),

    ("women_employees_pct", [
        r"women employees\D{0,20}?(\d{1,3}(?:\.\d+)?)\s*%",
    ], "float"),

    ("ltifr", [
        r"lost time injury frequency rate[^\d]{0,80}?(\d{1,3}(?:\.\d+)?)",
        r"LTIFR[^\d]{0,80}?(\d{1,3}(?:\.\d+)?)",
    ], "float"),

    ("fatalities_count", [
        r"number of fatalities\D{0,20}?(\d{1,3})",
    ], "int"),

    ("total_energy_consumption_gj", [
        r"total energy consumption[^\d]{0,40}?([\d,]+(?:\.\d+)?)",
    ], "float_comma"),

    ("renewable_energy_gj", [
        r"from renewable sources[^\d]{0,20}?([\d,]+(?:\.\d+)?)",
    ], "float_comma"),

    ("non_renewable_energy_gj", [
        r"from non-renewable sources[^\d]{0,20}?([\d,]+(?:\.\d+)?)",
    ], "float_comma"),

    ("scope1_emissions_tco2e", [
        r"total scope\s*1 emissions?[^\d]{0,30}?([\d,]+(?:\.\d+)?)",
        r"scope\s*1[^\d]{0,30}?([\d,]+(?:\.\d+)?)\s*(?:tonnes|mt|metric ton)",
    ], "float_comma"),

    ("scope2_emissions_tco2e", [
        r"total scope\s*2 emissions?[^\d]{0,30}?([\d,]+(?:\.\d+)?)",
        r"scope\s*2[^\d]{0,30}?([\d,]+(?:\.\d+)?)\s*(?:tonnes|mt|metric ton)",
    ], "float_comma"),

    ("scope3_disclosed", [
        r"total scope\s*3 emissions?[^\.]{0,60}",
    ], "presence_text"),  # captures the phrase so we can tell if it says "not disclosed"

    ("total_employees_permanent", [
        r"total employees?\s*\(permanent\)[^\d]{0,20}?([\d,]+)",
    ], "int_comma"),
]


def _parse_value(raw: str, value_type: str):
    if raw is None:
        return None
    raw = raw.strip()
    try:
        if value_type == "float":
            return float(raw)
        if value_type == "int":
            return int(raw)
        if value_type == "float_comma":
            return float(raw.replace(",", ""))
        if value_type == "int_comma":
            return int(raw.replace(",", ""))
    except ValueError:
        return None
    return raw


# ── Table-based extraction ──────────────────────────────────────────────
# Regex-over-flattened-text cannot reliably distinguish a table's HEADER row
# (e.g. "FY 2023-24") from its DATA row (e.g. "184,320") — both are just
# numbers in the text stream once flattened. BUG FOUND DURING TESTING: this
# caused "total_energy_consumption_gj" to extract 2023 (from the year in the
# header) instead of 184320 (the actual value). Fixed by extracting actual
# table structures instead of relying on regex for these specific fields.

TABLE_ROW_LABELS = {
    "total_energy_consumption_gj": ["total energy consumption"],
    "renewable_energy_gj": ["from renewable sources"],
    "non_renewable_energy_gj": ["from non-renewable sources"],
    "employee_turnover_rate_pct": ["employee turnover rate"],
    "women_employees_pct": ["women employees"],
}


def _extract_from_tables(pdf) -> dict:
    """Scans every table on every page; for each known row label, grabs the
    first numeric-looking cell in that row (skipping header-like cells)."""
    found = {}
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                if not row or row[0] is None:
                    continue
                row_label = str(row[0]).strip().lower()
                for field_name, label_variants in TABLE_ROW_LABELS.items():
                    if field_name in found:
                        continue
                    if any(variant in row_label for variant in label_variants):
                        # scan remaining cells in this row for the first
                        # parseable number that ISN'T a bare 4-digit year
                        # (guards against grabbing "2023" out of a stray
                        # "FY 2023-24" that ended up in the same row)
                        for cell in row[1:]:
                            if cell is None:
                                continue
                            cell_clean = str(cell).replace(",", "").strip()
                            m = re.match(r"^-?\d+(\.\d+)?$", cell_clean)
                            if m:
                                val = float(cell_clean)
                                if val == int(val) and 1900 <= val <= 2100:
                                    continue  # looks like a bare year, skip
                                found[field_name] = val
                                break
    return found


def extract_brsr_fields(pdf_path: str) -> dict:
    """Returns a dict of extracted fields plus a '_coverage' meta block."""
    pdf_path = Path(pdf_path)
    result = {"source_file": pdf_path.name}
    found_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        full_text_pages = [page.extract_text() or "" for page in pdf.pages]
        full_text = "\n".join(full_text_pages)
        table_values = _extract_from_tables(pdf)

    # Normalize: collapse runs of spaces/tabs, and — importantly — collapse
    # SINGLE newlines (mid-sentence line wraps, e.g. "Independent\nDirectors")
    # into spaces, while still treating blank-line gaps as paragraph breaks.
    normalized = re.sub(r"[ \t]+", " ", full_text)
    normalized = re.sub(r"\n{2,}", "<PARA>", normalized)   # protect real paragraph breaks
    normalized = normalized.replace("\n", " ")               # join wrapped lines
    normalized = normalized.replace("<PARA>", "\n")          # restore paragraph breaks

    for field_name, patterns, value_type in FIELD_PATTERNS:
        # Prefer a table-extracted value when we have one for this field —
        # it's more reliable than text regex for numeric table data.
        if field_name in table_values:
            result[field_name] = table_values[field_name]
            found_count += 1
            continue

        matched_value = None
        for pattern in patterns:
            m = re.search(pattern, normalized, flags=re.IGNORECASE)
            if m:
                raw = m.group(1) if m.groups() else m.group(0)
                if value_type == "presence_text":
                    matched_value = raw.strip()
                else:
                    matched_value = _parse_value(raw, value_type)
                if matched_value is not None:
                    break
        result[field_name] = matched_value
        if matched_value is not None:
            found_count += 1

    result["_coverage_fields_found"] = found_count
    result["_coverage_fields_total"] = len(FIELD_PATTERNS)
    result["_coverage_pct"] = round(100 * found_count / len(FIELD_PATTERNS), 1)

    return result


def extract_batch(pdf_dir: str, out_csv: str = "data/processed/brsr_extracted.csv"):
    """Runs extraction across every PDF in a directory, writes a combined CSV
    plus prints a coverage summary so you know where to focus manual review."""
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return None

    records = [extract_brsr_fields(str(p)) for p in pdf_files]
    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Processed {len(pdf_files)} PDF(s) -> {out_csv}\n")
    print("Coverage by file:")
    print(df[["source_file", "_coverage_fields_found", "_coverage_fields_total", "_coverage_pct"]]
          .to_string(index=False))
    print(f"\nAverage coverage: {df['_coverage_pct'].mean():.1f}%")
    print("\nPer-field hit rate (which fields are reliably extractable vs. need pattern fixes):")
    field_names = [f[0] for f in FIELD_PATTERNS]
    for f in field_names:
        hit_rate = df[f].notna().mean() * 100
        print(f"  {f:35s} {hit_rate:5.1f}%")

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if Path(target).is_dir():
            extract_batch(target)
        else:
            result = extract_brsr_fields(target)
            print(json.dumps(result, indent=2, default=str))
    else:
        # Default: run against the synthetic test PDF
        print("No path given — running against the synthetic test PDF.\n")
        result = extract_brsr_fields("data/raw/TESTCO_BRSR_FY23-24.pdf")
        print(json.dumps(result, indent=2, default=str))
