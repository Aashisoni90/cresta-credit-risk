"""
brsr_rows.py
============
Row-based BRSR field extraction, replacing regex-over-flattened-text.

WHY A SECOND EXTRACTOR. `brsr_extractor.py` matches a label and then allows a
bounded run of non-digits before the value (`\\D{0,30}` and friends). On the
synthetic test PDF that works. On all three real FY2022-23 filings it does not,
for two structural reasons that no gap width fixes:

  1. BRSR tables are COLUMN-LAID-OUT. The gap between a label and its value is
     whatever column padding the filer's template used -- 60+ spaces is normal,
     and it differs per filer. Any fixed budget is right for one layout and
     wrong for the next.
  2. A units column sits BETWEEN the label and the value:
         "Total Scope 1 emissions (Break-up of the   Metric tonnes of CO2   5,77,040.30   4,34,989.70"
     so the text between label and value is not padding at all -- it is more
     label. Widening the budget lets the pattern skip into the WRONG column
     rather than failing honestly.

So this module works on `pdftotext -layout` LINES: find the row whose text
matches the label, then read the numeric tokens out of that row positionally.
Column 0 is the current fiscal year in every FY2022-23 filing inspected --
verified against the explicit "FY 22-23 | FY 21-22" headers in all three.

A number is only a value if it stands alone. "CO2" and "Scope 1" contain digits
and are NOT values; the token pattern requires non-alphabetic boundaries, and
the search starts after the end of the label match so the label's own digits
are already consumed.

Indian digit grouping ("5,77,040.30") is the norm in these filings, so comma
positions carry no meaning and are simply stripped.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# A standalone number: optional Indian/Western comma grouping, optional
# decimals. Must not touch a letter on either side, so "CO2", "N2O", "SF6",
# "Scope 1" and "FY22-23" never register as values.
_NUM = re.compile(r"(?<![A-Za-z0-9.,-])(\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9])")

# Tokens that are years rather than measurements. A BRSR row should not have a
# bare year in a value column, but headers repeated mid-table do occur.
_YEARISH = re.compile(r"^(19|20)\d{2}$")


def pdf_lines(path: Path) -> list[str]:
    out = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.splitlines()


def _numbers_after(line: str, start: int) -> list[float]:
    vals = []
    for m in _NUM.finditer(line, start):
        tok = m.group(1)
        if _YEARISH.match(tok.replace(",", "")):
            continue
        try:
            vals.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return vals


# (field, label regex, column index, section anchor or None)
# The section anchor disambiguates labels that recur. "Board of Directors"
# appears in a dozen places; only the one under item 19 is the gender matrix.
SPECS = [
    ("ltifr",
     r"lost\s*time\s*injury\s*frequency\s*rate", 0, None),
    ("fatalities_count",
     r"\bno\.?\s*of\s*fatalities|number\s*of\s*fatalities", 0, None),
    ("total_energy_consumption_gj",
     r"total\s*energy\s*consumption\s*\(\s*A\s*\+\s*B\s*\+\s*C\s*\)", 0, None),
    ("renewable_energy_gj",
     r"total\s*energy\s*consumed\s*from\s*renewable\s*sources", 0, None),
    ("non_renewable_energy_gj",
     r"total\s*energy\s*consumed\s*from\s*non[- ]?renewable\s*sources", 0, None),
    ("scope1_emissions_tco2e",
     r"total\s*scope\s*1\s*emissions", 0, None),
    ("scope2_emissions_tco2e",
     r"total\s*scope\s*2\s*emissions", 0, None),
    # Item 19 gender matrix: columns are Total (A) | No. (B) | % (B/A).
    ("board_women_pct",
     r"board\s*of\s*directors", 2,
     r"participation/?\s*inclusion/?\s*representation\s*of\s*women"),
    ("board_women_count",
     r"board\s*of\s*directors", 1,
     r"participation/?\s*inclusion/?\s*representation\s*of\s*women"),
    ("board_total_directors",
     r"board\s*of\s*directors", 0,
     r"participation/?\s*inclusion/?\s*representation\s*of\s*women"),
    # Item 20 turnover: FY(t) Male | Female | Total, then FY(t-1), FY(t-2).
    ("employee_turnover_rate_pct",
     r"permanent\s*employees", 2,
     r"turnover\s*rate\s*for\s*permanent\s*employees"),
]


def extract(path: Path) -> dict:
    lines = pdf_lines(path)
    rec: dict = {"source_file": path.name}

    for field, label_re, col, anchor in SPECS:
        lab = re.compile(label_re, re.I)
        # Restrict the search window to after the section anchor, when given.
        start_idx = 0
        if anchor:
            anc = re.compile(anchor, re.I)
            hit = next((i for i, ln in enumerate(lines) if anc.search(ln)), None)
            if hit is None:
                rec[field] = None
                continue
            start_idx = hit

        value = None
        for ln in lines[start_idx:start_idx + 400 if anchor else len(lines)]:
            m = lab.search(ln)
            if not m:
                continue
            nums = _numbers_after(ln, m.end())
            if len(nums) > col:
                value = nums[col]
                break
        rec[field] = value

    found = sum(1 for k, v in rec.items() if k != "source_file" and v is not None)
    total = len(rec) - 1
    rec["_coverage_fields_found"] = found
    rec["_coverage_fields_total"] = total
    rec["_coverage_pct"] = round(100 * found / total, 1)
    return rec


if __name__ == "__main__":
    import json
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/brsr_pilot")
    paths = ([target] if target.is_file()
             else sorted(p for p in target.glob("*.pdf") if not p.name.startswith("._")))
    for p in paths:
        print(json.dumps(extract(p), indent=2))
