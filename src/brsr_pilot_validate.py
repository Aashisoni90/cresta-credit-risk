"""
brsr_pilot_validate.py
======================
Gate every BRSR PDF on the fiscal year it DECLARES ABOUT ITSELF before any
field is extracted from it.

WHY THIS EXISTS. The pilot's first three downloads were saved under correct
FY22-23 filenames and two of them were not FY2022-23 filings at all: one was
FY2023-24, one FY2025-26. A filename is an assertion by whoever typed it; the
"Financial year for which reporting is being done" field in BRSR Section A is
an assertion by the filer. Only the second one is evidence.

The year matters more here than it would elsewhere. The panel's last feature
year is fiscal_year_bbg = 2023 (Indian FY2022-23), and those features predict a
label in FY2023-24. A BRSR for FY2023-24 is published AFTER the outcome it
would help predict. Feeding it to the model is look-ahead (PRD §7.4) and it
would not announce itself in any metric -- it would just quietly inflate
Track B's apparent uplift, which is the exact claim this pilot exists to test.

Per invariant 4, a rejected file is LOGGED, never silently skipped: a filing
that cannot be year-verified is a fact about BRSR data availability, which is
itself a Track B finding.

Text is read with `pdftotext -layout` (poppler) rather than pdfplumber so the
gate runs even where the extraction stack is not installed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

TARGET_FY = "2022-23"
REJECT_LOG = Path("data/raw/brsr_pilot/rejected_filings.csv")

# BRSR Section A item 9 states the reporting year outright. Filers phrase the
# label consistently (SEBI prescribes it) but vary the separator and casing.
# Filers label item 9 consistently (SEBI prescribes the wording) but the label
# often wraps across lines, with a table header injected between its halves:
#     "Financial year for which"  /  "9  Start date  End date"  /  "reporting is
#     being done". So the two halves are matched independently rather than as
#     one contiguous phrase.
_DECLARED = re.compile(
    r"financial\s+year\s+for\s+which(?:.{0,120}?)reporting\s+is\s+being\s+done"
    r"\D{0,40}?(20\d{2})\s*[-–—/]\s*(\d{2,4})",
    re.I | re.S,
)
# Other filers answer item 9 only as a dated table, so no "2022-23" string
# exists anywhere near the label. Those state the window as a start/end pair on
# one laid-out row. The row is isolated first and the dates read out of it --
# matching across an arbitrary run of column padding needs a space budget, and
# any budget is wrong for the next filing's column widths.
_ROW = re.compile(r"^.*current\s+financial\s+year.*$", re.I | re.M)
_DATE = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b")

# A third variant: item 9 answered as a PROSE date range rather than a year
# label or a numeric table -- "1st April, 2022 to 31st March, 2023". Three
# filings, three encodings of the same mandated field; this is why the gate
# reads the filing instead of trusting a filename.
_PROSE = re.compile(
    r"financial\s+year\s+for\s+which(?:.{0,120}?)reporting\s+is\s+being\s+done"
    r".{0,80}?1\s*(?:st)?\s+april\s*,?\s*(20\d{2})"
    r".{0,40}?31\s*(?:st)?\s+march\s*,?\s*(20\d{2})",
    re.I | re.S,
)


def pdf_text(path: Path) -> str:
    """Whole-document text via poppler. Raises if poppler is absent."""
    out = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def declared_fiscal_year(text: str) -> tuple[str | None, str]:
    """Return (fy_string, how_it_was_found). fy_string is e.g. '2022-23'."""
    m = _DECLARED.search(text)
    if m:
        start, end = m.group(1), m.group(2)
        return f"{start}-{end[-2:]}", "section_a_item_9"

    m = _PROSE.search(text)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if end == start + 1:
            return f"{start}-{str(end)[-2:]}", "item_9_prose_date_range"

    for row in _ROW.findall(text):
        dates = _DATE.findall(row)
        if len(dates) < 2:
            continue
        (d1, m1, y1), (d2, m2, y2) = dates[0], dates[1]
        start, end = int(y1), int(y2)
        # An Indian FY runs 1 Apr Y to 31 Mar Y+1. Both the month pattern and
        # the year gap must hold, or this is some other dated row entirely.
        if (int(m1), int(m2)) == (4, 3) and end == start + 1:
            return f"{start}-{str(end)[-2:]}", "current_fy_date_range"

    return None, "not_found"


def validate(path: Path, target: str = TARGET_FY) -> dict:
    text = pdf_text(path)
    fy, how = declared_fiscal_year(text)
    principles = len(set(re.findall(r"PRINCIPLE\s+(\d)", text, re.I)))
    return {
        "file": path.name,
        "declared_fy": fy,
        "detected_via": how,
        "target_fy": target,
        "ok": fy == target,
        "reason": (
            "declared year matches target" if fy == target
            else "reporting year could not be read from the filing" if fy is None
            else f"filing is for FY{fy}, not FY{target}"
        ),
        "distinct_principles": principles,
        "chars": len(text),
    }


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("data/raw/brsr_pilot")
    pdfs = sorted(p for p in root.glob("*.pdf") if not p.name.startswith("._"))
    if not pdfs:
        print(f"no PDFs in {root}")
        return 1

    rows = [validate(p) for p in pdfs]
    df = pd.DataFrame(rows)

    for r in rows:
        mark = "PASS" if r["ok"] else "REJECT"
        print(f"[{mark}] {r['file']:<34} declared FY{r['declared_fy'] or '?'}"
              f"  ({r['detected_via']})  principles={r['distinct_principles']}/9")
        if not r["ok"]:
            print(f"         -> {r['reason']}")

    bad = df[~df.ok]
    if len(bad):
        REJECT_LOG.parent.mkdir(parents=True, exist_ok=True)
        bad.to_csv(REJECT_LOG, index=False)
        print(f"\n{len(bad)} of {len(df)} rejected; logged to {REJECT_LOG}")
    else:
        print(f"\nall {len(df)} filings are FY{TARGET_FY}")

    # Rejections are a reportable result, not a crash.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


# ---------------------------------------------------------------------------
# Internal-consistency checks on EXTRACTED values
# ---------------------------------------------------------------------------
# The financial panel is protected by the balance-sheet identity
# (TA == TE + TL within 2%), which has caught parser faults repeatedly. BRSR
# has an equivalent: Principle 6's energy tables must reconcile, because the
# Leadership table is a break-up of the Essential table's total.
#
# They do not always reconcile. Shree Renuka Sugars' FY2022-23 filing states a
# total of 4,325,818 GJ while its own renewable + non-renewable rows sum to
# 15,378,019 GJ -- 3.6x. Both figures were read from the correct FY22-23
# column; the filing disagrees with itself, because self-generated renewable
# power (bagasse) is excluded from the Essential table's "total".
#
# This matters because PRD §6.8's intensity features are RATIOS. A renewable
# share computed off these two tables is 312%, which is not a value any model
# should be trained on, and nothing downstream would flag it. So the ratio is
# refused when the identity fails, rather than clipped to 100% -- clipping
# would manufacture a plausible number out of a filing that does not support
# one. NaN here is signal, exactly as it is for debt_to_equity under negative
# equity (invariant 1).

ENERGY_IDENTITY_TOLERANCE = 0.02


def check_energy_identity(rec: dict, tol: float = ENERGY_IDENTITY_TOLERANCE) -> dict:
    """Does renewable + non-renewable reconcile with the stated total?

    Returns a verdict dict. `renewable_share_usable` is False whenever the
    identity fails or an input is absent; callers must not compute an
    intensity ratio when it is False.
    """
    total = rec.get("total_energy_consumption_gj")
    ren = rec.get("renewable_energy_gj")
    non = rec.get("non_renewable_energy_gj")

    if total is None or ren is None or non is None:
        return {"energy_identity": "incomplete", "rel_error": None,
                "renewable_share_usable": False,
                "detail": "one or more energy fields absent"}

    if total <= 0:
        return {"energy_identity": "degenerate", "rel_error": None,
                "renewable_share_usable": False,
                "detail": f"stated total is {total}, cannot form a ratio"}

    rel = abs((ren + non) - total) / total
    ok = rel <= tol
    return {
        "energy_identity": "pass" if ok else "FAIL",
        "rel_error": round(rel, 4),
        "renewable_share_usable": ok,
        "detail": (
            f"renewable {ren:,.0f} + non-renewable {non:,.0f} = {ren + non:,.0f} "
            f"vs stated total {total:,.0f} ({rel:.1%} apart)"
        ),
    }
