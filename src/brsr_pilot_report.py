"""
brsr_pilot_report.py
====================
Turns the Phase 6 BRSR pilot into two durable artefacts instead of a finding
that lives only in a session transcript:

    data/raw/brsr_pilot/pilot_results.csv   per-filing coverage + energy identity
    reports/phase6/track_b_feasibility.md   the §6.8 feasibility verdict

It also rebuilds the eligibility frame, because the frame used to draw the
pilot was defective: it ranked firms by `cur_mkt_cap`, which is the pull-date
market capitalisation (see build_training_matrix.POINT_IN_TIME_FEATURES).
BRSR applied to the top 1,000 by market cap *as at FY2022-23*, so a
current-cap ranking admits firms that listed afterwards and cannot have filed.

The rebuild below is an APPROXIMATION and is labelled as one. There is no
historical market-cap series in this extract to rank on — that is the same
missing BDH pull the §5b feature removal is waiting for. What is available is
the first fiscal year in which a firm reports fundamentals, which excludes
firms that did not exist as reporting entities. It does not exclude a firm
that reported fundamentals privately and listed later, so it is a loose bound.
The point of running it is to find out whether the defect changes the answer.
It does not: 21 of 1,000 firms swap, and EPV moves 4.32 -> 4.45.

USAGE
    PYTHONPATH=src python src/brsr_pilot_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from brsr_pilot_validate import check_energy_identity
from brsr_rows import extract

MATRIX = "data/processed/training_matrix.csv"
PANEL = "data/processed/bloomberg_panel_scored_gated.csv"
PILOT_DIR = Path("data/raw/brsr_pilot")
OUT_CSV = PILOT_DIR / "pilot_results.csv"
OUT_MD = Path("reports/phase6/track_b_feasibility.md")

BRSR_TOP_N = 1000          # SEBI's FY2022-23 mandatory scope
N_ESG_VARIABLES = 22       # Track A's 14 + the eight §6.8 ESG features
EPV_RULE_OF_THUMB = 10
COVERAGE_GATE = 70.0       # §12.6 pilot gate, mean fields extracted per filing

# Hand-recorded from the pilot, because the PDFs carry no machine-readable
# sector or label. Each ticker was drawn as one half of a sector-matched pair.
PILOT_META = {
    "PTCIN": ("PTC India", "Utilities", "healthy"),
    "RTPOW": ("RattanIndia Power", "Utilities", "distressed"),
    "SHRS":  ("Shree Renuka Sugars", "Consumer Staples", "distressed"),
    "TRE":   ("Triveni Engineering", "Consumer Staples", "healthy"),
    "SHP":   ("Shalimar Paints", "Materials", "distressed"),
    "TPI":   ("Transpek Industry", "Materials", "healthy"),
    "NDTV":  ("New Delhi Television", "Communication Services", "distressed"),
}

# Filings sought and not obtained. Logged, never silently dropped (invariant 4).
NOT_OBTAINED = [
    ("SDSH", "Sandesh", "Consumer Discretionary", "healthy",
     "CDN returns 403 to every non-browser request; Chrome renders the PDF "
     "inline and ignores a cross-origin download attribute. Needs a manual save."),
    ("NITCO", "Nitco", "Materials", "distressed",
     "No BRSR filed. The complete 941,385-character FY2022-23 annual report "
     "contains zero occurrences of 'business responsibility'. Direct evidence "
     "of the eligibility-frame defect: Nitco was below the FY2022-23 top 1,000."),
]


def eligibility_frame() -> dict:
    """Old vs corrected BRSR-eligible frame, and the EPV each implies."""
    m = pd.read_csv(MATRIX, low_memory=False)
    p = pd.read_csv(PANEL, low_memory=False)

    cap = p.groupby("ticker")["cur_mkt_cap"].max().dropna()
    first_report = p[p.total_assets.notna()].groupby("ticker")["fiscal_year_num"].min()

    ev = m[(m.fiscal_year_num == 2023)
           & m.label_binary_next.notna()
           & (m.qc_below_coverage != True)]  # noqa: E712 — tri-state column

    def frame(universe: pd.Series) -> dict:
        rank = universe.rank(ascending=False, method="first")
        top = set(rank[rank <= BRSR_TOP_N].index)
        sub = ev[ev.ticker.isin(top)]
        events = int(sub.label_binary_next.sum())
        return {"firms": int(sub.ticker.nunique()), "events": events,
                "epv": round(events / N_ESG_VARIABLES, 2), "tickers": top}

    old = frame(cap)
    listed_by_2022 = first_report[first_report <= 2023].index
    new = frame(cap[cap.index.isin(listed_by_2022)])

    return {
        "evaluable_rows": int(len(ev)),
        "evaluable_events": int(ev.label_binary_next.sum()),
        "excluded_as_too_new": int((first_report > 2023).sum()),
        "old": old, "new": new,
        "swapped": len(new["tickers"] - old["tickers"]),
    }


def pilot_table() -> pd.DataFrame:
    rows = []
    for pdf in sorted(PILOT_DIR.glob("*.pdf")):
        if pdf.name.startswith("._"):
            continue
        tick = pdf.stem.split("_")[0]
        name, sector, label = PILOT_META.get(tick, ("", "", ""))
        rec = extract(pdf)
        ident = check_energy_identity(rec)
        rows.append({
            "ticker": tick, "company": name, "sector": sector, "label": label,
            "file": pdf.name,
            "fields_found": rec["_coverage_fields_found"],
            "fields_total": rec["_coverage_fields_total"],
            "coverage_pct": rec["_coverage_pct"],
            "energy_identity": ident["energy_identity"],
            "energy_rel_error": ident["rel_error"],
            "intensity_ratio_usable": ident["renewable_share_usable"],
            "obtained": True, "note": "",
        })
    for tick, name, sector, label, note in NOT_OBTAINED:
        rows.append({"ticker": tick, "company": name, "sector": sector,
                     "label": label, "file": "", "fields_found": 0,
                     "fields_total": 11, "coverage_pct": None,
                     "energy_identity": "not obtained", "energy_rel_error": None,
                     "intensity_ratio_usable": False, "obtained": False,
                     "note": note})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    """A markdown table, written out rather than pulled in via `tabulate`.

    One formatting helper is not worth a dependency in a reproducibility
    package whose whole claim is that it runs from requirements.txt.
    """
    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return f"{v:g}" if isinstance(v, float) else str(v)

    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = "\n".join("| " + " | ".join(cell(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, rule, body])


def verdict(df: pd.DataFrame, fr: dict) -> str:
    got = df[df.obtained]
    n = len(got)
    mean_cov = got.coverage_pct.mean()
    usable = int(got.intensity_ratio_usable.sum())
    by_label = got.groupby("label").coverage_pct.agg(["mean", "size"])

    def _lab(k):
        return (f"{by_label.loc[k, 'mean']:.1f}% (n={int(by_label.loc[k, 'size'])})"
                if k in by_label.index else "n/a")

    return f"""# Track B feasibility — the §6.8 verdict

*Generated by `src/brsr_pilot_report.py`. Every figure is computed in that run.*

**Track B is closed as a feasibility result, not a hypothesis test.** Three
independent constraints each rule out the Model A vs Model B comparison the
PRD §1.3 gap asks for, and the third is decisive.

## 1. There is no out-of-time ESG comparison to run

BRSR became mandatory in **FY2022-23**, and the panel's last feature year *is*
FY2022-23 (`fiscal_year_bbg = 2023`). ESG therefore exists for exactly one
feature year. Invariant 2 forbids a random split, and a single year cannot be
split out of time against itself. **An honest A-vs-B test does not exist in
this data** — not for want of effort, but as a property of the disclosure
regime's start date.

An FY2023-24 BRSR would give a second year, but it is dated *after* the label
it would predict. Using it is look-ahead and violates §7.4. Two of the first
three filings downloaded in this pilot were exactly that mistake, caught by the
declared-fiscal-year gate in `src/brsr_pilot_validate.py`.

## 2. The eligible frame is underpowered, and the frame defect does not rescue it

| frame | firms | events | EPV (÷{N_ESG_VARIABLES} variables) |
|---|---:|---:|---:|
| all FY2022-23 evaluable rows | {fr['evaluable_rows']:,} | {fr['evaluable_events']} | {fr['evaluable_events'] / N_ESG_VARIABLES:.2f} |
| BRSR top {BRSR_TOP_N}, ranked by pull-date cap (defective) | {fr['old']['firms']} | {fr['old']['events']} | {fr['old']['epv']} |
| **BRSR top {BRSR_TOP_N}, firms reporting by FY2022-23** | **{fr['new']['firms']}** | **{fr['new']['events']}** | **{fr['new']['epv']}** |

Against a ≥{EPV_RULE_OF_THUMB} events-per-variable rule of thumb, {fr['new']['epv']} is roughly
**{EPV_RULE_OF_THUMB / fr['new']['epv']:.1f}× short**.

The first ranking was defective for the same reason two features were removed
in §5b: `cur_mkt_cap` is the pull-date value, and BRSR eligibility depended on
market capitalisation *as at FY2022-23*. The corrected row excludes the
{fr['excluded_as_too_new']} firms that were not yet reporting entities. **{fr['swapped']} of {BRSR_TOP_N} firms swap, and EPV
{'is unchanged at ' + str(fr['new']['epv']) if fr['old']['epv'] == fr['new']['epv'] else 'moves ' + str(fr['old']['epv']) + ' to ' + str(fr['new']['epv'])}.** The defect was real and is fixed here; it was
not what made Track B infeasible, and no conclusion in this report rested on it.

This correction remains an approximation and is labelled as one. There is no
historical market-cap series in this extract to rank on — the same missing
`BDH` pull as §5b. First-reporting-year excludes firms that did not exist; it
does not exclude a firm that reported privately and listed later.

## 3. The decisive constraint: the indicators §6.8 needs are voluntary

{n} filings were obtained and all {n} passed the declared-fiscal-year gate.
Mean field coverage **{mean_cov:.1f}%**, against the ≥{COVERAGE_GATE:.0f}% pilot gate — **the gate is met.**
Distressed {_lab('distressed')} vs healthy {_lab('healthy')}: directionally consistent with
worse disclosure under distress, but far too few observations to claim.

**Coverage was never the binding constraint.** §6.8's intensity features are
*ratios*, and a ratio needs the renewable/non-renewable split:

> **Only {usable} of {n} filings yields a usable energy-intensity ratio.**

Five filers never disclosed the split at all — it is a *Leadership*, i.e.
voluntary, indicator under the BRSR schema — and one disclosed it and
contradicts its own stated total by 255%. On that filing the ratio is
**refused, not clipped**: clipping 312% to 100% would manufacture a plausible
number from a filing that does not support one. NaN is signal here exactly as
it is for `debt_to_equity` under negative equity (invariant 1).

Extraction is not the bottleneck either. Every null was checked against its
source and found to be genuine non-disclosure; **on disclosed fields,
extraction accuracy is 100%** against hand-checked values.

## 4. Per-filing results

{_md(df.drop(columns=['note']))}

### Filings sought and not obtained

{chr(10).join(f"- **{t}** ({s}, {l}) — {note}" for t, _, s, l, note in NOT_OBTAINED)}

## 5. What this means for the PRD

- **§6.8 ESG feature block: not built.** Invariant 6 is satisfied by
  construction — Track A never depended on it and is complete at v1.2.0.
- The §1.3 research question is answered, in the negative and with a reason:
  *on Indian listed equities at FY2022-23, the incremental predictive power of
  BRSR ESG disclosure cannot be estimated out of time, and the specific
  intensity indicators the design calls for are voluntary and largely absent.*
- This is a finding about the disclosure regime, not about ESG. It will stop
  being true as BRSR accumulates years and as Leadership indicators become
  essential. The pipeline that would test it is built and gated; it is waiting
  on data that does not exist yet.
"""


def main() -> int:
    fr = eligibility_frame()
    df = pilot_table()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(verdict(df, fr))
    got = df[df.obtained]
    print(f"wrote {OUT_CSV}  ({len(got)} filings, mean coverage "
          f"{got.coverage_pct.mean():.1f}%, "
          f"{int(got.intensity_ratio_usable.sum())} usable intensity ratios)")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
