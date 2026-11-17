# CRESTA — Real Data Extraction Guide (v2)

**Purpose.** Step-by-step instructions for pulling the real Bloomberg Terminal data and
real BRSR PDFs this pipeline needs, in the exact shape `src/load_bloomberg_export.py`,
`src/load_filled_template.py`, `src/load_p2_export.py`, and `src/brsr_extractor.py`
expect. This is not a generic "how Bloomberg works" doc — every step below reflects a
mistake this project already made and fixed once (see `Project_Context.md` for the
full incident). Follow it in order; skipping steps reproduces bugs that are already
solved.

**Where things stand right now (so you don't redo work):**

| Artifact | Status |
|---|---|
| `data/processed/universe_800.csv` | EQS export processed, 2,999 real tickers |
| Fundamentals, 7 chunks x 18 fields x 11 years | **DONE** — filled, validated, accepted; panel at `data/processed/bloomberg_panel_scored_gated.csv` |
| `CUR_MKT_CAP` / `VOLATILITY_260D` historical series | **OUTSTANDING** — the delivered values are point-in-time, not historical. One `BDH` re-pull, two fields only (Part 2.3) |
| Ratings history (long-format, per-action) | **Not pulled** — and no longer blocking; the §7.5 financial-criteria fallback became the primary label after the census found 4 of 3,000 tickers rated (Part 3) |
| BRSR PDFs (Track B) | **CLOSED** — 7 filings, feasibility result reached (Part 4) |

**The one piece of extraction work still worth doing is Part 2.3.** Everything
else in this guide is a record of what was done, kept because the procedures
are the corrected ones and the traps are real.

---

## Part 0 — Before you touch the Terminal

1. Bloomberg Terminal is **Windows-only**. If you're on the Mac dev machine, you need
   physical/remote access to a Windows machine with the Terminal installed and licensed.
   `blpapi`/`xbbg` cannot be used from here — confirmed structurally impossible
   (see `Project_Context.md` §Phase II Step 6). Don't spend time installing them.
2. Automated/API bulk pulls are also an **explicit PRD non-goal** (§3.2) — licensing
   restricts programmatic redistribution. Every pull below is manual, via Excel + BDH.
3. Work in **Excel**, not the raw Terminal grid — BDH formulas need a spreadsheet to
   live in and recalculate.

---

## Part 1 — Universe (already mostly done — verify, don't redo)

`data/processed/universe_800.csv` already exists from an EQS screen. If you need to
regenerate or extend it:

1. In the Terminal, run **EQS** (Equity Screening).
2. Filters, **in this exact order** (each one fixes a real mistake made the first time):
   - `Country/Region of Domicile: India` — **only India**, not "Asia Pacific
     (Developed); India" (that OR's in all of developed Asia — caught this exact bug).
   - `Trading Status: Active; Inactive` — **both**, never just one. Active-only misses
     every delisted/bankrupt company (survivorship bias). Inactive-only gives zero
     healthy contrast companies.
   - `Common Stock` only, primary security where selectable.
   - Leave **market cap as display-only**, do NOT filter by it in the screen. A
     `Current Market Cap` floor silently deletes delisted companies (their current cap
     is 0/N/A) — apply the floor in code instead (`build_company_universe.py` does this
     for active companies only).
   - Exclude `Financials` sector — display-only/tag, don't hard-filter, so sector stays
     usable as a feature (PRD §6.7).
3. Export **all** matched rows. Bloomberg's results grid caps on-screen display at
   3,000 rows — confirm the actual export file row count matches the "N Securities"
   total shown, not the capped display.
4. Columns to include in the export: `Ticker`, `Company Name`, `GICS Sector Name`,
   `Trading Status`, `Delisting Date`, `Current Market Cap`, `ICRA Long Term Issuer
   Rating`, `Crisil LT Rating Lookup`.
5. Save as CSV into `data/raw/` (e.g. `eqs_export_3.csv`).
6. Run it through `build_company_universe.py::process_eqs_export()`. This already
   handles:
   - Dedup of `-PP` (partly-paid), `-RT` (rights), `-W` (warrants), `-BE` suffixes —
     `RELIANCE IN` genuinely appears 3x in raw EQS output; this is real, not an error.
   - Market-cap floor applied to active companies only.
   - Financial-sector tagging (kept, not dropped).

---

## Part 2 — Financial fundamentals pull (DONE — and one correction outstanding)

**Status: complete.** All seven chunks were pulled, validated and accepted; the
panel is `data/processed/bloomberg_panel_scored_gated.csv`. This part is kept
as the record of what was actually done, because the earlier version of it
described a `BDH` procedure and a field list that were both superseded. Do not
follow a procedure from memory of the first draft.

### 2.1 What was actually pulled — the corrected 18 mnemonics

`src/generate_formula_templates.py` is authoritative. The list is:

```
BS_TOT_ASSET, BS_CUR_ASSET_REPORT, BS_CUR_LIAB, BS_RETAIN_EARN,
EBIT, TOTAL_EQUITY, BS_TOT_LIAB2, RETURN_ON_ASSET,
TOT_DEBT_TO_TOT_EQY, INTEREST_COVERAGE_RATIO, CUR_MKT_CAP, CUR_RATIO,
NET_INCOME, SALES_REV_TURN, VOLATILITY_260D, SHORT_AND_LONG_TERM_DEBT,
CF_CASH_FROM_OPER, IS_INT_EXPENSE
```

**Two mnemonics in the first draft were invalid** and were corrected against a
live `BDP` test on `RELIANCE IN Equity` FY2023 before any bulk pull:

| first draft | returned | correct mnemonic |
|---|---|---|
| `TOTAL_LIABILITIES` | `#N/A Invalid Field` | `BS_TOT_LIAB2` |
| `INT_COVERAGE_RATIO` | `#N/A N/A` | `INTEREST_COVERAGE_RATIO` |

`TOTAL_ASSETS`, `CUR_CAPITAL`, `CUR_LIAB`, `RETAIN_EARN`, `TOT_EQUITY` and
`TOT_LIAB_AND_EQY` from the first draft are **not** the mnemonics that were
used. Testing one ticker's full field list on a throwaway sheet first is what
caught this, and it remains step zero for any new pull.

### 2.2 `BDP` per cell, not `BDH` per block

The templates carry one live `BDP()` per (ticker, year, field):

```
=BDP($A2, "BS_TOT_ASSET", "FUND_PER", "Y", "EQY_FUND_YEAR", $B2)
```

`FUND_PER="Y"` plus the `EQY_FUND_YEAR` override selects the annual figure for
that fiscal year, and the sheet is already one row per (ticker, fiscal_year) —
so no flattening step is needed. This replaced the `BDH` block approach, which
returns a vertical block per ticker that then has to be reshaped by hand; the
reshape is where the first real pull was corrupted.

**Batch size is a hard constraint, not a preference.** One chunk is
500 companies x 18 fields x 11 years, roughly **99,000 `BDP` requests** — close
to a one-day capacity ceiling. Chunk 5 returned `#N/A Daily Capacity` when
attempted too soon after chunk 4. Spread remaining chunks across separate days.

### 2.3 THE CORRECTION STILL OUTSTANDING — two fields need a `BDH` re-pull

**`EQY_FUND_YEAR` governs fundamental fields only.** `CUR_MKT_CAP` and
`VOLATILITY_260D` are real-time *market* fields: Bloomberg ignores the override
and returns the value as at the pull date for every year requested.

Measured on the delivered panel, over tickers reporting at least five years:

| field | tickers carrying ONE value across all years |
|---|---:|
| `cur_mkt_cap` | **98.4%** |
| `volatility_260d` | **95.4%** |
| every accounting field | **0.0%** |

So market cap on a FY2014 row was the 2026 figure. Both derived features
(`log_mkt_cap`, `equity_volatility`) were **removed from the model in v1.2.0**
as look-ahead — see `MODEL_CARD.md` §5b.

**To restore them, these two fields need a historical `BDH` pull**, which is
the one place `BDH` is the right tool:

```
=BDH($A2, "CUR_MKT_CAP", "3/31/"&$B2, "3/31/"&$B2, "Per=Y", "Days=A", "Fill=P")
```

Pull only these two fields, for the same ticker list and years 2015-2025. That
is 2,999 x 2 x 11 ≈ 66,000 requests — one day's budget, on its own day. The
result is a tidy `ticker, fiscal_year, CUR_MKT_CAP, VOLATILITY_260D` CSV that
joins onto the existing panel; nothing else needs re-pulling.

`tests/test_no_lookahead.py` will **fail** once that pull lands, by design: it
asserts the fields are still constant per ticker, so that whoever re-pulls is
told the exclusion can now be lifted.

### 2.4 Known traps — checked automatically, but verify by eye once

- **Test one ticker's full field list on a throwaway sheet before opening any
  large live-formula template.** Two invalid mnemonics and a wrong-security
  ticker were all caught this way. The cost of re-pulling a corrupted
  5,500-row chunk is far higher than the ten minutes this takes.
- **Ragged rows.** Real exports sometimes carry trailing or leading blank
  columns that make row width inconsistent. The loaders handle this
  (`_read_ragged_csv` in `load_bloomberg_export.py`), but re-saving a
  hand-edited CSV from Excel can silently reintroduce an unnamed column —
  this exact bug (`total_assets` becoming a date string) corrupted the first
  real pull. Export straight from Excel to CSV; do not hand-edit.
- **`#N/A` variants** — `#N/A N/A`, `#N/A Invalid Field`,
  `#N/A Field Not Applicable`, `#N/A Invalid Security`, `#N/A Authorization`,
  `#N/A Daily Capacity`, `#N/A Requesting Data...` and
  `#N/A Mandatory parameter [SECURITY] cannot be empty` are all handled by
  `clean_numeric()` and become typed NaN. **Never find-and-replace these with
  0** (invariant 1). `#N/A Daily Capacity` is the one that is *not* a data
  fact — it means re-pull that chunk tomorrow.
- **Banks and NBFCs** return `#N/A` across `current_assets`,
  `current_liabilities`, `ebit`, `interest_coverage` and `cur_ratio`. Expected
  and correct — 86.3% of financial-sector rows are null on the current-asset
  pair against 23.5% elsewhere. They are excluded from the model (§5.6); do
  not chase this as a pull error.
- **Negative-equity companies** show `#N/A` for `debt_to_equity`. Also
  expected: the ratio is undefined, which is signal, not a fetch failure.
- **Wrong-security tickers.** Sanity-check each chunk's `BS_TOT_ASSET` column
  for order-of-magnitude outliers before accepting a pull — the `CCL IN`
  incident (an entirely different company) was caught only this way.
  `TATAINVE IN Equity` FY2020 and `UVDH IN Equity` FY2017 remain unresolved
  suspects.
- **Dead tickers are logged, never dropped** — `data/raw/failed_tickers.csv`,
  171 so far. Distressed and delisted firms fail to resolve
  disproportionately, so a silent drop reintroduces survivorship bias
  (invariant 4).

---

## Part 3 — Ratings history (the current real blocker)

This is what the pipeline is actually stuck on. Without it, the primary label
(§7.2) can't be built and the project falls back to §7.5 financial criteria (with
the mandatory circularity guard — see `src/fallback_label.py`).

1. Field to search in Browse Fields: **`ICRA Long Term Issuer Rating`** —
   issuer-level, domestic-scale, exactly what's needed.
2. `CRISIL` returns mostly **instrument-level** ratings (Senior Unsecured Debt, Fixed
   Deposit, etc.), not issuer-level. If used, document it explicitly as a proxy
   substitution — don't silently pool instrument- and issuer-level ratings.
3. **Do not pull this as an EQS point-in-time snapshot only** — a snapshot tells you
   *current* rating, not the history needed to build a graded label per fiscal year.
   You need the **rating action history** per ticker: use `CRAT` (Bloomberg's
   ratings-history function) or export via the Fixed Income / Credit tab's ratings
   history view, one row per (ticker, rating_date, agency, rating).
4. **Do not use the Credit tab's DRSK fields** (`1Yr Default BBG Default Risk Prob`,
   `IG1` bands, CDS spreads) as the label. That's Bloomberg's own proprietary model
   output, not a realised outcome — it's earmarked as a *third benchmark* (PRD
   Appendix, `Project_Context.md` §I6), never as ground truth.
5. Expected output format for `label_construction.py`:
   ```
   ticker,rating_date,agency,rating
   ```
6. First step before a full pull: run the **coverage census** — count how many of the
   ~800 tickers have any non-`NR` ICRA/CRISIL rating at all in `universe_800.csv`'s
   `icra_rating`/`crisil_rating` columns. This determines whether ratings become the
   primary label or the §7.5 fallback is promoted — don't skip straight to a full
   ratings-history pull before checking coverage is worth it.

---

## Part 4 — BRSR PDFs (Track B pilot — CLOSED, 7 filings)

**Status: closed as a feasibility result.** See
`reports/phase6/track_b_feasibility.md`. No further BRSR collection is needed
for the current scope. This part is kept because the procedure below is the
one that worked, and because the pilot will be worth repeating once BRSR has
accumulated enough years to support an out-of-time test.

### 4.1 The rule that caught two look-ahead downloads

**Pull FY2022-23 BRSR, NOT FY2023-24.** The panel's last feature year is
`fiscal_year_bbg = 2023` (Indian FY2022-23), and those features predict a label
in FY2023-24. An FY2023-24 filing is dated *after* the outcome it would help
predict — using it is look-ahead and violates §7.4. FY2022-23 is also the first
mandatory year, so earlier filings mostly will not exist.

**Never trust the filename or the download page for the year.** Two of the
first three filings collected were the wrong year, and one company's IR
portal *defaults its year selector to the most recent year*. Gate every file
on what the filing declares about itself:

```bash
PYTHONPATH=src python src/brsr_pilot_validate.py <file.pdf>
```

That reads BRSR Section A item 9, "Financial year for which reporting is being
done". Three filings used three different encodings of that one SEBI-prescribed
field — a year label (`2022-23`), a numeric date table, and prose (`1st April,
2022 to 31st March, 2023`) — so the gate carries three detectors. Expect a
fourth. A rejected file is logged to
`data/raw/brsr_pilot/rejected_filings.csv`, never silently discarded.

### 4.2 Where to get them

**BSE corporate filings are the most reliable route.** Attachments under
`bseindia.com/xml-data/corpfiling/AttachHis/` are directly fetchable with a
`bseindia.com` referer, and BRSRs are often filed there as standalone
documents rather than buried in a 900-page annual report.

Routes that did not work, so they are not retried:
- **The NSE bulk archive** — its company field needs an autocomplete that does
  not render under automation, and its date fields reject typed input.
- **Some IR CDNs 403 every non-browser request.** One filing (Sandesh) could
  not be retrieved at all; a full Chrome header set did not help. That needs a
  manual save, and is logged as not obtained rather than dropped.
- **URL guessing.** Naming is unstable across years at the same company
  (`TEIL-BRSR-25-26.pdf` vs `BRSR-TEIL-2022-2023.pdf`).

Save as `data/raw/brsr_pilot/<TICKER>_BRSR_FY22-23.pdf`, where `<TICKER>` is
the first token of the Bloomberg ticker (`PTCIN IN Equity` -> `PTCIN`).

### 4.3 Extraction — rows, not regex over flattened text

Use `src/brsr_rows.py`, not `src/brsr_extractor.py`. The original regex
extractor scored 28-36% coverage on real filings and the cause was structural,
not tunable: column padding varies per filer, and a units column sits *between*
the label and the value, so widening the gap budget lets a pattern skip into
the wrong column rather than fail honestly.

`brsr_rows.py` works on `pdftotext -layout` *lines*: find the row matching the
label, then read numeric tokens positionally. On the same three filings that
scored 28.6 / 35.7 / 35.7%, it scores 81.8 / 100.0 / 81.8%.

```bash
PYTHONPATH=src python src/brsr_rows.py data/raw/brsr_pilot/
PYTHONPATH=src python src/brsr_pilot_report.py      # the full pilot verdict
```

### 4.4 Two things the pilot established that will not change

- **The energy identity is the ESG analogue of the balance-sheet identity.**
  Renewable + non-renewable must reconcile with the stated total within 2%. One
  filing of seven disagrees with itself by 255%. When it fails, the intensity
  ratio is **refused, not clipped** — clipping 312% to 100% manufactures a
  plausible number from a filing that does not support one.
- **Coverage is not the binding constraint; disclosure is.** Mean coverage was
  77.9%, comfortably over the 70% gate, and on disclosed fields extraction is
  100% accurate against hand-checked values. But **only 1 of 7 filings
  disclosed enough to form an energy-intensity ratio at all**: the
  renewable/non-renewable split is a voluntary *Leadership* indicator.

---

## Part 5 — After extraction, hand it back

Once you have real files, tell me which of these you have and I'll run them through
the actual loaders and report exactly what breaks:

- [ ] Filled `bloomberg_template_chunk_*.csv` (financials, Part 2)
- [ ] Ratings history CSV (Part 3)
- [ ] BRSR PDFs (Part 4, optional)

I do not need Bloomberg access myself — hand me the exported files and I run
`load_p2_export.py` / `load_bloomberg_export.py` / `label_construction.py` /
`quality_gates.py` against them directly and report the coverage numbers, gate
failures, and any new formatting issues, the same way every prior real pull was
diagnosed in `Project_Context.md`.
