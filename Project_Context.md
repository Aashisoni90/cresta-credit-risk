# Project Context — CRESTA Development Log

**Project:** CRESTA — Credit Risk Early-warning System with Transparent Analytics
**Subject:** ESG-Augmented Corporate Distress Prediction for Indian Listed Companies
**Institution:** MPSTME, NMIMS Indore — Capstone, AY 2026–27
**Document type:** Chronological development log / technical appendix
**Last updated:** 2026-09-19

> **Purpose of this document.** This is not a summary. It is a reconstruction of the
> entire development process in order, including hypotheses that proved wrong, code that
> failed, and directions that were abandoned. Failures are retained deliberately: several
> of the most consequential design decisions in this project came directly from things
> that broke, and a log that hides them is not reproducible.

---

## 1. Objective / Problem Definition

### 1.1 Primary objective (Track A)

Predict corporate financial distress in Indian listed companies **twelve months ahead**,
using audited financial and market data, with every prediction attributable to specific,
inspectable financial drivers via SHAP.

### 1.2 Secondary objective (Track B)

Test whether ESG signals — sourced from SEBI-mandated BRSR filings — provide
**statistically significant incremental** predictive power beyond financial ratios.

### 1.3 The stated research gap

No published study tests whether ESG risk signals provide statistically significant
incremental predictive power for corporate distress specifically in the **Indian listed
universe**, during the period in which SEBI's BRSR mandate has materially increased ESG
data availability.

### 1.4 The two-track constraint

Track A must be **fully operational with zero ESG data present**. ESG is an optional
feature block injected at feature-assembly time, never a required input at any interface.
Track B is ~15% of effort and must never block Track A.

A **negative result in Track B is a valid result.** Any design in which the project only
"works" if the hypothesis is confirmed is a badly designed study.

### 1.5 Success criteria (PRD §3.3)

| Metric | Target |
|---|---|
| AUC-ROC (out-of-time) | ≥ 0.80 |
| PR-AUC | ≥ 0.45 |
| KS statistic | ≥ 0.40 |
| Recall @ threshold | ≥ 0.70 |
| Uplift vs Altman Z″ | ≥ +0.10 AUC, significant |
| Batch scoring | 800 companies < 60s |
| Pipeline reproducibility | 100% deterministic |

---

## 2. Context and Initial Understanding

### 2.1 State at the start of this log

A working codebase existed, tested **only against synthetic data**:

| Script | Status entering this log |
|---|---|
| `build_company_universe.py` | Runs; produces 51-company starter CSV |
| `altman_zscore.py` | Tested on 3 synthetic companies across safe/grey/distress |
| `label_construction.py` | Tested on synthetic ratings; 0→1→2 progression correct |
| `brsr_extractor.py` | Tested on synthetic BRSR PDF; 2 real bugs found and fixed |
| `train_model.py` | Full pipeline runs end-to-end on synthetic data |

**Critical acknowledged gap:** nothing had been tested against real Bloomberg exports or
real BRSR PDFs. The synthetic tests proved the *logic*; only real data could surface
*formatting* failures.

### 2.2 Two data issues already confirmed before this log began

These came from a real 5-company Bloomberg test pull (Reliance, TCS, HDFC Bank, Yes Bank,
Suzlon Energy) via a column-block Excel template, 12 fields, 2015–2025 yearly.

**Issue 1 — Financial-sector companies fail on five fields.**
HDFC Bank and Yes Bank both returned `#N/A` for `current_assets`, `current_liabilities`,
`ebit`, `interest_coverage`, `cur_ratio` — *identically*. Banks have no meaningful
current/non-current balance sheet split, so these concepts do not exist rather than
failing to fetch.
→ **Decision:** exclude `sector_type == "financial"` from the primary model, consistent
with Altman's own exclusion of financial institutions.

**Issue 2 — `debt_to_equity` is `#N/A` where book equity is negative.**
Confirmed on Suzlon FY2015–FY2021. This is mathematically expected, not a fetch error: a
leverage ratio against a negative denominator is not interpretable.
→ **Decision:** (a) create a binary `negative_equity_flag` feature, (b) leave
`debt_to_equity` as NaN — never zero-fill, (c) check whether `altman_zscore.py`'s X4
calculation needs a guard.

---

## 3. Chronological Development Log

### PHASE I — Code review and the X4 guard question

**Step 1. Read all four core modules.**

Findings from `altman_zscore.py`:
- Implements Z″ = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4 (Altman, Hartzell & Peck 1995)
- `REQUIRED_COLUMNS` confirmed the financial-sector problem structurally: X1 needs
  current assets/liabilities, X3 needs EBIT — precisely the fields banks don't report
- **Line 75 confirmed the suspicion:** `X4 = total_equity_book / total_liabilities` had
  **no guard whatsoever**

**Step 2. Diagnosed exactly what was wrong with X4 — and what was NOT wrong.**

This distinction drove the entire fix and is worth preserving:

- *Not a bug:* a negative X4 from negative equity. This is economically **correct** — a
  firm whose liabilities exceed its assets *should* score deep in the distress zone.
  Suppressing it would delete the clearest insolvency signal in the sample.
- *Actual bug 1:* `total_liabilities == 0` produces `±inf` silently.
- *Actual bug 2 (subtler and more dangerous):* a non-positive denominator **flips the
  sign** of the ratio. For a distressed firm this converts a solvency measure into a
  spuriously *favourable* number.
- *Actual bug 3:* `altman_flag_extreme_value` used `|Z″| > 50` to mean "suspected units
  error". A legitimately catastrophic firm was therefore indistinguishable from a scaling
  bug, sending the analyst hunting a bug that doesn't exist.

**Step 3. Built `src/load_bloomberg_export.py`** for the wide column-block export shape
(ticker row 1, company name row 2, field headers row 3, data row 4+).

Design decisions taken here:
- Parse **generically against the shape**, not against one file — block detection walks
  columns; a ticker cell opens a block, merged-cell blanks continue it, a fully blank
  spacer column closes it
- `FIELD_ALIASES` matched on alphanumerics-only, so `Total Assets` / `TOTAL_ASSETS` /
  `BS_TOT_ASSET` all resolve identically
- Unknown headers **kept under a slug and reported**, never dropped
- `clean_numeric` handles all `#N/A` variants, thousands separators, `%`, and
  accounting-style negatives `(1,234)` → `-1234.0`
- Unparseable → NaN, **never 0**

**Step 4. Fiscal-year derivation centralised.** `parse_fiscal_year` maps a 31/03/2020
period end to `FY2019-20`, the same April-start convention as `label_construction.py`.
Rationale recorded in code: an off-by-one here misaligns *every* label by a year, so it is
derived in exactly one place.

**Step 5. Rewrote the X4 logic** with `_safe_divide` (NaN when denominator ≤ 0), and split
the diagnostic flags:
- `altman_flag_negative_equity`
- `altman_flag_x4_uncomputable`
- `altman_flag_extreme_value` (|Z″| > 50)
- `altman_flag_check_scaling` = extreme **AND NOT** negative equity ← the meaningful one

Also added a guard so that if the ingestion step already produced `negative_equity_flag`,
the scorer **does not clobber it** but warns on disagreement — a mismatch means the equity
column changed between ingestion and scoring.

---

### PHASE II — Scale-up investigation (Step 1 of the session brief)

**Step 6. Environment check for a Python-API bulk pull.**

| Check | Result |
|---|---|
| `blpapi` installed | No |
| `xbbg` installed | No |
| Bloomberg process running | No |
| Terminal installed | No |
| Port 8194 (Desktop API) | CLOSED |
| Platform | Darwin arm64 (macOS, Apple Silicon) |

**Key realisation, stronger than "not installed":** the Bloomberg Terminal is
Windows-only, and `blpapi`'s Desktop API works by connecting to a Terminal running locally
on `localhost:8194`. Installing `blpapi` on this Mac would produce a library with nothing
to connect to. **Step 2A was not a `pip install` away — it was structurally impossible on
this machine.**

**Step 7. Located the real filled data.** `bloomberg_pull_template_filled - Sheet1.csv`:
407 rows, 37 companies, 2015–2025.

**Step 8. Discovered a critical parsing defect.**

The header row names 16 columns plus a trailing empty one:
```
ticker,company_name,fiscal_year,date_range_used,total_assets,...,cur_ratio,
```
But every data row carries an **extra unlabelled period-end-date column** at position 5:
```
JETIN IN Equity,Jet Airways,2015,1/1/2015 - 12/31/2015,12/31/2015,191608.2,...
                                                        ^^^^^^^^^^ unnamed
```
Both rows have 17 fields, so **pandas raises no error**. It assigns
`total_assets = "12/31/2015"` and shifts *every* subsequent field by one.

This is the single most dangerous defect found in the project: silent, total corruption
with no warning, producing a model trained on garbage.

**Step 9. Proved the corrected mapping rather than assuming it.**

Applied the balance-sheet identity as a test:
```
TA == TE + TL on 331 complete rows
max absolute error   = 0.0001
max relative error   = 1.42e-05
within 0.1% tolerance = 100.0%
```
This is *proof*, not inference — a one-column shift destroys the identity immediately.
The check was then built into the loader to run on **every** load.

**Step 10. Profiled the real data.** Results:

| Field | Coverage |
|---|---|
| total_assets … cur_ratio (7 BS fields) | 81.3% |
| roa, cur_mkt_cap | 79.1% |
| interest_coverage | 76.7% |
| debt_to_equity | 66.1% |

**Issue 2 reproduced perfectly at scale:** negative equity in **64 rows across 15
companies**; `debt_to_equity` NaN in **62 of those 64**, and in **0 of 267** positive-equity
rows. The established handling was correct.

**Issue 1 did not appear** — zero companies showed the banks/NBFC signature, confirming the
financial exclusion had already been applied upstream.

**Step 11. Found two further data defects.**

*Defect: `CCL IN Equity` is the wrong security.* Median total assets **7.02** against
4.1×10⁴–4.0×10⁶ for every other company — 4–5 orders of magnitude off. Labelled "Coffee
Day Enterprises", but Coffee Day trades as `COFFEE IN`; `CCL IN` is CCL Products, a
different company. It also reported `debt_to_equity = 0.0` in negative-equity years, which
is meaningless.

*Defect: three companies returned nothing at all* — Videocon Industries (pulled under a
**US** ticker `0123831D US Equity`), Essar Steel India, ABG Shipyard. **All three are
distressed.** Preferentially losing distressed firms biases the model in the most damaging
possible direction.

**Step 12. Extracted the BDH formula** from the vertical template to recover the mnemonics:
```
=BDH(A2,"BS_TOT_ASSET,BS_CUR_ASSET_REPORT,BS_CUR_LIAB,BS_RETAIN_EARN,EBIT,
TOTAL_EQUITY,BS_TOT_LIAB2,RETURN_ON_ASSET,TOT_DEBT_TO_TOT_EQY,
INTEREST_COVERAGE_RATIO,CUR_MKT_CAP,CUR_RATIO","1/1/"&C2,"12/31/"&C2,"Per=CY")
```
`Per=CY` is significant: the `fiscal_year` column is the **calendar year in which the
fiscal year ends**. For a 31-March Indian year end, `fiscal_year=2020` = FY ending
31-Mar-2020 = `FY2019-20` — a **−1 offset** from `label_construction.py`'s convention.

**Step 13. Built `src/load_filled_template.py`** — reads by **position**, discards header
names, re-proves the balance-sheet identity every load, flags unit outliers by
order-of-magnitude (generic, not hardcoded to CCL), and writes dead tickers to
`data/raw/failed_tickers.csv` rather than dropping them.

---

### PHASE III — Reading the PRD, and finding a contradiction inside it

**Step 14. Read `CRESTA_PRD.docx` in full** (700 lines, 302 paragraphs, 64 tables).

Three conclusions changed as a direct result:

**(a) Automated Bloomberg ingestion is an explicit NON-GOAL.** PRD §3.2: *"Automated
Bloomberg ingestion in v1. Terminal licensing restricts programmatic redistribution;
extraction is a controlled, human-initiated step."* §5.1 confirms access mode is *"Manual
Excel export (BDH), controlled and human-initiated."*
→ The `bulk_bloomberg_pull.py` script would have **violated a documented scope decision**,
independent of the environment finding.

**(b) Two "recommended additions" turned out to be PRD-mandated gaps, not enhancements.**
- `debt_to_assets` is a **required** Block 1 feature (§6.1) — needs
  `SHORT_AND_LONG_TERM_DEBT`, absent from the 12-field pull
- `emissions_intensity` is defined in §6.8 as **(Scope 1 + Scope 2) ÷ revenue**, and
  `energy_intensity` as energy ÷ revenue — **both impossible without
  `SALES_REV_TURN`**. Track B was unbuildable as specified.

**(c) Most of Block 6 needs no re-pull.** All temporal-dynamics features
(`roa_yoy_change`, `altman_z_3yr_trend`, `consecutive_loss_years`, …) compute from the 12
fields already held.

**Step 15. ⚠️ Found a direct internal contradiction in the PRD.**

This is the single most consequential finding in the log.

| PRD section | Treats as |
|---|---|
| §6.1 Block 1 | `negative_equity_flag` — a **FEATURE** |
| §6.4 Block 4 | `interest_coverage`, `interest_coverage_below_1_flag` — **FEATURES** |
| §7.5 Fallback label | negative book equity — a **LABEL CRITERION** |
| §7.5 Fallback label | interest coverage < 1 for 2+ years — a **LABEL CRITERION** |

The same quantities appear on both sides of the equation. Using both makes the label a
near-deterministic function of the features. The model would score AUC ≈ 1.0, comfortably
clear the §3.3 target of ≥0.80, and SHAP would attribute the prediction almost entirely to
the defining features — **a result that looks excellent and means nothing.**

The forward-looking design in §7.4 (features FY(t) → label FY(t+1)) *softens* but does not
remove this: negative equity and sub-1 coverage are highly persistent year-over-year, so
FY(t) values predict FY(t+1) status nearly mechanically.

**Why this is dangerous rather than merely untidy:** §7.5 is the *fallback* label — it
activates precisely when ratings coverage disappoints, which §17's risk register rates as
Medium likelihood. **The trap springs exactly when the project is under pressure.**

**Step 16. Built `src/fallback_label.py`** implementing §7.5 verbatim, with a
`CIRCULAR_FEATURE_MAP` that prints the exact feature-exclusion list for whichever criteria
are enabled, on every run. Criteria can be toggled individually so a label independent of
the desired features can be constructed.

Ran on real data:
```
338/407 company-years scored
  label 0 (stable):        216 (63.9%)
  label 1 (deteriorating):  10 (3.0%)
  label 2 (distressed):    112 (33.1%)
Forward-looking target: 307 rows retain an FY(t+1) outcome
Forward positive rate: 35.5%
```
**Observation:** 33.1% positive vastly exceeds §8.3's anticipated 5–15%, because the
starter universe is ~50% distress-history *by construction*. Correct for this curated
sample, but it means metrics from these 37 companies are **not comparable** to the §3.3
targets. Those become meaningful only on the broad 800-company screen.

---

### PHASE IV — Bloomberg EQS universe construction (iterative, with corrections)

The user drove the Terminal; this log records the review of each iteration.

**Iteration 1 — `Country = India` alone.**
Rejected. EQS by default returns only **currently-active** securities, silently excluding
every company delisted after bankruptcy — i.e. the entire positive class. This is the same
survivorship trap that had already consumed 3 of the 37 companies.

Prescribed filters: include inactive/delisted; Common Stock only; primary security; exclude
Financials; market-cap floor.

**Iteration 2 — over-correction.**
The screen came back as `Trading Status: Inactive` **only** — 1,043 securities, exclusively
dead ones, zero live companies. The opposite failure: no healthy contrast group at all.
Corrected to `Active; Inactive`.

Also caught: `Country/Region of Domicile: Asia Pacific (Developed); India` — these are OR'd,
and **Asia Pacific (Developed) does not contain India** (India is Asia Pacific *Emerging*).
The screen was pulling all of developed Asia alongside India, which is why that line still
showed 216,266 matches. Corrected to India only → **6,640 securities**.

**Iteration 3 — market cap.**
Two problems identified before commit:
- **Currency:** with the selector on USD, `500M` means ≈₹4,200 crore — roughly 8× the
  intended floor. In INR millions: ₹500 cr = `5000`, ₹1,000 cr = `10000`.
- **The `Current` trap:** a delisted company's *current* market cap is zero or N/A, so
  `Current Market Cap > 5000M` would filter out **every inactive company just added** —
  silently rebuilding the survivorship bias while appearing to have handled it.

→ **Decision:** set market cap to **display-only**, export all 6,640, and apply the floor
in Python to *active companies only*. Rationale: one screen, fully auditable, filtering
logic lives in versioned code rather than an unreproducible Terminal session, and it yields
the universe-composition numbers §12.5 requires for the Data Quality screen.

**Iteration 4 — the duplicate-security error (an incorrect claim, later corrected).**

Earlier in this log it was asserted that Bloomberg composite `IN` tickers are
one-per-company, so primary-security deduplication was unnecessary. **This was wrong.**

The results grid disproved it: `RELIANCE IN` appeared **three times** —
`RELIANCE INDUSTR`, `RELIANCE INDU-PP`, `RELIANCE INDU-RT`. Same for Bharti Airtel. These
are not NSE/BSE dual listings but **partly-paid shares (`-PP`) and rights entitlements
(`-RT`)** carried as separate securities, evidenced by their blank Price and Total Return
columns. The identical repeated ratio rows on the Credit tab (`9.0 / 86% / 100% / 1.4x`)
are the same duplicates inheriting parent fundamentals.

→ Deduplication **is** required. Decision: handle it in code (drop `-PP`/`-RT`/`-W`
suffixes, collapse on ticker, log removals) rather than in the Terminal, because code also
catches renamed and re-listed entities.

**Iteration 5 — a row cap risk.**
The results grid read **"3000 of 6640 Securities"**. Flagged: if the export truncates at
3,000 while sorted by market cap descending, **the lost half is small-caps and delisted
names — exactly where distress concentrates**, rebuilding survivorship bias from the other
end. Must be verified before the export is trusted.

**Iteration 6 — the Credit tab misdirection.**
The user surfaced the Credit tab hoping for ratings. What it actually contains:
`1Yr Default BBG Default Risk Prob`, `IG1` bands, CDS spreads, Debt/EBITDA, EBIT/Int.

**This is Bloomberg's proprietary DRSK model output, not agency ratings.** Assessment:
- **Invalid as a label** — it is another vendor's *model output*, not a realised outcome.
  Training on it would measure agreement with DRSK, not ability to predict distress.
- **Invalid as a feature** — derived from the same financials, and opaque/proprietary,
  defeating the explainability that is the project's entire purpose (§2.2).
- **Excellent as a third benchmark** — alongside Altman Z″ and logistic regression, it
  upgrades the claim to "beats the classical model, a properly specified logistic
  regression, *and* a commercial vendor model". Recommended addition to §8.1.

Also noted: the user had typed `India` into the **column value filter**, which filters rows
by value, not fields by name — the wrong mechanism for finding rating fields.

**Iteration 7 — rating fields located.**
Searching `ICRA` in Browse Fields returned **`ICRA Long Term Issuer Rating`** as the first
hit: issuer-level, domestic-scale, long-term. Exactly what `label_construction.py` expects.

Searching `CRISIL` returned a list dominated by **instrument-level** ratings — Senior
Unsecured Debt, Fixed Deposit, Structured Obligation, Subordinated Debt, Preferred Stock.
This reflects genuine market practice: CRISIL predominantly rates bank facilities and
specific instruments in India rather than issuing standalone issuer ratings. Substitution
(`Crisil Long Term Rating Lookup` or Senior Unsecured Debt as proxy) must be **documented**,
since instrument and issuer ratings are not the same object.

**Critical caveat recorded:** these EQS fields are **point-in-time snapshots**, not history.
They will *not* produce the label directly. What they *do* deliver is the **coverage
census** — counting non-`NR` values across 6,640 names determines whether ratings become
the primary label or whether the §7.5 fallback is promoted. `NR` was already visible for
Axis Bank in the grid.

---

## 4. Approaches Attempted

| # | Approach | Outcome | Why |
|---|---|---|---|
| 1 | Column-block wide-format loader | **Kept** | Correct for the original template shape; retained for that layout |
| 2 | `blpapi`/`xbbg` bulk API pull (~800 companies) | **Abandoned** | Terminal is Windows-only; API needs local Terminal on :8194. Later found to also be an explicit PRD non-goal (§3.2) |
| 3 | 800-company single Excel template | **Rejected pre-emptively** | ~8,800 live BDH formulas: slow, and risks daily data-request limits |
| 4 | 8 × 100-company vertical templates | **Selected, pending ticker list** | Manageable per-file calc time; per user confirmation |
| 5 | `pd.read_csv` with header inference on filled template | **Failed silently** | Off-by-one shifted every field; replaced by positional read |
| 6 | Hardcoding the CCL ticker as a known-bad row | **Rejected** | Replaced with generic order-of-magnitude outlier detection |
| 7 | Ratings-based label as primary | **Deferred** | Coverage unknown; census in progress |
| 8 | Financial-criteria fallback label (§7.5) | **Built, running** | Unblocks the project; circularity guard mandatory |
| 9 | Bloomberg DRSK as label | **Rejected** | Vendor model output, not a realised outcome |
| 10 | Bloomberg DRSK as feature | **Rejected** | Opaque + derived from same inputs; defeats explainability |
| 11 | Bloomberg DRSK as third benchmark | **Recommended** | Strengthens the comparative claim at negligible cost |
| 12 | EQS market-cap floor at screen time | **Rejected** | `Current` basis would silently delete delisted companies |
| 13 | EQS primary-security dedup filter | **Rejected** | Handle in code; catches renames/re-listings too |
| 14 | EQS screen-time Financials exclusion | **Rejected** | Display-only keeps exclusion auditable and preserves sector as a feature |

---

## 5. Errors, Failures, and Debugging Process

### 5.1 `pandas.errors.ParserError: Expected 16 fields in line 3, saw 18`

**Context:** first run of the column-block loader's self-test.
**Cause:** real exports are **ragged** — ticker and company-name rows stop at the last
non-empty cell while data rows run wider. pandas' C parser refuses this.
**Fix:** `_read_ragged_csv()` — parse with the stdlib `csv` reader (correct quoting
handling) and pad every row to the widest. All values stay strings; typing is
`clean_numeric`'s job.
**Lesson:** this was not a test-fixture artefact — it is how real exports arrive.

### 5.2 Self-test passed while the output was visibly wrong

**Context:** the column-block self-test printed `✅ Self-test passed`, but the panel showed
Reliance's `total_liabilities` and Yes Bank's `total_equity_book` as NaN despite having
values in the fixture.
**Cause:** the fixture was built by hand-counting commas, and the next block's ticker was
placed **one column too early**, colliding with the previous block's last field.
**Why the test didn't catch it:** the assertions checked only a subset of fields and never
touched the last field of any block.
**Fix (two parts):**
1. Rebuilt the fixture from **explicit per-block column lists** instead of hand-counted commas
2. Added assertions covering **every field of every block**, with an explicit comment that
   the last field of a block is where block-detection off-by-ones surface first

**Lesson — the important one:** a green test on a wrong fixture is worse than no test. The
failure mode being tested for (column misalignment) was present *in the test itself*.

### 5.3 `KeyError: 'Column not found: total_assets'`

**Context:** first run of `load_filled_template.py` on real data.
**Cause:** `NUMERIC_COLUMNS[:-0]` — in Python, `[:-0]` evaluates to an **empty list**, not
"everything". Every numeric column was silently dropped from the output frame.
**Detection:** the failure cascaded visibly — `Cannot verify balance-sheet identity —
columns missing`, then `total_equity_book absent`, then the KeyError.
**Fix:** `NUMERIC_COLUMNS[:-0]` → `NUMERIC_COLUMNS`.
**Note:** the defensive checks *worked* — three separate guards reported the problem before
the crash, which is the behaviour they were written for.

### 5.4 No pandas in any Python on the machine

**Context:** attempting to inspect the universe CSV.
**Finding:** `pandas` absent from `/usr/local/bin/python3`, `/usr/bin/python3`,
`~/.local/bin/python`, and system pip.
**Fix:** created a project `.venv` with pandas, numpy, openpyxl, python-docx.
**Note:** `.venv/` is untracked and should be gitignored.

### 5.5 Incorrect claim about ticker deduplication

**Context:** advising on EQS primary-security filtering.
**Claim made:** Bloomberg composite `IN` tickers are one-per-company, so dedup is
unnecessary. Supporting argument offered: 6,640 for Indian common stock "is about right for
one row per company", and dual listings would show roughly double.
**Reality:** disproved by the results grid — `RELIANCE IN` appeared three times
(`INDUSTR`, `INDU-PP`, `INDU-RT`).
**Root cause of the error:** reasoning about the *wrong duplication mechanism*. The analysis
correctly ruled out NSE/BSE dual listings but never considered partly-paid shares and
rights entitlements, which duplicate under the same ticker root.
**Correction:** dedup required; implemented in code by suffix and ticker collapse.

### 5.6 `_consecutive_true` counted the run-breaking element (found by Phase 1 tests)

**Context:** first run of the Phase 1 test suite; 3 of 53 tests failed.
**Symptom:** for `[T, T, F, T, T, T]` with n=2, index 3 returned `True` when it should be
`False`.
**Cause:**
```python
run = f.groupby((~f).cumsum()).cumcount() + 1   # WRONG
```
`(~f).cumsum()` groups each run *together with the False row that opens it*, and
`cumcount()` counts every row in the group. So the first `True` after any break scored a
run of 2.
**Consequence on real data — not cosmetic:** a company with a *single* sub-1 coverage year
immediately following a good year was graded **2 (distressed)** instead of **1
(deteriorating)**. On the 37-company panel this affected **10 company-years**:

| | Before fix | After fix |
|---|---|---|
| label 1 (deteriorating) | 10 (3.0%) | **20 (5.9%)** |
| label 2 (distressed) | 112 (33.1%) | **102 (30.2%)** |
| forward positive rate | 35.5% | **32.2%** |

**Fix:** `f.groupby((~f).cumsum()).cumsum()` — cumulative sum of the boolean counts only
the `True` elements, so the opening `False` contributes zero.
**Lesson:** this bug produced a *plausible* label distribution and would never have been
noticed by inspection. It was caught only because the test asserted an exact expected
sequence rather than a property. The three failing tests were written **before** the bug
was known to exist — they were written against the §7.5 specification, and the
implementation disagreed with it.

### 5.7 EQS screen — two user-side errors, both caught before export

1. `Trading Status: Inactive` only → 1,043 dead securities, no live companies
2. `Country: Asia Pacific (Developed); India` → OR'd; developed-Asia does not contain India

Both corrected; screen now returns 6,640.

---

## 6. Key Insights and Turning Points

**I1 — Negative equity is signal, not noise.** The instinct to "fix" the `#N/A` was wrong.
The undefined-ness is itself one of the strongest distress indicators available, and the
correct engineering response is to *flag it as a feature* and let the negative X4 flow
through to the score.

**I2 — The accounting identity is a parser test.** `TA == TE + TL` doesn't validate the
data; it validates the **column mapping**. A one-column shift destroys it instantly. This
converted a guess about the header defect into a proof, and is now a permanent load-time gate.

**I3 — Silent corruption is the real enemy.** The two worst defects (header off-by-one,
`[:-0]`) both produced *no error*. Every subsequent design decision favoured loud failure:
assert-backed self-tests, identity checks on every load, coverage reports that annotate
expected gaps by name.

**I4 — Survivorship bias has two failure modes, not one.** The obvious one is excluding
delisted companies. The non-obvious one is a *current*-basis market-cap floor, which
deletes them just as effectively while appearing to be a size filter. Both were encountered.

**I5 — The PRD contains a self-invalidating configuration.** Features and label criteria
overlap (§6.1/§6.4 vs §7.5). Detecting this before training saved a result that would have
looked outstanding and been meaningless.

**I6 — A vendor's model is a benchmark, never a label.** DRSK looked like a solution to the
label problem and would have quietly redefined the research question from "can we predict
distress" to "can we approximate Bloomberg".

**I7 — Rating coverage is architectural, not administrative.** Whether ratings or financial
criteria form the primary label changes the methodology chapter, the circularity treatment,
and the feature set. It deserved a census, not an assumption.

**I8 — Indian rating market structure is asymmetric across agencies.** ICRA exposes a clean
issuer-level long-term rating; CRISIL is predominantly instrument-level. Any pooled label
must document this.

---

## 7. Constraints and Limitations Identified

### 7.1 Environmental
- Bloomberg Terminal is **Windows-only**; the development machine is macOS arm64
- `blpapi` Desktop API therefore unusable locally
- Automated ingestion is an explicit **PRD non-goal** (§3.2) regardless of environment
- Bloomberg data **cannot be redistributed** under licence — the reproducibility package
  ships code and universe list, not data (§18.3)

### 7.2 Data
- **Financial-sector issuers** structurally unmodellable with the standard ratio set
- **`debt_to_equity` undefined** under negative equity — and the current 12-field pull
  contains **no alternative leverage measure that survives it** (no `total_debt`)
- **No ratings history** in any pull to date → no label from `label_construction.py`
- **No revenue** → `emissions_intensity`/`energy_intensity` (§6.8) unbuildable → Track B
  blocked as specified
- **No cash-flow data** → the Beaver-class CFO/debt predictors absent
- 3 of 37 companies returned no fundamentals; **all three distressed**
- 1 of 37 (`CCL IN`) is the wrong security entirely
- Coverage declines sharply in later years: 31/37 in FY2014-15 → 23/37 in FY2024-25
- **Restatement risk:** Bloomberg fundamentals are restated; a true backtest wants
  point-in-time data

### 7.3 Temporal / methodological
- **9-month mismatch:** `cur_mkt_cap` returns at 31-Dec while fundamentals are at 31-Mar.
  §7.4 forbids look-ahead, so an explicit prediction date must be fixed and documented
- **FY convention ambiguity:** §8.5's split ("Train FY2015→FY2021") does not state whether
  it means Bloomberg's `fiscal_year` column or Indian FY notation. A −1 offset separates
  them. **Unresolved.**
- **Class balance non-representative:** 33.1% positive on the curated 37 vs §8.3's 5–15%
  expectation; §3.3 targets are not meaningful on this sample
- **BRSR window is short:** mandatory only from FY2022-23 → ~3–4 fiscal years for Track B,
  with correspondingly few distress events inside it

### 7.4 Structural
- EQS results grid caps display at 3,000 of 6,640 — export completeness unverified
- Rating fields in EQS are point-in-time, not historical
- CRISIL exposes instrument-level rather than issuer-level ratings

---

## 8. Decisions and Trade-offs

| Decision | Alternative rejected | Rationale |
|---|---|---|
| Exclude financial-sector issuers | Impute the 5 missing fields | The concepts don't exist for banks; imputation fabricates data. Matches Altman's own exclusion |
| `negative_equity_flag` as a feature; `debt_to_equity` NaN | Zero-fill | Zero encodes "no leverage" for the *most* levered firms — inverts the signal |
| Negative X4 flows through to Z″ | Clip/floor X4 | Economically correct; clipping deletes the insolvency signal |
| NaN when `total_liabilities ≤ 0` | Allow ±inf | Non-positive denominators flip the sign, producing spuriously favourable scores |
| Split `check_scaling` from `extreme_value` | Single extreme flag | Distinguishes genuine deep distress from a units bug |
| Read filled template **by position** | Trust header names | Header omits a column; names are actively misleading |
| Balance-sheet identity on every load | One-time verification | Protects against upstream layout changes |
| Generic order-of-magnitude outlier detection | Hardcode `CCL IN` | Catches the *class* of wrong-security/wrong-currency errors |
| Dead tickers logged, never dropped | Silent exclusion | They are disproportionately the positive class |
| Market cap display-only, filter in code | Screen-time floor | `Current` basis would delete delisted firms invisibly |
| Dedup `-PP`/`-RT` in code | EQS primary-security filter | Also catches renames and re-listings; auditable |
| Financials excluded downstream | Screen-time `!=` filter | Keeps exclusion countable and documented; preserves sector as a Block 7 feature |
| DRSK as benchmark only | As label or feature | Vendor model ≠ realised outcome; opacity defeats explainability |
| Fallback label built with circularity guard | Build it silently | Guard converts a hidden invalidator into a visible constraint |
| Manual templates, 8 × 100 | API bulk pull | Environment impossible + explicit PRD non-goal |

---

## 9. Current State of the Project

### 9.1 Code

| File | Status |
|---|---|
| `src/altman_zscore.py` | **Modified.** `_safe_divide` guard; 4 diagnostic flags; ingestion-flag preservation; assert-backed self-test |
| `src/load_bloomberg_export.py` | **New.** Wide column-block loader; ragged-CSV reader; sector merge; `--exclude-financial`; assert-backed self-test |
| `src/load_filled_template.py` | **New.** Positional loader for the vertical filled template; identity verification; unit-outlier and dead-ticker detection |
| `src/fallback_label.py` | **New.** PRD §7.5 financial-criteria label; `CIRCULAR_FEATURE_MAP`; `make_forward_looking()` for §7.4. Run-length bug fixed (§5.6) |
| `src/quality_gates.py` | **New.** All 7 PRD §5.4 gates with 5 distinct severities; rejection log |
| `tests/test_foundation.py` | **New.** 53 tests, all passing; every test targets an encountered defect or a silent-failure invariant |
| `src/label_construction.py` | Unchanged — awaiting ratings history |
| `src/brsr_extractor.py` | Unchanged — Track B, not yet started |
| `src/train_model.py` | Unchanged — synthetic-tested only |
| `src/build_company_universe.py` | Unchanged |

### 9.2 Data artefacts produced

| File | Contents |
|---|---|
| `data/processed/bloomberg_panel.csv` | 407 rows × 20 cols, 37 companies, FY2014-15→FY2024-25 |
| `data/processed/distress_labels_fallback.csv` | 407 label rows (338 scored) |
| `data/processed/distress_labels_fallback_forward.csv` | 307 forward-looking FY(t+1) rows |
| `data/raw/failed_tickers.csv` | 3 dead tickers with diagnosed reasons |

### 9.3 Validated on real data

- ✅ Balance-sheet identity: 100% within 0.1%, max rel. error 1.42×10⁻⁵ on 331 rows
- ✅ Issue 2 reproduced: 62/64 negative-equity rows have NaN `debt_to_equity`; 0/267 positive-equity rows do
- ✅ Issue 1 absent (financials already excluded from this pull)
- ✅ First working label — project no longer blocked on Bloomberg ratings

### 9.4 Open / blocked

- ⏳ Rating coverage census (EQS columns being added)
- ⏳ EQS export of 6,640 → dedup → ~800 universe
- ⏳ 8 × 100 templates (awaiting ticker list)
- ❌ FY convention (§8.5) unresolved
- ❌ Prediction-date / 9-month market-cap mismatch undecided
- ❌ Track B blocked (no revenue field)

---

## 10. Recommendations / Next Steps

**Immediate**
1. Complete the rating census — count non-`NR` across 6,640. Decides primary vs fallback label.
2. Verify the EQS export writes all 6,640 rows, not 3,000.
3. Resolve the FY convention by checking one company's total assets against a published annual report.

**Pull design (next template)**
4. Add `SHORT_AND_LONG_TERM_DEBT` (§6.1 `debt_to_assets` — the only leverage measure that survives negative equity).
5. Add `SALES_REV_TURN` (unblocks Track B §6.8 intensity features).
6. Add `CF_CASH_FROM_OPER`, `NET_INCOME`, `IS_INT_EXPENSE`.
7. Add `VOLATILITY_260D` (enables Merton distance-to-default), `GICS_SECTOR_NAME`, `Debt/EBITDA`.

**Methodology**
8. Decide and document the circularity treatment before any Model A training run.
9. Fix an explicit prediction date to resolve the 9-month market-cap gap.
10. Add DRSK to §8.1 as a third benchmark tier.
11. Update PRD §6 to include volatility/DtD and cash-flow features.

**Engineering**
12. Repair the 3 dead tickers and `CCL IN` before scale-up.
13. Gitignore `.venv/`.

---

## 11. Notes for Final Deliverable

### 11.1 Findings that belong in the paper

**Data-quality findings are contributions, not embarrassments.** The distress-prediction
literature rarely documents what real vendor exports actually look like. Concretely
reportable:
- Financial-sector issuers returning `#N/A` identically across 11 years on 5 fields — an
  empirical justification for Altman's exclusion, not merely an inherited convention
- `debt_to_equity` undefined in 62/64 negative-equity rows and 0/267 positive-equity rows —
  a clean natural experiment demonstrating that missingness is *structured and informative*
- The balance-sheet identity as a **parser-validation technique** — reusable methodology

**Survivorship bias deserves its own subsection.** This project encountered *both* failure
modes: outright exclusion of delisted firms, and the subtler current-basis market-cap
floor. 3 of 37 companies in the test pull failed to resolve, **all distressed**. That is a
concrete, quantified illustration of a bias most papers only mention.

**The circularity finding is methodologically publishable.** Financial-criteria distress
labels are common in the literature. The observation that such labels are frequently
constructed from the same quantities used as features — and the resulting inflated
performance — is a legitimate cautionary contribution.

### 11.2 Framing

- Lead with **Track A**. The ESG question is the research contribution, but a validated
  baseline is what makes it credible.
- State **H1–H3 in advance** (§9.1) and pre-empt the size/profitability confound (§9.3)
  before a reviewer raises it.
- Report the **achieved BRSR extraction rate honestly**; §9.5 anticipates 80–90% on E and G
  for table-based filings, lower for prose-heavy Social.
- Scope ESG claims as *"ESG as measured by BRSR-extracted indicators"*, never universally.
- **SHAP explains the model, not the world.** State this in Limitations and respect it in
  dashboard copy.

### 11.3 Must-disclose limitations

1. Sample size and class balance — the curated universe is distress-enriched by construction
2. Survivorship exposure — quantified, with the mitigation used named explicitly
3. Label proxy — ratings lag by construction; financial criteria risk circularity
4. Agency asymmetry — ICRA issuer-level vs CRISIL instrument-level
5. BRSR window ~3–4 years, few distress events within it
6. Restatement/point-in-time risk in Bloomberg fundamentals
7. Extraction noise in PDF-parsed ESG data
8. Causation caveat on all SHAP attributions

### 11.4 Reproducibility package (§18.3)

Ship: feature definitions, label construction logic, model config, evaluation code, and the
universe list. Data itself cannot be redistributed under Bloomberg licensing — **state that
restriction explicitly rather than leaving it implicit.**

---

## 12. Session Log — Real Bloomberg Financial Data Extraction (Track A, Phase 1 completion)

**Session date:** 2026-09-19
**Scope of this session:** move Phase 1 from "pipeline ready, waiting for export" to
"expanded 18-field real data actually in hand for the ~800-company universe." This
picks up directly from §9.4's open item `⏳ 8 × 100 templates (awaiting ticker list)` —
except the ticker list had, by this point, already grown into a full ~800-company
universe (`data/processed/universe_800.csv`, from the EQS work briefly summarised
below), and the template chunking had been redesigned since the log's last update.

### 12.1 State discovered at the start of this session (context reconstruction)

Work had continued between the 2026-08-05 log entry and this session without a
corresponding write-up. The following artefacts already existed on disk and were
inventoried before any new work began, so this session's changes are additive to them,
not a redo:

| Artefact | What it is |
|---|---|
| `data/raw/eqs_export_1.csv`, `eqs_export_2.csv` | Raw EQS screen exports, ~3,000 rows each, fields: Ticker, Trading Status, EPS T12M, Revenue T12M, Total Return YTD, P/E, Name, GICS Sector, Icra LT Issr Rtg, Crisil LT Rtg Lu, Market Cap, Price:D-1 |
| `data/processed/universe_800.csv` | Processed universe from the EQS export, **3,001 rows** (header included), one row per ticker with the above fields carried through plus `is_financials_sector` |
| `src/build_company_universe.py::process_eqs_export()` | Already implements the dedup (`-PP`/`-RT`/`-W`/`-BE`), active-only market-cap floor, and Financials tag-not-filter logic described in §Phase IV of this log |
| `src/generate_bloomberg_templates.py` | Generates empty 18-field-by-ticker CSV templates, `chunk_size=500` (superseded in practice by the formula-based approach below) |
| `src/generate_formula_templates.py` | Generates **live-formula** `.xlsx` templates: one row per (ticker, fiscal_year), `=BDP()` formulas per field, using `FUND_PER="Y"` + `EQY_FUND_YEAR` override to pull one exact fiscal year per row rather than a BDH date range |
| `data/raw/templates/bloomberg_formula_chunk_1.xlsx` … `_7.xlsx` | Output of the above: chunks 1–6 = 500 companies × 11 fiscal years (2015–2025) = 5,500 rows each; chunk 7 = the ~11-row tail |
| `p2_export_1 - Sheet1.csv` (project root) | An earlier, separate expanded-pull attempt, mostly `#N/A` — superseded by the chunk-based approach; not deleted, left as-is |
| `src/load_p2_export.py` | Loader written against `p2_export_1`'s column names (`TOTAL_ASSETS`, `TOT_EQUITY`, etc.) — **does not match** the formula-chunk column names below; needs a new/adapted loader once all 7 chunks are filled (open item, §12.5) |

**Key architectural point carried forward:** the formula-chunk design
(`generate_formula_templates.py`) is superior to the plain BDH-range design
(`generate_bloomberg_templates.py`) for this use case, because `BDP` with an
`EQY_FUND_YEAR` override returns one flat value per (ticker, fiscal_year, field) —
no vertical BDH block to flatten afterward, no risk of the header/column-shift defect
that corrupted the original 37-company pull (§Phase II Step 8 of this log). **This is
now the standard extraction method for the rest of the ~800-company pull.**

### 12.2 Field mnemonic validation and correction (before firing the real templates)

Per this project's established discipline (§I3 — "silent corruption is the real
enemy"), the 18 field mnemonics baked into the formula templates were tested on a
single known-good ticker (`RELIANCE IN Equity`, FY2023) in a throwaway test sheet
**before** any of the real 5,500-row chunk files were opened. This caught two invalid
mnemonics:

| Field as originally templated | Bloomberg response | Root cause | Corrected mnemonic |
|---|---|---|---|
| `TOTAL_LIABILITIES` | `#N/A Invalid Field` | Not a real Bloomberg field code | `BS_TOT_LIAB2` |
| `INT_COVERAGE_RATIO` | `#N/A N/A` | Close to, but not, the real mnemonic — Bloomberg returns empty rather than erroring on a near-miss | `INTEREST_COVERAGE_RATIO` |

Both corrected mnemonics were **cross-verified against the BDH formula mnemonic list
already recovered once before, in §Phase II Step 12 of this log** (`BS_TOT_LIAB2` and
`INTEREST_COVERAGE_RATIO` both appear there, extracted from an earlier real BDH pull) —
so this was a confirmation of already-known-correct values, not a fresh guess.

**Fix applied in code:** `src/generate_formula_templates.py`'s `fields` list updated in
place, with an inline comment recording the test that justified the change. **All 7
template chunk `.xlsx` files were then regenerated from the corrected script** — same
row structure and ticker order as before (verified: still 5,500/5,500/5,500/5,500/5,500/
5,500/11 rows), only the two field mnemonics changed. This regeneration happened
**before** the user had pulled any real data against the broken mnemonics, so no rework
was lost.

**Lesson reinforced (consistent with §I3 and §5.6 of this log):** test one ticker's full
field list on a throwaway sheet before opening a 5,500-row live-formula file. A silent
`#N/A N/A` on a near-miss mnemonic would otherwise have looked identical to a genuine
missing-data cell across the entire `INTEREST_COVERAGE_RATIO` column, and would very
likely have gone unnoticed until much later in the pipeline (e.g. as suspiciously low
coverage in `quality_gates.py`, with no obvious cause).

### 12.3 Per-chunk pull-and-validate loop established

For each of the 7 formula chunks, the following procedure was run (and should continue
to be run for the remaining chunks):

1. User opens `bloomberg_formula_chunk_N.xlsx` on the Bloomberg-connected Terminal
   machine, lets all ~5,500 × 18 `BDP()` formulas calculate.
2. User saves/exports the calculated sheet as CSV (this incidentally also solves the
   "convert formulas to static values" step that would otherwise be needed — a CSV
   cannot hold live formulas, so exporting to CSV bakes in the values automatically).
3. File is copied into `data/raw/bloomberg_formula_chunk_N_filled.csv`.
4. A validation pass is run (ad hoc script, not yet promoted into `src/`, see §12.5)
   that:
   - Replaces the full set of known Bloomberg NA-sentinel strings with typed `NaN`
     (`#N/A Field Not Applicable`, `#N/A Invalid Field`, `#N/A N/A`, `#N/A`,
     `#N/A Requesting Data...`, `#N/A Authorization`, and — newly added this session —
     `#N/A Daily Capacity`, see §12.4).
   - Reports per-field coverage (% non-null) across all 18 columns.
   - Groups by ticker and flags any ticker with **zero** usable data across all 18
     fields and all 11 years — these are appended to `data/raw/failed_tickers.csv`
     (never dropped, per the standing invariant in §6/§8 of this log and PHASE_PLAN.md's
     cross-phase invariant #4), with a `reason` string identifying which chunk pull
     they came from.
   - Re-runs the **balance-sheet identity check** (`Total Assets == Total Equity +
     Total Liabilities`, within 2% relative tolerance) on every row where
     `BS_TOT_ASSET`, `TOTAL_EQUITY`, and `BS_TOT_LIAB2` are all present — this is the
     same parser-validation technique established in §Phase II Step 9 and named as a
     reusable methodology contribution in §11.1 of this log. A pass rate materially
     below ~99% on this check would indicate a column-mapping or shift defect and
     would have blocked acceptance of that chunk.

### 12.4 Results, chunk by chunk

| Chunk | Rows | Unique tickers | Fully-dead tickers | BS-identity pass (within 2%) | Field coverage range | Status |
|---|---|---|---|---|---|---|
| 1 | 5,500 | 499 | 37 | 99.4% (n=3,809) | 41.4% – 84.8% | ✅ accepted |
| 2 | 5,500 | 500 | 29 | 99.1% (n=4,019) | 47.3% – 88.6% | ✅ accepted |
| 3 | 5,500 | 500 | 29 | 99.5% (n=3,955) | 45.6% – 87.0% | ✅ accepted |
| 4 | 5,500 | 500 | 29 | 99.7% (n=3,934) | 44.2% – 88.6% | ✅ accepted |
| 5 | 5,500 (attempted) | — | — | — | — | ❌ **blocked**, see below |
| 6 | not yet attempted | | | | | ⏳ pending |
| 7 | not yet attempted (11-row tail) | | | | | ⏳ pending |

**Observations consistent across all four accepted chunks:**
- `IS_INT_EXPENSE` and `INTEREST_COVERAGE_RATIO` are consistently the weakest-covered
  fields (41–57%) — expected, since interest-expense reporting is the least
  standardised line item across small/mid-cap Indian issuers, and this is the same
  field this session had to correct the mnemonic for (§12.2). Low coverage here is a
  genuine data-availability finding, not a residual mnemonic problem — the balance-
  sheet identity check (which does not depend on this field) passing at ~99% on the
  same chunks confirms the extraction mechanism itself is sound.
- `CUR_MKT_CAP` and `VOLATILITY_260D` are consistently the best-covered (79–89%) —
  market data is more universally available than fundamental accounting line items,
  as expected.
- Dead-ticker rate is stable at ~6–7% per chunk (29–37 of 500), consistent with a
  universe that intentionally includes small-cap and inactive/delisted names per the
  survivorship-bias mitigation in §5.7/§Phase IV — **do not treat this rate as a
  pull error; it is the expected shape of a survivorship-corrected universe.**

**`data/raw/failed_tickers.csv` status:** started this session at 3 rows (the original
Videocon/Essar Steel/ABG Shipyard entries from the 37-company pull). After chunks 1–4:
**127 rows total** (3 original + 37 + 29 + 29 + 29 from chunks 1–4). Schema unchanged:
`ticker, company_name, n_years, n_with_assets, n_with_mktcap, reason`. The
`company_name` field is blank for all chunk-sourced entries (the formula-chunk CSVs
carry no company-name column, only ticker) — **resolving company names for these dead
tickers is an open item**, useful for the eventual manual-resolution pass PHASE_PLAN.md
already anticipates but does not block continued extraction.

**Chunk 5 — blocked on a Bloomberg daily data-request capacity limit.**
Every cell in the chunk 5 pull returned `#N/A Daily Capacity` rather than a value or
a normal field-level `#N/A`. This is a **Terminal-account-level throttle**, not a
per-field or per-ticker problem: the account has exhausted its daily allowance of
data-point requests (4 chunks × 5,500 rows × 18 fields ≈ **396,000 individual BDP
requests** already issued in this session alone, on top of whatever the account used
earlier in the day). This is functionally identical in kind — though larger in scale —
to the data-request-limit risk flagged pre-emptively back in §Phase II Step 6 / the
"Approaches Attempted" table row #3 ("800-company single Excel template... rejected...
risks daily data-request limits"), which is exactly the failure mode now observed
against the chunked approach as well, just later than expected because the per-chunk
size (500 companies) is larger than the originally-recommended 100-company batches.

**Immediate consequence:** chunks 5, 6, and 7 (and any retry of 1–4 if ever needed)
must wait for the Terminal account's daily capacity to reset before proceeding.
Retrying immediately does not help and does not consume further quota once already at
the cap — there is nothing to fix in the file or the formulas.

**Decision for next session:** treat chunk size (500 companies × 18 fields × 11 years
= 99,000 requests per chunk) as **the practical ceiling for this account's daily
capacity when four such chunks are pulled in a single day.** If the same throttle
recurs on chunk 5 alone after a reset, the remaining chunks should be spread across
multiple days (e.g. two chunks per day) rather than attempted back-to-back.

### 12.5 Code and artefact changes made this session

| File | Change |
|---|---|
| `src/generate_formula_templates.py` | `fields` list corrected: `TOTAL_LIABILITIES` → `BS_TOT_LIAB2`, `INT_COVERAGE_RATIO` → `INTEREST_COVERAGE_RATIO`; inline comment records the validation test and date |
| `data/raw/templates/bloomberg_formula_chunk_1..7.xlsx` | Regenerated from the corrected script (same tickers/rows/order; only the two field mnemonics changed) |
| `data/raw/bloomberg_formula_chunk_1_filled.csv` … `_4_filled.csv` | New — the real, calculated, validated Bloomberg pulls for chunks 1–4 |
| `data/raw/failed_tickers.csv` | Appended 124 new rows (37+29+29+29) across chunks 1–4, deduplicated on `ticker` |
| `Data_Extraction_Guide_v2.md` (project root) | **New.** Full manual extraction guide — EQS universe screening recap, the BDP/`EQY_FUND_YEAR` template-filling procedure, ratings-history sourcing options (Bloomberg `CRAT` + public agency sites + IBBI), and BRSR PDF sourcing. Fills the gap left by `README.md`'s reference to a same-named file that had never actually been created. |
| `Project_Context.md` (this file) | This §12 appended |

**Not yet done (explicitly deferred, not forgotten):**
- The ad hoc validation script used in §12.3/§12.4 was run interactively and was
  **not yet promoted into a reusable `src/` module.** Once all 7 chunks are in, this
  logic (NA-sentinel cleaning, per-field coverage report, dead-ticker detection,
  balance-sheet identity check) should become a proper loader — either a new
  `src/load_formula_chunks.py`, or a rewrite of `src/load_p2_export.py` — since its
  column names (`BS_TOT_ASSET`, `BS_TOT_LIAB2`, `TOTAL_EQUITY`,
  `INTEREST_COVERAGE_RATIO`, etc.) do not match what `load_p2_export.py` currently
  expects (`TOTAL_ASSETS`, `TOT_EQUITY`, `INT_COVERAGE_RATIO`, etc. — the pre-correction
  names from the earlier, now-superseded `p2_export_1` attempt).
- Chunks 1–4 have **not yet been merged into a single panel**, and no fiscal-year
  convention offset has been applied to them (recall the unresolved −1 offset issue
  from §Phase II Step 12 / §7.3 of this log — `load_p2_export.py` applies
  `fiscal_year_num = fiscal_year - 1`, but this has not yet been re-verified against
  the formula chunks' `EQY_FUND_YEAR`-based fiscal year values specifically).
- No quality-gate run (`src/quality_gates.py`) against the merged chunks yet — that
  should happen once all 7 are in, not chunk-by-chunk, since several gates (coverage,
  temporal continuity) are only meaningful across a company's full time series.
- Company-name backfill for the 124 newly-logged dead tickers (currently blank).

### 12.6 Immediate next steps (in order)

1. **Wait for the Bloomberg Terminal account's daily data capacity to reset**, then
   re-attempt chunk 5. Check the Terminal's own message/notification panel for a
   specific reset time before assuming a fixed 24-hour window.
2. Pull chunks 5, 6, and 7 — **spread across separate days if the capacity limit
   recurs**, rather than attempting all three (or even two) in one session, given the
   observed ~99,000-request cost per chunk against an apparent daily ceiling
   somewhere above 4×99,000 but below 5×99,000 requests.
3. Once all 7 chunks are filled and individually validated (same per-chunk procedure
   as §12.3): merge into a single panel, resolve the fiscal-year convention explicitly
   against one company's published annual report (per Recommendation #3 in §10 of this
   log — still unresolved), and run `src/quality_gates.py` against the full merged
   panel.
4. Promote the ad hoc validation logic into a real loader module (§12.5).
5. Only after the merged, gated financial panel exists: resume work on the ratings
   coverage census (§Phase III Step 15–16, still the primary blocker for the label —
   unaffected by this session's work and not addressed here) and the label
   construction decision (ratings-based vs. §7.5 fallback).
6. BRSR/ESG extraction (Track B) remains correctly out of scope until Track A's
   financial panel is complete, per the PRD's own sequencing rule (§16.1) — no BRSR
   work should begin before item 3 above is done.

---

## 13. Session Log — Financial Panel Completed, Merged and Gated (Track A, Phase 1 exit)

**Session date:** 2026-10-27
**Scope:** chunks 5–7 pulled by the user and handed back; all seven chunks validated,
the ad hoc validation logic promoted into a real loader module (the §12.5 deferral),
all seven chunks merged into one panel, and `quality_gates.py` run against it. This
closes the extraction work that §12.6 left open at items 1–4.

### 13.1 Chunks 5–7 received and validated

Files added by the user: `data/raw/bloomberg_formula_chunk_5_filled.csv`,
`_6_filled.csv`, `_7_filled.csv`. Validated with the same procedure as chunks 1–4
(§12.3), now executed by code rather than interactively.

| Chunk | Rows | Tickers | Dead tickers | BS-identity pass (2%) | Field coverage | Status |
|---|---|---|---|---|---|---|
| 5 | 5,489 | 499 | 22 (4.4%) | **99.63%** (n=4,097) | 46.3% – 90.6% | ✅ accepted |
| 6 | 5,500 | 500 | 22 (4.4%) | **99.75%** (n=4,052) | 47.1% – 89.4% | ✅ accepted |
| 7 | 11 | 1 | 0 | **100.00%** (n=11) | 81.8% – 100% | ✅ accepted |

**The daily-capacity throttle is cleared.** A sentinel census across all seven chunks
found **zero `#N/A Daily Capacity` cells** anywhere, confirming chunks 5–7 are complete
pulls rather than partially-throttled ones. This was checked explicitly rather than
assumed, because a throttled cell and a genuinely-missing cell are indistinguishable
once parsed to NaN.

### 13.2 Two universe-construction defects found (pre-existing, not pull errors)

The sentinel census surfaced two junk rows that have been in the pipeline since August
and had never been noticed:

| Defect | Where | Signature | Cost |
|---|---|---|---|
| **Bloomberg legal disclaimer carried in as a ticker** | chunk 5, 11 rows | ticker cell = the full "The BLOOMBERG PROFESSIONAL service…" trademark notice; all 18 fields `#N/A Invalid Security` | 198 wasted BDP requests |
| **Null ticker row** | chunk 1, 11 rows | empty ticker; all 18 fields `#N/A Mandatory parameter [SECURITY] cannot be empty` | 198 wasted BDP requests |

Traced to source: **the disclaimer is a row in both `data/raw/eqs_export_1.csv` and
`eqs_export_2.csv`** — Bloomberg appends its legal notice to EQS exports, and
`build_company_universe.py::process_eqs_export()` has no filter for it, so it was
carried through `universe_800.csv` as if it were a security and then templated into
chunk 5 by `generate_formula_templates.py`. The null row is `universe_800.csv` index 396.

→ **`universe_800.csv` therefore contains 2,999 real tickers, not 3,000.** The headline
"~800-company universe" naming is also now stale by a factor of ~3.75; the screen
yielded 3,001 rows.
→ **Open item:** add a junk-row filter to `process_eqs_export()` (non-matching ticker
pattern → log and drop) so a re-run of the universe build does not reproduce this.
Not done this session; the defect is now neutralised downstream instead.

### 13.3 Validation logic promoted into `src/load_formula_chunks.py` (§12.5 deferral closed)

**New module.** Implements the full §12.3 procedure as reusable code:

- Quote-aware CSV parse with header and row-width assertions. (A naive `awk -F,`
  field count reported chunk 5 as having 39-field rows; that was the parser, not the
  data — the Bloomberg disclaimer contains commas inside a quoted field. Recorded
  because it is a plausible false alarm for a future session to re-encounter.)
- Junk-row removal by ticker pattern, **logged, never silent**.
- NA-sentinel cleaning, now including `#N/A Daily Capacity`, `#N/A Invalid Security`
  and `#N/A Mandatory parameter [SECURITY] cannot be empty`. Never zero-fills.
- Per-field coverage report across all 18 fields.
- Dead-ticker detection appended to `data/raw/failed_tickers.csv` with a chunk-tagged
  reason, deduplicated on ticker (cross-phase invariant #4).
- Balance-sheet identity at 2% tolerance, reported per chunk with max relative error.
- Bloomberg mnemonics renamed to the §5.2 canonical panel names.

**Why not `load_p2_export.py`:** it expects the pre-correction mnemonics
(`TOTAL_ASSETS`, `TOT_EQUITY`, `INT_COVERAGE_RATIO`) and would silently produce an empty
frame against these files. It also derived `total_liabilities` from `TOTAL_LIAB_AND_EQY`,
which is liabilities **plus** equity — wrong by construction. It is now superseded for
this input shape and should not be pointed at the formula chunks.

**Self-validation of the new loader:** re-run against chunks 1–4 it reproduces the
numbers §12.4 recorded interactively — BS identity 99.45 / 99.13 / 99.54 / 99.69%
against the logged 99.4 / 99.1 / 99.5 / 99.7%, and dead-ticker counts 37 / 29 / 29 / 29
matching exactly. The promotion therefore preserved behaviour rather than quietly
changing it.

### 13.4 Merged panel

`data/processed/bloomberg_panel_full.csv` — **32,989 rows × 2,999 tickers × FY2015–2025.**

Three independent consistency checks, all passing:
- 2,999 tickers = 3,001 universe rows − 1 disclaimer − 1 null ✅
- 32,989 rows = 2,999 × 11 years exactly — every ticker has a complete series ✅
- 0 duplicate `(ticker, fiscal_year)` keys ✅

### 13.5 Quality gates (PRD §5.4) — BATCH ACCEPTED

Run via `src/quality_gates.py`. Output: `data/processed/bloomberg_panel_full_gated.csv`
(32,973 rows), rejection log `data/processed/quality_rejections_full.csv` (10,158 records).

| Gate | Severity | Offending | Checked | Result |
|---|---|---|---|---|
| primary_key | HARD_FAIL_BATCH | 0 | 32,989 | ✅ unique and non-null |
| accounting_identity | FLAG | 102 | 23,877 | 99.6% within 2% |
| sign_sanity | HARD_FAIL_ROW | 16 | 32,989 | 12 non-positive total_assets, 4 negative cur_ratio — **removed** |
| range_plausibility | FLAG | 801 | 32,989 | 156 ROA, 653 interest_coverage out of range — retained, marked |
| coverage | EXCLUDE_TRAINING | 9,239 | 32,989 | below 70% on 7 core features — retained, marked unusable |
| temporal_continuity | WARN | 0 | 2,999 | ✅ no gaps |
| scale_sanity | FLAG | — | — | correctly skipped; Z″ not computed yet (Phase 2) |

**Trainable subset: 23,735 rows across 2,783 tickers (72.0% of the gated panel).**

**On the 102 identity failures:** scattered across 18 distinct tickers and individual
years at 0.2–0.3% per chunk, *not* systematic. This is the diagnostic that matters — a
column-mapping shift breaks ~100% of rows, so the mapping is confirmed sound and these
are per-filing reporting quirks. Per §5.4 they are FLAG severity (retain + investigate),
not a batch reject. Two are a different class and need follow-up before training:
`TATAINVE IN Equity` FY2020 (book equity 68× total assets) and `UVDH IN Equity` FY2017
(7.6×) carry the order-of-magnitude signature of the `CCL IN` wrong-security incident
(§Phase II Step 11). Tata Investment Corporation is additionally an investment/NBFC
entity and likely falls under the §3.2 financial-sector exclusion regardless.

### 13.6 Findings on the merged panel

**Issue 2 reproduced at full scale — a much stronger version of the §11.1 result.**
`debt_to_equity` is NaN in **1,993 of 2,159** negative-equity rows (92.3%) and in only
**160 of 21,791** positive-equity rows (0.7%). The 37-company pull showed 62/64 vs 0/267.
The pattern now holds across 2,999 companies and 11 years, which makes "missingness in
this domain is structured and informative" an empirically demonstrated claim rather than
an illustrative anecdote.

**Coverage rises over time — and it runs against the §8.5 split.**

| FY (Bloomberg basis) | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_assets non-null | 57.5% | 61.3% | 64.5% | 63.0% | 69.0% | 71.8% | 76.2% | 79.6% | 84.8% | 85.6% | 82.9% |

This is the **opposite** of the 37-company pull, where coverage declined in later years
(§7.2). The cause is universe composition: the broad EQS screen includes many companies
that listed after 2015 and inactive names whose series stop early. **Methodological
consequence, not yet resolved:** PRD §8.5 puts training at FY2015–2021 and test at
FY2023–2025 — i.e. the training window is systematically the *thinner* data and the test
window the *richer*. That needs to be stated as a limitation at minimum, and may warrant
revisiting the split boundary. Flagged here; no decision taken.

**Dead tickers: `data/raw/failed_tickers.csv` now 171 rows** (3 original + 168 across the
seven chunks: 37/29/29/29/22/22/0). Rate is 4.4–7.4% per chunk and *falls* in the later
alphabetical chunks. Consistent with a survivorship-corrected universe per §5.7 — not a
pull error. `company_name` remains blank for all chunk-sourced entries.

### 13.7 State after this session

**Done:** all 7 chunks pulled, validated, merged, gated. Phase 1's last open exit
criterion — "~800-company universe built from EQS export with dedup + survivorship
accounting" — is now satisfied with real data behind it.

**Still open, unchanged by this session:**
- ❌ **FY convention (−1 offset) still unresolved** (§7.3, §Phase II Step 12). The merged
  panel deliberately carries `fiscal_year_bbg` (Bloomberg `EQY_FUND_YEAR`, the calendar
  year the FY ends in) and **no offset has been applied**. `load_formula_chunks.py`
  refuses to apply one silently. Must be settled against a published annual report before
  any label join. The gate run used this column unchanged; `temporal_continuity` only
  measures gaps, so it is convention-independent and unaffected.
- ❌ Ratings-history census and the primary-vs-fallback label decision.
- ❌ Altman Z″ not yet computed on the real panel (blocks `scale_sanity`).
- ❌ Company-name backfill for the 168 chunk-sourced dead tickers.
- ❌ EQS junk-row filter in `build_company_universe.py` (§13.2).
- ⬜ Track B / BRSR — correctly still out of scope per §16.1 sequencing.

### 13.8 Immediate next steps (in order)

1. **Resolve the FY convention** against one company's published annual report — this is
   now the single highest-value open item, because every label join depends on it and it
   has been deferred three times (§10 rec. 3, §12.5, here).
2. Compute Altman Z″ on the merged panel and re-run gates so `scale_sanity` actually fires.
3. Investigate the two order-of-magnitude identity failures (`TATAINVE`, `UVDH`).
4. Ratings-coverage census, then the label decision. **New input available:** a Bloomberg
   RATC rating-actions export exists at `~/Downloads/Reliance RATC 2015 (1).xlsx` — see
   §13.9.
5. Only then: feature assembly (Phase 2), which can now use the expanded fields
   (`total_debt`, `total_revenue`, `cash_from_operations`, `net_income`, `volatility_260d`)
   that were unavailable to the 12-field pull.

### 13.9 Ratings history — a usable source appeared, with a cap problem

`~/Downloads/Reliance RATC 2015 (1).xlsx` (dated 2026-09-21, not previously logged) is a
Bloomberg **RATC rating-actions export**. Despite the filename it is not Reliance-specific;
it is market-wide Indian rating actions.

- **Schema is right for `label_construction.py`:** `Company Name, Date, Rating Type, Agency,
  Curr Rtg, Last Rtg, Ctry/Reg, Industry Type, Security Name` — one row per rating action,
  ticker in `Security Name`, and `Last Rtg` gives the prior notch directly, so downgrades
  are readable per-action rather than inferred year-over-year.
- **It is truncated.** Exactly 5,000 rows, spanning only **2015-07-31 → 2015-12-31**. It hit
  a 5,000-row export cap and lost roughly the first seven months of 2015 — the same class of
  silent cap as the EQS "3,000 of 6,640" issue in §7.4.
- **Yield against the universe is thin.** 3,750 of 4,364 securities are numeric-coded
  unlisted/private entities irrelevant to a listed universe; 6.5% of rows carry no ticker at
  all. After filtering to domestic agencies + Long Term + tickers present in
  `universe_800.csv`: **241 rows covering 218 of 2,999 universe tickers (7.3%)**, for half of
  one fiscal year.
- Agencies: CRIS 2,183 / CARE 1,226 / ICRA 977 / IND-RA 315 / Brickwork 254, plus 45
  international rows (S&P, Moody's, Fitch, DBRS) that §7.3 requires be filtered and logged.
  258 `D` (default) ratings present — real positives.

**⚠️ Code defect this will hit immediately:** `label_construction.py`'s `DOMESTIC_AGENCIES`
contains `"CRISIL"` but **not `"CRIS"`**, which is the string RATC actually emits. Run as-is
it would discard CRISIL's 2,183 rows — 44% of the file and the largest single agency — as
"non-domestic". It prints a warning rather than failing silently, so it would be caught, but
it must be fixed before any census. Related: watch-flagged ratings (`BBB+ *`, `AA- *-`) and
`NR` / `WD` do not map to `RATING_RANK` and would be dropped as unmapped — these need an
explicit decision, not a silent drop.

**Open question for the next session:** at 7.3% universe coverage per half-year, a
year-by-year RATC export is a poor exchange rate for the ~22 exports it would take to cover
FY2015–2025. Establish whether RATC can be filtered to a ticker list or an index membership
before committing to that path.

### 13.10 Files changed this session

| File | Change |
|---|---|
| `src/load_formula_chunks.py` | **New.** Promoted loader/validator (§13.3) |
| `data/processed/bloomberg_panel_full.csv` | **New.** Merged 7-chunk panel, 32,989 rows |
| `data/processed/bloomberg_panel_full_forgates.csv` | **New.** Gate input; adds `fiscal_year`/`fiscal_year_num` aliases of `fiscal_year_bbg` |
| `data/processed/bloomberg_panel_full_gated.csv` | **New.** Post-gate panel, 32,973 rows |
| `data/processed/quality_rejections_full.csv` | **New.** 10,158 gate records |
| `data/raw/failed_tickers.csv` | 127 → **171** rows (chunks 5–6 added 44) |
| `Project_Context.md` | This §13 appended |

**Environment note:** the project lives on an exFAT external drive (`/Volumes/VARAD`) which
unmounted and remounted itself mid-session. A long-running filesystem scan crossing that
window returned incomplete results and was discarded. Worth a backup; a `Capstone.zip`
snapshot dated 2026-10-17 exists at the volume root.

---

## 14. Fiscal-Year Convention — RESOLVED (2026-10-27)

**Status: CLOSED.** This question was opened in §Phase II Step 12, restated in §7.3 as
"**Unresolved**", deferred in §10 (rec. 3), again in §12.5, and again in §13.7. It is now
settled against published annual-report figures and asserted in code.

### 14.1 Why the prior reasoning could not simply be carried over

§Phase II Step 12 inferred a −1 offset from the **BDH** formula's `Per=CY` argument. The
formula chunks are **BDP with `FUND_PER="Y"` + `EQY_FUND_YEAR`** — a different mechanism.
The earlier inference was therefore suggestive but not transferable, which is precisely why
it kept being re-deferred rather than confirmed. It needed external ground truth.

### 14.2 The test

Panel values for `fiscal_year_bbg == 2024` were compared against published results for the
**year ended 31 March 2024**:

| Company | Published (FY2023-24) | Panel `SALES_REV_TURN` at `fiscal_year_bbg=2024` | Verdict |
|---|---|---|---|
| TCS | consolidated revenue **₹240,893 cr** | 2,408,930 INR mn = **₹240,893 cr** | ✅ exact |
| Reliance | revenue from operations **₹901,064 cr** | 9,010,640 INR mn = **₹901,064 cr** | ✅ exact |
| UltraTech | revenue from operations ₹70,908 cr | 698,095 INR mn = ₹69,810 cr | level −1.5% (definition), **growth +12.0% vs reported +12.1%** ✅ aligned |

**On the Reliance figure — a trap worth recording.** RIL headlines FY2023-24 as
**₹10,00,122 cr "Value of Sales and Services"**, and most secondary sources quote that.
Its *revenue from operations*, net of GST recovered, is **₹9,01,064 cr** — which is what
Bloomberg's `SALES_REV_TURN` returns and what the panel holds, to the rupee. A future
session checking RIL against the headline number will appear to find a mismatch and should
not conclude the panel is wrong.

UltraTech's level differs because `SALES_REV_TURN` excludes some operating income there.
Level differences across vendors are expected; **year alignment** is what was under test,
and its growth profile settles it. Had `2024` meant FY2024-25, UltraTech would read
~₹75,000 cr rather than ₹69,810 cr.

### 14.3 The resolution

> **`fiscal_year_bbg` is the calendar year in which the Indian fiscal year ENDS.**
>
>     fiscal_year_bbg = 2024  ==  year ended 31-Mar-2024  ==  Indian "FY2023-24"
>
>     fiscal_year_num = fiscal_year_bbg − 1        (Indian FY START year)
>     fiscal_year     = "FY{start}-{end:02d}"      e.g. 2024 -> "FY2023-24"

**The −1 offset is confirmed.** `load_p2_export.py`'s `fiscal_year_num = fiscal_year - 1`
was correct all along — it was simply never verified, so it could not be relied upon.

### 14.4 Encoded, not left as a convention

`src/load_formula_chunks.py` now derives both columns via `derive_fiscal_year()`, with the
evidence above recorded in the module docstring and a plausible-range assertion on the input.
The panel carries all three columns — `fiscal_year` (Indian label), `fiscal_year_num`
(Indian start year), `fiscal_year_bbg` (raw Bloomberg) — so the raw value is never lost and
any future disagreement can be re-checked without re-pulling.

**Join compatibility verified end-to-end, not assumed.** `label_construction.py` maps a
rating action dated April(y)–March(y+1) to `"FY{y}-{y+1}"`. A test action dated 2023-06-01
produced `FY2023-24` and joined successfully to the panel row with `fiscal_year_bbg == 2024`.
The label and panel conventions now agree by construction.

### 14.5 Consequence for the §8.5 out-of-time split — needs a decision

The gated panel spans **Indian FY2014-15 → FY2024-25** (`fiscal_year_bbg` 2015–2025).
PRD §8.5 specifies "Train FY2015→FY2021, Validation FY2022, Test FY2023→FY2025" without
stating its convention. Read in Indian notation, that now maps unambiguously to:

| §8.5 window | Indian FY | `fiscal_year_bbg` |
|---|---|---|
| Train FY2015–FY2021 | FY2015-16 … FY2021-22 | 2016 … 2022 |
| Validation FY2022 | FY2022-23 | 2023 |
| Test FY2023–FY2025 | FY2023-24 … FY2025-26 | 2024 … 2026 |

**Two open issues this exposes, neither decided here:**
1. **The test window overruns the data.** FY2025-26 does not exist in the panel (max is
   FY2024-25 / bbg 2025), so the test window is effectively two years, not three.
2. **FY2014-15 (bbg 2015) falls outside every window** and would go unused under a literal
   reading — 2,999 rows discarded for no methodological reason.

Combined with §13.6's finding that coverage *rises* over time (57.5% in bbg 2015 → 85.6% in
bbg 2024), the split boundary deserves an explicit decision and a stated justification
rather than a literal transcription of §8.5. Flagged for Phase 2; **not** resolved here.

### 14.6 Files changed

| File | Change |
|---|---|
| `src/load_formula_chunks.py` | `derive_fiscal_year()` added; docstring records the evidence; panel now emits `fiscal_year`, `fiscal_year_num`, `fiscal_year_bbg` |
| `data/processed/bloomberg_panel_full.csv` | Regenerated with the three FY columns (32,989 rows, unchanged otherwise) |
| `data/processed/bloomberg_panel_full_gated.csv` | Regenerated; gate results identical (batch ACCEPTED, 32,973 rows, 23,735 trainable) |
| `data/processed/bloomberg_panel_full_forgates.csv` | **Deleted** — the intermediate alias file is obsolete now that the panel carries `fiscal_year`/`fiscal_year_num` natively |

---

## 15. Altman Z″ Computed on the Real Panel (2026-10-27)

Closes two Phase 2 items: "Altman Z″ integration + hand-verified sample" and the
`scale_sanity` gate, which had self-skipped since Phase 1 because Z″ did not exist.

### 15.1 Result

`data/processed/bloomberg_panel_scored_gated.csv` — 32,973 rows × 2,999 tickers.
Z″ computed on **20,812 rows (63.1%)**; the remainder lack one or more of the seven
required inputs. Of the 23,735 trainable rows, **20,812 carry a Z″** and **19,364 carry a
non-degenerate Z″** (see §15.3).

No scaling conversion was needed. Z″ uses only INR-million **level** fields, so X1–X4 come
out as decimals, which is the convention `altman_zscore.py` expects. (The §6.5 coefficient-
scaling trap does not bite here; `roa` is the percentage-valued field and Z″ does not use it.)

### 15.2 Hand verification — Phase 2 exit criterion satisfied

Four rows recomputed from raw inputs, independently of `altman_zscore.py`:

| Ticker | FY | X1 | X2 | X3 | X4 | Z″ manual | Z″ code | diff |
|---|---|---|---|---|---|---|---|---|
| TCS | FY2023-24 | +0.4502 | +0.6209 | +0.3928 | +1.6949 | 9.396195 | 9.396195 | 0.00e+00 |
| SRF | FY2022-23 | +0.0458 | +0.5076 | +0.1575 | +1.2254 | 4.300145 | 4.300145 | 0.00e+00 |
| UltraTech | FY2023-24 | −0.0372 | +0.5402 | +0.0975 | +1.4878 | 3.734165 | 3.734165 | 0.00e+00 |
| MIHI | FY2016-17 | +0.9820 | +0.0039 | −0.0556 | **+17668.93** | 18558.456245 | 18558.456245 | 0.00e+00 |

**The arithmetic is exact.** Healthy large-caps land where economic sense says they should
(all "safe"). The implementation is not in question — what the real data exposed is a
property of the *ratio*, below.

### 15.3 ⚠️ Finding — Z″ degenerates on two near-zero denominators

1,265 rows (6.1% of scored) have |Z″| > 50. **This is not a units error.** It is two
distinct degenerate denominators, each economically meaningless:

| | Mechanism | Worst case |
|---|---|---|
| **Thin liabilities** | X4 = equity ÷ liabilities explodes **positive**. Debt-free holding shells where assets ≈ equity. Dominates 61.7% of extremes. | `MIHI IN Equity` FY2016-17: TL = 0.014 INR mn vs TA = 247.4 → X4 = 17,669, Z″ = 18,558 |
| **Thin assets** | X1, X2 and X3 all divide by total assets, so all three explode, usually **negative**. Collapsed shells. | `NBF IN Equity` FY2024-25: TA = 0.001 INR mn (≈ ₹1,000) vs 15.9 mn liabilities → Z″ = −642,105 |

**Why this matters more than the magnitude suggests.** MIHI FY2016-17 is classified
**"safe"** despite EBIT of −13.7 — a loss-making shell scored as healthy purely because it
carries no debt. Across the panel, 883 degenerate rows are "safe" with negative EBIT. Left
raw, these would both corrupt training and make the SHAP attribution unreadable, while
looking like ordinary data.

Note this is the *inverse* of the §I1 negative-equity case. There, an extreme value was real
signal and had to be preserved. Here, an extreme value is an artefact of a vanishing
denominator and carries no distress information. The two must not be treated alike, which is
exactly why `altman_flag_check_scaling` separates them.

### 15.4 Treatment

New script **`src/score_altman_panel.py`**. `altman_zscore.py` is deliberately **not**
modified — it is correct and carries Phase 1 test coverage.

Two companion columns; the raw `altman_z_double_prime` is preserved untouched:

- **`altman_denominator_degenerate`** (+ `altman_flag_thin_liabilities`,
  `altman_flag_thin_assets`) — deterministic row-level flags. Purely a function of the row,
  so no fitted statistic and therefore **no train/test leak**. Flagged, never dropped
  (invariant #4). **1,448 rows (7.0% of scored)**: 1,271 thin-liabilities, 182 thin-assets.
- **`altman_z_double_prime_capped`** — winsorised to **fixed** bounds [−50, +50].

**Why fixed bounds rather than PRD §6.4's "1st/99th percentile":** a percentile is a
statistic fitted to the data, and fitting it on the full panel before the out-of-time split
would leak test information into training (§7.4, invariant #3). ±50 is also not a free
parameter — it is the threshold PRD §5.4's own `scale_sanity` gate already declares
implausible. `feature_engineering.py` sets the same precedent: `interest_coverage_capped`
uses fixed [−10, 50], not percentiles. **Open item:** PRD §6.4's text should be reconciled
with this, since as written it specifies a leaking transformation.

**Threshold calibration (5% of total assets), recorded so it is not mistaken for a free
parameter:**

| Threshold | Rows flagged | % of scored | % of `scale_sanity` rows covered |
|---|---|---|---|
| 1% | 656 | 3.2% | 59.9% |
| **5%** | **1,448** | **7.0%** | **98.0%** |
| 10% | 2,197 | 10.6% | 98.5% |

5% is the knee. It is also principled: below TL = 5% of TA, the 1.05·X4 term alone exceeds
21 — eight times the entire "safe" cutoff of 2.6 — so the score is decided by the absence of
debt rather than by any distress signal.

### 15.5 Distributions

| | raw (all scored) | raw (non-degenerate) | capped |
|---|---|---|---|
| n | 20,812 | 19,364 | 20,812 |
| std | 5,973.70 | 67.30 | 14.78 |
| 1% | −260.26 | −88.20 | −50.00 |
| median | 3.65 | 3.39 | 3.65 |
| 99% | 313.92 | 20.60 | 50.00 |
| min / max | −642,105 / 18,558 | −4,044 / 31.8 | −50 / 50 |

**Zones** (cutoffs 1.1 / 2.6), non-degenerate rows: safe 11,623 (60.0%), distress 4,591
(23.7%), grey 3,150 (16.3%).

**The remaining negative tail is real and must stay.** Of the 328 non-degenerate rows with
Z″ < −50, **312 (95.1%) are negative-equity rows** — genuine catastrophic insolvency, e.g.
`RIDS IN Equity` FY2021-22 with accumulated losses of ₹1,240 mn against ₹1.3 mn of remaining
assets. Per §I1 this signal is preserved in the raw column; the capped column exists for
modelling, not to delete it.

### 15.6 All seven quality gates now fire

| Gate | Severity | Offending | Checked | Result |
|---|---|---|---|---|
| primary_key | HARD_FAIL_BATCH | 0 | 32,973 | ✅ |
| accounting_identity | FLAG | 102 | 23,864 | 99.6% within 2% |
| sign_sanity | HARD_FAIL_ROW | 0 | 32,973 | ✅ (cleared in the prior pass) |
| range_plausibility | FLAG | 801 | 32,973 | 156 ROA, 653 interest_coverage |
| coverage | EXCLUDE_TRAINING | 9,238 | 32,973 | below 70% on core features |
| temporal_continuity | WARN | 1 | 2,999 | `NIVABUPA IN Equity`, 5y gap |
| **scale_sanity** | FLAG | **810** | 20,812 | **now active** — |Z″| > 50 unexplained by negative equity |

**BATCH ACCEPTED.**

Two notes on the newly-visible results:
- **`scale_sanity` 810 vs the degeneracy flag's 1,448** — they are different tests, not a
  contradiction. The gate excludes negative-equity rows by design; the degeneracy flag does
  not. The 5% threshold covers 98.0% of what the gate flags.
- **`NIVABUPA IN Equity`'s 5-year gap is an artefact, not new corruption.** It had all 11
  years pre-gate; `sign_sanity` removed FY2016-17…FY2019-20 in the earlier pass, creating the
  gap. Niva Bupa is a health insurer — a **financial-sector issuer**, excluded from the
  primary model under §3.2/§5.6 regardless. No action needed beyond recording it.

### 15.7 Files changed

| File | Change |
|---|---|
| `src/score_altman_panel.py` | **New.** Panel-level Z″ application + degeneracy handling |
| `src/altman_zscore.py` | **Unchanged** — deliberately |
| `data/processed/bloomberg_panel_scored_full.csv` | **New.** Scored panel, 32,973 rows |
| `data/processed/bloomberg_panel_scored_gated.csv` | **New.** Post-gate, all 7 gates active |
| `data/processed/quality_rejections_scored.csv` | **New.** 10,952 gate records |

### 15.8 Open items arising

1. **Decide which Z″ column feeds Block 5** — raw, capped, or non-degenerate-only. Not
   decided here; it is a methodology choice for feature assembly.
2. **Reconcile PRD §6.4** with the fixed-bounds winsorisation actually used (§15.4).
3. The 2 order-of-magnitude accounting-identity suspects (`TATAINVE`, `UVDH`, §13.5) are
   still uninvestigated.

---

## 16. Ratings Census Completed + Circularity Guard Found Bypassed (2026-10-27)

Two findings, the second more serious than the first.

### 16.1 Ratings census — the primary label is NOT achievable from this Bloomberg account

The census open since §Phase III / §9.4 is now done, against `universe_800.csv`'s EQS
snapshot columns. Result, on 3,000 tickers:

| Field | Blank | `NR` (not rated) | `#N/A` error | **Actual ratings** |
|---|---|---|---|---|
| `icra_rating` | 2,459 | 537 | 0 | **4** (all `AAA`) |
| `crisil_rating` | 1 | 0 | **2,999 × `#N/A Authorization`** | **0** |
| **Either agency** | | | | **4 / 3,000 = 0.13%** |

Two distinct causes, needing different responses:

1. **CRISIL is an entitlement failure, not a data gap.** 2,999 of 3,000 cells returned
   `#N/A Authorization` — **the Terminal account is not licensed for that field.** This is
   the risk PRD §17 already records as "Bloomberg entitlement gaps — Confirmed occurred"
   (previously observed on ESG). No amount of re-pulling fixes it; it needs a licensing
   change, which is outside this project's control.
2. **ICRA is genuine non-coverage.** 537 explicit `NR` responses confirm the field works
   and that ICRA simply does not rate most of this universe — expected for a 3,000-name,
   small/mid-cap-heavy universe, consistent with §I8 (ICRA issuer-level ratings cover a few
   hundred large issuers).

**Decision forced: the §7.2 ratings-based graded label cannot be the primary label.
The §7.5 financial-criteria fallback is promoted to primary.** This has been the
contingency since the PRD was written; it is now the active path, not a hypothetical.

**On the RATC export (§13.9) as a rescue:** it does not solve this. RATC returns rating
*actions*, so a company appears only when its rating CHANGES. That is fine for detecting
downgrades, but it cannot distinguish "rated and stable" (label 0) from "never rated"
(unlabelled) — and supplying that denominator is exactly what the EQS snapshot was for,
which returned 0.13%. RATC remains useful as corroboration on the ~218 covered names and
for a downgrade-based robustness check, not as the primary label source.

### 16.2 ⚠️ The circularity guard was being bypassed by name drift

Promoting the fallback label to primary makes §Phase III Step 15's circularity trap live.
An audit of what the guard actually catches found **it was letting the worst offender
straight through.**

**Single-feature AUC against the forward fallback label** (22,299 rows, 22.9% positive):

| AUC | Feature | Guard verdict (before fix) |
|---|---|---|
| **0.941** | `consecutive_ic_below1_years` | **ALLOWED THROUGH** |
| 0.937 | `interest_coverage_capped` | blocked |
| 0.937 | `interest_coverage` | blocked |
| 0.926 | `interest_coverage_below_1_flag` | blocked |
| 0.721 | `equity_to_assets` | **ALLOWED THROUGH** |
| 0.721 | `X4_equity_to_liabilities` | blocked |

`consecutive_ic_below1_years` **is** §7.5 criterion 1 — "interest coverage below 1.0 for two
or more consecutive years" — reproduced verbatim as a feature. It leaks *harder than the
features the guard does block*, and it was invisible.

**Root cause: name drift, not a wrong decision.** `CIRCULAR_FEATURE_MAP` lives in
`fallback_label.py` and was written **before** `feature_engineering.py` existed. That module
later introduced:
- new derivatives the map never knew about (`consecutive_ic_below1_years`,
  `liab_to_assets_yoy_change`),
- a differently-named duplicate (`ic_yoy_change` vs the map's `interest_coverage_yoy_change`
  — a pure spelling miss),
- and `altman_zscore.py` added `altman_flag_negative_equity`, a duplicate of
  `negative_equity_flag` under another name.

The guard was correct when written and silently decayed. **This is §I5 re-opening**, and it
is the second time this project has caught the same class of defect before training rather
than after.

**Critical distinction applied in the fix.** A high AUC is NOT itself evidence of
circularity. `ebit_to_assets` (0.870), `roa` (0.839) and `consecutive_loss_years` (0.806) are
economically correlated with distress — unprofitability genuinely precedes it — and are
legitimate, retained signal. The test used is: **is the feature a deterministic function of
a label criterion?** Only then is it excluded.

### 16.3 Fix

`src/fallback_label.py`:

- **`CIRCULAR_FEATURE_MAP` expanded** from 9 to 15 entries. Added under `negative_equity`:
  `altman_flag_negative_equity`, `equity_to_assets`, `liab_to_assets_yoy_change`. Added under
  `low_interest_coverage`: `consecutive_ic_below1_years`, `ic_yoy_change`. The inclusion test
  is documented inline so future additions are judged consistently.
- **New `PARTIALLY_CIRCULAR`** set for the honest grey area. Altman Z″ carries the
  X4 = equity/liabilities term (1 of 4 terms, weight 1.05), so it is partly contaminated by
  the negative-equity criterion (AUC 0.766) while also being a **PRD-mandated Block 5
  feature** (§6.5) central to the project's "classical scores as features" thesis. It is
  **not** auto-excluded — this is a methodology judgement that must be made explicitly and
  stated, not silently resolved.
- **New `assert_no_circular_leak(feature_columns, criteria, strict=False)`** — called with
  the actual column list immediately before fitting. **Raises** (never warns-and-continues),
  because a silent inflated AUC is precisely the failure being guarded against. `strict=True`
  additionally rejects the Altman-derived set. This replaces trusting a static list, which is
  what decayed.

**Verified:** the guard now raises on the full matrix naming all 11 offenders; passes on the
cleaned set; and flags 3 further Altman-derived columns under `strict=True`.

### 16.4 Feature budget under the fallback label

| Policy | Usable features |
|---|---|
| All populated | 31 |
| Minus clear-cut circular | **20** |
| Minus circular + Altman-derived (strict) | 16 |

Top single-feature AUCs in the clean 20: `ebit_to_assets` 0.870, `roa` 0.839,
`consecutive_loss_years` 0.806, `altman_z_double_prime` 0.766. No leak-level predictors
remain. Note `X1`/`X2`/`X3` duplicate `working_capital_to_assets` /
`retained_earnings_to_assets` / `ebit_to_assets` exactly — redundancy to deduplicate at
assembly (multicollinearity), not leakage.

### 16.5 Open decisions arising

1. **Altman Z″ under a financial-criteria label — include or not?** The honest tension: with
   §7.5 as the primary label, *most* financial features are somewhat entangled with it. Z″ at
   AUC 0.766 is the clearest case needing an explicit, stated choice.
2. **An expected-AUC caveat for the paper.** With a financial-criteria label and ~20 clean
   features, a high AUC is partly structural. §3.3's ≥0.80 target was framed around a
   ratings-based label; its interpretation changes under the fallback and this must be said
   plainly rather than reported as a clean win.
3. `label_construction.py`'s `DOMESTIC_AGENCIES` still lacks `"CRIS"` (§13.9). Not fixed
   here — it is now lower priority since ratings are not the primary label, but it must be
   fixed before any RATC-based robustness check.

### 16.6 Files changed

| File | Change |
|---|---|
| `src/fallback_label.py` | `CIRCULAR_FEATURE_MAP` expanded 9 → 15; `PARTIALLY_CIRCULAR` added; `assert_no_circular_leak()` added |

---

## 17. Altman Z″ Circularity Decided + Training Matrix Built (2026-10-27)

Closes Phase 2's two remaining open decisions and produces the model-ready matrix.

### 17.1 DECISION — Z″ as a feature: use the non-circular 3-term variant

**The question.** With the §7.5 financial-criteria label promoted to primary (§16.1),
canonical Z″ = 6.56·X1 + 3.26·X2 + 6.72·X3 + **1.05·X4** embeds X4 = equity ÷ liabilities,
and label criterion 2 is "negative book equity". Is Z″ therefore circular?

**Evidence gathered rather than argued.** Decomposed against the forward label:

| Variant | AUC | n |
|---|---|---|
| Z″ full (4 terms) | 0.7661 | 17,846 |
| **Z″ without the X4 term** | **0.8069** | 17,853 |
| X4 term alone | 0.7209 | 20,487 |

> `sign(X4) == (book equity < 0)` on **100.00%** of 20,487 rows.
> `corr(Z″ full, Z″ ex-X4) = 0.9943`

X4's sign is a **deterministic restatement of label criterion 2** — circular beyond
argument. The decisive and unexpected part: **removing it IMPROVES AUC by +0.041.** X4 is
unbounded and carries the thin-liabilities degeneracy from §15.3, so it was contributing
noise as well as leakage.

**Decision:**
- **As a FEATURE** → the 3-term **`altman_z_ex_x4`**, with `altman_z_ex_x4_yoy_change` and
  `altman_z_ex_x4_3yr_trend` rebuilt from it.
- **As the §8.1 Tier-1 BENCHMARK** → **canonical Z″, unchanged.** Benchmarking against a
  modified score would be meaningless; the benchmark must be the classical model as
  published.

**Why this is defensible rather than convenient.** There is normally a tension between
methodological cleanliness and performance; here there is none, which removes the usual
suspicion that a leak was retained for the metric. It must still be **stated explicitly in
the methodology** — a reviewer who sees "Altman Z″" in a feature list alongside a
financial-criteria label will rightly ask, and the answer is that the X4 term was removed,
with the AUC decomposition above as evidence.

**Residual honesty.** Dropping X4 does not make the remaining terms independent of the
label. Under a financial-criteria label, most financial features are entangled with the
target to some degree. The claim is narrower and defensible: no feature in the final matrix
is a *deterministic function* of a label criterion.

### 17.2 Block 5 was contributing nothing — fixed

Building the matrix exposed a second defect. `build_block5_zscore()` carried a **hardcoded
column list** and *recomputed* Z″ from the panel rather than reading precomputed columns, so
`altman_z_ex_x4` never reached the feature matrix. Combined with the circularity exclusions,
**PRD §6.5's entire "Classical Composite Scores" block was reduced to two diagnostic flags**
— the "exploit fifty years of accumulated domain knowledge as a prior" rationale silently lost.

Fixed in `feature_engineering.py`: Block 5 now emits `altman_z_ex_x4`; Block 6 builds its
temporal derivatives; `build_feature_matrix` passes the column through; the feature catalogue
is updated. Feature count 13 → **16**.

### 17.3 New module: `src/build_training_matrix.py`

Assembles blocks 1–7 + forward label + split, and makes three decisions explicit in code:

1. **Label** — §7.5 criteria, forward-looking FY(t) → FY(t+1).
2. **Z″** — non-circular variant per §17.1 (`--allow-altman` escape hatch, not recommended).
3. **Split** — out-of-time, never random.

**Hard guard:** `assert_no_circular_leak(strict=True)` runs on the real column list before
the matrix is written. It **raises**, so a circular feature cannot reach training even if a
future edit reintroduces one.

**Explicitly NOT done here:** no scaling, imputation or encoding. Those must be fitted on the
training window only (§7.4, invariant #3); this module emits raw features plus a `split`
column and leaves fitting downstream.

**Redundant aliases dropped** (`X1_/X2_/X3_` duplicate `working_capital_to_assets` /
`retained_earnings_to_assets` / `ebit_to_assets` exactly; `altman_ems_score` = Z″ + 3.25 is
perfectly collinear). Keeping twins would split SHAP attribution across identical columns and
degrade the explanation layer — which is the project's entire purpose.

### 17.4 The out-of-time split

PRD §8.5's literal windows do not fit the data once the FY convention is resolved (§14.5):
the test window overruns the panel by a year and FY2014-15 falls outside every window. The
split keeps §8.5's **shape** (long train / 1 validation year / multi-year test) while using
every available year. Features at FY(t) predict FY(t+1), so t runs 2014–2023.

| Split | Indian FY range | Rows | Positives | Rate | Tickers |
|---|---|---|---|---|---|
| train | FY2014-15 … FY2020-21 | 14,675 | 3,514 | 23.9% | 2,580 |
| validation | FY2021-22 | 2,566 | 543 | 21.2% | 2,566 |
| test | FY2022-23 … FY2023-24 | 5,058 | 1,046 | 20.7% | 2,649 |

After the coverage gate: train 13,242 (25.6% positive), validation 2,305 (22.8%),
test 4,830 (21.3%).

**⚠️ Class balance is ~21–26% positive, far above PRD §8.3's anticipated 5–15%.** This is a
property of the §7.5 financial-criteria label, not of the universe — it counts *states*
(negative equity, sustained sub-1 coverage) rather than *events* (defaults). It must be
reported, and §3.3's metric targets must be interpreted against it rather than against the
ratings-label framing they were written for (§16.5 item 2).

### 17.5 Final matrix

`data/processed/training_matrix.csv` — **22,299 rows × 24 cols, 16 features.**

Single-feature AUC on the **training window only** (13,242 rows), to confirm no residual leak:

| AUC | Feature | | AUC | Feature |
|---|---|---|---|---|
| 0.862 | ebit_to_assets | | 0.682 | cur_ratio_conservative |
| 0.845 | roa | | 0.682 | cur_ratio |
| 0.799 | **altman_z_ex_x4** | | 0.651 | altman_z_ex_x4_yoy_change |
| 0.796 | consecutive_loss_years | | 0.630 | log_mkt_cap |
| 0.760 | retained_earnings_to_assets | | 0.607 | equity_volatility |
| 0.700 | working_capital_to_assets | | 0.578 | roa_yoy_change |
| 0.684 | altman_z_ex_x4_3yr_trend | | 0.551 | log_total_assets |
| | | | 0.507 / 0.500 | the two Altman diagnostic flags |

**Max single-feature AUC 0.862** (`ebit_to_assets`) — high but economically legitimate:
unprofitability genuinely precedes distress. No leak-level predictors (cf. the 0.941 that
§16.2 caught). The audit is run on train only, so it does not itself touch the test window.

### 17.6 Files changed

| File | Change |
|---|---|
| `src/build_training_matrix.py` | **New.** Matrix assembly, circularity guard, split assignment |
| `src/score_altman_panel.py` | `altman_z_ex_x4` (+ capped, yoy, 3yr trend) added |
| `src/feature_engineering.py` | Block 5 emits `altman_z_ex_x4`; Block 6 builds its derivatives; catalogue updated |
| `data/processed/training_matrix.csv` | **New.** 22,299 × 24, 16 features |
| `data/processed/bloomberg_panel_scored_{full,gated}.csv` | Regenerated with the ex-X4 columns |

### 17.7 Phase 2 status

All exit criteria met except the two reporting items now satisfied above. **Phase 3 (Model A)
is unblocked.** Remaining caveats to carry into it: the elevated class balance (§17.4), the
reinterpretation of §3.3's targets under a financial-criteria label, and the 2 unresolved
accounting-identity suspects (`TATAINVE`, `UVDH`).

---

## 18. Phase 3 — Model A Trained and Benchmarked (2026-10-27)

First real training run on real data. New modules: `src/metrics.py` (shared §8.6
evaluation, §8.7 DeLong + bootstrap) and `src/train_model_a.py` (the §8.1 hierarchy).

### 18.1 Headline results — out-of-time test, scored once

Train FY2014-15…FY2020-21 (13,242 rows) · Validation FY2021-22 (2,305) ·
Test FY2022-23…FY2023-24 (4,830). 16 non-circular features.

| Model | n | Cov. | AUC | PR-AUC | KS | Brier | Recall | Prec. |
|---|---|---|---|---|---|---|---|---|
| **Persistence (label at t)** — naive | 4,830 | 1.00 | **0.8893** | 0.7120 | 0.7786 | — | — | — |
| Altman Z″ (fitted threshold) | 4,251 | 0.88 | 0.7708 | 0.6563 | 0.4852 | — | 0.745 | 0.389 |
| Altman Z″ @ conventional 1.1 | 4,251 | 0.88 | 0.7708 | 0.6563 | 0.4852 | — | 0.575 | 0.656 |
| Logistic (L2, balanced) | 4,830 | 1.00 | 0.8855 | 0.6549 | 0.6828 | 0.1582 | 0.943 | 0.366 |
| **XGBoost** | 4,830 | 1.00 | **0.9589** | **0.8481** | **0.8005** | **0.0819** | 0.965 | 0.602 |

Against §3.3: AUC ≥0.80 ✅ · PR-AUC ≥0.45 ✅ · KS ≥0.40 ✅ · Recall ≥0.70 ✅ ·
uplift vs Altman ≥+0.10 significant ✅ (+0.1839, DeLong p<0.001).

**Every target is met — and that is precisely why the next section exists.**

### 18.2 ⚠️ The pooled AUC is inflated by label persistence

AUC 0.9589 is far above what the distress literature reports (~0.80–0.90), so it was
treated as suspect rather than as success. The cause is not leakage this time; it is the
label's dynamics:

> The §7.5 label is **the same in FY(t) and FY(t+1) on 90.7% of rows.**
> Using the contemporaneous label as the entire prediction scores **AUC 0.858** on the
> panel, **0.8893** on the test window.

This is §Phase III Step 15's warning materialising exactly as written: *"negative equity
and sub-1 coverage are highly persistent year over year, so FY(t) values predict FY(t+1)
status nearly mechanically."* The forward-looking design softened it; it did not remove it.

**Persistence was therefore added as a formal Tier-0 benchmark**, and it reorders the
findings completely (DeLong, same test rows):

| vs persistence (AUC 0.8893) | ΔAUC | p | Verdict |
|---|---|---|---|
| Altman Z″ | **−0.1130** | <0.001 | **SIGNIFICANTLY WORSE than doing nothing** |
| Logistic regression | −0.0038 | 0.620 | **NOT significantly different** |
| XGBoost | **+0.0696** | <0.001 | **SIGNIFICANTLY BETTER** |

**Only XGBoost adds genuine value over "assume the condition continues."** The logistic
baseline's apparently respectable 0.8855 is, in substance, a reproduction of persistence.
Reporting the Altman comparison alone (+0.18, p<0.001) without this would have been
materially misleading — the classical benchmark loses to a baseline that requires no model
at all.

### 18.3 Stratified evaluation — the honest early-warning number

Pooled metrics mix two different tasks. Splitting by state at FY(t):

| Subset | n | Prevalence | Altman | Logistic | **XGBoost** |
|---|---|---|---|---|---|
| All test rows | 4,830 | 21.3% | 0.7708 | 0.8855 | 0.9589 |
| Already distressed at t (predicting *recovery*) | 1,046 | 81.5% | 0.6391 | 0.6655 | 0.7097 |
| **Healthy at t → genuine early warning** | **3,784** | **4.6%** | 0.6581 | 0.8490 | **0.9548** |

**The third row is the real result.** Every row in it is healthy at FY(t), so persistence
scores 0.5 there *by construction* — the 0.9548 is genuine predictive skill, not state
repetition. Altman manages only 0.6581 on the same rows. XGBoost PR-AUC there is 0.4636,
just clearing §3.3's ≥0.45.

Note also that this subset's **4.6% prevalence sits inside PRD §8.3's anticipated 5–15%
range**, unlike the pooled 21.3%. It is the methodologically appropriate population for the
headline claim, and should be what the paper leads with.

Predicting *recovery* (row 2) is markedly harder for every model — a genuine finding worth
reporting rather than hiding.

### 18.4 §8.4 cost-ratio sensitivity — the 10:1 default misfires here

Required by §8.4 ("the ratio must be stated as an assumption and its sensitivity
reported"). Logistic model, threshold re-chosen on validation at each ratio:

| FN:FP | Threshold | Recall | Precision | F1 | Flagged |
|---|---|---|---|---|---|
| 1:1 | 0.680 | 0.659 | 0.648 | 0.654 | 21.6% |
| 3:1 | 0.485 | 0.811 | 0.612 | **0.698** | 28.2% |
| 5:1 | 0.440 | 0.846 | 0.571 | 0.682 | 31.5% |
| **10:1 (PRD default)** | 0.325 | 0.943 | 0.366 | 0.528 | **54.7%** |
| 20:1 | 0.305 | 0.955 | 0.338 | 0.499 | 60.1% |

**At this label's 21–26% prevalence the PRD's 10:1 default flags 54.7% of the portfolio** —
near-useless for the §4.2 triage journey, where the product promise is "where to look
first." At 22.8% validation prevalence, flag-everything costs 0.772/row against
flag-nothing's 2.282, so aggressive flagging genuinely is cost-optimal; the assumption, not
the optimiser, is the problem. §8.4's 10:1 was framed for the 5–15% prevalence of an
event-based label. **Recommendation: operate at 3:1 (best F1, 28.2% flagged) and report
10:1 as the PRD-specified sensitivity case.** Not unilaterally changed — §8.4 is explicit
that the ratio is a stated assumption, so this is the user's call.

### 18.5 A second degenerate threshold, caught and fixed

The first run fitted the Altman benchmark threshold at **Z″ ≤ 259.96** (recall 0.996,
accuracy 0.244 — flag everything). Cause: the candidate grid was built from *quantiles of
raw Z″*, whose validation 99th percentile is ~260 because of the §15.3 degenerate tail. The
grid now spans a plausible score range (−10 to +10), giving Z″ ≤ 3.70, and the conventional
Z″ < 1.1 zone boundary is reported alongside it as a cost-assumption-free reference.

### 18.6 Invariants enforced in code

- **Out-of-time split** — `_assert_out_of_time()` verifies train < validation < test with
  no overlap and raises otherwise. The split is read from the matrix, never resampled.
- **Preprocessing fitted on train only** — `_assert_fitted_on_train_only()` re-fits on the
  training window alone and asserts the statistics are identical; it also reports the
  train-vs-full median drift (1.1984), whose being non-zero is the positive evidence that
  train statistics are *not* full-sample statistics. This is §15's "most valuable test".
- **Never zero-fill** — median imputation with TRAIN medians plus explicit
  `*__was_missing` indicators; XGBoost receives raw NaN and uses native handling.
- **Threshold on validation only** — test is scored once, at the end.
- **All three tiers reported together** (§8.1), not just the winner.

### 18.7 Caveats to carry forward

1. **The headline must be the stratified early-warning number (AUC 0.9548 on healthy-at-t),
   not the pooled 0.9589.** The pooled figure is substantially persistence.
2. **Altman Z″ loses to a no-model baseline.** Report it; it is a legitimate finding about
   a 1968-calibrated score on this label, not an embarrassment.
3. **The logistic baseline adds nothing over persistence** (p=0.62). §8.1's claim of
   "beating a properly specified logistic regression" holds, but the baseline itself is
   weaker than it appears.
4. Class balance (21–26%) and the §3.3 reinterpretation from §16.5/§17.4 still apply.
5. LightGBM (§8.1 comparator) is **not installed**; not run.
6. Still outstanding: rolling-origin folds (§8.5 supplementary), SHAP (§8.8), model card
   (§14.1), and the `TATAINVE`/`UVDH` identity suspects.

### 18.8 Files changed

| File | Change |
|---|---|
| `src/metrics.py` | **New.** §8.6 metrics, §8.4 cost threshold, DeLong, bootstrap CI |
| `src/train_model_a.py` | **New.** §8.1 hierarchy + persistence tier, stratification, cost sensitivity, leakage assertions |
| `reports/phase3/model_comparison.csv` | **New.** Test-set results |

---

## §19 — Phase 3 (cont.): SHAP Explainability Layer (§8.8)
*2026-10-27*

Built `src/explain_model_a.py`, delivering all four §8.8 levels against **the
XGBoost object that produced §18's table** — `train_model_a.run()` now returns
its fitted estimator and split frames in `store`, so nothing is re-fitted. A
second fit "with the same seed" is not the same object, and explaining a model
you did not report is worse than not explaining one.

### 19.1 Additivity asserted, not assumed
TreeSHAP is exact for tree ensembles, so `base + Σ contributions` must reproduce
the model margin. Verified: **max reconstruction error 4.4e-07 log-odds**;
base value **+0.0625 → 51.56% base rate** (the `scale_pos_weight` rebalanced
prior, not the sample prevalence — important when reading waterfalls). The check
`raise`s rather than warns: if attributions are not a decomposition of the
prediction, every number downstream is decorative.

### 19.2 Global ranking (test set, mean |SHAP| in log-odds)

| Feature | Block | share | mean signed | corr(value, SHAP) |
|---|---|---:|---:|---:|
| `ebit_to_assets` | profitability | **33.97%** | −1.4022 | −0.054 |
| `roa` | profitability | 11.43% | −0.3333 | −0.122 |
| `retained_earnings_to_assets` | profitability | 10.85% | −0.2618 | −0.156 |
| `log_total_assets` | controls | 7.44% | **+0.1131** | **+0.828** |
| `altman_z_ex_x4` | z-score | 6.83% | −0.1739 | −0.113 |
| `log_mkt_cap` | controls | 6.52% | −0.1813 | **−0.813** |
| `consecutive_loss_years` | temporal | 5.29% | −0.1872 | +0.593 |
| `working_capital_to_assets` | liquidity | 4.80% | −0.049 | −0.136 |

Remaining eight features share 9.9%. Full table in
`reports/phase3/shap_global_ranking.csv`.

### 19.3 Block view — and a reporting trap caught before it shipped
`FEATURE_GROUPS` files `ebit_to_assets` under `block4_debt_service` (it is the
numerator of interest coverage). But **no interest-coverage feature reached this
matrix** — those await the expanded pull — so block4 holds that one feature and
nothing else. Aggregating the catalogue as-is produces a committee chart reading
*"debt servicing drives 34% of the model"* when the model sees **no
debt-servicing variable at all**. EBIT/TA is a return measure, so the
attribution view reassigns it to profitability via an explicit
`BLOCK_ATTRIBUTION_OVERRIDE`, and the substitution is **printed beneath the
table every run** rather than silently applied. The catalogue itself is
unchanged.

| Block | share | n features |
|---|---:|---:|
| profitability | **56.25%** | 3 |
| controls | 16.75% | 3 |
| z-score | 11.18% | 5 |
| liquidity | 8.96% | 3 |
| temporal | 6.86% | 2 |

The model is overwhelmingly a **profitability** model. That is a finding, not a
flaw — but it is also the honest answer to "what does CRESTA add over Altman":
it weights current earnings power far more heavily than Altman's fixed coefficients do.

### 19.4 The two size controls carry OPPOSITE signs — and that is the interesting result
`log_total_assets` has corr(value, SHAP) = **+0.83** (bigger assets → model
pushes risk UP) while `log_mkt_cap` has **−0.81** (bigger market cap → risk
DOWN). Read together the model has learned **a high asset base relative to
market valuation is a distress signal** — economically coherent (the market
declining to capitalise the balance sheet), and not something either control
expresses alone. Worth stating explicitly in the paper, because an unexplained
positive size coefficient reads as a bug to a reviewer.

This also pre-empts §9.3's confound warning on the Track A side: size is not
acting as a monotone "big = safe" proxy here.

### 19.5 Dead feature
`altman_flag_x4_uncomputable` received **exactly zero** attribution — XGBoost
never split on it across 400 trees. Reported each run rather than quietly
dropped; it is near-constant in the gated panel. Candidate for removal, but
removing it changes the matrix, so it is logged, not actioned.

### 19.6 Local + counterfactual
Five local waterfalls drawn from the **early-warning stratum** (healthy at t,
distressed at t+1), highest confidence first — these are the predictions the
product exists to make; explaining an already-distressed row largely restates
its current state. All five FY2022: SKIL (99.1%), SVPGLOB (98.8%), MRSP (98.4%),
SNGI (98.4%), STAEL (98.3%).

Counterfactuals match each case to its nearest peer that was **healthy at t,
healthy at t+1, and scored p < 0.10**, inside a ±0.5 sd size band, Euclidean in
feature space standardised on the healthy pool. The contrast is therefore
against a genuinely healthy comparator, not merely a low-scored one. In all five
pairs `ebit_to_assets` is the largest Δ-SHAP (+2.9 to +4.1 log-odds).

Artefacts: `reports/phase3/shap_{global_ranking,block_ranking,local_cases,counterfactuals}.csv`,
`shap_summary.json`, `reports/phase3/figures/` (beeswarm, block bar, 5 waterfalls).

### 19.7 Scale inconsistency found in the feature matrix (logged, not fixed)
`roa` is inherited from Bloomberg `RETURN_ON_ASSET` **in percent** (median 1.98,
range −161,400 to +16,550), while `ebit_to_assets` is a **decimal** (median
0.0377). Two homogeneous return measures on different scales. Harmless to
XGBoost (splits are scale-invariant) and to the standardised logistic, but it is
exactly the §2.2 scaling trap, and the dashboard must not render ROA = 34.1
beside EBIT/TA = −0.002 without units. **Not changed unilaterally** — rescaling
`roa` alters the matrix and would invalidate §18's numbers.

### 19.8 The caveat is carried by the artefacts, not by the prose
`CAVEAT` ("SHAP explains the MODEL, not the world…") is printed with every
table, stamped on every figure, and stored in `shap_summary.json`, so it cannot
be detached from the numbers when they are copied into the report or the
dashboard. §8.8 names conflating attribution with causation as a visible and
damaging error; this makes separating them require deliberate effort.

### 19.9 Files
| File | Change |
|---|---|
| `src/explain_model_a.py` | **New.** §8.8 four-level explainability layer |
| `src/train_model_a.py` | Returns `_frames` / `_matrix` in `store` so the explained model is the reported model |
| `reports/phase3/shap_*.csv`, `shap_summary.json`, `figures/*.png` | **New.** |

---

## §20 — Phase 3 (cont.): Rolling-Origin (Walk-Forward) Validation (§8.5)
*2026-10-27*

Built `src/rolling_origin.py`. §8.5's primary protocol yields ONE estimate from
ONE arbitrary split date; this yields a distribution across regimes.

### 20.1 Fold design, and where it extends the PRD sketch
§8.5 sketches four folds of the form `train 2015-2019 -> test 2020`, with **no
validation year**. But a cost-optimal threshold (§8.4) has to be chosen
somewhere, and choosing it on the fold's own test set is leakage. Each fold here
therefore holds out the **last training year** as that fold's validation:

| Fold | Train | Val | Test | n train | n test | test prev. |
|---|---|---|---|---:|---:|---:|
| 1 | 2014-2017 | 2018 | 2019 | 6,977 | 2,080 | 30.3% |
| 2 | 2014-2018 | 2019 | 2020 | 8,957 | 2,205 | 26.8% |
| 3 | 2014-2019 | 2020 | 2021 | 11,037 | 2,305 | 22.8% |
| 4 | 2014-2020 | 2021 | 2022 | 13,242 | 2,436 | 22.0% |
| 5 | 2014-2021 | 2022 | 2023 | 15,547 | 2,394 | 20.5% |

Ranking metrics (AUC, PR-AUC, KS) do not depend on a threshold and are
unaffected by this addition; recall/precision would be unreportable without it.
`fiscal_year_num` is the Indian FY **start** year (§14), so fold 5 tests FY2023-24.

**Everything is re-fitted inside each fold** — preprocessing statistics, class
weights, both models — on that fold's training window alone (§7.4, invariant #3).
A single preprocessor fitted once on the full panel and reused across folds is
the most common way walk-forward validation is quietly invalidated.

### 20.2 Results

| Fold (test yr) | persistence | Altman | logistic | XGBoost | Altman EW | logistic EW | **XGBoost EW** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (2019) | 0.8548 | 0.7594 | 0.8897 | 0.9399 | 0.6482 | 0.8419 | 0.9132 |
| 2 (2020) | 0.8858 | 0.7534 | 0.8743 | 0.9435 | 0.6178 | 0.8305 | 0.9385 |
| 3 (2021) | 0.9077 | 0.7746 | 0.8736 | 0.9547 | 0.6876 | 0.8239 | 0.9221 |
| 4 (2022) | 0.8830 | 0.7635 | 0.8814 | 0.9592 | 0.6233 | 0.8469 | 0.9571 |
| 5 (2023) | 0.8963 | 0.7783 | 0.8924 | 0.9612 | 0.7028 | 0.8538 | 0.9528 |

**Mean ± sd across folds**

| Model | pooled AUC | early-warning AUC | PR-AUC |
|---|---|---|---|
| persistence | 0.8855 ± 0.0198 | 0.5 by construction | 0.7216 ± 0.0100 |
| Altman Z″ | 0.7659 ± 0.0104 | 0.6560 ± 0.0380 | 0.6745 ± 0.0256 |
| logistic | 0.8823 ± 0.0086 | 0.8394 ± 0.0121 | 0.6902 ± 0.0515 |
| **XGBoost** | **0.9517 ± 0.0095** | **0.9367 ± 0.0190** | **0.8551 ± 0.0076** |

`"EW"` = the HEALTHY-at-t early-warning subset (§18.7). Persistence scores 0.5
there by construction, so it carries no EW column; every fold reports both so
the persistence inflation is visible five times, not restated five times.

### 20.3 The §18 single-split estimate is not a lucky split
§18's reported test AUC **0.9589** corresponds to fold 4's 0.9592 and sits
**+0.0072 above the five-fold mean — well inside one sd (0.0095)**. The headline
is a representative draw, not a favourable one. This is the specific claim
rolling-origin exists to support, and it is now evidenced.

One honest qualification: §18's **early-warning** figure of 0.9548 sits
**+1.8pp above** the walk-forward EW mean of 0.9367. The paper should quote the
**distribution** for the early-warning claim, not the single-split point.

### 20.4 Every model is stable; only Altman is both weak AND unstable
XGBoost's pooled sd is 0.0095 and its PR-AUC sd just 0.0076 across five
origins — no regime dependence worth reporting. Altman's **early-warning** sd is
0.0380, four times XGBoost's, so the classical benchmark is not merely worse on
average, it is less reliable fold to fold.

Fold 2 (test FY2020-21, the COVID shock) is where that shows: **Altman EW falls
to its minimum 0.6178** while XGBoost EW holds at 0.9385, above its own fold
mean. The learned model absorbed the regime break that the fixed-coefficient
model did not. Fold 1 is XGBoost's weakest (0.9399 / 0.9132 EW) on the smallest
training window — monotone improvement with training size thereafter, which is
the expected shape and a mild argument that more history would still help.

### 20.5 Incidental fix — `penalty="l2"` deprecation
sklearn 1.8 deprecates naming `penalty="l2"` (removal in 1.10). L2 is the
default, so the argument was dropped in both `train_model_a.py` and
`rolling_origin.py`. **Verified behaviour-identical**: re-running
`train_model_a.py --xgb` reproduces §18 exactly — persistence 0.8893, Altman
0.7708, logistic 0.8855, XGBoost 0.9589, to four decimals.

### 20.6 Files
| File | Change |
|---|---|
| `src/rolling_origin.py` | **New.** §8.5 walk-forward validation, 5 folds, per-fold re-fit |
| `src/train_model_a.py` | Dropped deprecated `penalty="l2"` (behaviour-identical, verified) |
| `reports/phase3/rolling_origin_folds.csv`, `rolling_origin_summary.csv` | **New.** |
| `reports/phase3/figures/rolling_origin_auc.png` | **New.** Stability chart, pooled vs early-warning |

---

## §21 — Phase 3 (cont.): Model Card + Artefact Registration (§14.1)
*2026-10-27*

Built `src/model_card.py`. It trains Model A, registers the fitted artefact with
reproducibility hashes, and generates `MODEL_CARD.md` (320 lines, all ten §14.1
sections). **Every number in the card is computed in the generating run** — none
is transcribed. A card whose performance table was pasted in by hand is a card
that goes stale silently.

### 21.1 DEFECT FOUND: §5.6 / invariant 5 is not enforced anywhere
Writing the card's required "known exclusions" section forced the question of
whether financial-sector issuers are actually excluded. **They are not.**
`grep -i financ` across `build_training_matrix.py`, `quality_gates.py`,
`load_formula_chunks.py` and `score_altman_panel.py` returns only comments.
**1,763 matrix rows across 217 bank/NBFC tickers are in the training data**, and
398 rows / 206 tickers are in the test set.

Measured impact on the reported figures:

| | n (test) | AUC | early-warning AUC |
|---|---:|---:|---:|
| As reported in §18–§20 (financials included) | 4,830 | 0.9589 | 0.9548 |
| Financials removed at scoring time | 4,432 | 0.9557 | 0.9502 |

**−0.0031 pooled, −0.0046 early-warning.** The §18–§20 conclusions do not depend
on it — Altman still loses to persistence, XGBoost still wins. But the mechanism
matters more than the magnitude: financials score **AUC 0.9888 (early-warning
0.9989)** at a prevalence of 8.5%, the best of any sector by a distance. That is
not the model understanding banks. It is the model reading the **missingness
signature** of a bank — absent current assets, current liabilities and EBIT, the
exact §5.6 signature `altman_zscore.sanity_check_report()` already warns about —
and inferring "rarely labelled distressed". It is a shortcut, and it inflates
the aggregate.

**Remediation requires a retrain, not a scoring-time filter**, so it is
documented and left as an explicit decision rather than patched silently. Until
then every figure in §18–§21 is "financials included", and the card says so in
its own numbered section rather than in a footnote.

### 21.2 Per-sector breakdown — and the real weak spot
Sector is joined from `universe_800.csv` (`gics_sector`); the panel carries
none, and it is missing for ~16% of matrix rows, reported as `(sector unknown)`
rather than imputed or dropped.

| Sector | n | prev. | AUC | EW AUC |
|---|---:|---:|---:|---:|
| Industrials | 852 | 18.3% | 0.9714 | 0.9555 |
| Consumer Discretionary | 828 | 21.6% | 0.9547 | 0.9731 |
| (sector unknown) | 780 | 26.4% | 0.9412 | 0.8941 |
| Materials | 661 | 18.0% | 0.9657 | 0.9747 |
| Financials | 398 | 8.5% | 0.9888 | 0.9989 |
| Consumer Staples | 351 | 25.1% | 0.9574 | 0.9502 |
| Information Technology | 304 | 20.1% | 0.9504 | 0.9501 |
| Health Care | 232 | 17.7% | 0.9640 | 0.9530 |
| **Real Estate** | 208 | **38.0%** | **0.8691** | **0.7874** |
| Communication Services | 145 | 37.2% | 0.9668 | 0.9444 |
| Energy | 36 | 8.3% | 1.0000 | — |
| Utilities | 35 | 20.0% | 0.9592 | — |

**Real Estate is the weak spot**: lowest AUC, lowest early-warning AUC by far
(0.7874), at the highest prevalence in the panel. §12.4 requires the model's
weaknesses to be visible, not just its averages — this is the row a reviewer
will look for, so it is named in the card rather than buried in a table.

### 21.3 Artefact registration
`models/model_A-1.0.0-real.pkl` + `models/registry.json`. The registry entry
records version, UTC training time, owner, git commit, config hash, Python
version, matrix hash, row counts, threshold, the full metric set, and the
sector-exclusion audit above.

**`models/model_A.pkl` was deliberately NOT overwritten.** That artefact is the
**synthetic-data** model `cresta-app/` currently serves; overwriting it would
silently repoint the running app at a different model. The real-data artefact
gets its own versioned filename and the collision is stated in card §1.

**Config hash** = sha256 over the nine pipeline source files plus the training
matrix and scored panel. Verified stable across runs:
`409661a383cb…81c2b1`. A changed hash under an unchanged version number is the
signal that someone edited the pipeline without re-versioning. (The *artefact*
sha256 does change per run — `trained_utc` is inside the pickle payload — so the
config hash, not the artefact hash, is the reproducibility anchor.)

### 21.4 Governance gap: no git history
§14.1 requires a git commit hash per model version. **This directory is not a
git repository**, so the field is recorded as `null` in the registry and the
card prints an explicit "UNAVAILABLE" with the reason, rather than a placeholder
that would read as a real hash. The config hash pins *content* but not
*provenance* — it cannot tell you what changed or who changed it. Recommend
`git init` before Phase 4; it is a two-minute fix that closes a stated
compliance requirement.

### 21.5 What the card says that the metrics tables do not
Sections written to be read by a validator, not generated from a dataframe:
- **Misuse** — explicitly not investment advice, not a credit rating, not a
  default probability, not applicable to financials, not an automated decision.
- **Human-in-the-loop (§14.5)** as a non-negotiable operating constraint.
- **Fairness framed on the honest axis** — not demographic but **size and
  disclosure quality**: smaller issuers have thinner data, so they are both more
  likely to be imputed and more likely to be distressed, and the model may
  penalise opacity as much as weakness. The `*__was_missing` indicators make
  that legible instead of hidden.
- **Eleven enumerated limitations**, including the label proxy, persistence
  inflation, the financials defect, the `roa` percent/decimal mismatch, the
  16-feature subset, and "SHAP explains the model, not the world".

### 21.6 Files
| File | Change |
|---|---|
| `src/model_card.py` | **New.** §14.1 card generator + artefact registry |
| `MODEL_CARD.md` | **New.** 320 lines, all ten §14.1 sections, fully generated |
| `models/model_A-1.0.0-real.pkl`, `models/registry.json` | **New.** Registered artefact |

### 21.7 Open items added by this section
1. **Decide on the §5.6 financial-sector exclusion** — retrain excluding 1,763
   rows / 217 tickers, or document a reasoned deviation. Blocks the Phase 3
   exit criterion as literally written.
2. `git init` to close the §14.1 commit-hash requirement.
3. `cresta-app/` still serves the synthetic model — repointing it at
   `model_A-1.0.0-real.pkl` is Phase 4 work, not done here.

---

## §22 — Phase 3 (cont.): §5.6 Financial-Sector Exclusion Implemented; Full Retrain
*2026-10-27*

Acting on the §21.1 defect. The exclusion is now enforced in
`build_training_matrix.py` and the whole Phase 3 stack was re-run end to end.

### 22.1 CORRECTION to §21.1 — right conclusion, wrong evidence
§21.1 claimed the model was reading "the missingness signature of a bank —
absent current assets, current liabilities **and EBIT**". Measured, that joint
test does **not** discriminate: all three missing together occurs on 18.9% of
financial rows vs 21.8% elsewhere. The claim as written was wrong.

The signature is real but narrower, and it is the two current-account fields
alone:

| Field | null rate, Financials | null rate, other known sectors |
|---|---:|---:|
| `current_assets` | **86.3%** | 23.5% |
| `current_liabilities` | **85.4%** | 23.5% |
| `ebit` | 19.8% | 22.0% |
| `total_revenue` | 12.2% | 21.2% |

Banks have no operating cycle, so Bloomberg reports no current assets or
liabilities — but EBIT is reported at the same rate as everywhere else. Three of
sixteen features (`working_capital_to_assets`, `cur_ratio`,
`cur_ratio_conservative`) are therefore structurally NaN for these firms, and
XGBoost's native NaN branch can isolate them on that absence. The §21 card text
has been regenerated with the corrected statement.

### 22.2 Exclusion rule — and the judgement call in it
Implemented as `flag_financial_sector()` in `build_training_matrix.py`,
default-on, disabled with `--keep-financials`.

1. **`gics_sector == "Financials"`** — 2,021 rows / 231 tickers.
2. **Name fallback, applied ONLY to sector-unknown tickers** — 436 rows / 52
   tickers. `gics_sector` is missing for 692 of 2,999 tickers, so without this
   a known-nonzero number of lenders stays in the training data.

**Total excluded: 2,457 rows / 283 tickers**, written to
`data/processed/excluded_financials.csv` with a per-row reason. Never silently
dropped — the rows you remove are exactly the ones a reviewer asks about.

The name regex (`bank|financ|capital|invest|securit|insur|nbfc|housing fin|
leasing|credit|asset manag|broking|fincorp|finvest`) was **validated against the
labelled tickers before use**: it matches **62.7% of known Financials and 0.7%
of known non-financials**. High precision, moderate recall. Excluding a few
genuine non-financials is the conservative direction of error, but it IS a
heuristic and it is listed in the card's limitations, with `--no-name-heuristic`
to turn it off.

Matrix: 22,299 → **19,842 rows**. Archived the previous matrix as
`data/processed/training_matrix_with_financials.csv` so the comparison stays
reproducible.

### 22.3 Retrained results (v1.1.0) vs the defective v1.0.0

| | v1.0.0 (financials in) | **v1.1.0 (excluded)** |
|---|---:|---:|
| Test rows | 4,830 | 4,340 |
| Test prevalence | 21.3% | 22.7% |
| Persistence AUC | 0.8893 | 0.8859 |
| Altman Z″ AUC | 0.7708 | 0.7729 |
| Logistic AUC | 0.8855 | 0.8820 |
| **XGBoost AUC** | 0.9589 | **0.9550** |
| **XGBoost PR-AUC** | 0.8481 | **0.8487** |
| **Early-warning AUC** | 0.9548 | **0.9490** |
| EW subset n / prevalence | 3,784 / 4.6% | 3,334 / 5.1% |

**Every conclusion survives, with the same signs and the same significance:**

| vs persistence | v1.0.0 | v1.1.0 |
|---|---|---|
| Altman Z″ | −0.1130, p<0.001 **worse** | **−0.1110, p<0.001 worse** |
| Logistic | −0.0038, p=0.620 not different | **−0.0039, p=0.622 not different** |
| XGBoost | +0.0696, p<0.001 better | **+0.0691, p<0.001 better** |

The headline fell 0.0039. **PR-AUC actually rose slightly** (0.8481 → 0.8487),
which is the more informative metric under imbalance. The retained-financials
model scored higher only because 398 test rows at 8.5% prevalence scored 0.9888
and flattered the aggregate.

### 22.4 Rolling-origin re-run — stability unchanged

| Model | pooled AUC | early-warning AUC | PR-AUC |
|---|---|---|---|
| persistence | 0.8817 ± 0.0199 | 0.5 by construction | 0.7233 ± 0.0129 |
| Altman Z″ | 0.7679 ± 0.0091 | 0.6602 ± 0.0333 | 0.6777 ± 0.0282 |
| logistic | 0.8819 ± 0.0111 | 0.8398 ± 0.0182 | 0.7014 ± 0.0585 |
| **XGBoost** | **0.9473 ± 0.0099** | **0.9317 ± 0.0187** | **0.8562 ± 0.0112** |

Single-split 0.9550 sits inside one sd of the 0.9473 fold mean — still a
representative draw. Altman's early-warning sd (0.0333) is still ~3.4× XGBoost's.

### 22.5 SHAP re-run — ranking essentially invariant
`ebit_to_assets` 34.0% → **36.0%**; profitability block 56.2% → **57.8%**.
`log_total_assets` corr +0.83 → **+0.84**, `log_mkt_cap` −0.81 → **−0.79** — the
opposite-signs finding (§19.4) is not an artefact of the financial rows.
`altman_z_ex_x4` moved up to 4th. `altman_flag_x4_uncomputable` still receives
exactly zero attribution.

### 22.6 Model card v1.1.0 — and a stale-numbers bug in the generator
Card §5 is now **"Financial-sector exclusion — ENFORCED"**, and the generator
**asserts** it: `build()` raises if any GICS-Financials row survives into the
test split. A card that asserts an exclusion without verifying it is how the
defect reached §21 in the first place.

While regenerating, **the card's prose was still quoting v1.0.0 numbers** —
`0.9517 ± 0.0095`, `90.7%`, `+0.1881`, `~34%`, `+0.83/−0.81`, the whole
stratified table — because those sentences were hardcoded in the template even
though `model_card.py`'s own docstring warns that a hand-pasted table goes
stale. Exactly the failure it was written to prevent. All of them are now
computed in the generating run: persistence rate, all DeLong deltas and
p-values, the stratified table, the rolling-origin sentence (including whether
the single split falls inside one sd — the verdict flips automatically if it
ever does not), and the SHAP top feature and size correlations.

Also fixed: `_tbl()` decided integer-vs-decimal per *cell*, so an AUC of exactly
1.0000 rendered as `1` next to `0.9663`. Now decided per *column*. And a DeLong
p-value that underflows to 0.0 now prints `< 1e-300`, not `p = 0.0`, which
claims an exactness the arithmetic does not have.

Artefact: `models/model_A-1.1.0-real.pkl`, registered in `models/registry.json`
alongside v1.0.0 (history retained).

### 22.7 Files
| File | Change |
|---|---|
| `src/build_training_matrix.py` | `flag_financial_sector()`, `--keep-financials`, `--no-name-heuristic`; exclusion logged |
| `src/model_card.py` | §5 rewritten; exclusion assertion; **all prose numbers de-hardcoded**; per-column formatting; p-value underflow |
| `data/processed/excluded_financials.csv` | **New.** 2,457 rows with reasons |
| `data/processed/training_matrix_with_financials.csv` | **New.** Archived v1.0.0 matrix |
| `data/processed/training_matrix.csv` | Rebuilt, 19,842 rows |
| `models/model_A-1.1.0-real.pkl`, `MODEL_CARD.md`, `reports/phase3/*` | Regenerated |

### 22.8 Phase 3 exit criteria — now met
- Metric targets met (§3.3): AUC 0.9550, PR-AUC 0.8487, KS 0.7959, recall 0.9838.
- Beats both baselines significantly: Altman +0.1813 (p<0.001), persistence
  +0.0691 (p<0.001).
- Artefact registered with a model card.
- **Invariant 5 now enforced in code**, not just in prose.

Remaining open (neither blocks the exit criterion): `git init` for the §14.1
commit hash, and the LightGBM comparator (not installed).

---

## §23 — Repository Initialised; §14.1 Commit-Hash Requirement Closed
*2026-10-27*

### 23.1 Licence control came before `git init`, not after
The project directory holds **1.0 GB in `data/processed` and 14 MB in
`data/raw`**, all Bloomberg terminal output. PRD §18.3 is explicit: the
reproducibility package ships **code and the universe list, never the data**.
Committing first and cleaning up later is not an option — a commit that reaches
a remote cannot be recalled, and history rewriting does not help once it is
cloned. So `.gitignore` was written and **verified against the actual staged
file list** before the first commit.

`.gitignore` excludes `data/`, every `*.csv` / `*.xlsx` / `*.xls` in the tree,
`models/*.pkl` (fitted on that data — derived works), `node_modules/`
(`cresta-app/frontend` alone is 3.5 GB), `.venv/`, and `.env`.

**One subtlety worth recording:** git cannot re-include a file inside a
fully-excluded directory, so the first attempt at
`!data/reproducibility/universe_tickers.csv` silently did nothing — the file
was simply absent from the staged list. The path has to be opened level by
level (`data/*`, `!data/reproducibility/`, `data/reproducibility/*`, then the
negation). Caught by checking the staged list rather than trusting the rule.

### 23.2 The shippable universe artefact is NOT universe_800.csv
§18.3 permits shipping "the universe list". `universe_800.csv` also carries
EPS, Revenue, P/E, total return and market cap **straight from Bloomberg**,
which it does not permit. So `data/reproducibility/universe_tickers.csv` was
generated — identifier plus GICS sector only, 3,000 tickers, 2,307 with a
sector tag — and that single file is un-ignored. The screen output stays out.

### 23.3 `.env` audited, not assumed
Checked before staging: it contains only `PYTHONPATH`. No credential. It is
gitignored regardless, by convention.

### 23.4 Initial commit
`7aae63a` — 139 files, **12.4 MB**, zero Bloomberg data. Verified post-stage
that nothing matching `^data/`, `*.xlsx`, `*.pkl` or `node_modules` was present
except the one permitted ticker list.

The two 3.4 MB PDFs (`final_revised_paper.pdf`,
`final_turnitin_revision.pdf`) and `reports/figures/*` are committed but are
**synthetic-data-era artefacts** — they predate the real panel and their numbers
do not match §18–§22. Flagged so they are not mistaken for current results.

### 23.5 §14.1 commit-hash requirement now satisfied
`model_card.py` picks the hash up automatically; regenerated card §1 reads
`7aae63abc4c0418328360c721b2f1c796ff93dab`, and `models/registry.json` records
it for v1.1.0. The card's limitation 10 is now **conditional on
`entry["git_commit"]`** — it previously hardcoded "No git history", which would
have been false the moment the repo existed, the same stale-prose failure
caught in §22.6. It now reads "Provenance is one commit deep", which is the
honest statement: the hash pins the current state but carries no history of how
the pipeline got here.

### 23.6 ⚠️ GITHUB REMOTE — BLOCKED PENDING USER ACTION
The user has stated that **this project must use a different GitHub account**
than the one currently authenticated. **Nothing has been pushed, no remote has
been added, and no `gh auth` command has been run.** Do not create a repo, add
a remote, or push until the user has logged in under the correct account and
says to proceed. Local git work (commits, branches) is fine meanwhile.

### 23.7 Files
| File | Change |
|---|---|
| `.gitignore` | Rewritten as a licence control with the §18.3 rationale inline |
| `data/reproducibility/universe_tickers.csv` | **New.** The only shippable data artefact |
| `src/model_card.py` | Limitation 10 conditional on whether a commit hash exists |
| `MODEL_CARD.md`, `models/registry.json` | Regenerated with commit `7aae63a` |

---

## §24 — Phase 4 Begun: Backend Repointed from Synthetic to Real Data
*2026-10-27*

The FastAPI backend was wired end to end to `synthetic_panel.csv` and
`models/model_A.pkl` / `model_B.pkl` — artefacts fitted on generated data with
an ESG effect injected by construction (γ = −4.0). Serving those beside real
company names would have presented **fabricated risk estimates about
identifiable listed issuers**. That is the defect Phase 4 opens by closing.

### 24.1 Scope of the coupling
All 17 endpoints referenced synthetic-only columns: `e_score`, `s_score`,
`g_score`, `carbon_intensity`, `is_distress_event`, and a Model B on every
prediction. None of those exist in the real panel.

### 24.2 New foundation — `services/real_data.py`
Single source of truth for the real artefacts. Loads panel, matrix, universe
and the registered model once at startup and **scores all 19,842 rows up
front** (well under a second), so the portfolio view is O(1) per request and
the §12.6 "interactive in under 2s" requirement holds without per-request
inference.

Three decisions worth recording:

1. **The model is loaded from the registry payload, not a bare pickle.** The
   artefact carries its exact feature order, validated threshold and version,
   and `load_all()` **refuses to serve** if the matrix is missing any feature
   the model was fitted on. A reindexed frame scored against the wrong columns
   fails silently otherwise.
2. **Risk bands are derived from the model's own threshold**, not fixed at
   0.30/0.50/0.70. With a cost-optimal cut at 0.200, the old scheme would have
   labelled a company "low risk" while the flag beside it read *flagged* — a UI
   arguing with itself. Bands are now `t/2`, `t`, `t+(1−t)/2`, and a contract
   test asserts no flagged company is ever banded "low".
3. **Company names come from `universe_800.csv`** — the panel has no
   `company_name` column at all. When a name is absent the API returns the
   ticker rather than inventing one.

### 24.3 Track B is reported unavailable, never numerically
`get_model_b()` raises `TrackBUnavailable` instead of falling back to
`models/model_B.pkl`. A silent fallback there would ship **fabricated ESG
findings about named companies under a real-data banner** — the worst failure
this service could have. `/api/predict` returns `model_b: null` plus a
`track_b_status` object stating the code, and a contract test pins it.

### 24.4 SECTOR_UNSUPPORTED implemented (§5.6)
Now that financials are genuinely excluded (§22), "out of scope" and "no data"
are **different answers and the API must not conflate them**:

- financial-sector ticker → **422** `SECTOR_UNSUPPORTED`, with the exclusion
  reason from the log
- unknown ticker → **404** `NO_DATA`

422 rather than 404 because the request is well-formed; the issuer is simply
outside what this model is entitled to speak about. Enforced on
`/predict`, `/predict/explanation` and `/predict/counterfactual` alike.

### 24.5 Endpoints now served (all on real data)
`/model/info` `/model/metrics` `/model/strata` `/model/rolling-origin`
`/model/curves` (ROC + PR + decile calibration) `/model/card` `/model/registry`
`/model/shap/global` `/model/shap/blocks` `/predict` `/predict/portfolio`
`/predict/explanation` `/predict/counterfactual` `/threshold` `/data-quality`
`/features` `/companies*` `/dashboard/summary`.

Two §8.8 levels are computed **live** per company — the local waterfall and the
nearest-healthy-peer counterfactual — while global and block attribution are
read from the Phase 3 artefacts produced by the *same* fitted model, so the API
and the paper cannot disagree.

`/model/metrics` refuses to serve the pooled AUC bare: the persistence caveat
travels in the payload, with the early-warning stratum one call away at
`/model/strata`.

### 24.6 Verified on a real distressed company
`POST /api/predict {"ticker": "SPII IN Equity"}` → SUPREME INFRASTRUCTURE
INDIA, FY2023-24, **p = 0.9981**, band `critical`, horizon FY2024-25, data
quality flag `negative book equity`. SHAP waterfall: `log_total_assets` +1.56,
`working_capital_to_assets` +1.21 (value −1.40), `retained_earnings_to_assets`
+0.95, `roa` +0.77 (value −22.1). Counterfactual peer WESTLIFE FOODWORLD
(p = 0.073), largest gap `ebit_to_assets` (−0.008 vs +0.082).

### 24.7 Contract tests — 23, all passing
`cresta-app/backend/tests/test_api_contract.py`. Not smoke tests: each pins a
property an invariant depends on, so a later refactor cannot quietly undo it.
Track B never returns a number; financials answer SECTOR_UNSUPPORTED;
out-of-scope stays distinguishable from no-data; **SHAP additivity is asserted
per request** (max error 1.7e-05 — the waterfall must add up to the probability
beside it); risk band never contradicts the flag; the §14.5 human-in-the-loop
disclaimer travels with every estimate; `/model/metrics` discloses the
persistence inflation.

One real bug caught by them: NaN is not valid JSON, and
`corr_feature_vs_shap` is legitimately NaN for a constant feature. It now
serialises as `null` — not dropped, and not coerced to 0.0, which would read
as "no relationship" rather than "not defined".

### 24.8 Stale test fixture repaired (pre-existing, not Phase 4)
`tests/test_foundation.py::TestRealData` pointed at
`data/processed/bloomberg_panel.csv` — an **Aug 2026, 499-ticker artefact in
which `total_equity_book` is entirely NaN**, so its negative-equity assertions
had been silently testing an empty frame. Repointed at the current gated panel.

`test_issue_2_reproduces` then genuinely failed, which was worth running down:
**166 of 2,159 negative-equity rows do carry a `debt_to_equity` value**. Every
one is exactly `0.0`, and `total_debt` is 0 or NaN on all 166 — these are
debt-free firms, so the zero is real, not a zero-fill. The assertion is now the
sharper one it should always have been: no negative-equity row may carry a
**non-zero** ratio, which is what a sign-flipped denominator would produce.
`debt_to_equity` is not a model feature, so no result is affected.

Root suite: **96 passed**. Backend suite: **23 passed**. `pytest` and `httpx2`
installed into `.venv` (note: `.venv/bin/pip` has a stale shebang from the old
`~/Documents` path — use `.venv/bin/python -m pip`).

### 24.9 Files
| File | Change |
|---|---|
| `cresta-app/backend/config.py` | Repointed to real artefacts; Model B path deliberately `None` |
| `cresta-app/backend/services/real_data.py` | **New.** Loader, scorer, scope and banding |
| `cresta-app/backend/services/company_service.py` | Rewritten against the real matrix |
| `cresta-app/backend/ml/model_manager.py` | Real registry payload; `TrackBUnavailable` |
| `cresta-app/backend/api/{prediction,explainability,model_eval,threshold,data_quality,features}.py` | Rewritten |
| `cresta-app/backend/tests/test_api_contract.py` | **New.** 23 contract tests |
| `tests/test_foundation.py` | Stale fixture repointed; issue-2 assertion corrected |

### 24.10 Remaining in Phase 4
Caching headers / response-time budget, the frontend still consuming the old
ESG-shaped payloads (Phase 5), and `models/model_A.pkl` + `model_B.pkl` left on
disk as synthetic reference — **not deleted, but no longer wired to anything.**

---

## §25 — Phase 4 (cont.): Response-Time Budget and Caching (§12.6)
*2026-10-27*

§12.6 states the budget as a requirement: *"portfolio view interactive < 2s on
800 rows"*. Profiled first rather than caching blind — the result was that the
budget was **breached**, and the fix was algorithmic, not a cache.

### 25.1 Measured: the portfolio view was over budget
`/api/predict/portfolio?limit=800` took **2,387 ms**. Profiling put the time in
pandas string comparison, 2 calls per company: `data_quality_flags()` filtered
the full **32,973-row panel** with `panel["ticker"] == ticker` for every row
returned. O(n·m) — and because the cost is per row, it scaled linearly with the
page size, which is exactly the direction the §12.6 requirement constrains.

A cache would have hidden this on the second request and left the first one
over budget. The lookup was fixed instead.

### 25.2 Fixes — all algorithmic
1. **Quality-flag index built at startup.** One pass over the panel yields
   `{(ticker, year): [flag, ...]}` for the **14,049 flagged company-years**.
   Clean rows are omitted rather than stored as empty lists. O(n) per request → O(1).
2. **Name and sector dicts** replace `universe.loc[universe.ticker == t]`,
   which was another full scan per company in the same loop.
3. **TreeSHAP built at startup** (`real_data.warm_explainer()`, called from the
   lifespan hook). It cost **2,463 ms on first use** — paid by the analyst on
   their first click into a company, the single most important interaction the
   product has.
4. **Vectorised risk banding** in the dashboard summary: `np.searchsorted` over
   the band edges replaces `iterrows()`.

### 25.3 Result

| Endpoint | before | after | |
|---|---:|---:|---:|
| `/predict/portfolio?limit=800` | 2,387 ms | **56 ms** | 42× |
| `/predict/portfolio?limit=200` | 592 ms | **13 ms** | 45× |
| `/predict/explanation` (first call) | 2,463 ms | **12 ms** | 204× |
| `/dashboard/summary` | 49 ms | **6 ms** | 8× |

Startup now takes 2.63 s — the SHAP construction moved there deliberately. A
one-time cost at boot is the right place for it; a per-click cost is not.

### 25.4 Caching keyed on model version, not time
Everything served derives from artefacts fixed at a model version: the matrix
is scored once at startup and never mutates while the process runs. So the
ETag is **derived from `MODEL_A_VERSION` + path + query**, which means
**deploying a new model version invalidates every cached response
automatically**. A time-based TTL alone cannot do that — it would keep serving
v1.1.0 numbers after v1.2.0 shipped, which for a risk product is the failure
that matters.

- static report endpoints (`/model/*`, `/features`, `/data-quality`,
  `/threshold`): `max-age=3600`
- per-company and portfolio responses: `max-age=300`
- `/health`: `no-store`
- **every error response and every POST: `no-store`** — a
  `SECTOR_UNSUPPORTED` 422 must never be cached as though it were data
- `X-Response-Time-ms` and `X-Model-Version` on every response
- GZip above 1 KiB (the curve arrays, threshold sweep and model card are large
  and highly compressible)

### 25.5 Tests — the optimisation must not have changed an answer
`tests/test_performance_contract.py`, 14 tests. Two kinds:

**Budget guards**, so the regression cannot return quietly — including
`test_portfolio_scales_sublinearly`, which fails if 800 rows ever costs more
than 10× what 50 rows costs, the signature of a per-row scan returning.

**Equivalence checks**, because a speed-up that changes an answer is a bug, not
an optimisation:
- `test_vectorised_banding_matches_the_scalar_classifier` — the `searchsorted`
  path must agree exactly with `classify_risk()` across the whole current year.
  Verified: `{low: 1286, critical: 607, high: 243, moderate: 87}` identical.
- `test_flag_index_matches_a_direct_panel_scan` — the O(1) index must return
  exactly what the O(n) scan returned, checked on a random sample of 40 rows.

Timings are measured in-process, so they exclude network latency and are a
**lower bound** on what a browser sees; thresholds are set with that headroom
rather than at the budget itself, and the test says so.

Backend suite: **37 passed**. Root suite: **96 passed**.

### 25.6 Files
| File | Change |
|---|---|
| `cresta-app/backend/main.py` | Cache/ETag middleware, GZip, startup warm-up, timing header |
| `cresta-app/backend/services/real_data.py` | Flag index, name/sector dicts, `warm_explainer()` |
| `cresta-app/backend/services/company_service.py` | Vectorised banding |
| `cresta-app/backend/api/explainability.py` | Uses the warmed explainer |
| `cresta-app/backend/tests/test_performance_contract.py` | **New.** 14 tests |

### 25.7 Phase 4 remaining
Only one item left: decide whether to retire `models/model_A.pkl` /
`model_B.pkl`. They are synthetic-data artefacts, now wired to nothing, but
deleting them would break any old notebook that loads them — so this is a call
to make, not a cleanup to perform unilaterally.

---

## §26 — Phase 5: Frontend Repointed to Real Data
*2026-10-27*

The Next.js dashboard was written against the synthetic backend and still
asserted its results as fact. Rewritten end to end and verified in a browser
against the live API.

### 26.1 The types did the work
Rewriting `src/types/api.ts` to the real contract surfaced **89 TypeScript
errors across 9 files** — every place the UI depended on a shape that no
longer exists. That list was the actual work plan, and it is why the types
were rewritten first rather than patched alongside the pages.

### 26.2 Bugs found before any page was touched
Two were in `src/lib/api.ts` itself:

1. **`predict()` hardcoded `threshold: 0.5`**, overriding the model's
   validated operating point of **0.200**. The UI was reporting "not flagged"
   for companies the model flags. Removed entirely — omitting the field lets
   the backend apply its own cost-optimal threshold, which is the only
   defensible default.
2. **`ApiError` stringified a structured body.** The backend returns
   `{code, message}`, so `String(body.detail)` rendered **"[object Object]"** —
   including for `SECTOR_UNSUPPORTED`, the one error state §12.6 explicitly
   requires real copy for. `ApiError` now carries `detail` with
   `isSectorUnsupported` / `isNoData` / `isArtefactMissing` helpers.

Dead endpoints removed (`/model/comparison`, `/model/classifiers`,
`/model/pdp`, `/model/shap/individual`) and query parameters corrected
(`n_points` not `steps`, `fiscal_year` not `year`) — the wrong names were
being silently ignored and served defaults.

### 26.3 A defect the browser caught that the tests did not
The model screen rendered **AUC 0.9452 on "all test rows" while the headline
said 0.9550**. Same rows, two numbers.

Cause: **`train_model_a` applies the coverage gate; `real_data` did not.** The
API was serving 4,525 test rows while the model card reports 4,340. Both
behaviours are defensible on their own — a thin-coverage company should still
get a marked estimate rather than silently vanishing (invariant 4) — but
conflating them makes the dashboard contradict the card.

Fixed by separating the two populations explicitly:
`matrix["is_evaluation_row"] = qc_below_coverage != True`. **19,842 rows are
served; 18,216 are evaluated.** `/model/metrics`, `/model/strata`,
`/model/curves` and `/threshold` filter on it; the portfolio and company
endpoints do not, and surface the coverage flag instead.

A second, smaller mismatch followed: the early-warning stratum read 3,333
against the card's 3,334, because the endpoint re-derived `label_now` by
shifting the forward label. It now calls the same `build_fallback_labels()`
the training pipeline uses. **All three strata now match MODEL_CARD.md
exactly** (0.9550 / 0.7189 / 0.9490).

### 26.4 Hardcoded synthetic figures, stated as fact
The chrome and landing page asserted numbers from the generated testbed:

| Where | Was | Now |
|---|---|---|
| Landing headline | "Model B ROC-AUC 0.6117, ΔAUC +0.0774" | fetched early-warning AUC, pooled AUC, persistence baseline |
| Landing footer | "60,000 firms, synthetic testbed" | real scope + §14.5 limitations |
| Status strip | "cresta-b-v1.3 · 0.2278% · synthetic validation" | live version, window, prevalence, Track B status |
| Side rail build | "cresta-b-v1.3" | live `A-1.1.0-real` |
| Base-rate strip | "264 events in 115,873 firm-years" | **985 in 4,340**, fetched |
| Company page | "financial only · 34 features" | live **16** |
| Spine | "6 circular features severed" | live **19** |

The base-rate strip is the one that mattered most: the synthetic class balance
was **two orders of magnitude rarer** than the real panel's, so every claim
about accuracy under imbalance was wrong by 100×.

### 26.5 Risk bands no longer contradict the flag beside them
`bandFromProb()` hardcoded 0.3 / 0.5 / 0.7 while the server bands at
0.1 / 0.2 / 0.6 — the same contradiction fixed server-side in §24.2, living a
second time in the client. `RiskGauge` now takes the server's own
`risk_category` and the live band edges; `bandFromProb` keeps non-model
defaults only as an explicit last resort, documented as such.

Related: the company page's threshold slider opened at 0.5 and then labelled
that value *"cost-optimal at a 10:1 cost ratio"* — a provenance it did not
have. It now opens at the model's operating point, and says **"manual
override"** when the user moves it away.

### 26.6 Screens rewritten
- **Model performance (§12.4)** — rebuilt around the real §8.1 hierarchy
  (persistence / Altman / logistic / XGBoost) instead of A-vs-B ΔAUC. The
  stratified panel sits directly under the metric tiles, not below the fold,
  because the pooled AUC is not the headline. Adds decile calibration and the
  rolling-origin fold distribution. Accuracy is deliberately absent from the
  comparison table.
- **Portfolio (§12.2)** — ESG histogram replaced with a risk-band breakdown;
  ESG column dropped; `is_distressed` → `flagged` at the operating threshold.
- **Company (§12.3)** — Model B gauge replaced with the Track B unavailability
  notice; ESG scores replaced with data-quality flags; the financial snapshot
  renders a missing ratio as **"n/d"**, not 0 or blank — `debt_to_equity`
  under negative equity is the canonical case, and invariant 1 is now visible
  in the UI.
- **Data & Features (§12.5)** — pipeline flow, gates as hard-vs-warn, and the
  circularity list read from `fallback_label`'s own guard so the screen cannot
  drift from what the pipeline actually blocks.
- **Risk trajectory** — was ROA plus an averaged E/S/G score; now the
  predicted probability series against the operating threshold, which is what
  §12.3's mock actually specifies.

### 26.7 Verified live, not just compiled
`next build` clean, **0 TypeScript errors**, both servers run, and every screen
checked in a browser against the live API. Supreme Infrastructure India
(SPII) renders: p = **0.9981**, band critical, horizon FY2023-24, flag
*negative book equity*, base rate 0.5156 with the rebalanced-prior note, the
§8.8 caveat and the §14.5 disclaimer.

One environment note worth recording: `NEXT_PUBLIC_API_URL` is baked at
**build** time, so setting it on `next start` does nothing — the first browser
check showed "Can't reach the CRESTA API" for exactly that reason. The error
states degraded correctly, which is how it was obvious.

Root suite **96 passed**, backend suite **37 passed**.

### 26.8 Files
| File | Change |
|---|---|
| `src/types/api.ts`, `src/lib/api.ts` | Rewritten to the real contract |
| `src/app/{page,layout}.tsx` | Live figures; honest metadata and footer |
| `src/app/{model,portfolio,data,company/[ticker]}/page.tsx` | Rewritten |
| `src/components/shell/{StatusStrip,SideRail}.tsx` | Live version/window/prevalence |
| `src/components/ui/{BaseRateStrip,RiskGauge,Spine}.tsx` | Live values; server-driven bands |
| `src/components/charts/*` | Retyped; PDP and ESG histogram removed; trajectory rewritten |
| `src/lib/format.ts` | `bandFromProb` takes live band edges |
| `backend/services/real_data.py` | `is_evaluation_row`; `label_now` from the training source |
| `backend/api/{model_eval,threshold,features,data_quality}.py` | Evaluation-population filter; blocks; circularity list; gate verdicts |
| `backend/services/company_service.py` | Sector distribution; financial snapshot |

### 26.9 Remaining in Phase 5
Accessibility pass (§12.6 WCAG 2.1 AA — risk must never be encoded by colour
alone; the band label is present everywhere, but this has not been audited),
CSV/PDF export with model version and timestamp stamped, and skeleton loaders
on the remaining data-dense panels.

---

## §27 — Phase 5 Complete: Accessibility, Export, Loading States (§12.6)
*2026-10-27*

The three remaining §12.6 non-functional requirements. Each was **measured
before being changed** — the accessibility item in particular, because "looks
fine" is not an audit.

### 27.1 Accessibility — WCAG 2.1 AA
Computed every foreground/background contrast pair in the palette against the
three surface colours. Two tokens failed, and one of them carried most of the
interface's text:

| Token | Before | After | Floor |
|---|---:|---:|---|
| `--ink-500` | **2.72:1** ✗ | **4.53:1** ✓ | 4.5 (normal text) |
| `--ink-600` | **1.79:1** ✗ | **3.03:1** ✓ | 3.0 (large / non-text) |

`ink-500` had **65 text call sites plus every chart axis label**. Lightening
the token fixed all of them at once; it is used as a background in exactly one
place, so there was no design trade-off to make. All four risk colours already
passed comfortably (5.20–10.26:1), as did `amber` and both model channels.

**Risk is never encoded by colour alone** — verified, not assumed:
`RiskBadge` renders a text label in every branch of `RiskGauge`, the portfolio
table and the band breakdown. §12.6's specific requirement is met.

Two genuine **Level A** failures in `DataTable`:
- **Clickable rows had no keyboard path.** `<tr onClick>` with no `tabIndex`,
  `role` or `onKeyDown` — the entire company explorer was unreachable without a
  pointer (WCAG 2.1.1). Now focusable, Enter/Space activated, with a visible
  focus ring.
- **Sort direction was a chevron glyph only**, which a screen reader does not
  read. `aria-sort` now states it on the header cell, and the sort buttons
  carry labels.

`RiskGauge` gained `role="img"` with a full text summary, and its decorative
zone track is `aria-hidden` so assistive tech hears "0.9981, Critical risk,
threshold 0.20" rather than a list of coloured segments. A global
`:focus-visible` outline and a `prefers-reduced-motion` block were already
present.

### 27.2 Export, stamped (§12.6)
New `api/export.py`: `/export/portfolio.csv`, `/export/company.csv`,
`/export/manifest`.

**Why the stamp is the point.** An exported CSV outlives the session that made
it; six months later the file itself is the only record of which model,
threshold and cost assumption produced a flag. Every export opens with a
commented preamble carrying model version, **config hash, git commit**,
training and generation timestamps, operating threshold, cost ratio, the §5.6
scope exclusion, the label-is-a-proxy note, and the §14.5 human-review
constraint — because a spreadsheet circulating without that last line is
precisely how an analytical estimate becomes a decision.

Comment lines use `#`, which Excel and `pd.read_csv(..., comment="#")` both
skip. Verified: the preamble does not break the file it annotates.

Two details worth recording:
- **The threshold is repeated on every row**, not only in the header, so a
  filtered or pasted subset still says what `flagged` meant.
- **The company export carries the model's own 16 inputs**, so the estimate can
  be audited rather than taken on trust.
- Exports are `no-store`: caching one would hand back a stale timestamp.

The export's config hash was cross-checked against `registry.json` and
`MODEL_CARD.md` — all three read `f6df3ed0…`, so a downloaded file is
traceable to an exact build.

**PDF is a documented deviation, not an omission.** §12.6 asks for CSV and
PDF. The PDF half is better served by the browser's print pipeline against a
print stylesheet than by a second server-side rendering engine that would
drift from what the screen shows; `/export/manifest` hands the front end the
same stamp for the print header. Stated in the module docstring rather than
quietly dropped.

### 27.3 Skeleton loaders
§12.6: *"Skeleton loaders; never a bare spinner on a data-dense view."*
There were **13 bare spinners**; there are now **zero**. Added `SkeletonRows`,
`SkeletonTable` and `SkeletonChart`, shaped like the content so the layout
does not jump when data lands — on a dense analytical screen that is the
difference between a usable wait and a page reflowing under the cursor. Each
is `role="status"`, `aria-busy`, and labelled, so a screen reader hears
"loading SHAP attribution" rather than silence then a sudden tableful.

### 27.4 Tests
`tests/test_export_contract.py`, 12 tests, pinning: the version and timestamp
stamp, the config hash and commit, the cost assumption, the §14.5 constraint,
the scope notes, that the preamble parses with the documented recipe, that
every row carries its threshold and that `flagged` is consistent with it, that
a financial-sector issuer cannot be exported, and that the company export
includes all 16 model inputs.

One test bug of my own, caught and fixed: `Series == pytest.approx(x)` does
not compare elementwise. Replaced with `np.allclose`.

Backend suite **49 passed**, root suite **96 passed**, `next build` clean,
**0 TypeScript errors**. (`next lint` is not configured and prompts
interactively — left alone rather than setting up ESLint unasked.)

### 27.5 Files
| File | Change |
|---|---|
| `frontend/src/app/globals.css` | `--ink-500`, `--ink-600` lightened to clear AA, with the measured ratios in comments |
| `frontend/src/components/ui/DataTable.tsx` | Keyboard-operable rows; `aria-sort`; labelled sort buttons |
| `frontend/src/components/ui/RiskGauge.tsx` | `role="img"` summary; decorative track hidden |
| `frontend/src/components/ui/primitives.tsx` | `SkeletonRows` / `SkeletonTable` / `SkeletonChart` |
| `frontend/src/components/ui/ExportButton.tsx` | **New.** |
| `backend/api/export.py` | **New.** Stamped CSV export + manifest |
| `backend/tests/test_export_contract.py` | **New.** 12 tests |

### 27.6 Phase 5 is complete
All four screens on real data, every §12.6 non-functional requirement met or
documented as a deviation. **Next is Phase 7 (hardening and docs), not Phase 6
(ESG)** — the ESG arithmetic is unchanged at ~5.3 events per variable, and the
largest outstanding task is that `final_turnitin_revision.tex` and
`final_revised_paper.tex` are written against the SYNTHETIC results. Every
number in them is now wrong. That rewrite is not currently itemised in
PHASE_PLAN.

---

## §28 — Phase 6 Begun: BRSR Pilot, First Three Filings (2026-10-28)

Phase 6 was opened after the user asked whether it was being skipped. It was, and the
reason recorded in `PHASE_PLAN.md` ("gated behind Phase 4") had stopped being the real
reason some time ago. The honest reason is below, measured rather than estimated.

### 28.1 The ESG window is smaller than the earlier estimate — corrected

A prior session put Track B at ~5.3 events per variable. That used *our matrix's* top
1,000 by market cap. SEBI ranks the **whole market**, financials included, so the real
BRSR-eligible set reaches less far down our non-financial universe:

| Frame | Firms | Events |
|---|---:|---:|
| FY2023 evaluable rows (all) | 2,151 | 479 |
| Our matrix's own top 1,000 by market cap | 1,000 | 126 |
| **Market-wide top 1,000, financials excluded, in the FY2023 eval set** | **860** | **93** |

At 16 financial + ~6 ESG features, that is **93 / 22 ≈ 4.2 EPV**, not 5.3, against the
≥10 rule of thumb. The earlier number was optimistic by a quarter.

**The binding constraint is not power, it is the split.** BRSR is mandatory only from
FY2022-23, and the panel's last feature year *is* FY2022-23. ESG therefore exists for
exactly one feature year. Invariant 2 forbids a random split, so there is no out-of-time
comparison of Model A against Model B available here — not a weak one, none. Any Model B
number from this data would be in-sample or randomly split.

**Phase 6 is therefore scoped to a feasibility + power result, not a hypothesis test.**
Per §1.4 a negative result is valid; "underpowered to test, and here is the measurement"
is a stronger claim than an AUC computed on 93 events, and it speaks directly to the §1.3
research gap, which is about data availability under the BRSR mandate.

### 28.2 Pilot sample — matched pairs, after a first draw was rejected

24 companies from the BRSR-eligible frame. The first draw stratified on sector and label
independently and produced a healthy arm skewed large (ONGC at cap rank 9) against a
mid-cap distressed arm. Large firms file better BRSRs, so that design would have measured
a **size** effect on extraction coverage and reported it as a distress effect.

Redrawn as matched pairs: each distressed firm paired to the nearest-cap-rank healthy firm
in the same sector. Result: **12/12 sector-matched**, median cap rank 502 (distressed) vs
498 (control). Written to `data/raw/brsr_pilot_sample.csv`; `data/raw/brsr_pilot/MANIFEST.csv`
carries the 24 expected filenames.

### 28.3 ⚠️ Two of the first three downloads were the wrong fiscal year

The user downloaded three filings under correct `FY22-23` filenames. Reading the year each
filing **declares about itself** (BRSR Section A item 9):

| File | Declared | Verdict |
|---|---|---|
| `SHRS_BRSR_FY22-23.pdf` | 2022-23 | PASS |
| `SHP_BRSR_FY22-23.pdf` | 2023-24 | REJECT |
| `TRE_BRSR_FY22-23.pdf` | 2025-26 | REJECT |

Both rejects are **look-ahead**: the panel's features sit at FY2022-23 and predict a label
in FY2023-24, so an FY2023-24 BRSR postdates the outcome it would help predict (§7.4).
Triveni's filing postdates it by three years. Neither would surface as anything but a
*better* Track B result — which is the precise claim this pilot exists to test.

A filename is an assertion by whoever typed it; item 9 is an assertion by the filer. Only
the second is evidence. New module **`src/brsr_pilot_validate.py`** gates every filing on
its declared year before any field is extracted, and logs rejects to
`data/raw/brsr_pilot/rejected_filings.csv` rather than skipping them (invariant 4) — a
filing that cannot be year-verified is itself a fact about BRSR availability.

`Data_Extraction_Guide_v2.md` Part 4 was corrected: its worked example used
`FY23-24`, i.e. the guide was teaching the trap.

Two detection bugs of mine surfaced from real layout, both found by debugging against the
text rather than reasoning about the regex:
1. Item 9's label wraps with a table header injected mid-phrase — `"Financial year for
   which"` / `"9  Start date  End date"` / `"reporting is being done"`. Fixed by matching
   the halves independently.
2. The date columns are padded wider than any `\D{1,40}` budget. Fixed by isolating the
   row and reading dates out of it; **any space budget is wrong for the next filing's
   column widths**, so the budget was removed rather than enlarged.

### 28.4 Extraction result on the one valid filing — and why 35.7% is not the answer

`brsr_extractor.py` ran for the first time ever on a real filing (it had only seen the
synthetic PDF; `pdfplumber` was not even installed). It reported **5/14 fields, 35.7%**,
against the agreed ≥70% gate.

That number is wrong in both directions, and the pilot existed to find out which.

**Checked field by field against the source, every one of the nine nulls is present in the
filing:**

| Field | In the filing | Why it was missed |
|---|---|---|
| `scope1_emissions_tco2e` | 577,040.30 | unit column interleaved into the label text |
| `scope2_emissions_tco2e` | 168,334.28 | same |
| `board_women_pct` | 8.33 (1 of 12) | Section A item 19 matrix layout |
| `board_independent_directors_pct` | 50% (6 of 12) | stated as **prose**, not a table |
| `employee_turnover_rate_pct` | 20.09 | item 20, three-FY × male/female/total grid |

So **data availability is not the bottleneck on this filing; the regexes are.** 35.7%
understates what is reachable. Regex tuning is deliberately deferred until 3 valid filings
are in hand — tuning against one layout is how an extractor gets overfitted to one filer.

### 28.5 ⚠️ A filing that disagrees with itself — the ESG balance-sheet identity

The three energy fields that *were* extracted are each correct, read from the correct
`FY 22-23` column, and **mutually impossible**:

| Source table (same filing) | FY22-23 |
|---|---:|
| Essential Indicator 1 — total energy (A+B+C) | 4,325,818 GJ |
| Leadership Indicator 1 — renewable (A+B+C) | 13,504,795 GJ |
| Leadership Indicator 1 — non-renewable (D+E+F) | 1,873,224 GJ |
| **renewable + non-renewable** | **15,378,019 GJ — 3.6×the stated total** |

Not a parser fault. The Essential table books 19,124 GJ of electricity, matching the
Leadership table's *non*-renewable electricity exactly, so ~11M GJ of self-generated
renewable power (bagasse — Shree Renuka is a sugar producer) is excluded from the
"total". A naive renewable share off these two tables is **312%**.

This is the Track B analogue of the balance-sheet identity that has repeatedly caught
parser faults in the financial panel — except here it catches a **filer** fault, which is
worse, because no amount of parser correctness removes it. Implemented as
`check_energy_identity()` with a 2% tolerance mirroring the TA == TE + TL check.

On failure the renewable-share ratio is **refused, not clipped**. Clipping 312% to 100%
would manufacture a plausible number from a filing that does not support one. NaN is
signal here exactly as it is for `debt_to_equity` under negative equity (invariant 1).

**This belongs in the paper.** §11.1 already argues that data-quality findings are
contributions; a SEBI-mandated disclosure whose own two tables disagree by 3.6× is a
direct, quantified caution about ESG-ratio features built from regulatory filings, and
the distress literature does not document it.

### 28.6 Files

| File | Status |
|---|---|
| `src/brsr_pilot_validate.py` | **New.** Declared-FY gate + `check_energy_identity()` |
| `data/raw/brsr_pilot_sample.csv` | **New.** 24 matched-pair companies |
| `data/raw/brsr_pilot/MANIFEST.csv` | **New.** Expected filenames, sector, cap rank, label |
| `data/raw/brsr_pilot/rejected_filings.csv` | **New.** 2 rejects with reasons |
| `Data_Extraction_Guide_v2.md` | Part 4 rewritten — FY trap corrected, pilot paths |
| `.venv` | `pdfplumber 0.9.0` installed (was absent) |

### 28.7 Open

- ⏳ `SHP` and `TRE` to be re-downloaded for FY2022-23 (blocked on the user)
- ⬜ Regex tuning, deferred until 3 valid filings are in hand
- ⬜ `SHP` shows 8/9 distinct Principles — check whether its filing is partial
- ⬜ Remaining 21 of 24 filings

---

## §29 — Phase 6 (cont.): All Three Filings Valid; Extraction Rebuilt Row-Based (2026-10-28)

### 29.1 The two wrong-year filings were replaced via the browser

Using Claude in Chrome at the user's request. Both were found on the companies' own IR
sites and fetched directly:

- **Shalimar Paints** — `/uploads/SPL_-_Business_Responsibility_and_Sustainability_Report_FY_2022-23.pdf`.
  The site has a dedicated BRSR document per year behind a year dropdown.
- **Triveni Engineering** — `/storage/files/brsr/BRSR-TEIL-2022-2023.pdf`.
  **Root cause of the FY2025-26 download found:** Triveni's year selector *defaults to
  2025-2026*. Nothing on the page indicates the default is the newest year rather than
  the one being browsed. Note also that the FY25-26 file is named `TEIL-BRSR-25-26.pdf`
  and the FY22-23 file `BRSR-TEIL-2022-2023.pdf` — **the naming convention is not stable
  across years**, so URL-guessing from a known file would have failed. The year must be
  selected in the page and the resulting link read.

All three filings now PASS the declared-FY gate, 9/9 Principles each.

NSE's BRSR archive (1,264 records, with a CSV export) would have been the efficient
bulk route, but its company combobox needs an autocomplete selection that does not
render under automation and its date fields reject typed input. Abandoned after three
attempts rather than ground on; per-company IR sites work.

### 29.2 The regex extractor was structurally wrong, not merely untuned

`brsr_extractor.py` scored 28.6 / 35.7 / 35.7% on the three filings. Diagnosed before
touching the patterns; it is **not** a matter of widening the gap budgets:

1. BRSR tables are **column-laid-out**. The distance between a label and its value is
   the filer's column padding -- 60+ spaces is normal and differs per filer. Any fixed
   `\D{0,30}` budget is right for one layout and wrong for the next.
2. A **units column sits between label and value**:
   `"Total Scope 1 emissions (Break-up of the   Metric tonnes of CO2   5,77,040.30"`.
   The text in between is not padding, it is more label. Widening the budget lets the
   pattern skip into the *wrong column* rather than failing honestly -- strictly worse
   than a null.

New module **`src/brsr_rows.py`** works on `pdftotext -layout` **lines**: locate the row
matching the label, then read numeric tokens out of that row positionally. Column 0 is
the current fiscal year in all three filings (verified against their explicit
`FY 22-23 | FY 21-22` headers). A value must be a standalone number, so `CO2`, `N2O`,
`SF6` and `Scope 1` cannot register. Recurring labels are disambiguated by section
anchor -- `"Board of Directors"` appears throughout; only the instance under item 19 is
the gender matrix.

| Filing | regex extractor | row-based |
|---|---:|---:|
| SHP | 28.6% | **81.8%** |
| SHRS | 35.7% | **100.0%** |
| TRE | 35.7% | **81.8%** |

Every SHRS value was checked by hand against the source and matches exactly.

### 29.3 The remaining nulls are non-disclosure, not extraction failure

Checked rather than assumed:

- **SHP** — the string "renewable" occurs **zero times** in the filing. The renewable /
  non-renewable split is a *Leadership* indicator (voluntary); Shalimar did not file it.
- **TRE** — the Scope 1 and Scope 2 rows exist but hold **prose**: "Emissions under
  scope 1 & 2 are technically estimated". There is no number to extract.

**On disclosed fields, extraction accuracy is 100% across all three filings.** The
>=70% coverage gate is met on all three. Extraction is therefore *not* the Track B
bottleneck -- which inverts the working assumption carried since §9.

### 29.4 The energy identity now has a control

TRE: renewable 19,295,577.93 + non-renewable 28,392.56 = 19,323,970.49 = stated total
**exactly** (rel_err 0.0000). SHP: incomplete, correctly refused. SHRS: **FAIL at 255%**.

One filing reconciling perfectly is what makes the SHRS failure attributable to the
*filer* rather than to the parser. Without that control the 3.6x discrepancy would have
remained a suspected bug in our own code.

**Tally so far: 1 of 3 filings cannot support a renewable-share ratio because its own
two tables disagree; 1 of 3 cannot because it did not disclose the split.** Only 1 of 3
yields a usable energy-intensity feature. For §6.8 intensity features that is the real
constraint, and it compounds with the 93-event ceiling from §28.1.

### 29.5 Files

| File | Status |
|---|---|
| `src/brsr_rows.py` | **New.** Row-based positional extractor; 11 fields |
| `src/brsr_extractor.py` | Superseded for these fields; kept for its table layer |
| `data/raw/brsr_pilot/SHP_BRSR_FY22-23.pdf` | Replaced — correct year |
| `data/raw/brsr_pilot/TRE_BRSR_FY22-23.pdf` | Replaced — correct year |
| `data/raw/brsr_pilot/rejected_filings.csv` | Removed — no rejects remain |

### 29.6 Open

- ⬜ 21 remaining filings — per-company IR navigation; NSE bulk route did not work
- ⬜ `board_independent_directors_pct` still unextracted — stated as **prose**
      ("Out of 12 Directors, 6 Are independent directors"), not a table row
- ⬜ Decide: 21 more downloads, or report the §28.1 power result on a 3-filing
      feasibility base

---

## §30 — Phase 6 (cont.): Pilot Extended to 7 Filings (2026-10-28)

Middle path taken at the user's direction: four matched pairs rather than all 24.

### 30.1 ⚠️ The sampling frame has a look-ahead defect

`brsr_pilot_sample.csv` ranked firms by **current** market cap. BRSR eligibility in
FY2022-23 depended on market cap **at that time**. Several high-ranking names in the
sample listed *after* FY2022-23 and cannot have a FY2022-23 filing at all:
Signatureglobal (IPO Sept 2023), Sri Lotus (2025), Vinsys (2024), Rishabh (2023).

This is the same class of error as the `Current`-basis market-cap floor caught in
Phase I -- a present-day attribute used to define a historical population. Pairs were
re-picked among long-listed firms. **The full 24-company frame must be rebuilt on
FY2022-23 market cap before any coverage rate is quoted as representative.**

### 30.2 Filings obtained — and two sourcing lessons

| Ticker | Route |
|---|---|
| PTCIN | IR site, BRSR inside the annual report |
| RTPOW | **BSE** `xml-data/corpfiling/AttachHis/` — BRSRs are filed as standalone attachments |
| NDTV | IR site, inside the annual report |
| TPI | BSE attachment |

- **BSE's corpfiling attachments are directly fetchable** with a `bseindia.com` referer.
  This is the most reliable route found; IR sites vary wildly.
- **Sandesh could not be retrieved.** Its CDN returns 403 to every non-browser request,
  and a cross-origin `download` attribute is ignored by Chrome. Pair dropped; NDTV
  retained unpaired. Logged rather than quietly skipped (invariant 4).
- NSE's bulk BRSR archive (1,264 records, CSV export) remains the right route in
  principle but its combobox and date fields do not work under automation.

### 30.3 ⚠️ Nitco filed no BRSR at all

Nitco's FY2022-23 annual report (941,385 characters, the complete document) contains
**zero** occurrences of "business responsibility". Not a parse failure -- Nitco, at
current cap rank 494, was not BRSR-obligated in FY2022-23. Direct confirmation of the
§30.1 frame defect.

### 30.4 A third encoding of the same mandated field

Transpek answers item 9 as **prose** -- "1st April, 2022 to 31st March, 2023" -- where
Shree Renuka writes "2022-23" and Shalimar gives a numeric table. **Three filings,
three encodings of one SEBI-prescribed field.** `_PROSE` added as a third detector.
This is the clearest possible argument for gating on the filing rather than the filename.

### 30.5 Results across 7 filings

| Ticker | Sector | Label | Coverage | Energy identity |
|---|---|---|---:|---|
| PTCIN | Utilities | healthy | 90.9% | incomplete |
| RTPOW | Utilities | distressed | 45.5% | incomplete |
| SHRS | Cons Staples | distressed | 100.0% | **FAIL (255%)** |
| TRE | Cons Staples | healthy | 81.8% | pass (0.0000) |
| SHP | Materials | distressed | 81.8% | incomplete |
| TPI | Materials | healthy | 81.8% | incomplete |
| NDTV | Comm Services | distressed | 63.6% | incomplete |

**Mean coverage 77.9% — the >=70% gate is met.** Distressed 72.7% vs healthy 84.8%
(n=4 vs 3): directionally consistent with worse disclosure under distress, far too few
observations to claim it.

### 30.6 The finding that actually decides Track B

**Only 1 of 7 filings yields a usable energy-intensity ratio.** Five did not disclose
the renewable/non-renewable split at all (it is a *Leadership*, i.e. voluntary,
indicator); one disclosed it and contradicts its own total by 255%.

PRD §6.8's intensity features are ratios. On this evidence they are unbuildable for
~86% of filers, and the ESG feature block would be dominated by missingness -- on top
of the 93-event / 4.2 EPV ceiling (§28.1) and the absence of any out-of-time split.

Coverage was never the binding constraint. **Disclosure of the specific indicators
§6.8 needs is.** That is a cleaner, more defensible Track B result than any AUC this
data could produce, and it is directly responsive to the §1.3 research gap.

### 30.7 Open

- ⬜ Rebuild the 24-company frame on FY2022-23 market cap (§30.1)
- ⬜ Sandesh — CDN-blocked; needs a manual download if the pair is wanted
- ⬜ RTPOW at 45.5% — lowest in the set, not yet diagnosed

---

## §31 — The Point-in-Time Leakage Defect, Found and Fixed (2026-10-28)

### 31.1 How it surfaced

While rebuilding the BRSR eligibility frame on FY2022-23 market cap (the §30.1 open
item), I checked whether `cur_mkt_cap` in the panel was historical. It is not.

```
tickers reporting >=5 years, share carrying ONE value across all years
    cur_mkt_cap       98.4%
    volatility_260d   95.4%
    debt_to_equity     4.9%
    total_debt         4.1%
    every accounting field  0.0%
```

The pull applied `FUND_PER="Y"` + `EQY_FUND_YEAR` uniformly to all 18 fields. **That
override governs fundamental fields only.** `CUR_MKT_CAP` and `VOLATILITY_260D` are
real-time *market* fields: Bloomberg ignores the override and returns the pull-date
value for every year requested.

So `log_mkt_cap` on a FY2014 row was the firm's market cap as at the 2026 pull date —
a value postdating the FY2015 label it was predicting by eleven years. Same for
`equity_volatility`. Together: **9.6% of SHAP attribution was look-ahead.**

### 31.2 The leakage signature, measured not argued

AUC of `-log_mkt_cap` alone against the label, by feature year:

```
FY2014 0.5976   FY2017 0.6364   FY2020 0.6538   FY2022 0.6730
FY2015 0.6308   FY2018 0.6495   FY2021 0.6611   FY2023 0.6718
FY2016 0.6332   FY2019 0.6497
```

Monotone rising as the gap to the pull date closes. A genuinely contemporaneous size
control does not trend that way. `log_total_assets` rises too (0.530 → 0.599) but it
varies per year on 100% of tickers, so that trend is panel composition, not leakage.

**Why nothing caught it.** `_assert_out_of_time()` checks the *years* do not overlap —
they did not. The leakage was inside a column, not across the split. An out-of-time
split guarantees nothing if a feature's value is dated outside its row's fiscal year,
and that is a property of the column.

### 31.3 The fix and what it cost

`build_training_matrix.POINT_IN_TIME_FEATURES` drops both by default;
`--keep-point-in-time` restores them for the sensitivity run only. Retrained as
**v1.2.0**, 14 features.

| | feat | AUC | early-warning | PR-AUC | logistic |
|---|---:|---:|---:|---:|---:|
| v1.1.0 (leaked) | 16 | 0.9550 | 0.9490 | 0.8487 | 0.8820 |
| **v1.2.0** | **14** | **0.9507** | **0.9434** | **0.8315** | **0.8742** |

**−0.0043 pooled, −0.0056 early-warning.** Small, and that is the honest reading: the
model recovers nearly all of it from legitimate fundamentals. Every significance
conclusion holds — Altman still significantly worse than persistence (−0.1110,
p<1e-300), logistic still indistinguishable (−0.0117, p=0.165), XGBoost still better
(+0.0648, p<1e-300). Block attribution shifts to profitability 64.7% (was 57.8%).

### 31.4 Guards added

- `model_card.py` raises if any point-in-time feature reaches `feats` — same shape as
  the §5.6 financials assertion, added for the same reason: the defect got here
  because nothing checked.
- `tests/test_no_lookahead.py` (16 tests). It asserts the *panel fields are still
  constant*, so that a future BDH re-pull makes the test **fail** and tells whoever
  ran it that the exclusion can be lifted.
- Backend contract test `test_no_point_in_time_feature_is_served`. The old
  `assert n_features == 16` was replaced with a check against the artefact — a magic
  number in a test is a second place for the truth to live.

### 31.5 Remediation documented

`Data_Extraction_Guide_v2.md` Part 2 rewritten (it still described the superseded BDH
procedure and the two invalid mnemonics). New §2.3 specifies the two-field BDH re-pull
that would restore both features honestly: ~66,000 requests, one day's budget.

---

## §32 — Phase 6 Closed (2026-10-28)

### 32.1 The frame defect was real and immaterial

Rebuilt the BRSR-eligible frame excluding firms not yet reporting by FY2022-23:

| frame | firms | events | EPV |
|---|---:|---:|---:|
| all FY2022-23 evaluable | 2,151 | 471 | 21.4 |
| top 1000 by pull-date cap (defective) | 857 | 95 | 4.32 |
| **top 1000, reporting by FY2022-23** | **862** | **95** | **4.32** |

16 firms excluded as too new, **6 of 1,000 swap, EPV unchanged at 4.32.** The defect
was real — and it is the *same* defect as §31, since `cur_mkt_cap` is what both the
ranking and the removed feature were built from. It was not what made Track B
infeasible, and no conclusion rested on it.

The correction remains an approximation and is labelled as one: there is no historical
market-cap series to rank on (that is the same missing BDH pull), so first-reporting-
year is a loose bound. It excludes firms that did not exist; it does not exclude a firm
that reported privately and listed later.

### 32.2 Decision: report the power result, do not build an ESG feature block

`reports/phase6/track_b_feasibility.md`, generated by `src/brsr_pilot_report.py`.
Three independent constraints, the third decisive:

1. **No out-of-time comparison exists.** BRSR mandatory from FY2022-23 = the panel's
   last feature year. One feature year cannot be split out of time against itself.
2. **4.32 EPV** against ≥10 — roughly 2.3× short.
3. **Only 1 of 7 filings yields a usable energy-intensity ratio.** Coverage 77.9%
   (gate met); on disclosed fields extraction is 100% accurate. Disclosure is the
   binding constraint, not extraction.

Invariant 6 is satisfied by construction — Track A never depended on ESG.

### 32.3 Also closed

- `data/raw/brsr_pilot/pilot_results.csv` — per-filing durable record, including the
  two filings **sought and not obtained** (SDSH CDN-blocked, NITCO filed no BRSR),
  logged rather than dropped (invariant 4).
- `models/model_A.pkl` / `model_B.pkl` moved to `models/deprecated_synthetic/` — the
  last Phase 4 open item. Moved, not deleted: the synthetic paper drafts need them.
  Moved rather than left in place so a stale `joblib.load("models/model_A.pkl")`
  raises instead of silently scoring a named company on invented data.

---

## §33 — Phase 7: Hardening & Docs (2026-10-28)

### 33.1 Monitoring — `src/monitoring.py`, §14.3

Four triggers with thresholds, not four charts. Output is a decision
(`fire`/`watch`/`ok`/`insufficient`), and `--fail-on-trigger` exits non-zero for cron.

Two implementation choices that matter:

- **Missingness is bucketed explicitly.** The standard non-null-only PSI scores a
  feature that went 5% → 60% null at **0.00**. In a distress panel that is the signal:
  firms under stress stop filing.
- **An emptied bucket is floored and FLAGGED**, not epsilon-padded. The usual epsilon
  turns "this value never occurs now" — the most informative drift there is — into a
  moderate number. The result says `lower bound` so the caller knows.
- Low-cardinality features take a categorical path; quantile edges collapse on a
  boolean flag and the numeric path would skip them as degenerate.
- Decay is measured on the **early-warning stratum**. Pooled AUC is 89.8% persistence
  and would mask a total collapse in the model's actual job.

Current run: no trigger fired. Top PSI is `consecutive_loss_years` 0.4295 (floored
buckets) and `altman_z_ex_x4_3yr_trend` 0.4123 — the latter driven by a **−23.9pp fall
in missingness**, i.e. it is more widely computable now, not differently distributed.
Trend features do this by construction at the panel's left edge. The generated report
says so rather than leaving a reader to work it out.

12 tests in `tests/test_monitoring.py`, each targeting a way a PSI implementation
produces confident nonsense.

### 33.2 Outcomes analysis — `src/outcomes_analysis.py`, §14.4

§14.4 requires every false negative reviewed individually, asking: *signal present but
unused, or genuinely absent?* **24 false negatives of 985 events** at threshold 0.245
(97.6% recall):

| verdict | n |
|---|---:|
| SIGNAL ABSENT | 12 |
| SIGNAL UNUSED | 11 |
| DATA MISSING | 1 |

"Adverse" is calibrated per feature on the **train window** — each feature's 20%/80%
tail, kept only if it carries ≥1.25× base prevalence. Letting the test window define
what counts as a warning sign would be circular. `DATA MISSING` is kept separate
deliberately: folding it into `SIGNAL ABSENT` would credit the data with an absence it
never had the chance to demonstrate.

### 33.3 The calibration finding

Decile calibration surfaced something the AUC hides. **The model is systematically
over-confident**: mean gap −0.175 across deciles 7–10; decile 8 predicts 69.8% against
a realised 45.2%.

Cause is `scale_pos_weight`, which inflates probabilities by construction. **Ranking is
unaffected** — AUC, KS and the threshold are rank statistics, and the threshold was
chosen on validation under the same distortion, so the flag decision is sound. But a
user reading "70%" is misinformed. Either present the score as a rank (§12.3 bands
already do) or fit a calibration layer on validation only before quoting a probability.
Added as limitation 7 in the model card.

### 33.4 Validation report — `reports/VALIDATION_REPORT.md`

§14.1–14.5 on the three-component standard. It **computes nothing**: every figure is
read from the artefact that produced it, so it cannot drift from `MODEL_CARD.md`. It
**fails** rather than degrades when an input is missing — a validation report with a
gap in it reads as a complete one.

### 33.5 The paper — `paper/cresta_real.tex`, generated

Rewritten against real results, and **generated by `src/generate_paper.py`** rather
than written. The three superseded drafts all carry hand-transcribed numbers and the
pipeline has been rerun three times since, twice for a defect. A paper that cannot be
regenerated goes stale silently.

New framing: *Persistence, Leakage, and Disclosure — What Survives an Out-of-Time
Evaluation of Corporate Distress Prediction in India.* Three contributions, each of
which only ever lowers a reported number:

1. The persistence baseline, and Altman Z″ being significantly **worse** than it.
2. **Point-in-time field substitution** named as a leakage class, with a diagnostic
   that detects it from the delivered panel alone (§31).
3. The ESG negative result as a finding about the **disclosure regime**, with the
   three constraints stated separately so future work can see which expire.

Plus the calibration result, which "is the kind of result that disappears from papers".

The three synthetic drafts carry a SUPERSEDED banner; `paper/README.md` explains why
they are kept. **Not compiled** — no LaTeX toolchain in this environment. Syntax-
checked only (balanced environments, escaped data-derived strings).

### 33.6 Extraction guide

Part 2 rewritten (was stale: BDH procedure, pre-correction mnemonics). Part 4 rewritten
to the closed Track B state, including the three encodings of BRSR item 9, the BSE
`AttachHis` route that works, and the routes that do not.

### 33.7 State

Tests: **124 root + 50 backend = 174 passing.** Frontend 0 type errors, build clean.
Still open: GitHub remote (needs the correct account — the user will handle login);
the two-field BDH re-pull; LightGBM comparator; cost-ratio decision (3:1 vs 10:1).
