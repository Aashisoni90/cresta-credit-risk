# CRESTA — Session Handoff Prompt

Paste everything below this line to a new AI session (any model) to bring it fully
aligned with the CRESTA project before continuing work.

---

You are picking up an in-progress capstone project called **CRESTA** (Credit Risk
Early-warning System with Transparent Analytics) — an explainable machine learning
system predicting financial distress in Indian listed companies 12 months ahead, with
a secondary research track testing whether ESG signals add incremental predictive
power. This is a real, partially-built codebase with real (not synthetic-only) data
extraction in progress. Do not treat this as a fresh design exercise — your job is to
understand exactly where the project stands and continue from that exact point.

## Read these files, in this exact order, before responding to anything else

1. **`CRESTA_PRD.docx`** — the authoritative product requirements document (18
   sections + 2 appendices). This defines every target metric, every architectural
   constraint, every non-goal, and the two-track design. Treat it as the spec. It is a
   binary `.docx` — use `python-docx` (install via the project's `.venv` if present, or
   `pip install python-docx`) to extract paragraphs and tables in document order;
   `Read`-ing it directly as text will fail.
2. **`Project_Context.md`** — the full chronological development log, including every
   bug found and fixed, every data-quality finding, every design decision and its
   rejected alternatives, and (most importantly for you) **§12, the most recent
   session log**, which describes exactly what was done in the last working session on
   real Bloomberg data extraction, what's validated, what's blocked, and what the
   immediate next steps are. **Read this file fully, not just the tail** — later
   sections assume the earlier findings (e.g. the label-circularity trap in §Phase III,
   the survivorship-bias controls in §Phase IV) as established fact, and re-deriving or
   contradicting them without reading the reasoning first will waste time and risk
   re-introducing already-fixed mistakes.
3. **`PHASE_PLAN.md`** — the phased build plan with hard exit criteria per phase.
   Cross-reference this against §12.6 of `Project_Context.md` to see exactly which
   Phase 1/2 exit criteria are still open.
4. **`Data_Extraction_Guide_v2.md`** — the operational guide for how real Bloomberg
   Terminal data (financials, ratings, and BRSR ESG PDFs) is actually being pulled by
   the user. If you are about to give guidance on data extraction, read this first —
   it already encodes lessons from real mistakes made during this project (invalid
   field mnemonics, EQS screening traps, survivorship-bias pitfalls) and repeating
   generic Bloomberg advice instead of this project's specific, corrected procedure is
   a regression.
5. **`README.md`** — shorter, slightly stale overview of the script-level workflow;
   useful for orientation but `Project_Context.md` and `PHASE_PLAN.md` are more
   current and take precedence wherever they disagree.
6. Skim `src/` directory listing (`ls src/`) and open **`src/generate_formula_templates.py`**
   specifically — it was modified in the last session (corrected two Bloomberg field
   mnemonics) and is the current source of truth for what the 18 pulled financial
   fields are actually called.

## Current state in one paragraph (as of 2026-09-19)

Track A (the primary financial-distress model) is in **Phase 1 (Foundation)**,
specifically the real-data-extraction step. A ~800-company universe
(`data/processed/universe_800.csv`, 3,001 rows) already exists from a completed EQS
screen. Seven Bloomberg pull templates with live `BDP()` formulas
(`data/raw/templates/bloomberg_formula_chunk_1..7.xlsx`, ~500 companies × 11 fiscal
years each) were generated, had two invalid field mnemonics found and corrected, and
are being filled one chunk at a time by the user on a Bloomberg-connected Terminal
machine. **Chunks 1–4 are filled, validated, and accepted** (balance-sheet identity
check passing ~99% on each, dead tickers logged, no corruption found). **Chunk 5 hit a
Bloomberg daily data-request capacity limit** (`#N/A Daily Capacity` on every cell) and
is blocked until the Terminal account's quota resets. Chunks 6 and 7 have not been
attempted yet. Ratings-history data (needed for the primary distress label) and BRSR
ESG PDFs (needed for Track B, the secondary research track) have **not** been pulled
yet and are explicitly scoped as later work — do not suggest starting them before the
financial panel (chunks 1–7) is complete, per the PRD's own Track A-before-Track B
sequencing rule.

## What to do next, concretely

Check with the user first whether the Bloomberg daily capacity has reset and whether
chunks 5, 6, 7 have since been pulled. Do not assume — ask. Then:

- If chunk 5 (and possibly 6/7) is now available: validate it exactly the way chunks
  1–4 were validated (see `Project_Context.md` §12.3 for the precise procedure — NA-
  sentinel cleaning, per-field coverage report, dead-ticker detection into
  `data/raw/failed_tickers.csv`, and the balance-sheet identity check at 2% tolerance).
  Do not skip the balance-sheet identity check — it is this project's standard proof
  that column mapping is correct, not optional diligence.
- If all 7 chunks are eventually in: the next real engineering task is to (a) merge
  them into a single panel, (b) resolve the fiscal-year convention ambiguity
  explicitly (there is a documented, still-unresolved −1 offset question between
  Bloomberg's `EQY_FUND_YEAR` and Indian FY notation — see `Project_Context.md` §7.3
  and §Phase II Step 12), (c) promote the ad hoc validation logic used on chunks 1–4
  into a real loader module in `src/` (this was explicitly deferred, see
  `Project_Context.md` §12.5), and (d) run `src/quality_gates.py` against the merged
  panel.
- Only after that: resume the ratings-coverage census and the primary-vs-fallback
  label decision (still open — see `Project_Context.md` §Phase III and PHASE_PLAN.md
  Phase 2 exit criteria).

## Non-negotiable invariants — do not violate these even if they seem like reasonable engineering shortcuts

These are established, tested-against-real-data decisions, not stylistic preferences.
Violating any of them reintroduces a bug this project already found and fixed once:

1. **Never zero-fill a financial ratio.** NaN is signal (e.g. `negative_equity_flag`,
   `debt_to_equity` undefined under negative equity). Zero-filling fabricates distress
   signal.
2. **Never use a random train/test split on this data.** Out-of-time split only — see
   PRD §8.5. A random split leaks future information into training.
3. **Fit all preprocessing (scalers, imputers, encoders) on the training window only**,
   then apply unchanged to validation/test.
4. **Never silently drop a failed/dead ticker.** Log it to `data/raw/failed_tickers.csv`
   with a reason. Distressed and delisted companies are disproportionately the ones
   that fail to resolve, so dropping them silently reintroduces survivorship bias.
5. **Financial-sector issuers (banks/NBFCs) are excluded from the primary model** —
   this is empirically justified (§5.6 of `Project_Context.md`), not an oversight to
   "fix" by imputing their missing fields.
6. **Track A must work with zero ESG data present, always.** Track B (ESG) is
   injected as an optional feature block at feature-assembly time only, never a
   dependency of Track A at any layer.
7. **Watch for label circularity.** Do not let a feature and a label-construction
   criterion be built from the same underlying quantity (e.g. `negative_equity_flag`
   as both a feature and a fallback-label criterion) without the circularity guard in
   `src/fallback_label.py`'s `CIRCULAR_FEATURE_MAP`. This exact contradiction exists
   in the PRD itself (§6.1/§6.4 vs §7.5) and was caught before any training run — see
   `Project_Context.md` §Phase III Step 15 for why this matters and what it would have
   done to the results if missed.
8. **Test one ticker's full field list on a throwaway sheet before opening any large
   live-formula Bloomberg template.** This project has twice now caught invalid or
   near-miss field mnemonics this way before wasting a large pull on them (see
   `Project_Context.md` §12.2). Do not skip this step to save time — the cost of
   re-pulling a corrupted 5,500-row chunk is far higher.
9. **When automating manual Bloomberg extraction guidance, batch size matters.** A
   single chunk of 500 companies × 18 fields × 11 years (~99,000 `BDP()` requests) is
   apparently close to a one-day capacity ceiling when several chunks are pulled back
   to back. Advise spreading remaining chunks across multiple days rather than
   assuming they can all be pulled in one sitting.

## How this project likes to work (process norms, not just technical facts)

- The user pulls data manually from a Bloomberg Terminal (Windows-only, no API access
  from the Mac dev machine — this is a confirmed environmental + licensing constraint,
  not a temporary inconvenience to solve around). Automated/API Bloomberg ingestion is
  an explicit PRD non-goal (§3.2).
- The user does the manual Bloomberg/Excel work; the AI assistant's job is to (a) give
  precise, step-by-step manual instructions one step at a time when asked, (b) validate
  every file the user hands back immediately using the project's real Python tooling
  (not just visually inspect it), and (c) log findings back into `Project_Context.md`
  and `PHASE_PLAN.md` so state is never lost between sessions.
- Work proceeds one concrete step at a time when the user is doing hands-on manual
  work (e.g. "what's step 1", confirm it worked, "what's step 2"). Don't dump the
  entire remaining roadmap on the user when they've asked for the next single step —
  but do keep the full roadmap in mind so each step is coherent with the next.
- When something breaks (a bad mnemonic, a throttled pull, a data-quality anomaly),
  diagnose it concretely against real evidence (run the actual validation code) before
  proposing a fix — this project's whole documented history is built on catching
  silent failures with proof (e.g. the balance-sheet identity check), not assumption.
- Every session's real progress should be appended to `Project_Context.md` in the same
  detailed, dated, numbered-section style already used throughout that file — chunk-
  level results tables, exact file changes, explicit "not yet done" lists, and
  concrete "immediate next steps" — so that a future session (human or AI) can resume
  without re-deriving anything already settled here.

## First thing to say to the user in the new session

Confirm you've read the above files, briefly state your understanding of exactly where
extraction stands (which chunks are done, what's blocked, what's next), and ask the
user directly whether the Bloomberg daily capacity has reset yet before proposing any
next action.
