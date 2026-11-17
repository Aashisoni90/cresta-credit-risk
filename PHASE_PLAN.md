# CRESTA — Phased Build Plan

Derived from PRD §16.1, adjusted for what has already been built and what is currently
blocked. Each phase has **hard exit criteria**: the next phase does not begin until they
pass. This is the mechanism for "minimising errors" — most defects found so far were
silent, and phase gates are what make them loud.

**Sequencing rule (PRD §16.1):** Track B (P6) begins only after Model A is validated and
serving. If BRSR extraction proves harder than estimated, P6 descopes to documented future
work and the project still delivers. The reverse — a half-finished primary system with an
ESG experiment bolted on — is not acceptable under any timeline pressure.

---

## Phase 1 — Foundation ✅ COMPLETE (2026-10-27)

**Goal:** every byte of real data that enters the pipeline is validated, and every rejection
is logged rather than silently dropped.

| Item | Status |
|---|---|
| Ingestion: wide column-block layout | ✅ `load_bloomberg_export.py` |
| Ingestion: vertical filled layout | ✅ `load_filled_template.py` |
| Sentinel cleaning (`#N/A` variants → typed NULL) | ✅ `clean_numeric` |
| Negative-equity handling | ✅ flag + `debt_to_equity` NaN rule |
| Balance-sheet identity verification | ✅ runs on every load |
| Unit-outlier / dead-ticker detection | ✅ with `failed_tickers.csv` |
| **Quality gates (PRD §5.4, all 7)** | ✅ `quality_gates.py` — **all 7 now firing on real data** as of 2026-10-27; `scale_sanity` activated once Z″ existed (§15.6) |
| **Test suite** | ✅ `tests/` — **96 tests, all passing** |
| Universe builder EQS pipeline | ✅ `build_company_universe.py` — EQS export processed; ⚠ needs a junk-row filter, see `Project_Context.md` §13.2 |
| Formula-chunk loader + validator | ✅ `load_formula_chunks.py` — **new 2026-10-27**, closes the §12.5 deferral |
| Real 18-field panel, all 7 chunks | ✅ 32,989 rows × 2,999 tickers, merged + gated |
| PostgreSQL schema + Docker | ⏸ deferred — see note |

**Deferred deliberately:** PostgreSQL and Docker (PRD §5.2, §10.2). At 37 companies, CSV
is sufficient and a database adds failure surface without adding correctness. Introduce at
Phase 3 when the ~800-company universe makes it worthwhile. The canonical schema in §5.2
is still the contract — the CSV columns match it.

**Exit criteria**
- [x] Real export loads with zero silent corruption
- [x] All 7 §5.4 gates implemented and firing
- [x] Rejected rows logged, never dropped
- [x] ~800-company universe built from EQS export with dedup + survivorship accounting
      ← **DONE 2026-10-27.** All 7 Bloomberg formula chunks pulled, validated, merged and
      gated. Panel: 32,989 rows × 2,999 tickers × FY2015–2025
      (`data/processed/bloomberg_panel_full_gated.csv`). All 7 §5.4 gates run, batch
      ACCEPTED; 23,735 rows trainable. Loader promoted to `src/load_formula_chunks.py`.
      See `Project_Context.md` §13.

---

## Phase 2 — Features & Labels ✅ COMPLETE (2026-10-27)

**Goal:** a model-ready feature panel with a defensible, non-circular label.

| Item | Status |
|---|---|
| Feature blocks 1–7 (§6.1–6.7) | ✅ `feature_engineering.py` — 30/40 features computable now |
| Expanded-pull features (debt, revenue, CF, vol) | ✅ **unblocked 2026-10-27** — `total_debt`, `total_revenue`, `cash_from_operations`, `net_income`, `volatility_260d`, `interest_expense` all present in the merged panel |
| Altman Z″ integration + hand-verified sample | ✅ **DONE 2026-10-27** — computed on the real panel (20,812 rows); hand-verified exact on 4 companies; degenerate-denominator handling added in `score_altman_panel.py`. See `Project_Context.md` §15. |
| **Circularity resolution** — decide and document | ✅ **RESOLVED 2026-10-27** — guard fixed (§16.2) and Z″ decided (§17.1): feature-side uses 3-term `altman_z_ex_x4` (AUC 0.807 vs 0.766 full), canonical Z″ stays the §8.1 benchmark |
| Rating-history ingestion | ❌ **not achievable** — census 2026-10-27 found 4/3,000 tickers rated (0.13%); CRISIL field returns `#N/A Authorization` (entitlement). §7.5 fallback **promoted to primary label**. See §16.1. |
| Reporting-lag enforcement (§7.4) | ✅ `make_forward_looking()` in `fallback_label.py` |

- Feature blocks 1–7 (§6.1–6.7) — Block 6 temporal features need no new data ✅
- Altman Z″ integration + hand-verified sample ← Z″ computed; hand-verify 1 company
- **Circularity resolution** — decide and document before any training run
- Rating-history ingestion **or** formal promotion of the §7.5 fallback to primary
- Reporting-lag enforcement (§7.4) ✅

**Exit criteria**
- [x] Feature panel materialised ← **DONE 2026-10-27**, `data/processed/training_matrix.csv`, 22,299 rows × 16 non-circular features (§17.5)
- [x] Label distribution reported with class balance ← **DONE** — train 25.6% / val 22.8% / test 21.3% positive; **well above §8.3's 5-15% expectation**, a property of the §7.5 label (§17.4)
- [x] Circularity treatment documented ← **DONE 2026-10-27** (§16.2, §17.1). Guard enforced by `assert_no_circular_leak(strict=True)`; Z″ feature-side uses the non-circular `altman_z_ex_x4`, canonical Z″ retained as the §8.1 benchmark
- [x] FY convention resolved and asserted in code ← **DONE 2026-10-27**, verified against TCS/Reliance published FY2023-24 accounts; `derive_fiscal_year()` in `load_formula_chunks.py`. See `Project_Context.md` §14.
- [x] Hand-verified sample of Z″ scores against manual calculation ← **DONE 2026-10-27**, 4 companies, diff 0.00e+00 (§15.2)

**Blocked on:** nothing external. Rating census **COMPLETE 2026-10-27** — outcome: ratings
label unavailable (0.13% coverage, CRISIL entitlement gap), §7.5 fallback promoted to primary
(§16.1). FY convention **RESOLVED** (§14). Expanded field pull landed.

**Remaining Phase 2 work is now internal:** finalise the feature matrix (20 clean features),
decide the Altman Z″ circularity question (§16.5), and report the label distribution.

**New Phase 2 decision needed:** PRD §8.5's split now maps unambiguously onto the data, and
doing so exposes that its test window overruns the panel by a year while FY2014-15 falls
outside every window. Needs an explicit, justified boundary — see §14.5.

**New Phase 2 inputs available:** a Bloomberg RATC rating-actions export exists but is
capped at 5,000 rows / half of FY2015 and covers only 7.3% of the universe; and
`label_construction.py` must be fixed to recognise agency code `CRIS` before any census
(it currently only knows `CRISIL` and would discard 44% of the file). See §13.9.

---

## Phase 3 — Model A  ✅ COMPLETE (2026-10-28; v1.2.0, see `Project_Context.md` §18–§22, §31)

- Altman Z″ threshold benchmark, regularised logistic regression, XGBoost (+LightGBM)
- **Bloomberg DRSK as third benchmark** (new — see `Project_Context.md` §6, I6)
- Out-of-time split + rolling-origin folds (§8.5)
- Cost-weighted threshold at 10:1 (§8.4)
- Bootstrapped CI + DeLong on AUC differences
- **Leakage tests** (§15 — "the most valuable test in the suite")

**Exit criteria:** metric targets met or deviation documented; beats both baselines
significantly; artefact registered with model card.

- [x] Altman Z″ benchmark + logistic baseline + XGBoost trained and compared (§18.1)
- [x] Out-of-time split and train-only preprocessing **enforced by assertion** (§18.6)
- [x] DeLong + bootstrap CI on ΔAUC (§18.1)
- [x] Cost-ratio sensitivity reported (§18.4)
- [x] All §3.3 metric targets met — **but see §18.2: pooled AUC is inflated by label
      persistence; the honest headline is the stratified early-warning AUC **0.9434** (v1.2.0)**
- [x] Rolling-origin (walk-forward) folds — 5 folds, XGBoost **0.9428 ± 0.0101** pooled / **0.9284 ± 0.0160** early-warning (v1.2.0); `src/rolling_origin.py`, see §20, §22.4
- [x] SHAP explainability layer (§8.8) — all four levels; `src/explain_model_a.py`, see §19
- [x] Model card + registered artefact (§14.1) — `MODEL_CARD.md`, `models/model_A-1.2.0-real.pkl`, `models/registry.json`; see §21, §22.6
- [x] **§7.4 point-in-time look-ahead REMOVED** — `log_mkt_cap`, `equity_volatility`; retrained as v1.2.0, 14 features, AUC 0.9550 → 0.9507 (§31)
- [x] **§5.6 financial-sector exclusion ENFORCED** — 2,457 rows / 283 tickers excluded and logged; retrained as v1.1.0. XGBoost AUC 0.9550 (was 0.9589), PR-AUC 0.8487 (up), early-warning 0.9490; all significance conclusions unchanged. See §22.
- [x] `git init` — repo initialised, commit `7aae63a`; §14.1 commit-hash requirement closed. Bloomberg data excluded by licence control; **GitHub remote pending the user's correct account** (§23)
- [ ] LightGBM comparator — **not installed**

**Key finding (v1.2.0, financials excluded, point-in-time leakage removed):** Altman
Z″ is *significantly worse* than a naive persistence baseline (ΔAUC −0.1110,
p<1e-300) and the logistic baseline is *not significantly different* from it
(−0.0117, p=0.165). Only XGBoost beats persistence (+0.0648, p<1e-300). Pooled AUC is
inflated by 89.8% label persistence — quote the early-warning stratum (XGBoost
**0.9434**, n=3,334, 5.1% prevalence), not the pooled **0.9507**.

**And the model is over-confident**: decile 8 predicts 69.8% against a realised 45.2%
(§14.4). The score is a rank, not a probability.

---

## Phase 4 — Serving  ✅ COMPLETE (2026-10-28, see `Project_Context.md` §24–§25, §32.3)
FastAPI, SHAP endpoints, `SECTOR_UNSUPPORTED` path, caching, contract tests.

- [x] Backend repointed from synthetic panel/models to real artefacts (§24.2)
- [x] Model A v1.1.0 served from the registry payload; refuses to serve on feature drift
- [x] Track B reported unavailable, never numerically (§24.3)
- [x] `SECTOR_UNSUPPORTED` (422) distinct from `NO_DATA` (404) (§24.4)
- [x] SHAP endpoints — global, block, live local waterfall, live counterfactual
- [x] Contract tests — 23 passing, pinning the invariants (§24.7)
- [x] Caching / response-time budget (§12.6) — portfolio 800 rows 2,387 ms → 56 ms; first explanation 2,463 ms → 12 ms; version-keyed ETags; 14 perf/caching contract tests (§25)
- [x] Synthetic `model_A.pkl` / `model_B.pkl` moved to `models/deprecated_synthetic/` — moved, not deleted (the synthetic paper drafts need them); moved rather than left in place so a stale load raises instead of silently scoring on invented data (§32.3)

## Phase 5 — Dashboard  ✅ COMPLETE (2026-10-27, see `Project_Context.md` §26–§27)
Four screens (§12.2–12.5); all journeys completable end-to-end.

- [x] Frontend repointed from the synthetic backend to real Model A v1.1.0 (§26)
- [x] All four screens rewritten; 0 TypeScript errors; `next build` clean
- [x] Every hardcoded synthetic figure replaced with a fetched one (§26.4)
- [x] Risk bands server-driven — the flag and the band can no longer disagree (§26.5)
- [x] Verified in a browser against the live API, not just compiled (§26.7)
- [x] Accessibility audit (§12.6 WCAG 2.1 AA) — contrast measured; `ink-500` 2.72→4.53:1, `ink-600` 1.79→3.03:1; keyboard-operable table rows and `aria-sort` (two Level-A fixes); risk never colour-alone (§27.1)
- [x] CSV export stamped with model version, config hash, commit, timestamp, threshold and the §14.5 constraint; 12 contract tests. **PDF deferred to the browser print pipeline — documented deviation** (§27.2)
- [x] Skeleton loaders — 13 bare spinners → 0, shaped and `aria-busy` (§27.3)

## Phase 6 — Track B  ✅ COMPLETE (2026-10-28, see `Project_Context.md` §28–§30, §32)
**Closed as a feasibility result, NOT a hypothesis test** — and that is the finding.
Full verdict: `reports/phase6/track_b_feasibility.md`.

- [x] Power/feasibility arithmetic measured against the real matrix (§28.1)
- [x] Matched-pair pilot drawn; 12/12 sector-matched (§28.2)
- [x] Declared-fiscal-year gate — `src/brsr_pilot_validate.py`. Two of the first three
      downloads were look-ahead and were caught; three filings used three different
      encodings of BRSR item 9 (§28.3, §30.4)
- [x] Energy-identity check — the ESG analogue of the balance-sheet identity; the
      ratio is **refused, not clipped**, when a filing disagrees with itself (§28.5)
- [x] Row-based positional extractor replacing regex-over-flattened-text —
      28.6/35.7/35.7% → 81.8/100.0/81.8% on the same filings (§30.3)
- [x] 7 filings obtained, all gated PASS; mean coverage **77.9%** (≥70% gate met);
      on disclosed fields extraction is **100% accurate** against hand-checked values
- [x] Sampling frame rebuilt excluding firms not yet reporting by FY2022-23 — 6 of
      1,000 swap, **EPV unchanged at 4.32** (§32.1). Same root cause as §31.
- [x] Two filings sought and not obtained logged, never dropped — SDSH (CDN 403),
      NITCO (filed no BRSR at all) (§32.3)
- [x] **DECISION: report the power result, do not build an ESG feature block.**
      Invariant 6 satisfied by construction — Track A never depended on ESG.

**The three constraints, the third decisive:** (1) BRSR is mandatory only from
FY2022-23 and the panel's last feature year *is* FY2022-23, so **no out-of-time
A-vs-B comparison exists in this data**; (2) the eligible frame carries 95 events
over ~22 variables = **4.32 EPV** against ≥10; (3) **only 1 of 7 filings yields a
usable energy-intensity ratio** — the renewable split is a voluntary *Leadership*
indicator. Coverage was never the binding constraint; disclosure was.

Not done, and not blocking: `board_independent_directors_pct` (stated as prose, not a
table row); RTPOW's 45.5% coverage undiagnosed; the Sandesh pair.

## Phase 7 — Hardening & Docs  ✅ COMPLETE (2026-10-28, see `Project_Context.md` §31, §33)

- [x] **Point-in-time look-ahead found and fixed** (§31). `CUR_MKT_CAP` and
      `VOLATILITY_260D` are real-time market fields; `EQY_FUND_YEAR` does not apply to
      them, so every historical row carried the 2026 pull-date value — 98.4% and 95.4%
      of tickers held ONE value across eleven years. Retrained as **v1.2.0**, 14
      features. Cost −0.0043 pooled / −0.0056 early-warning. All conclusions unchanged.
- [x] Guards: assertion in `model_card.py`, `tests/test_no_lookahead.py` (16 tests,
      written to FAIL when a BDH re-pull fixes the panel), backend contract test
- [x] **Monitoring** (§14.3) — `src/monitoring.py`, four triggers with thresholds;
      PSI with missingness as an explicit bucket; decay on the early-warning stratum;
      `--fail-on-trigger` for cron. 12 tests. `reports/phase7/MONITORING.md`
- [x] **Outcomes analysis** (§14.4) — `src/outcomes_analysis.py`. All 24 false
      negatives reviewed individually against the TRAIN window's distribution:
      12 signal absent / 11 signal unused / 1 data missing
- [x] **Calibration finding** — the model is systematically over-confident (mean gap
      −0.175 over deciles 7–10; decile 8 predicts 69.8% vs realised 45.2%). Ranking
      unaffected; the number must not be read as a probability. Model card limitation 7
- [x] **Validation report** (§14) — `reports/VALIDATION_REPORT.md`, assembled from
      artefacts only. Fails rather than degrades on a missing input
- [x] **Paper rewritten against real results** — `paper/cresta_real.tex`, GENERATED by
      `src/generate_paper.py` so no figure is transcribed. Three superseded synthetic
      drafts banner-marked and kept. **Not compiled — no LaTeX toolchain here**
- [x] `Data_Extraction_Guide_v2.md` Parts 2 and 4 rewritten (Part 2 was stale:
      superseded BDH procedure, pre-correction mnemonics)

**Open, and waiting on someone other than the pipeline:**
- ⬜ **GitHub remote** — commits are local only. Needs the *correct* account; the user
      handles the login. Public-vs-private also undecided.
- ⬜ **Two-field BDH re-pull** (`Data_Extraction_Guide_v2.md` §2.3) to restore
      `log_mkt_cap` / `equity_volatility` honestly. ~66,000 requests, one day's budget.
- ⬜ **Cost ratio** — 3:1 maximises F1; §8.4 defaults to 10:1. The user's call.
- ⬜ LightGBM comparator not installed; Bloomberg DRSK unused as a third benchmark.

---

## Cross-phase invariants

1. **Never zero-fill a financial ratio** (§5.5)
2. **Never a random split** — out-of-time only (§8.5). And an out-of-time split
   guarantees nothing if a feature's *values* are dated outside their row's fiscal
   year — see §31 and `tests/test_no_lookahead.py`
3. **Fit preprocessing on the training window only** (§7.4)
4. **Failed rows logged, never silently dropped** (§5.7)
5. **Every prediction carries its `model_version`** (§11.1)
6. **Track A never depends on ESG data at any layer** (§6.8)
