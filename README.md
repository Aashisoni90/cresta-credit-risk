# CRESTA — Credit Risk Early-warning System with Transparent Analytics

Explainable machine learning for predicting financial distress in Indian listed
companies one fiscal year ahead, with a secondary track testing whether
mandated ESG disclosure adds incremental predictive power.

**Status: all seven phases complete.** Current model `A-1.2.0-real`.

> This README is the map. The authoritative documents are
> [`PHASE_PLAN.md`](PHASE_PLAN.md) (what was built, with exit criteria),
> [`Project_Context.md`](Project_Context.md) (the full dated development log —
> 33 sections, every defect and its diagnosis), [`MODEL_CARD.md`](MODEL_CARD.md)
> and [`reports/VALIDATION_REPORT.md`](reports/VALIDATION_REPORT.md).

---

## The headline, stated honestly

On an out-of-time test set of 4,340 firm-years, the gradient-boosted model
scores **AUC 0.9507**. That number alone is misleading, and most of this
project is about why.

The distress label is a *state*: it is identical in consecutive years on
**89.8%** of rows. A naive "the condition continues" rule therefore scores
**AUC 0.8859** by itself. Against that baseline:

| Model | ΔAUC vs persistence | DeLong *p* | Verdict |
|---|---:|---:|---|
| Altman Z″ | **−0.1110** | <1e-300 | **significantly WORSE than no model** |
| Logistic (L2) | −0.0117 | 0.165 | not distinguishable |
| XGBoost | **+0.0648** | <1e-300 | significantly better |

**Quote the early-warning stratum, not the pooled AUC.** Restricted to firms
healthy at *t* — the task the product actually performs — XGBoost scores
**0.9434** (n=3,334, 5.1% prevalence) and Altman scores 0.6628.

**And the score is a rank, not a probability.** Decile 8 predicts 69.8%
against a realised 45.2%. See [§14.4](reports/phase7/OUTCOMES_ANALYSIS.md).

---

## Three findings worth the reader's time

**1. Persistence should be a mandatory baseline in this literature.** The
sharpest result here needed no model: a one-line rule beats a benchmark cited
for five decades. That is a statement about *evaluation*, not about Altman.

**2. Point-in-time field substitution is a leakage class.** Bloomberg's
`EQY_FUND_YEAR` override governs fundamental fields only. `CUR_MKT_CAP` and
`VOLATILITY_260D` are real-time market fields and ignore it, so every
historical row carried the pull-date value — 98.4% and 95.4% of tickers held
one value across eleven years. Two features were look-ahead and are removed
([§31](Project_Context.md), [`MODEL_CARD.md` §5b](MODEL_CARD.md)). An
out-of-time *split* guarantees nothing about whether a *column's values* are
dated inside their row's fiscal year.

**3. Track B is a negative result about a disclosure regime.** BRSR is
mandatory only from FY2022-23 and the panel's last feature year *is*
FY2022-23, so no out-of-time ESG comparison exists in this data. The eligible
frame carries 4.3 events per variable. And in a seven-filing pilot reaching
77.9% coverage, **only 1 of 7 filings disclosed enough to build the
energy-intensity ratio the design specifies**.
See [`reports/phase6/track_b_feasibility.md`](reports/phase6/track_b_feasibility.md).

Each of the three corrections applied during this project **lowered** the
reported number. A validation record whose metrics only improve is not
recording validation.

---

## Non-negotiable invariants

These are enforced in code and in tests, not by convention.

1. **Never zero-fill a financial ratio.** NaN is signal — debt-to-equity is
   undefined under negative equity, and a zero there fabricates distress.
2. **Never a random split.** Out-of-time only, asserted. And an out-of-time
   split guarantees nothing if a feature's values are dated outside their
   row's fiscal year (`tests/test_no_lookahead.py`).
3. **Fit all preprocessing on the training window only**, verified by re-fitting.
4. **Never silently drop a failed ticker.** Distressed and delisted firms fail
   to resolve disproportionately; a silent drop is survivorship bias.
5. **Financial-sector issuers are excluded** — measured, not assumed.
6. **Track A works with zero ESG data, always.**
7. **Circular features are banned by assertion**, not by review.

---

## Layout

```
├── src/                      pipeline (see "Running it" below)
├── tests/                    124 tests
├── data/processed/           panel, training matrix, exclusion logs
├── models/                   versioned artefacts + registry.json
│   └── deprecated_synthetic/ pre-real-data models, retired not deleted
├── reports/
│   ├── phase3/               metrics, rolling-origin folds, SHAP
│   ├── phase6/               Track B feasibility verdict
│   ├── phase7/               monitoring, outcomes analysis, calibration
│   └── VALIDATION_REPORT.md  §14, assembled from artefacts only
├── paper/
│   ├── cresta_real.tex       GENERATED — src/generate_paper.py
│   └── README.md             and why the three synthetic drafts are kept
└── cresta-app/
    ├── backend/              FastAPI, 50 contract tests
    └── frontend/             Next.js 14, four screens
```

---

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# pipeline, in order
PYTHONPATH=src .venv/bin/python src/build_training_matrix.py
PYTHONPATH=src .venv/bin/python src/train_model_a.py --xgb
PYTHONPATH=src .venv/bin/python src/rolling_origin.py --xgb
PYTHONPATH=src .venv/bin/python src/explain_model_a.py
PYTHONPATH=src .venv/bin/python src/model_card.py

# governance (§14)
PYTHONPATH=src .venv/bin/python src/monitoring.py          # --fail-on-trigger for cron
PYTHONPATH=src .venv/bin/python src/outcomes_analysis.py
PYTHONPATH=src .venv/bin/python src/validation_report.py
PYTHONPATH=src .venv/bin/python src/generate_paper.py

.venv/bin/python -m pytest tests -q
```

```bash
# app
cd cresta-app/backend  && PYTHONPATH=. ../../.venv/bin/python -m uvicorn main:app --reload
cd cresta-app/frontend && npm install && npm run dev
```

---

## Data availability

**The Bloomberg panel cannot be redistributed.** Vendor licensing covers
derived works, which includes the fitted model artefacts — `models/**/*.pkl`
is gitignored. The reproducibility package ships **code and the ticker list**,
never the data. Everything above regenerates from a licensed terminal pull
following [`Data_Extraction_Guide_v2.md`](Data_Extraction_Guide_v2.md).

Bloomberg has no API access from the development machine; extraction is manual
and the guide reflects the corrected procedure, not generic terminal advice.

---

## Known open items

- **Two-field `BDH` re-pull** to restore market cap and equity volatility
  honestly (`Data_Extraction_Guide_v2.md` §2.3). `tests/test_no_lookahead.py`
  is written to *fail* when that lands, so whoever runs it is told the
  feature exclusion can be lifted.
- **Cost ratio undecided** — 3:1 maximises F1; §8.4 defaults to 10:1. The
  deployed ratio must be stated wherever a flag is shown.
- **The paper has not been compiled** — no LaTeX toolchain in this
  environment. It is syntax-checked only.
- LightGBM comparator not installed; Bloomberg DRSK unused as a third benchmark.

---

## This is not investment advice

CRESTA produces analytical estimates, never automated decisions. It is not a
credit rating, not a default probability, and not applicable to financial-sector
issuers. **No credit, lending or investment action may be taken on a model
output without qualified human review** (§14.5). See
[`MODEL_CARD.md`](MODEL_CARD.md) §2 and §10.


---

## Authors & Capstone Team

- **Aashi Soni** — Computer Engineering, SVKM's NMIMS STME (Indore) | [GitHub Profile](https://github.com/Aashisoni90)
- **Varad Pal** — Computer Engineering, SVKM's NMIMS STME (Indore)
- **Himanshu Sinner** — Computer Engineering, SVKM's NMIMS STME (Indore)

**Faculty Mentor:** Dr. Praveen Thakur  
**Institution:** School of Technology Management and Engineering (STME), SVKM's NMIMS University, Indore Campus  
**Academic Year:** 2026–2027  
**Degree:** B.Tech in Computer Engineering

---

## License & Intellectual Property

This project is licensed under the **Academic Portfolio & Recruiter Evaluation License** (see [`LICENSE`](LICENSE)).  
Recruiters and hiring managers are welcome to clone, inspect, and evaluate the code for employment assessment. Unauthorized copying, partial replication, or re-submission for academic credit at any university is strictly prohibited.
