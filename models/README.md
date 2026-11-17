# Model artefacts

| File | Data | Served by the API? |
|---|---|---|
| `model_A-1.2.0-real.pkl` | **Real** Bloomberg panel, 14 features | **Yes** — current |
| `model_A-1.1.0-real.pkl` | **Real**, 16 features incl. two point-in-time look-ahead features | No — superseded, kept for the §5b leakage comparison |
| `model_A-1.0.0-real.pkl` | **Real**, financials wrongly included | No — superseded, kept for the §22.3 comparison |
| `deprecated_synthetic/` | ⚠️ **SYNTHETIC** — see that directory's README | **No**, and no longer loadable by their old paths |
| `registry.json` | Version history, hashes, metrics | — |

## Version history, and why each one was superseded

- **v1.0.0 → v1.1.0** — financial-sector issuers were in the training data in
  violation of §5.6 / invariant 5. 2,457 rows across 283 tickers excluded.
  Pooled AUC fell 0.9589 → 0.9550: the retained banks were *flattering* the
  aggregate at 8.5% prevalence without the model understanding one of them.
- **v1.1.0 → v1.2.0** — `log_mkt_cap` and `equity_volatility` were look-ahead.
  Bloomberg's `CUR_MKT_CAP` and `VOLATILITY_260D` are real-time market fields;
  the `EQY_FUND_YEAR` override does not apply to them, so every historical row
  carried the 2026 pull-date value. Pooled AUC 0.9550 → 0.9507, early-warning
  0.9490 → 0.9434. See `MODEL_CARD.md` §5b.

Both corrections cost accuracy, which is the point: each number that fell was
measuring something the model was not entitled to know.

## The synthetic artefacts have been moved

`model_A.pkl` and `model_B.pkl` now live in `deprecated_synthetic/`. They were
moved rather than deleted so the pre-real-data paper drafts stay reproducible,
and moved rather than left in place so that a stale
`joblib.load("models/model_A.pkl")` fails loudly instead of quietly scoring a
named company with a model trained on invented data.

## There is no real Model B

Track B closed as a **feasibility result**, not a deferral — see
`reports/phase6/track_b_feasibility.md`. BRSR is mandatory only from FY2022-23
and the panel's last feature year *is* FY2022-23, so no out-of-time ESG
comparison exists in this data; the eligible frame carries 4.3 events per
variable against a ≥10 rule of thumb; and only 1 of 7 pilot filings disclosed
enough to form the energy-intensity ratio §6.8 specifies. The API raises
`TrackBUnavailable`, so an ESG estimate about a named company can never be
served from synthetic weights.

## Pickles are gitignored

`models/**/*.pkl` is excluded: these are fitted on Bloomberg data and are
derived works under the same licence (PRD §18.3). Regenerate with:

```
PYTHONPATH=src .venv/bin/python src/build_training_matrix.py
PYTHONPATH=src .venv/bin/python src/train_model_a.py --xgb
PYTHONPATH=src .venv/bin/python src/rolling_origin.py --xgb
PYTHONPATH=src .venv/bin/python src/explain_model_a.py
PYTHONPATH=src .venv/bin/python src/model_card.py
```
