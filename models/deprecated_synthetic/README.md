# Deprecated — synthetic-data artefacts

`model_A.pkl` and `model_B.pkl` were fitted on `generate_synthetic_data.py`
output, before any Bloomberg extraction existed. They are **retired, not
deleted**: every figure in `final_revised_paper.tex` and
`final_turnitin_revision.tex` was produced from them, so deleting them would
make that record unreproducible.

**They are wired to nothing.** The API serves the real-data model named in
`cresta-app/backend/config.py::MODEL_A_VERSION` from `../registry.json`. They
were moved into this directory rather than left beside the real artefacts
precisely so that a future `joblib.load("models/model_A.pkl")` raises
`FileNotFoundError` instead of silently loading a model trained on invented
numbers.

`model_B.pkl` is the more dangerous of the two: it carries an ESG effect that
was *injected* at γ = −4.0 by the generator. Serving it would be a fabricated
ESG claim. The backend raises `TrackBUnavailable` instead — see
`cresta-app/backend/ml/model_manager.py` and invariant 6.

Real artefacts and their provenance: `../registry.json`, `../../MODEL_CARD.md`.
