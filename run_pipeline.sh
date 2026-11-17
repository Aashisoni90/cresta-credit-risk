#!/bin/bash
set -x
echo "=== STEP 1: Generate Data ==="
python src/generate_synthetic_data.py --seed 42
echo "=== STEP 2: Quality Gates ==="
python src/quality_gates.py data/processed/synthetic_panel.csv -o data/processed/synthetic_panel_gated.csv
echo "=== STEP 3: Feature Engineering ==="
python src/feature_engineering.py data/processed/synthetic_panel_gated.csv --out data/processed/synthetic_feature_matrix.csv
echo "=== STEP 4: Train Model ==="
python src/train_model.py
echo "=== STEP 5: Diagrams ==="
python src/generate_paper_diagrams.py
echo "=== STEP 6: Expanded Results ==="
python src/generate_expanded_results.py
