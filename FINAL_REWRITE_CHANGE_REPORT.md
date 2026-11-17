# FINAL REWRITE CHANGE REPORT

**Date:** August 13, 2026
**Source:** [professional_conference_paper.tex](file:///Users/himanshu/.gemini/antigravity-ide/brain/58f19f6b-0f14-41a7-8c35-2a0a664b11bf/professional_conference_paper.tex) (original)
**Output:** [final_revised_paper.tex](file:///Users/himanshu/Desktop/CAPSTONE-PROJECT/esg_distress_project/final_revised_paper.tex) (revised)

---

## Structural Changes

| Aspect | Original | Revised |
|--------|----------|---------|
| Sections | 7 + Appendix | 6 + Appendix (more logical grouping) |
| Subsections | ~15 | 35 (substantially expanded methodology) |
| Figures | 9 | 9 (same set, revised captions) |
| Tables | 3 | 3 (1 completely replaced, 1 revised) |
| Equations | 4 | 7 (added ESG observation model, coefficient mapping, temporal target) |
| References | 14 | 14 (same set, Jones year noted correctly in text) |
| Approx. words | ~4,000 | ~5,200 |

### Section Structure Comparison

| Original | Revised |
|----------|---------|
| 1. Introduction | 1. Introduction (rewritten; 3 RQs redefined; 4 contributions) |
| 2. Extended Literature Review | 2. Related Work (reorganized into 6 themed subsections) |
| 3. Theoretical Framework and Stochastic Simulation | 3. Methodology (§3.1–§3.15, massively expanded) |
| 4. The CRESTA Pipeline Architecture | *(merged into §3)* |
| 5. Modeling Strategy and Objective Functions | *(merged into §3)* |
| 6. Empirical Results and Evaluation | 4. Results (§4.1–§4.9, restructured) |
| 7. Advanced Explainability as Proof (RQ1) | *(merged into §4 Results)* |
| 8. Discussion and Limitations | 5. Discussion (§5.1–§5.5, substantially expanded) |
| 9. Conclusion and Future Work | 6. Conclusion and Future Work |

---

## Methodological Additions

1. **ESG observation model equation** (Eq. 5): Explicitly defines the noisy observation process.
2. **Theoretical coefficient mapping** (Eq. 6): Derives the attenuation formula and explains the gap between −0.04 and −0.035.
3. **Temporal target equation** (Eq. 7): Formal definition of the forward-shift mechanism.
4. **Quality gate details**: All seven gates enumerated with exact counts and severity levels.
5. **38 vs 44 feature distinction**: Consistent three-level pipeline throughout (50 → 44 → 38).
6. **PR-AUC method specification**: Explicitly states trapezoidal rule via `sklearn.metrics.auc(recall, precision)`.
7. **Bootstrap limitation disclosure**: Full paragraph explaining row-level vs. cluster-bootstrap concern.
8. **Five-stage signal diagnostic table**: New Table 3 with exact coefficients, standard errors, and p-values at all 5 stages.
9. **PDP non-monotonicity disclosure**: Described as "broadly decreasing" with local fluctuations.
10. **Coverage-flagged rows**: Correctly stated as "retained in the feature matrix and participated in model training."
11. **No hyperparameter optimization**: Explicitly acknowledged as a limitation.

---

## Writing Improvements

1. **Title**: Changed "Architecture" to "Evaluation Framework" to better reflect the contribution.
2. **Abstract**: Completely rewritten with 6 logical components; all numbers verified; no overclaiming.
3. **RQs redefined**: From orthogonality/circularity/significance to circularity-effect/signal-recovery/imbalance-interpretation.
4. **Contributions**: Specific, evidence-backed, numbered list of 4 items. No "revolutionary" or "groundbreaking."
5. **Related Work**: Reorganized from chronological narrative to 6 themed subsections with clear gap synthesis.
6. **Methodology**: Expanded from ~1.5 columns to ~3.5 columns. Full reproducibility specifications.
7. **Discussion**: Structured around RQ answers, accuracy paradox, and categorized limitations.
8. **Conclusion**: Honest scope statement; explicitly states what the experiment does NOT prove.

---

## Table and Figure Changes

### Tables

| Table | Original | Revised | Change |
|-------|----------|---------|--------|
| Table 1 (Hyperparameters) | ✅ Present | ❌ Removed | Hyperparameters now inline in Methodology |
| Table 1 (Benchmark) | Stale values; ranked by accuracy; XGBoost=98.30% | Freshly reproduced; metrics include Bal.Acc, ROC-AUC, Pos.Prec, Pos.Recall, Pos.F1, Accuracy | **COMPLETELY REPLACED** |
| Table 2 (Ablation) | Same format | Minor wording update | Preserved |
| Table 3 (5-Stage Diagnostic) | Not present | New table with coefficients, SE, p-values | **NEW** |

### Figures

| Figure | Caption Change |
|--------|---------------|
| Fig 3 (Confusion matrix) | Removed any reference to 98.30%; added verified accuracy 66.97% context |
| Fig 8 (PDP) | Changed "monotonic downward slope validates" → "broadly decreasing trend... consistent with" |
| All others | Minor caption refinements for consistency |

---

## Claims Corrected

| Original Claim | Correction | Reason |
|----------------|-----------|--------|
| XGBoost accuracy = 98.30% | XGBoost accuracy = 66.97% | Stale value from prior pipeline run |
| "mathematically proven" | Removed entirely | ESG recovery is empirical evidence, not mathematical proof |
| "eradicate label circularity" | "address" / "mitigate" / "ablate" | Circularity control is not absolute elimination |
| "monotonically decreasing PDP" | "broadly decreasing" | 12 of 49 grid steps show increases |
| "2,097 rows excluded from modelling" | "retained and participated in training" | Code audit shows they are not filtered |
| Training prevalence "~0.4%" | "0.3680%" | Exact verified value |
| "44 features" (model input) | "38 final model-input features" | After 6 circular features ablated |
| "proof-of-concept" (multiple) | "controlled validation" / "evaluation framework" | More accurate scope |

---

## Claims Weakened

| Original Claim | Weakened To |
|----------------|------------|
| "proves orthogonality" | "provides evidence consistent with orthogonal contribution" |
| "SHAP... prove the orthogonal mechanism" | "confirms ESG features contribute as independent predictive inputs" |
| "this explicitly proves" | "the global direction is consistent with" |
| "novel architecture" | "evaluation framework" |
| "immune to circularity" | "designed to address circularity" |
| Bootstrap "rigorously proves significance" | "rejects the null hypothesis... with the caveat that cluster bootstrap would be more conservative" |

---

## CCS Concept Fix

Changed `Neural networks` → `Supervised learning by classification` (no neural networks are used in this paper).
