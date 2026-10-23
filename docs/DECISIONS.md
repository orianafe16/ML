# Design decisions

Each decision is listed with its justification and the course material it is based on, so it
can be defended orally. Decisions taken in later stages are appended to the corresponding section.

## Problem definition

| Decision | Justification | Source |
|---|---|---|
| Classification target: `Grade` (LGG vs GBM), binary | Clinically meaningful and balanced enough (≈58% / 42%) for a binary classifier | Course requirement: regression, classification and unsupervised tasks (Curso_Marco_y_Evaluacion.md, Presentation III) |
| `Project` and `Primary_Diagnosis` excluded as predictors of `Grade` | `Project` (TCGA-LGG / TCGA-GBM) is the target under another name, and the histological diagnosis determines the grade: using them is target leakage | Leakage example in EDA_for_Extreme_OSA_Classification_Opus5_5.ipynb (section 12, `IAH` added as a predictor of `OSA`) |
| Regression target: `Age_at_diagnosis`; `Grade` not used as a predictor | Age is the only continuous variable; excluding `Grade` keeps the regression a question about clinical and mutation features only | Decision of the author |
| Unsupervised analysis (PCA, MDS with Jaccard, k-means, hierarchical) inside the EDA | Clustering may be included as part of the EDA section | PRDL_MLLB_summary_23_10_2026.txt (Chap10, Clustering) |
| Multiclass `Primary_Diagnosis` and causal analysis postponed to January | Scope of the 1 November draft | Decision of the author |

## Stage 1 — Data preparation

| Decision | Justification | Source |
|---|---|---|
| Rebuild the clean dataset from `TCGA_GBM_LGG_Mutations_all.csv` instead of loading the UCI file | Keeps `Case_ID` (traceability) and `Primary_Diagnosis` (leakage demonstration, January multiclass task) | — |
| Check the rebuilt dataset value by value against `TCGA_InfoWithGrade.csv` (notebook stops if any check fails) | Proves that the cleaning is exactly the published one; 839 patients, same order, every column identical | UCI dataset page (839 instances) |
| Count `--` and `not reported` explicitly as missing | The TCGA export does not use empty cells, so `isna()` finds nothing | — |
| Remove the 23 incomplete patients (4 with every clinical field missing, 18 with race not reported, 1 with age missing) | Same exclusion as the UCI version; the removed patients are split almost evenly between classes (12 LGG, 11 GBM), so the removal does not bias the class balance | `results/logs/01_data_preparation.json` |
| Age = years + days/365, rounded to 2 decimals | Reproduces the UCI ages exactly | Check in notebook 01 |
| Drop `Project` | Identical to `Grade` for all 862 patients (target under another name) | Check in notebook 01 |
| Keep `Primary_Diagnosis` only as metadata | Glioblastoma ↔ GBM and every other diagnosis ↔ LGG: it determines the grade | Cross-tabulation in notebook 01 |
| Keep the UCI integer codes, but treat `Race` as nominal (one-hot encoding inside the model pipelines, unknown categories ignored) | Codes 0–3 have no order; one category has a single patient and may be missing from a training fold | — |
| Keep all 20 genes, including those mutated in < 5% of patients | Low prevalence does not imply no information; relevance is measured in the EDA | — |
| Regression predictors exclude `Grade`; classification predictors include age | Decision taken before the project started (see Problem definition) | — |

## Validation protocol

| Decision | Justification | Source |
|---|---|---|
| Nested cross-validation, 10 outer folds and 3 inner folds with `GridSearchCV` | The outer loop estimates generalisation; the inner loop selects hyperparameters without seeing the outer test fold | Example_ExtrOSA_ModelTuning_NestedLoop.ipynb; ML_Validacion_y_Seleccion_Variables.md (nested CV, train/validation/test) |
| k = 10 for the outer loop | k = 5 or 10 balances bias and variance of the CV estimate | ML_Validacion_y_Seleccion_Variables.md (Chap5, bias–variance trade-off for k-fold CV) |
| `StratifiedKFold` for classification (change from the course example, which uses `KFold`) | Every fold keeps the class proportions | EDA_for_Extreme_OSA_Classification_Opus5_5.ipynb (section 12) |
| Plain `KFold` (shuffled) for regression | Stratification by class does not apply to a continuous target | — |
| Scaling and encoding inside a scikit-learn `Pipeline` | Fitting a scaler on all data leaks test-fold information | bounty-guideline-example.md; EDA_for_Extreme_OSA_Classification_Opus5_5.ipynb (section 12); ML_Validacion_y_Seleccion_Variables.md ("cross-validation: right and wrong") |
| Metrics beyond accuracy: ROC-AUC, sensitivity, specificity; MAE, RMSE, R² with a mean baseline | Accuracy alone hides the trade-off between missed GBM cases and false alarms; a regression result only means something relative to the trivial predictor | EDA_for_Extreme_OSA_Classification_Opus5_5.ipynb (section 9.2, ROC curve, sensitivity and specificity); Chap3_ML_LinearRegression.md (RSE, R², MAE) |
| 95% bootstrap confidence intervals; overlapping intervals → no declared winner | The bootstrap quantifies the uncertainty of an estimator | ML_Validacion_y_Seleccion_Variables.md (Chap5, The Bootstrap) |
| Final interpretable model refitted on all data with the selected hyperparameters | Nested CV estimates the performance of the procedure, not of one fitted model; the final model is fitted once on all data, as in the course example with the decision tree | Example_ExtrOSA_ModelTuning_NestedLoop.ipynb ("Train a tree with best hyperparameters using all data") |
| Seed 42 everywhere (`src/config.py`) | Reproducibility | Project instructions |

## Engineering

| Decision | Justification |
|---|---|
| Conda environment with Python 3.11, libraries pinned with `==` in `requirements.txt` | Same versions on any computer; the pinned set was installed and tested together before being fixed |
| Notebooks call functions from `src/` | The same code is used by every notebook and can be read on its own during the defense |
| All outputs written to `results/` (figures, LaTeX tables, one JSON summary per stage) | The reports are built from these files; no number is typed by hand |
| Raw data never modified; stored byte-for-byte in git (`data/raw/** -text`) | Traceability to the UCI download |
