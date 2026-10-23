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
