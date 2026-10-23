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

## Stage 2 — Exploratory data analysis

| Decision | Justification | Source |
|---|---|---|
| Structure of the EDA copied from the course notebook and adapted to binary features | Same questions (quality, target and leakage, univariate, normality, tests + effect size, outliers, redundancy, relevance, conditional probability, multivariate view, baseline) | EDA_for_Extreme_OSA_Classification_Opus5_5.ipynb |
| Mann–Whitney U for age (Welch t-test only for comparison) | Shapiro–Wilk rejects normality of age in both classes and the Q–Q plots deviate in the tails | EDA notebook, section 5 |
| Fisher's exact test for binary features | Exact for 2×2 tables; several genes have very few mutated patients, where the chi-square approximation is unreliable | — |
| Race tested as white vs non-white | Three of the four categories have very few patients (one has a single patient) | `results/logs/01_data_preparation.json` |
| Effect sizes reported with every test (AUC for all features, Cohen's d for age, odds ratio with +0.5 correction for binaries) | A p-value does not measure the size of a difference; the AUC links the EDA to the evaluation metric | EDA notebook, section 6.2 |
| Bonferroni and Benjamini–Hochberg corrections | 23 simultaneous tests; Bonferroni controls any false positive (conservative), BH the false discovery rate | EDA notebook, section 6.2 (Bonferroni); BH not in the course material |
| Bootstrap 95% CI (2000 resamples, stratified by class) for single-feature AUCs | Suggested in the course notebook ("Use bootstrap to compute confidence intervals for the AUC of each feature") | EDA notebook, suggestions; Chap5 (bootstrap) |
| Age outliers kept | They are plausible ages of real patients, not errors | EDA notebook, section 7 (rule: medical extremes are usually kept) |
| No feature is removed on the basis of the EDA | Selecting features with the labels outside cross-validation leaks information; low VIF values show no problematic collinearity; Lasso/L1 inside the nested CV can select features legitimately | ML_Validacion_y_Seleccion_Variables.md ("Cross-validation: right and wrong"); Chap3 (VIF rule of thumb 5–10) |
| Correlated mutations (IDH1–ATRX–TP53, CIC–FUBP1) noted for interpretation | They do not inflate VIF above the threshold, but coefficients of linear models must be read jointly | EDA notebook, section 8 |
| LDA projection shown as visualisation only | Fitted on all patients, so its separation is optimistic | EDA notebook, section 11.3 |
| Quick baseline with the same 10-fold stratified CV and seed as the models, preprocessing inside a Pipeline | Sanity check of the EDA and demonstration of leakage (adding the diagnosis) | EDA notebook, section 12 |
| Regression EDA added (age vs each feature, Mann–Whitney + BH, effect sizes) | The course notebook only covers classification; the regression target needs its own exploration | — |
| Fixed class colours (LGG blue, GBM orange), checked for colour-blind readers; other variables never use the class colours | Same colour = same class in every figure | — |

## Stage 3 — Unsupervised analysis

| Decision | Justification | Source |
|---|---|---|
| Input: the 20 gene indicators only; the grade is used only after clustering, to describe the clusters | Unsupervised learning must not see the label | Chap10; weekly summary 23 Oct 2026 |
| PCA computed both centred and scaled, both reported | Centring is mandatory; scaling 0/1 genes gives a mutation in a 3% gene the same weight as IDH1, which is a modelling choice worth showing | Weekly summary 23 Oct 2026 (centring; variance explained; biplot) |
| Biplot arrows drawn with one common scale factor for both axes | Separate factors per axis distort the arrow directions (see AI usage log) | — |
| Jaccard distance for MDS and t-SNE | Ignores shared absences, which dominate sparse mutation data | Weekly summary 23 Oct 2026 ("need for different distance metrics beyond Euclidean") |
| k-means on unscaled 0/1 profiles, 50 random initialisations | Squared Euclidean distance on 0/1 data = number of differing genes; k-means depends on the random start, so it is repeated and the best WCV kept | Weekly summary 23 Oct 2026 (k-means initialisation) |
| Number of clusters chosen by the agreement (ARI) between k-means and Ward, fixed before seeing the result; k + 1 also reported as secondary/exploratory | WCV has no clear elbow and silhouette grows slowly with k; two algorithms with different assumptions finding the same partition is evidence of real structure | — |
| Hierarchical clustering with Jaccard reported as a negative result; Ward on Euclidean used for the comparison with k-means | With Jaccard, the 67 patients without mutations are at distance 1 from all others; complete linkage ties at 1 and gives a single cluster, average/single linkage chain and isolate the 67 | Notebook 03, section 5 |
| Clusters compared with the grade by ARI and NMI; diagnosis shown only descriptively | ARI is corrected for chance; NMI is a complementary information-based measure | — |

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
