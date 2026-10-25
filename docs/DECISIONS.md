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
| Multiclass task, SHAP and causal analysis included in the 1 November version | The draft is meant to be the complete work, changed afterwards only with the instructors' feedback | Decision of the author (25 October) |

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

## Stage 4 — Regression (age at diagnosis)

| Decision | Justification | Source |
|---|---|---|
| Predictors: gender, race (one-hot) and the 20 genes; no grade | Decision taken at the start; the question is what the clinical/mutation profile alone explains | Problem definition |
| Models from simple to flexible: mean baseline, OLS, Ridge, Lasso, tree, random forest, XGBoost, SVR | Covers the course chapters (3, 6, 8, 9) and gives a trivial reference | Chap3, Chap6, Chap8, Chap9 |
| Inner selection criterion RMSE; reported MAE, RMSE, R² | RMSE matches the squared-error loss of most models; MAE is easier to interpret in years; R² compares with the mean baseline | Chap3 (RSE, R², MAE) |
| Plain shuffled `KFold` (10 outer, 3 inner) | Stratification by class does not apply to a continuous target | — |
| Metrics on pooled out-of-fold predictions with bootstrap CIs (same resamples for every model); fold mean ± SD also reported | Pooled predictions use all 839 patients for one estimate; fold SD shows the variability between test sets | Chap5 (bootstrap) |
| Overlapping 95% intervals → no winner; paired bootstrap difference reported as a complement | Rule decided before the analysis; the paired difference is more sensitive because both models are evaluated on the same resampled patients | — |
| Small grids (11 to 27 configurations per model) | 839 patients; the goal is a sound comparison; nested CV multiplies the cost by 30 | — |
| Linear models and SVR on standardised inputs; trees unscaled | Penalties and kernels depend on the scale of the inputs; trees do not | Weekly summary 23 Oct 2026 (trees work with any predictors) |
| Regression tree limited by `max_depth` / `min_samples_leaf` (pre-pruning); classification tree pruned by `ccp_alpha` | Both control tree size; using the two approaches shows both | Chap8; weekly summary 23 Oct 2026 |
| Final models = the same GridSearchCV (3 folds) applied once to all patients | The nested CV estimates the performance of the procedure; the final model applies that procedure | Example_ExtrOSA_ModelTuning_NestedLoop.ipynb |
| OLS interpreted with statsmodels on unscaled 0/1 predictors (coefficients in years), white as race reference | Coefficient = difference in mean age, other predictors fixed; p-values adjusted with BH | Chap3 |
| Residual diagnostics: residuals vs fitted, Q–Q, scale-location, Cook's distance, Breusch–Pagan, Shapiro | The six "potential problems" of Chapter 3 | Chap3 |
| One patient with leverage 1 (the only American Indian / Alaska Native) reported and excluded from the diagnostic plots | Its fit is exact, so Cook's distance is undefined (0/0); keeping the category keeps the same predictors as the CV models | Chap3 (high-leverage points) |
| Permutation importance on the outer test folds | Importance on unseen patients, comparable between models | — |
| Out-of-fold predictions coloured by grade (not an input), with their correlation with both the grade and the IDH status | The predictions follow the IDH status much more closely than the grade; a first reading that attributed them to the grade was corrected (AI_USAGE_LOG, stage 7) | — |

## Stage 5 — Classification (LGG vs GBM)

| Decision | Justification | Source |
|---|---|---|
| Predictors: age, gender, race and the 20 genes; `Project` and `Primary_Diagnosis` excluded (asserted in the notebook) | Both encode the grade (leakage demonstrated in notebook 02: AUC = 1.000 with the diagnosis) | EDA notebook, section 3.2 |
| GBM as positive class | Sensitivity = share of GBM (the more aggressive tumour) detected | — |
| Stratified outer (10) and inner (3) folds | Same class proportions in every fold | EDA notebook (stratified CV) |
| Inner selection criterion ROC-AUC; also reported: balanced accuracy, sensitivity, specificity, accuracy at threshold 0.5 | AUC does not depend on the threshold; accuracy alone hides the trade-off and the class imbalance | EDA notebook, section 9.2 |
| Bootstrap CIs stratified by class | Each resample keeps the class sizes, as the folds do | — |
| Logistic regression with the penalty type (L1/L2) as a hyperparameter | Both regularisations of Chapter 6 compared inside the same nested CV | Chap6 |
| Naive Bayes as BernoulliNB with age cut into quantile bins (bins tuned) | All predictors become binary, so the Bernoulli model applies to every column; Gaussian NB would treat 0/1 mutations as normal variables | — |
| Classification tree post-pruned with cost-complexity pruning (`ccp_alpha` and Gini/entropy tuned) | "A tree must be pruned to avoid overfitting"; impurity measures Gini or deviance | Weekly summary 23 Oct 2026 (Chap8) |
| SVM with linear and RBF kernels in the same grid; `probability=True` (Platt scaling) | AUC needs a score; kernels and C/gamma are the hyperparameters highlighted in the course | Weekly summary 23 Oct 2026 (Chap9) |
| Threshold kept at 0.5; trade-off curve shown but no threshold tuned | Tuning the threshold on the same out-of-fold predictions would bias the reported metrics | — |
| Logistic coefficients read from the penalised final model, no unpenalised p-values | Mutations present only in LGG (e.g. NOTCH1) cause quasi-complete separation, with infinite maximum-likelihood coefficients | Notebook 02, section 4.2 |
| Flatness of the final inner-CV surface reported | Explains why the final hyperparameters may differ from those chosen in the outer folds | — |
| Tree nodes recoloured with the project colours | sklearn's default paints class 0 (LGG) orange, the opposite of the project convention | — |

## Stage 5b — Multiclass classification (histological family)

| Decision | Justification | Source |
|---|---|---|
| Four classes: Astrocytoma (NOS + anaplastic), Mixed glioma, Oligodendroglioma (NOS + anaplastic), Glioblastoma | NOS and anaplastic differ in grade (2 vs 3), which point mutations can hardly resolve; the family is what mutations can inform. Mixed glioma kept as a class to test whether it is separable | Decision of the author (25 October) |
| `Grade` and `Project` excluded as predictors | Glioblastoma = GBM and every other family = LGG: the grade would give one class away | Same leakage reasoning as stage 5 |
| Macro one-vs-one AUC (Hand–Till) as inner criterion and main metric; balanced accuracy and macro F1 as secondary | Threshold-free and insensitive to class proportions (Glioblastoma is 42% of the patients); balanced accuracy gives each family the same weight | Hand and Till (2001); Chap4 (multiclass logistic regression, softmax) |
| Models: multinomial logistic regression (saga, L1/L2), pruned tree, random forest, XGBoost; same grids as stage 5 | Direct multiclass extensions; Naive Bayes (worst calibrated) and SVM (one model per pair of classes with Platt scaling, costly in nested CV) not repeated | Chap4 (multinomial regression), Chap8 |
| Recompute the k = 3 clusters of notebook 03 and cross-tabulate them with the families | Tests whether the unsupervised structure corresponds to the histological families | Notebook 03 |

## Stage 5c — SHAP values

| Decision | Justification | Source |
|---|---|---|
| Explain the winning models only (best + those whose paired AUC difference includes 0) | Explaining models that perform worse adds nothing; the tied models may use the data differently | Decision of the author ("winning models, with a short discussion with the literature") |
| TreeSHAP (random forest, tree, XGBoost) and linear SHAP (logistic regression), via the `shap` library | Exact values for these model types; the additivity of every explanation is checked in the notebook | Lundberg and Lee (2017); Lundberg et al. (2020) |
| Compare models by the share of mean \|SHAP\| and by ranking, not raw values | Units differ (probability for forests, log-odds for XGBoost and logistic regression) | — |
| SHAP computed on the final models (all patients), compared with permutation importance computed on held-out folds | SHAP describes how the model uses the inputs; performance claims stay with the held-out estimates | — |

## Stage 5d — Causal analysis

| Decision | Justification | Source |
|---|---|---|
| Question: how much of the IDH–age association goes through the grade (mediation) | Turns the reading of the regression (H2) into an explicit causal question | PRDL_MLLB_summary_25_9_2026.txt ("link these ideas with a causal analysis"; *The Effect*, chapters 6 and 13) |
| Exposure IDH mutation = IDH1 or IDH2 | The WHO classification uses IDH-mutant status (either gene) | WHO 2021 (louis2021) |
| Covariates sex and race (white vs other) | Fixed before the tumour arises, so they can only be confounders; the small race categories would give leverage-1 patients | Notebook 04 (leverage problem) |
| Product-of-coefficients decomposition with OLS, HC3 intervals and 2000 bootstrap resamples | Exact identity c = c' + ab for OLS; residuals are heteroscedastic (notebook 04) | Baron and Kenny (1986); MacKinnon et al. (1995); Hayes and Cai (2007) |
| Report several adjustment sets and the IDH × grade interaction | Shows that the estimate depends on what is adjusted for, and tests the no-interaction assumption | *The Effect*, chapter 13 |

## Stage 6 — Reports

| Decision | Justification | Source |
|---|---|---|
| Reports written in LaTeX from the Overleaf templates; cover and structure kept. The Ensemble Learning appendices of the templates were removed and will go to the Deep Learning reports | Course templates; ensemble learning fits the Deep Learning reports better | Templates; decision of the author (25 October) |
| Every number in the text is a macro generated from `results/logs/*.json` by `report/make_numbers.py` | No number typed by hand; re-running the notebooks updates both reports | Project instructions |
| Tables are `\input` from `results/tables/`; figures copied from `results/figures/` by `report/build_zips.py` | Same principle; one zip per report for Overleaf Free (no Git sync) | Project instructions |
| PRDL = theory, hypotheses and critical analysis; MLLB = tools, implementation and graphics; figures and text not duplicated | Course FAQ on PRDL vs MLLB | Curso_Marco_y_Evaluacion.md (Presentation III) |
| Literature claims written only from the part of a NotebookLM statement supported by its verbatim quote; anything else marked `\todoverify{}` (red in the PDF) | Extracted statements were often broader than their quotes | AI usage log, stage 6 |
| Bibliography in BibTeX (IEEEtran style); each entry's `note` says how it was checked; DOIs checked against Crossref with `report/check_references.py` | Traceability of every reference | — |
| AI use and Bounty sheets reported as Appendix A in both reports (scientific assistants in PRDL, code assistant in MLLB) | Graded item (0.5 points); the guideline places the Bounty sheets in Appendix A | bounty-guideline-example.md |

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
