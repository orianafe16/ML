# AI usage log

Record of every interaction with AI assistants that shaped the code or the reports. It feeds the
"Use of AI assistants" section of both reports and the AI Error Bounty sheets (Appendix A of the
January reports, see `bounty-guideline-example`).

**Assistant:** Claude (Anthropic), Claude desktop app with read/write access to the project folder.
**Workflow:** Claude writes the files; the author runs the notebooks in VS Code; Claude reads the
outputs from `results/`, checks them and explains them; the author verifies and decides.

Entry format:

- **Stage / date**
- **Request** (summary of the prompt)
- **What the AI produced**
- **Verification** (how it was checked)
- **Corrections** (what was wrong and how it was fixed)
- **Bounty candidate** (yes / no, and why)

---

## Stage 0 — Repository and environment (2026-10-23)

- **Request:** create the repository structure, environment files, README, `.gitignore` and
  `src/config.py` for a nested-CV project on the TCGA glioma dataset, following the decisions
  listed in `docs/DECISIONS.md`.
- **What the AI produced:** folder structure, `environment.yml` (conda, Python 3.11),
  `requirements.txt` with pinned versions, `src/config.py`, `src/check_env.py`, README,
  `.gitignore`, `.gitattributes`, `docs/DECISIONS.md`, this log.
- **Verification:**
  - The pinned package set was installed in a clean Python 3.11 environment by the AI and a
    `GridSearchCV` + `XGBClassifier` smoke test was run, to rule out the known incompatibility
    between some XGBoost and scikit-learn versions.
  - The dataset citation (authors, year, DOI 10.24432/C5R62J, CC BY 4.0, 839 instances) was
    checked on the UCI dataset page before entering `references.bib`.
  - The author runs `python -m src.check_env` on her computer; the result is stored in
    `results/logs/environment.json`.
- **Corrections:**
  - The AI first wrote "Code: MIT" in the README license section without being asked; removed,
    because the license of the code is the author's decision.
  - In `docs/DECISIONS.md` the AI first cited a student report (Engvall) as the source for a
    methodological choice and a course chapter it had not checked; both replaced by sources it had
    actually read (rule: student reports are format references only).
- **Bounty candidate:** no (minor, caught by the AI itself before delivery).

## Stage 0 (continued) — GitHub push (2026-10-23)

- **Issue:** `git push` returned 403 because Windows had cached the credentials of another GitHub
  account; after fixing it, the push was rejected (non-fast-forward) because the commit had been
  amended after an earlier successful push.
- **AI advice:** put the account name in the remote URL; check the commit author; before
  overwriting, inspect the remote (`git fetch`, `git log origin/main`) and compare contents
  (`git diff origin/main main --stat` was empty), then use `--force-with-lease` instead of `--force`.
- **Verification:** the author ran the commands; `git log` shows the correct author and the remote
  holds a single commit.
- **Bounty candidate:** no.

## Stage 1 — Data preparation (2026-10-23)

- **Request:** notebook `01_data_preparation` and `src/data.py`: load, check quality, clean and
  encode the data, avoiding leakage.
- **What the AI produced:** `src/data.py`, `src/reporting.py` (LaTeX tables and JSON summaries),
  `notebooks/01_data_preparation.ipynb`.
- **Verification:**
  - The AI reverse-engineered the UCI cleaning (which rows were removed and how age was converted)
    and the notebook asserts that the rebuilt dataset equals the UCI file in every value.
  - The AI executed the notebook in a clean copy of the repository before delivering it.
  - The author runs it on her computer; the AI reads `results/logs/01_data_preparation.json`.
- **Corrections:** none so far.
- **Bounty candidate:** no.

## Stage 2 — Exploratory data analysis (2026-10-23)

- **Request:** notebook `02_eda` following the structure of the course EDA notebook, adapted to
  binary mutations (Fisher's exact test), Mann–Whitney for age and multiple-comparison correction.
- **What the AI produced:** `src/eda.py` (tests, effect sizes, VIF, relevance, conditional
  probabilities), `src/evaluation.py` (rank-based AUC, bootstrap CI), `src/plotting.py`
  (style and colours), `src/pipelines.py` (preprocessing inside a Pipeline),
  `notebooks/02_eda.ipynb`.
- **Verification:** the AI executed the notebook in a clean copy and inspected every figure
  before delivering it. The author runs it and the AI compares her `results/logs/02_eda.json`
  with its own run.
- **Errors found by the AI in its own first version (fixed before delivery):**
  1. Crash in the error bars: Wilson intervals for P(GBM) = 0 give a lower bound that is
     negative by floating-point error (−6.9e-18 for NOTCH1, which has no mutated GBM patient); matplotlib rejects negative error lengths.
     Fixed by clipping the error lengths at 0.
  2. `KeyError: 'Feature'` after re-ordering a table: `reset_index()` named the column `index`
     because the index had lost its name. Fixed with `rename_axis("Feature")`.
  3. Misleading figures: (a) the ROC curves of single features used the class colours (blue and
     orange) for features, which suggests a link with LGG/GBM; (b) the bar chart of mutations per
     patient used a categorical axis that silently skipped empty counts (7, 8, 10…), hiding that
     a few patients have many mutated genes; (c) legends covered data points. All fixed and checked
     visually.
- **Bounty candidate:** 1 and 2 are ordinary bugs. 3(b) is a candidate example of a plausible but
  misleading visualisation produced by an AI (an axis that hides part of the distribution),
  detected by visual inspection. Caught by the AI itself, so the author should decide whether
  it qualifies.
- **Note for the defense:** the author should be able to explain why Fisher instead of chi-square,
  why Mann–Whitney instead of t-test, what Bonferroni and BH control, and why no feature is
  removed after the EDA.

## Stage 3 — Unsupervised analysis (2026-10-23)

- **Request:** notebook `03_unsupervised`: PCA (scaling, biplot, explained variance), MDS with
  Jaccard distance, k-means and hierarchical clustering of the mutation profiles, compared with
  the grade.
- **What the AI produced:** `src/unsupervised.py`, `notebooks/03_unsupervised.ipynb`; t-SNE added
  because the course summary presents it as a stronger visualisation than PCA.
- **Verification:** exploratory runs before writing the notebook (cluster sizes for every
  distance/linkage pair, k-means stability across seeds); full execution in a clean copy;
  every figure inspected.
- **Errors and decisions found by the AI in its own work (fixed before delivery):**
  1. **Distorted biplot.** The first version scaled the loading arrows with a different factor
     on each axis. This changes the direction of the arrows, so the plot showed wrong angles
     between genes and principal components (and some arrows fell outside the axes and were not
     drawn). Detected by visual inspection (IDH1, ATRX and TP53 arrows missing in the scaled
     biplot). Fixed with a single common factor. **Bounty candidate:** yes, as an example of a
     plausible-looking but mathematically wrong visualisation; the author should check that she
     can explain why a common factor is needed before claiming it.
  2. `TypeError: Object of type ndarray is not JSON serializable` when saving the summary; fixed
     in `src/reporting.py`.
  3. **Risk of choosing k after seeing the result.** The pre-declared rule (highest
     k-means–Ward agreement) gives k = 2, while k = 3 gives clusters that look more interesting.
     The AI kept the rule's answer as the main solution and reports k = 3 explicitly as
     exploratory, instead of switching the rule. Worth mentioning in the defense as a
     methodological choice (avoiding post-hoc selection).
- **Negative result reported, not hidden:** hierarchical clustering with Jaccard distance
  degenerates (one cluster with complete linkage; a cluster of the 67 patients without mutations
  with average/single linkage).

## Stage 4 — Regression (2026-10-23)

- **Request:** notebook `04_regression` with nested CV (10 outer / 3 inner folds, GridSearchCV),
  mean baseline, linear regression (VIF, residuals), Ridge, Lasso, visualised tree, random forest,
  XGBoost and SVR; bootstrap intervals and the overlap rule; final models refitted on all data.
- **What the AI produced:** `src/models.py` (pipelines and grids), nested CV, final-model fitting,
  bootstrap comparison and paired differences in `src/evaluation.py`, `notebooks/04_regression.ipynb`.
- **Verification:** grids timed before fixing them; full run in a clean copy; every figure
  inspected; the author's run is compared value by value with the AI's run.
- **Errors found by the AI in its own first version (fixed before delivery):**
  1. **Cook's distance = NaN**, so the summary reported "max Cook's distance = nan". Cause: the only
     American Indian / Alaska Native patient has leverage exactly 1 (a one-hot column that is 1
     for a single patient), so the model fits that patient exactly and the standardised residual is
     0/0 (statsmodels warned "invalid value encountered in sqrt"). Fixed by detecting leverage-1
     patients, reporting them and excluding them from the diagnostics. **Bounty candidate:** a good
     example of a silent statistical problem (a NaN that `max()` propagated and the plot dropped
     without warning) linked to course theory on high-leverage points.
  2. Hyperparameter grids printed as `np.float64(0.0316)` and chosen values with 16 decimals
     (NumPy 2 representation); fixed with a formatting helper.
  3. The Lasso path used matplotlib's default colours instead of the project palette, and the
     tree figure had unreadable text; both fixed.
- **Point for the defense:** with the grade excluded, the models still reach R² ≈ 0.4 mainly
  through IDH1/IDH2/ATRX, and their predictions correlate strongly with the grade: the regression
  partly recovers the grade through the mutations. This is not leakage, but it limits what the
  coefficients mean.

## Stage 5 — Classification (2026-10-23)

- **Request:** notebook `05_classification` with the same nested CV (stratified), logistic
  regression (L1/L2), Naive Bayes, pruned and visualised tree, random forest, XGBoost and SVM;
  AUC, sensitivity and specificity with bootstrap intervals.
- **What the AI produced:** classification pipelines and grids in `src/models.py`,
  `classification_metrics` and `search_flatness` in `src/evaluation.py`, `recolor_tree` in
  `src/plotting.py`, `notebooks/05_classification.ipynb`.
- **Verification:** full run in a clean copy; every figure inspected; the order of the tree node
  boxes was checked against the node sample counts before recolouring them.
- **Errors and issues found by the AI (fixed or documented before delivery):**
  1. **Inverted class colours in the tree plot.** `plot_tree(filled=True)` paints class 0 (LGG) in
     orange and class 1 (GBM) in blue, the opposite of every other figure, so a reader would
     misread the leaves. Fixed by recolouring the boxes. **Bounty candidate:** a silent,
     convention-level error that only visual cross-checking reveals.
  2. Grid table showed `age` instead of `age__n_bins` (the parameter name was split at the first
     `__`); fixed.
  3. Pruning-path plot showed a negative range of `ccp_alpha` (symmetric log axis); fixed.
  4. **Documented, not an error:** the final logistic search on all patients chose C = 100 (almost
     no penalty), a value no outer fold chose; the inner-CV surface is flat (see the flatness table),
     and with weak penalty NOTCH1, mutated only in LGG patients, gets a large coefficient
     (quasi-separation). The interpretation must not rely on that coefficient's size.
  5. **Documented:** the pooled AUC of the constant baseline is 0.496, not 0.5, because the
     constant probability differs slightly between folds.
  6. **Non-reproducible logistic regression.** Two runs of the notebook chose different
     penalties/C in the outer folds. Cause: `LogisticRegression(solver="liblinear")` shuffles the
     data internally and its `random_state` had not been set, although the project fixes seed 42
     everywhere. Detected by comparing two runs of the AI's own notebook; fixed by passing
     `random_state=42` and checked by running the nested CV twice (identical out-of-fold
     probabilities). **Bounty candidate:** strong — a reproducibility failure that a single run
     cannot reveal.
  7. **Wrong rule in the tree figure.** With `precision=0`, `plot_tree` rounded the split threshold
     0.5 to "IDH1 <= 0.0", a rule that reads as false (IDH1 takes values 0 or 1, and the split is
     0 vs 1). Detected by reading the figure against `export_text`; fixed with one decimal.
     **Bounty candidate:** a visualisation that silently states a wrong rule.

## Stage 6 — Literature search with Perplexity (2026-10-23)

- **Tool:** Perplexity (AI scientific assistant), used by the author.
- **Prompts:** (1) WHO 2021 classification, clinical differences LGG/GBM, role and co-occurrence of
  the markers, limitations of histological grading, molecular testing; (2) the UCI dataset paper,
  the TCGA LGG/GBM publications and ML studies on glioma grading with clinical/mutation data. Each
  prompt asked for DOI, open-access status, and the exact claim supported.
- **Problems detected in the output (to be confirmed by the author against the PDFs):**
  - Off-topic sources returned as supporting the requested points: a 12/15-lipoxygenase study in
    glioblastoma cell biology (12.pdf) and, twice, a cardiac epigenetic reprogramming paper
    (20.pdf, 27.pdf), unrelated to glioma.
  - Likely wrong attribution: Parsons et al. 2008 (Science) labelled as a TCGA publication
    (23.pdf); the 2008 TCGA glioblastoma paper is a different article.
  - Inconsistent metadata: one title given with two different first authors (21.pdf / 25.pdf), and
    a methylation-classification paper given with mixed authors (02.pdf).
  - Supplements of the same TCGA LGG paper listed as separate sources (10.pdf, 22.pdf).
- **Verification method:** comparing each returned title, author list and topic with the requested
  point; to be completed by reading the PDFs (first page and author list).
- **Bounty candidate:** yes (PRDL, AI scientific assistant): sources that do not support the
  claims they were returned for, and misattributed authorship.

## Stage 6 — Source inventory with NotebookLM (2026-10-23)

- **Tool:** NotebookLM, with all downloaded PDFs uploaded. Prompt: one table row per file with the
  bibliographic data *as printed in the document*, document type, topics covered, duplicates and
  unrelated files ("not stated" instead of guessing).
- **Use:** the AI decided which sources to keep from this table only (no PDF was read by the AI, to
  save tokens). Kept 25 files; removed duplicates (10, 22, 25, 45), off-topic files (12, 20, 27),
  a conference abstract (35), sources not used by the analysis (02, 03, 06, 15, 16, 32), and
  files with unreliable metadata (23 as a TCGA paper, 43, 44, 46, 47).
- **Errors in the NotebookLM output (to be confirmed by the author):**
  - Flagged a 2025 publication year as "a future publication year" — wrong, the inventory was made
    in October 2026 (the model's notion of the current date is outdated).
  - Assigned Parsons et al. 2008 (Science) to the topic "TCGA LGG or GBM project publications",
    repeating Perplexity's misattribution.
  - Marked all methods papers (nested CV, leakage, BH, Mann–Whitney, t-SNE, TRIPOD+AI, PROBAST) as
    "unrelated to glioma": literally true but not useful for the decision; the AI kept them.
  - Marked 12.pdf as related to glioma while assigning it no topic.
  - File 42.pdf is missing from the inventory (numbering jumps from 41 to 43).
- **Bounty candidate:** yes (PRDL): a temporal error ("future" 2025) and a repeated misattribution
  across two AI tools, detected by cross-checking the table against the current date and the author
  lists.

## Stage 6 — Drafting the reports (2026-10-25)

- **Request:** full drafts of the PRDL and MLLB reports in LaTeX, from the templates, with results
  read from `results/` and literature from the NotebookLM extraction (the AI does not read PDFs).
- **What the AI produced:** `report/prdl/` and `report/mllb/` (main.tex + sections),
  `report/common/references.bib`, `report/make_numbers.py`, `report/build_zips.py`,
  `report/check_references.py`.
- **Verification:** both reports compiled locally (pdflatex + bibtex, no errors, no undefined
  citations); every page rendered and inspected; numbers come from macros only.
- **Problems found in the AI tools' literature output and handled in the text:**
  - NotebookLM statements broader than their quotes (e.g. glioblastoma incidence/survival numbers
    attributed to a file whose quote only covers lower-grade glioma survival): only the quoted part
    used, the rest marked `[verify]`.
  - 21.pdf: title and authors given by Perplexity and NotebookLM ("Machine Learning and Statistical
    Evaluation of Glioma Datasets", Garbulowski & Komorowski) contradicted by NotebookLM's own
    extraction (Sánchez-Marqués et al.); the AI checked the DOAJ record, which confirms
    Sánchez-Marqués, García and Sánchez, Sci Rep 2024, doi:10.1038/s41598-024-68291-0.
  - 28.pdf: NotebookLM gave the Varma & Simon (2006) title as "Estimating Error Rates in Genomic Data
    Analysis: Nested Cross-Validation Versus Standard Cross-Validation"; the AI's knowledge gives
    "Bias in error estimation when using cross-validation for model selection". To be settled with
    `check_references.py`.
  - 42.pdf is cited by NotebookLM for the reclassification of TCGA cases (134 changed, 811
    unchanged) but is missing from the inventory: its bibliographic data are still needed.
- **Errors of the AI found while compiling (fixed):** generated tables printed `$\pm$` literally
  (escaping bug in `src/reporting.py`); several wide tables overflowed the margin; long `\texttt`
  paths ran into the margin.

## Stage 6 — Reference check with Crossref (2026-10-25)

- **Tool:** `report/check_references.py` (written by the AI, run by the author: the AI's
  environment cannot reach the Crossref API).
- **Result:** 29 of 38 DOIs matched title and year. Of the 9 flagged: tasci2022dataset is a DataCite
  DOI (404 in Crossref, verified on the UCI page); zakharova2023, weller2021 and efron1993 differ only
  in the year (online vs print / reprint; efron1993 changed to 1994 to match its DOI); yan2009,
  kaufman2012, james2021 and chen2016 differ only in title formatting (italics, subtitle).
- **Real errors found:**
  1. **sahm2023 had a wrong DOI** (10.1093/neuonc/noad108, given by NotebookLM from 17.pdf): Crossref
     resolves it to an unrelated article on CAR-T cells targeting CD317 in glioblastoma. Correct DOI
     10.1093/neuonc/noad100, verified on the EANO guideline page. **Bounty candidate:** strong — a
     wrong DOI would have pointed every reader to another paper.
  2. **varma2006 title:** NotebookLM gave "Estimating Error Rates in Genomic Data Analysis: Nested
     Cross-Validation Versus Standard Cross-Validation"; Crossref confirms "Bias in error estimation
     when using cross-validation for model selection".

## Stage 6 — Closing [verify] items with NotebookLM quotes (2026-10-25)

- **Prompt:** one item per open claim; file, page and verbatim quote (max 30 words) or "no source".
- **Used as quoted:** WHO 2021 three adult-type diffuse glioma types and grades (01.pdf); grade-4
  markers CDKN2A/B, TERT, EGFR amplification, +7/−10 (05.pdf, EANO 2021); glioblastoma 14.3 % of
  primary CNS tumours, median age 65, 5-year survival 6.8 % (04.pdf, CBTRUS 2014–2018, US data);
  treatment "predicated on tumor grade" (19.pdf); CIC/FUBP1/NOTCH1 in codeleted and TP53 (94 %) /
  ATRX (86 %) in non-codeleted IDH-mutant gliomas (09.pdf); hyperparameters of 21.pdf tuned on an
  80–20 stratified holdout (21.pdf).
- **Not used as stated by NotebookLM:**
  - 42.pdf: the quote "811 samples ... maintained consistent classification across both methods,
    while 134 cases ... changed" compares two methods; it is not clear that it compares the original
    TCGA labels with WHO 2021 labels, so the numbers were removed from the text (still [verify]).
  - "Neither paper performed nested CV" for 19.pdf: the supporting quote is about feature selection,
    not hyperparameters; left as [verify].
- **New finding:** 42.pdf (Leiria de Mendonça et al., Sci Data 2025) has the same DOI that the
  inventory attributed to 43.pdf with a different title ("MONET") and garbled authors: 43.pdf was
  probably a second copy whose metadata NotebookLM misread.
- **Errors of the AI in the bibliography (fixed):** (1) the checking notes were stored in the BibTeX
  `note` field, which the styles print in the reference list (and an underscore in one note broke
  the compilation); renamed to `comment`, which is not printed. (2) Author lists written as
  "First and others and Last" printed "First, others, and Last"; BibTeX requires `others` at the end.
  The first preview PDFs sent to the author contained both problems.

## Stage 6 — Second targeted query to NotebookLM (2026-10-25)

- **Prompt:** for 42.pdf, what exactly are the "two methods" of the 811/134 result and whether one is
  the original TCGA label; for 19.pdf, how the classifier hyperparameters were chosen.
- **42.pdf (p. 3, Results / Table 1):** the two methods are the authors' own 2016-based and
  2021-based re-annotation pipelines; neither is the original TCGA label. The numbers are now quoted
  in PRDL Section 2 with that caveat, not as an error rate of the TCGA labels.
- **19.pdf (p. 5, Section 2.1):** "The default values were assigned as the corresponding parameter
  values for the utilized classifiers". PRDL Section 6 now states that Tasci et al. did not tune
  hyperparameters.
- **Confirmed AI errors (Bounty):** both earlier extracted statements went beyond their quotes —
  the 811/134 count had been presented as a change from the original labels, and "no nested CV" had
  been supported by a feature-selection quote. The targeted query corrected both.
- **Human check pending:** the author reads the two quoted pages in the PDFs.

## Stage 7 — Multiclass task, SHAP and causal analysis (2026-10-25)

- **Request of the author:** include the multiclass classification, a SHAP analysis of the winning
  model(s) with a short discussion with the literature, and the causal analysis in this version, so
  that it is the complete work; notebooks may be changed and re-run.
- **Decisions asked to the author** (multiple choice): four histological families (chosen over three
  classes without mixed gliomas, or the six original diagnoses); SHAP on the winning models; a bounded
  causal analysis (diagram + mediation through the grade).
- **What the assistant built:** `src/explain.py`, `src/causal.py`, multiclass support in
  `src/evaluation.py` (`multiclass_auc`, `multiclass_metrics`, `task="multiclass"` in `nested_cv`) and
  `src/models.py` (`multiclass_models`); notebooks 06_multiclass, 07_shap and 08_causal; new macros in
  `report/make_numbers.py`; new text in both reports.
- **Verification:** `multiclass_metrics` compared with sklearn on random data (identical values);
  SHAP additivity checked for every model (maximum error ≤ 2e-6); the identity c = c' + ab checked
  (error < 1e-8); notebook 07 asserts that the refitted models have the hyperparameters saved by 05
  and 06; notebook 04 re-run: all previous values identical.
- **Error of the assistant (Bounty candidate) — interpretation of the regression.** In stage 4 the
  assistant had written that the regression predictions "mostly recover the grade through IDH1, IDH2
  and ATRX", based on a correlation of 0.71 between predicted age and grade, and the PRDL conclusion of
  H2 said that "the mutations explain age largely through the grade". The causal analysis estimated
  that only about a fifth of the IDH–age association goes through the grade, which contradicted that
  reading. Checking the alternative explanation: the correlation of the predictions with IDH status is
  −0.93, and within GBM the IDH-mutant patients are predicted about 20 years younger than the
  IDH-wildtype ones. **Correction:** notebook 04 now reports both correlations and the mean prediction
  by grade × IDH; the PRDL regression discussion, the conclusion of H2 and DECISIONS.md were rewritten.
  Lesson: a correlation between a prediction and a variable that was not an input does not show that
  the model uses that variable, when that variable is correlated with another one.
- **Errors of the assistant in figures (fixed before delivery):** the SHAP summary plots used the
  project's sequential palette, whose lowest value is almost white, so non-mutated patients were
  invisible; a long label overlapped the next panel; a plot title was lower-cased ("macro auc").
- **Dependency:** `shap` installed without pins pulled the newest numba; pinned shap 0.48.0 with numba
  0.61.2 and llvmlite 0.44.0, tested with numpy 2.2.6.
- **Literature still to be checked with NotebookLM:** WHO 2016 on oligoastrocytoma; WHO 2021 on
  IDH-wildtype diffuse astrocytic gliomas with glioblastoma features; previous feature-importance or
  SHAP results on these data. Marked `[verify]` in PRDL Section 6.
