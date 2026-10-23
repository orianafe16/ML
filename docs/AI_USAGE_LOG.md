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
