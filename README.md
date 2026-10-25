# Glioma grading and age prediction from clinical and mutation features

Machine Learning project for the courses **PRDL** (Predictive and Descriptive Learning) and
**MLLB** (Machine Learning Lab), Master in Signal Theory and Communications (MUTSC),
ETSI Telecomunicación, Universidad Politécnica de Madrid, 2026/27.

Author: Oriana Andrea Ferraz De Jesus

## Problem

Using the clinical and somatic-mutation profile of TCGA glioma patients, the project addresses:

| Task | Target | Notes |
|---|---|---|
| Classification | `Grade` (LGG = 0, GBM = 1) | `Project` and `Primary_Diagnosis` are excluded as predictors (they encode the target) |
| Regression | `Age_at_diagnosis` (years) | `Grade` is not used as a predictor |
| Unsupervised (within the EDA) | mutation profiles | PCA, MDS with Jaccard distance, k-means and hierarchical clustering, compared with `Grade` |
| Multiclass classification | histological family (astrocytoma, mixed glioma, oligodendroglioma, glioblastoma) from `Primary_Diagnosis` | `Grade` and `Project` are excluded as predictors |
| Model explanations | winning models of both classification tasks | SHAP values, compared with permutation importance |
| Causal analysis | `Age_at_diagnosis` | causal diagram and mediation of the IDH effect through `Grade` |

## Data

*Glioma Grading Clinical and Mutation Features* dataset, UCI Machine Learning Repository
(Tasci, Camphausen, Krauze and Zhuge, 2022), DOI [10.24432/C5R62J](https://doi.org/10.24432/C5R62J),
licensed under CC BY 4.0. It is derived from the TCGA-LGG and TCGA-GBM projects.

| File | Content |
|---|---|
| `data/raw/TCGA_GBM_LGG_Mutations_all.csv` | Original labels (862 patients, includes `Case_ID`, `Project`, `Primary_Diagnosis`) |
| `data/raw/TCGA_InfoWithGrade.csv` | Encoded version published by UCI (839 patients, 23 features + `Grade`) |

The raw files are never modified; every derived file is written to `data/processed/`.

## Repository structure

```
ML/
├── data/raw/            original CSV files (unchanged)
├── data/processed/      cleaned data produced by notebook 01 (not versioned)
├── src/                 shared code imported by the notebooks
│   ├── config.py        paths, seed, validation settings, variable names
│   ├── data.py          loading, cleaning, encoding, feature sets
│   ├── reporting.py     LaTeX tables and JSON summaries in results/
│   ├── eda.py           statistical tests and effect sizes for the EDA
│   ├── evaluation.py    nested CV, final models, metrics, bootstrap CIs, model comparison
│   ├── models.py        model pipelines and hyperparameter grids (regression, classification, multiclass)
│   ├── pipelines.py     preprocessing inside scikit-learn Pipelines
│   ├── unsupervised.py  PCA, Jaccard MDS / t-SNE, k-means, hierarchical clustering
│   ├── explain.py       SHAP values and summary plots
│   ├── causal.py        mediation decomposition, robust OLS and causal diagram
│   ├── plotting.py      common figure style and class colours
│   └── check_env.py     environment check
├── notebooks/           01_data_preparation … 08_causal (run in order)
├── results/             figures/, tables/ (LaTeX) and logs/ (one JSON summary per stage)
├── report/              LaTeX sources of the PRDL and MLLB reports
├── docs/                AI usage log and design decisions
├── environment.yml      conda environment (Python 3.11)
└── requirements.txt     pinned library versions
```

Every number and figure in the reports is read from `results/`; nothing is typed by hand.

## Setup

With Anaconda (from the repository root, in an Anaconda Prompt):

```bash
conda env create -f environment.yml
conda activate glioma-ml
python -m ipykernel install --user --name glioma-ml --display-name "Python (glioma-ml)"
python -m src.check_env
```

Without conda: create a Python 3.11 virtual environment and run `pip install -r requirements.txt`.

## Reproducing the results

Open the notebooks in `notebooks/` with the `Python (glioma-ml)` kernel and run them in order
(01 → 08). All random processes use the seed defined in `src/config.py` (`SEED = 42`).
Notebooks 04, 05 and 06 run nested cross-validation and take a few minutes each; notebook 07 reads
the results of 05 and 06, so it must run after them.

| Notebook | Content |
|---|---|
| `01_data_preparation` | cleaning, encoding, leakage check, comparison with the UCI file |
| `02_eda` | tests, effect sizes, correlations, VIF, leakage demonstration |
| `03_unsupervised` | PCA, Jaccard MDS / t-SNE, k-means and hierarchical clustering |
| `04_regression` | age at diagnosis: nested CV of 8 models, OLS diagnostics |
| `05_classification` | LGG vs GBM: nested CV of 7 models |
| `06_multiclass` | histological family: nested CV of 5 models, comparison with the clusters |
| `07_shap` | SHAP values of the winning models of 05 and 06 |
| `08_causal` | causal diagram and mediation analysis |

To run everything from scratch without opening the notebooks (from the repository root, with the
`glioma-ml` environment active):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb notebooks/02_eda.ipynb notebooks/03_unsupervised.ipynb notebooks/04_regression.ipynb notebooks/05_classification.ipynb notebooks/06_multiclass.ipynb notebooks/07_shap.ipynb notebooks/08_causal.ipynb
```

Every figure, table and number used in the reports is regenerated in `results/`.

## Building the reports

```bash
python report/make_numbers.py        # numbers.tex from results/logs/*.json
python report/build_zips.py          # report/dist/PRDL_Overleaf.zip and MLLB_Overleaf.zip
python report/check_references.py    # checks every DOI of references.bib against Crossref
```

Upload each zip to its Overleaf project (New Project > Upload Project). No number is typed by hand
in the reports: every value comes from `numbers.tex` or from a table generated by the notebooks.
The macro `\todoverify{}` marks in red any claim still to be checked against a source.

## Methodology in brief

- Nested cross-validation on the full dataset: 10 stratified outer folds to estimate
  performance and 3 inner folds with `GridSearchCV` to select hyperparameters.
- All preprocessing (scaling, encoding) is fitted inside a scikit-learn `Pipeline`,
  so no information from a test fold reaches the training step.
- Classification metrics: ROC-AUC, accuracy, sensitivity, specificity; regression metrics:
  MAE, RMSE, R², always compared with a mean-prediction baseline.
- 95% bootstrap confidence intervals; a model is not declared better than another when
  their intervals overlap.

Design decisions and their justification are listed in `docs/DECISIONS.md`.

## Use of AI assistants

This project was developed with AI assistance: Claude for the code and the report drafts,
Perplexity for the literature search and NotebookLM for extracting quoted claims from the papers.
Every interaction that shaped the code or the reports, and every error detected in the AI output,
is recorded in `docs/AI_USAGE_LOG.md`; the literature PDFs are not distributed (copyright).

## Data license

The dataset is distributed under CC BY 4.0 by its original authors (see above).
