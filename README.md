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
│   └── check_env.py     environment check
├── notebooks/           01_data_preparation … 05_classification (run in order)
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
(01 → 05). All random processes use the seed defined in `src/config.py` (`SEED = 42`).

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

This project was developed with AI assistance (Claude). Every interaction that shaped the code or
the reports, and every error detected in the AI output, is recorded in `docs/AI_USAGE_LOG.md`.

## Data license

The dataset is distributed under CC BY 4.0 by its original authors (see above).
