"""Project-wide configuration: paths, random seed, validation settings and variable names.

Every notebook and module imports its constants from here, so that a single change
(e.g. the number of outer folds) propagates to the whole project.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved from this file, so they work from any working directory)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
LOGS = RESULTS / "logs"

RAW_MUTATIONS_ALL = DATA_RAW / "TCGA_GBM_LGG_Mutations_all.csv"  # original labels, 862 rows
RAW_INFO_WITH_GRADE = DATA_RAW / "TCGA_InfoWithGrade.csv"        # UCI encoded version, 839 rows

for _folder in (DATA_PROCESSED, FIGURES, TABLES, LOGS):
    _folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Validation protocol (nested cross-validation, as in the course example
# Example_ExtrOSA_ModelTuning_NestedLoop.ipynb, with stratified folds)
# ---------------------------------------------------------------------------
N_OUTER_FOLDS = 10   # outer loop: estimates generalisation performance
N_INNER_FOLDS = 3    # inner loop: GridSearchCV hyperparameter selection
N_BOOTSTRAP = 2000   # resamples for bootstrap confidence intervals
CI_LEVEL = 0.95

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
ID_COL = "Case_ID"
CLASSIFICATION_TARGET = "Grade"            # LGG (0) vs GBM (1)
REGRESSION_TARGET = "Age_at_diagnosis"     # years
MULTICLASS_TARGET = "Primary_Diagnosis"    # reserved for the January version

# Columns that must never be used as predictors because they encode the
# classification target (Project = TCGA-LGG / TCGA-GBM is the same as Grade,
# and Primary_Diagnosis determines Grade).
LEAKAGE_COLS = ["Project", "Primary_Diagnosis"]

CLINICAL_COLS = ["Gender", "Age_at_diagnosis", "Race"]

GENE_COLS = [
    "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16", "PIK3CA", "NF1",
    "PIK3R1", "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A",
    "IDH2", "FAT4", "PDGFRA",
]

CLASS_LABELS = {0: "LGG", 1: "GBM"}

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIG_DPI = 300
FIG_FORMAT = "png"
