"""Data loading, cleaning and encoding.

The cleaned dataset is built from the original-label file (TCGA_GBM_LGG_Mutations_all.csv)
so that the patient identifier and the histological diagnosis are kept, and it is then checked
value by value against the encoded file published by UCI (TCGA_InfoWithGrade.csv).
"""
import re

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Encodings (identical to the ones used by UCI in TCGA_InfoWithGrade.csv)
# ---------------------------------------------------------------------------
GRADE_MAP = {"LGG": 0, "GBM": 1}
GENDER_MAP = {"Male": 0, "Female": 1}
RACE_MAP = {
    "white": 0,
    "black or african american": 1,
    "asian": 2,
    "american indian or alaska native": 3,
}
MUTATION_MAP = {"NOT_MUTATED": 0, "MUTATED": 1}

MISSING_MARKER = "--"            # empty field in the TCGA export
RACE_NOT_REPORTED = "not reported"
DAYS_PER_YEAR = 365              # convention that reproduces the UCI ages exactly


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw_mutations():
    """Original-label file (862 patients). `utf-8-sig` removes the byte-order mark of the file."""
    return pd.read_csv(config.RAW_MUTATIONS_ALL, encoding="utf-8-sig")


def load_uci_encoded():
    """Encoded file published by UCI (839 patients)."""
    return pd.read_csv(config.RAW_INFO_WITH_GRADE)


def load_processed():
    """Clean dataset produced by notebooks/01_data_preparation.ipynb."""
    path = config.DATA_PROCESSED / "glioma_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run notebooks/01_data_preparation.ipynb first.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def count_invalid_markers(df):
    """Number of '--' and 'not reported' entries per column (only columns that have any)."""
    counts = pd.DataFrame({
        "missing ('--')": (df == MISSING_MARKER).sum(),
        "'not reported'": (df == RACE_NOT_REPORTED).sum(),
        "NaN": df.isna().sum(),
    })
    return counts[counts.sum(axis=1) > 0]


def parse_age(text):
    """Convert 'X years Y days' (or 'X years') to years as a float. Returns NaN if missing."""
    if not isinstance(text, str) or text.strip() == MISSING_MARKER:
        return np.nan
    match = re.fullmatch(r"(\d+) years(?: (\d+) days)?", text.strip())
    if match is None:
        raise ValueError(f"Unexpected age format: {text!r}")
    years = int(match.group(1))
    days = int(match.group(2) or 0)
    return years + days / DAYS_PER_YEAR


def removal_reasons(df):
    """Boolean table with one column per exclusion reason, evaluated in order.

    A patient is assigned only to the first reason that applies, so the counts add up.
    """
    all_missing = (df[["Gender", "Race", "Primary_Diagnosis"]] == MISSING_MARKER).all(axis=1)
    race_not_reported = (df["Race"] == RACE_NOT_REPORTED) & ~all_missing
    age_missing = (df["Age_at_diagnosis"] == MISSING_MARKER) & ~all_missing & ~race_not_reported
    return pd.DataFrame({
        "All clinical fields missing": all_missing,
        "Race not reported": race_not_reported,
        "Age at diagnosis missing": age_missing,
    })


# ---------------------------------------------------------------------------
# Cleaning and encoding
# ---------------------------------------------------------------------------
def clean_and_encode(raw):
    """Remove incomplete patients, parse age and encode every variable as integers.

    Returns
    -------
    clean : DataFrame with Case_ID, Primary_Diagnosis (metadata), Grade, clinical and gene columns.
    removal_table : DataFrame with the number of patients removed and remaining at each step.
    removed : DataFrame with the removed patients and the reason (for inspection).
    """
    reasons = removal_reasons(raw)

    rows, remaining = [], len(raw)
    rows.append({"Step": "Original file", "Removed": 0, "Remaining": remaining})
    for reason in reasons.columns:
        n = int(reasons[reason].sum())
        remaining -= n
        rows.append({"Step": reason, "Removed": n, "Remaining": remaining})
    removal_table = pd.DataFrame(rows)

    removed_mask = reasons.any(axis=1)
    removed = raw[removed_mask].copy()
    removed["Reason"] = reasons[removed_mask].idxmax(axis=1)

    kept = raw[~removed_mask].reset_index(drop=True)

    clean = pd.DataFrame({
        config.ID_COL: kept[config.ID_COL],
        config.MULTICLASS_TARGET: kept[config.MULTICLASS_TARGET],
        "Grade": kept["Grade"].map(GRADE_MAP),
        "Gender": kept["Gender"].map(GENDER_MAP),
        "Age_at_diagnosis": kept["Age_at_diagnosis"].map(parse_age).round(2),
        "Race": kept["Race"].map(RACE_MAP),
    })
    for gene in config.GENE_COLS:
        clean[gene] = kept[gene].map(MUTATION_MAP)

    if clean.isna().any().any():
        bad = clean.columns[clean.isna().any()].tolist()
        raise ValueError(f"Unmapped values remain after encoding in columns: {bad}")

    int_cols = ["Grade", "Gender", "Race"] + config.GENE_COLS
    clean[int_cols] = clean[int_cols].astype(int)
    return clean, removal_table, removed


def check_leakage_columns(raw):
    """Check that `Project` is the classification target under another name."""
    project_as_grade = raw["Project"].str.replace("TCGA-", "", regex=False)
    return bool((project_as_grade == raw["Grade"]).all())


def verify_against_uci(clean, uci):
    """Compare the clean dataset with the UCI encoded file, column by column and row by row."""
    checks = {"Same number of patients": len(clean) == len(uci)}
    cols = [c for c in uci.columns if c != "Age_at_diagnosis"]
    if checks["Same number of patients"]:
        for col in cols:
            checks[f"{col} identical"] = bool((clean[col].values == uci[col].values).all())
        checks["Age_at_diagnosis identical (2 decimals)"] = bool(
            np.allclose(clean["Age_at_diagnosis"].values, uci["Age_at_diagnosis"].values, atol=1e-9)
        )
    return checks


def save_processed(clean):
    path = config.DATA_PROCESSED / "glioma_clean.csv"
    clean.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Feature sets for each task
# ---------------------------------------------------------------------------
def feature_columns(task):
    """Predictor columns for each task. Leakage columns are never included.

    task = "classification": target Grade; predictors = Gender, Age, Race, 20 genes.
    task = "regression": target Age_at_diagnosis; predictors = Gender, Race, 20 genes (no Grade).
    task = "unsupervised": the 20 gene mutation indicators.
    """
    if task == "classification":
        return ["Gender", "Age_at_diagnosis", "Race"] + config.GENE_COLS
    if task == "regression":
        return ["Gender", "Race"] + config.GENE_COLS
    if task == "unsupervised":
        return list(config.GENE_COLS)
    raise ValueError(f"Unknown task: {task}")


def data_dictionary():
    """Description of every variable of the clean dataset, with its role in the project."""
    genes = ", ".join(config.GENE_COLS)
    return pd.DataFrame([
        {"Variable": "Case_ID", "Type": "Identifier", "Encoding": "TCGA patient barcode",
         "Role": "Identifier (not a predictor)"},
        {"Variable": "Primary_Diagnosis", "Type": "Nominal (6 classes)", "Encoding": "Histological diagnosis (text)",
         "Role": "Excluded: determines Grade (leakage)"},
        {"Variable": "Project", "Type": "Nominal (2 classes)", "Encoding": "TCGA-LGG / TCGA-GBM",
         "Role": "Dropped: identical to Grade"},
        {"Variable": "Grade", "Type": "Binary", "Encoding": "0 = LGG, 1 = GBM",
         "Role": "Classification target"},
        {"Variable": "Age_at_diagnosis", "Type": "Continuous", "Encoding": "Years (years + days/365, 2 decimals)",
         "Role": "Regression target; predictor for classification"},
        {"Variable": "Gender", "Type": "Binary", "Encoding": "0 = male, 1 = female", "Role": "Predictor"},
        {"Variable": "Race", "Type": "Nominal (4 classes)",
         "Encoding": "0 = white, 1 = black or African American, 2 = Asian, 3 = American Indian or Alaska Native",
         "Role": "Predictor (one-hot encoded inside the model pipelines)"},
        {"Variable": f"20 genes ({genes})", "Type": "Binary", "Encoding": "0 = not mutated, 1 = mutated",
         "Role": "Predictors; input of the unsupervised analysis"},
    ])
