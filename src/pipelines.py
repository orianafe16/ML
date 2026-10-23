"""Preprocessing pipelines shared by the EDA baseline and the models.

Everything that learns from the data (means and standard deviations for scaling, the list of
categories for one-hot encoding) lives inside a scikit-learn Pipeline. When the pipeline is
cross-validated, it is re-fitted on the training folds only, so no information from the test
fold reaches the training step.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NOMINAL_COLS = ["Race"]


def make_preprocessor(columns, scale=True):
    """Column-wise preprocessing for a list of predictor columns.

    - Age (continuous): standardised (mean 0, std 1) if `scale` is True.
    - Race (nominal): one-hot encoded (categories not seen in a training fold are ignored),
      then standardised like the other columns if `scale` is True.
    - Binary columns (gender, mutations, one-hot diagnosis): standardised too if `scale` is True,
      so that all coefficients of penalised linear models and SVMs are on the same scale
      (a penalty such as Lasso or Ridge treats every coefficient equally only if the inputs
      have the same scale).
    Tree-based models do not need scaling: use scale=False.
    """
    nominal = [c for c in columns if c in NOMINAL_COLS]
    other = [c for c in columns if c not in NOMINAL_COLS]
    transformers = []
    if other:
        transformers.append(("num", StandardScaler() if scale else "passthrough", other))
    if nominal:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("nominal", make_pipeline(onehot, StandardScaler()) if scale else onehot, nominal))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


RACE_NAMES = {"Race_0": "Race_white", "Race_1": "Race_black", "Race_2": "Race_asian", "Race_3": "Race_native"}


def readable_names(names):
    """Replace encoder names by readable ones: Race_0 ... Race_3 -> race labels, and the age bins
    of the Naive Bayes discretiser (Age_at_diagnosis_0.0 ...) -> 'Age bin 1 (youngest)' ..."""
    import re
    out = []
    for n in names:
        m = re.fullmatch(r"Age_at_diagnosis_(\d+)\.0", n)
        out.append(f"Age bin {int(m.group(1)) + 1}" if m else RACE_NAMES.get(n, n))
    return out


def format_value(v):
    """Short, readable representation of a hyperparameter value (for tables)."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
