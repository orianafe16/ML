"""Model definitions and hyperparameter grids for regression (stage 4) and classification (stage 5).

Every model is a Pipeline (preprocessing + estimator), so that scaling and one-hot encoding are
re-fitted inside every training fold. Grids are kept small on purpose: the dataset has 839
patients and the goal is a sound comparison, not squeezing the last decimal.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src import config
from src.pipelines import make_preprocessor


def _pipe(columns, estimator, scale):
    return Pipeline([("prep", make_preprocessor(columns, scale=scale)), ("model", estimator)])


def regression_models(columns):
    """name -> (pipeline, parameter grid). An empty grid means no tuning (inner loop skipped).

    Scaling: linear models and SVR need it (penalties and distances depend on the scale of
    the inputs); trees and tree ensembles are invariant to monotonic transformations, so the
    inputs are passed unscaled.
    """
    seed = config.SEED
    return {
        "Mean baseline": (_pipe(columns, DummyRegressor(strategy="mean"), scale=False), {}),
        "Linear regression": (_pipe(columns, LinearRegression(), scale=True), {}),
        "Ridge": (_pipe(columns, Ridge(), scale=True),
                  {"model__alpha": np.logspace(-2, 3, 11)}),
        "Lasso": (_pipe(columns, Lasso(max_iter=50_000), scale=True),
                  {"model__alpha": np.logspace(-3, 1, 9)}),
        "Decision tree": (_pipe(columns, DecisionTreeRegressor(random_state=seed), scale=False),
                          {"model__max_depth": [2, 3, 4, 5, 6, 8],
                           "model__min_samples_leaf": [5, 10, 20, 40]}),
        "Random forest": (_pipe(columns, RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1),
                                scale=False),
                          {"model__max_features": [0.33, 0.66, 1.0],
                           "model__min_samples_leaf": [1, 5, 10, 20]}),
        "XGBoost": (_pipe(columns, XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist"), scale=False),
                    {"model__n_estimators": [100, 300],
                     "model__learning_rate": [0.03, 0.1],
                     "model__max_depth": [2, 3, 5]}),
        "SVR": (_pipe(columns, SVR(kernel="rbf"), scale=True),
                {"model__C": [1, 10, 100],
                 "model__epsilon": [0.5, 2, 5],
                 "model__gamma": ["scale", 0.01, 0.1]}),
    }


REGRESSION_GRID_NOTES = {
    "Ridge": "alpha: strength of the L2 penalty (larger = coefficients shrunk towards 0)",
    "Lasso": "alpha: strength of the L1 penalty (larger = more coefficients set exactly to 0)",
    "Decision tree": "max_depth and min_samples_leaf limit the size of the tree (pre-pruning against overfitting)",
    "Random forest": "max_features: share of features tried at each split (decorrelates the trees); "
                     "min_samples_leaf: smallest leaf; 300 trees fixed (more trees do not overfit, only cost time)",
    "XGBoost": "n_estimators x learning_rate (shrinkage) control how far boosting goes; max_depth: size of each tree",
    "SVR": "C: penalty of errors outside the tube; epsilon: half-width of the tube (years) in which errors "
           "are not penalised; gamma: reach of each support vector in the RBF kernel",
}


# ---------------------------------------------------------------------------
# Classification (stage 5)
# ---------------------------------------------------------------------------


def naive_bayes_preprocessor(columns):
    """Naive Bayes with binary features: age is cut into quantile bins and one-hot encoded,
    race is one-hot encoded, and the 0/1 columns are passed as they are. BernoulliNB then models
    every resulting column as a Bernoulli variable given the class."""
    age = [c for c in columns if c == "Age_at_diagnosis"]
    race = [c for c in columns if c == "Race"]
    binary = [c for c in columns if c not in age + race]
    transformers = [("binary", "passthrough", binary)]
    if race:
        transformers.append(("race", OneHotEncoder(handle_unknown="ignore", sparse_output=False), race))
    if age:
        transformers.append(("age", KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile"), age))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def classification_models(columns):
    """name -> (pipeline, parameter grid) for Grade (LGG = 0, GBM = 1)."""
    seed = config.SEED
    return {
        "Majority baseline": (_pipe(columns, DummyClassifier(strategy="prior"), scale=False), {}),
        "Logistic regression": (_pipe(columns, LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed), scale=True),
                                {"model__penalty": ["l1", "l2"], "model__C": np.logspace(-3, 2, 11)}),
        "Naive Bayes": (Pipeline([("prep", naive_bayes_preprocessor(columns)), ("model", BernoulliNB())]),
                        {"prep__age__n_bins": [3, 5, 8], "model__alpha": [0.1, 0.5, 1.0]}),
        "Decision tree (pruned)": (_pipe(columns, DecisionTreeClassifier(random_state=seed), scale=False),
                                   {"model__ccp_alpha": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                                    "model__criterion": ["gini", "entropy"]}),
        "Random forest": (_pipe(columns, RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
                                scale=False),
                          {"model__max_features": ["sqrt", 0.33, 0.66],
                           "model__min_samples_leaf": [1, 5, 10, 20]}),
        "XGBoost": (_pipe(columns, XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist",
                                                 eval_metric="logloss"), scale=False),
                    {"model__n_estimators": [100, 300],
                     "model__learning_rate": [0.03, 0.1],
                     "model__max_depth": [2, 3, 5]}),
        "SVM": (_pipe(columns, SVC(probability=True, random_state=seed), scale=True),
                [{"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1, 10]},
                 {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]}]),
    }


CLASSIFICATION_GRID_NOTES = {
    "Logistic regression": "penalty: L1 (Lasso-type, can set coefficients to 0) or L2 (Ridge-type); "
                           "C = 1/lambda (smaller C = stronger penalty)",
    "Naive Bayes": "n_bins: number of age intervals; alpha: Laplace smoothing of the Bernoulli probabilities",
    "Decision tree (pruned)": "ccp_alpha: cost-complexity pruning (larger = smaller tree); criterion: Gini or entropy (deviance)",
    "Random forest": "max_features: features tried at each split; min_samples_leaf: smallest leaf; 300 trees",
    "XGBoost": "n_estimators x learning_rate (shrinkage); max_depth of each tree",
    "SVM": "kernel: linear or RBF; C: penalty of margin violations; gamma: reach of each support vector (RBF)",
}


# ---------------------------------------------------------------------------
# Multiclass classification (notebook 06): histological family
# ---------------------------------------------------------------------------


def multiclass_models(columns):
    """name -> (pipeline, parameter grid) for the four histological families.

    The same model families and grids as the binary task where they extend directly to K classes:
    multinomial logistic regression (one coefficient vector per class, softmax link), a pruned
    tree, a random forest and XGBoost (one tree per class and boosting round). Naive Bayes and the
    SVM are left out: Naive Bayes was the worst-calibrated binary model, and a multiclass SVM with
    Platt probabilities needs one model per pair of classes, which multiplies the cost of the
    nested CV without a model of a new kind.
    """
    seed = config.SEED
    return {
        "Prior baseline": (_pipe(columns, DummyClassifier(strategy="prior"), scale=False), {}),
        "Logistic regression": (_pipe(columns, LogisticRegression(solver="saga", max_iter=20_000, random_state=seed),
                                      scale=True),
                                {"model__penalty": ["l1", "l2"], "model__C": np.logspace(-3, 2, 11)}),
        "Decision tree (pruned)": (_pipe(columns, DecisionTreeClassifier(random_state=seed), scale=False),
                                   {"model__ccp_alpha": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                                    "model__criterion": ["gini", "entropy"]}),
        "Random forest": (_pipe(columns, RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
                                scale=False),
                          {"model__max_features": ["sqrt", 0.33, 0.66],
                           "model__min_samples_leaf": [1, 5, 10, 20]}),
        "XGBoost": (_pipe(columns, XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist",
                                                 objective="multi:softprob", eval_metric="mlogloss"), scale=False),
                    {"model__n_estimators": [100, 300],
                     "model__learning_rate": [0.03, 0.1],
                     "model__max_depth": [2, 3, 5]}),
    }
