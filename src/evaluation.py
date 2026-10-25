"""Evaluation utilities: rank-based ROC-AUC, bootstrap confidence intervals, nested
cross-validation, final-model fitting and model comparison.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from src import config


def auc_score(y_true, score):
    """ROC-AUC computed from ranks (Mann-Whitney formulation), with average ranks for ties.

    AUC = P(score of a random positive > score of a random negative) + 0.5 P(tie).
    Equivalent to sklearn.metrics.roc_auc_score, but fast enough to be bootstrapped.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(score)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(statistic, *arrays, n_boot=config.N_BOOTSTRAP, level=config.CI_LEVEL,
                 seed=config.SEED, stratify=None):
    """Percentile bootstrap confidence interval of statistic(*arrays).

    Patients are resampled with replacement (n_boot times) and the statistic is recomputed on
    each resample. If `stratify` is given (e.g. the class labels), resampling is done within
    each class so that every resample keeps the original class sizes.

    Returns (point_estimate, lower, upper).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = statistic(*arrays)

    if stratify is not None:
        stratify = np.asarray(stratify)
        groups = [np.flatnonzero(stratify == g) for g in np.unique(stratify)]

    values = np.empty(n_boot)
    for b in range(n_boot):
        if stratify is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values[b] = statistic(*[a[idx] for a in arrays])

    alpha = (1 - level) / 2
    lower, upper = np.nanquantile(values, [alpha, 1 - alpha])
    return point, lower, upper


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

REGRESSION_SCORING = "neg_root_mean_squared_error"   # inner-loop selection criterion


def regression_metrics(y_true, y_pred):
    """MAE and RMSE (in years) and R^2 = 1 - RSS/TSS.

    Written with numpy (same formulas as sklearn's mean_absolute_error, mean_squared_error and
    r2_score) because they are recomputed thousands of times in the bootstrap.
    """
    y_true = np.asarray(y_true, dtype=float)
    err = y_true - np.asarray(y_pred, dtype=float)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": float(1 - np.sum(err ** 2) / tss)}


CLASSIFICATION_SCORING = "roc_auc"                  # inner-loop selection criterion
THRESHOLD = 0.5


def classification_metrics(y_true, proba, threshold=THRESHOLD):
    """ROC-AUC (threshold-free) and, at the given probability threshold, accuracy, sensitivity
    (recall of GBM), specificity (recall of LGG) and balanced accuracy (their mean).

    GBM (class 1) is the positive class: sensitivity is the share of GBM patients detected.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1)); tn = np.sum((pred == 0) & (y_true == 0))
    fp = np.sum((pred == 1) & (y_true == 0)); fn = np.sum((pred == 0) & (y_true == 1))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"AUC": float(auc_score(y_true, proba)), "Accuracy": float((tp + tn) / len(y_true)),
            "Sensitivity": float(sens), "Specificity": float(spec),
            "Balanced accuracy": float((sens + spec) / 2)}


MULTICLASS_SCORING = "roc_auc_ovo"             # inner-loop criterion: macro one-vs-one AUC


def multiclass_auc(y_true, proba):
    """Macro one-vs-one AUC of Hand and Till: for every pair of classes (i, j), the AUC of
    P(class i) separating i from j and the AUC of P(class j) separating j from i are averaged;
    the result is the mean over all pairs. It does not depend on the class proportions.
    Same value as sklearn's roc_auc_score(multi_class="ovo", average="macro"), but fast enough
    to be bootstrapped (it reuses auc_score).
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    classes = np.arange(proba.shape[1])
    pair_aucs = []
    for a in classes:
        for b in classes[a + 1:]:
            mask = (y_true == a) | (y_true == b)
            if not mask.any():
                continue
            ya = (y_true[mask] == a).astype(int)
            pair_aucs.append((auc_score(ya, proba[mask, a]) + auc_score(1 - ya, proba[mask, b])) / 2)
    return float(np.mean(pair_aucs))


def multiclass_metrics(y_true, proba):
    """Macro AUC (Hand-Till), accuracy, balanced accuracy (mean recall over classes) and macro F1
    of the class with the highest predicted probability."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    k = proba.shape[1]
    pred = proba.argmax(axis=1)
    cm = np.bincount(k * y_true + pred, minlength=k * k).reshape(k, k)   # rows: true, columns: predicted
    tp = np.diag(cm).astype(float)
    recall = np.divide(tp, cm.sum(axis=1), out=np.full(k, np.nan), where=cm.sum(axis=1) > 0)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(k), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    return {"Macro AUC": multiclass_auc(y_true, proba), "Accuracy": float(tp.sum() / len(y_true)),
            "Balanced accuracy": float(np.nanmean(recall)), "Macro F1": float(np.mean(f1))}


def _clean_params(params):
    """'model__alpha' -> 'alpha', numpy scalars -> Python scalars."""
    out = {}
    for k, v in params.items():
        v = v.item() if hasattr(v, "item") else v
        out[k.split("__", 1)[-1]] = v
    return out


def make_cv(task, n_splits, seed=config.SEED):
    """Stratified folds for classification (same class proportions in every fold), plain
    shuffled folds for regression."""
    if task in ("classification", "multiclass"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------
def nested_cv(pipeline, grid, X, y, task, scoring, metric_fn, n_outer=config.N_OUTER_FOLDS,
              n_inner=config.N_INNER_FOLDS, importance=True, n_repeats=10):
    """Nested cross-validation (scheme of Example_ExtrOSA_ModelTuning_NestedLoop.ipynb).

    Outer loop (n_outer folds): each fold is held out once as a test set.
    Inner loop (n_inner folds, inside the outer training part only): GridSearchCV chooses the
    hyperparameters, and the best configuration is refitted on the whole outer training part.
    The refitted model predicts the held-out fold, which it has never seen in any way.

    Returns
    -------
    dict with
      oof        : out-of-fold prediction for every patient (probability of class 1 for
                   classification, an n x K matrix of class probabilities for multiclass), each
                   made by a model that did not see that patient;
      folds      : one row per outer fold (test metrics, inner CV score, chosen hyperparameters);
      importance : permutation importance of each original column on each outer test fold
                   (drop in the score when that column is shuffled), or None.
    """
    y = np.asarray(y)
    outer = make_cv(task, n_outer)
    # multiclass: one column of predicted probability per class (classes 0..K-1)
    oof = np.full((len(y), len(np.unique(y))), np.nan) if task == "multiclass" else np.full(len(y), np.nan)
    rows, importances = [], []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
        if grid:
            search = GridSearchCV(clone(pipeline), grid, scoring=scoring, cv=make_cv(task, n_inner),
                                  n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr)
            model, params, inner_score = search.best_estimator_, _clean_params(search.best_params_), search.best_score_
        else:
            model, params, inner_score = clone(pipeline).fit(X_tr, y_tr), {}, np.nan
        if task == "classification":
            pred = model.predict_proba(X_te)[:, 1]
        elif task == "multiclass":
            pred = model.predict_proba(X_te)
        else:
            pred = model.predict(X_te)
        oof[te] = pred
        rows.append({"Fold": k + 1, "n test": len(te), "Inner CV score": inner_score,
                     **metric_fn(y_te, pred), **params})
        if importance:
            pi = permutation_importance(model, X_te, y_te, scoring=scoring, n_repeats=n_repeats,
                                        random_state=config.SEED, n_jobs=1)
            importances.append(pi.importances_mean)
    imp = pd.DataFrame(importances, columns=X.columns) if importance else None
    return {"oof": oof, "folds": pd.DataFrame(rows), "importance": imp}


def fit_final(pipeline, grid, X, y, task, scoring, n_inner=config.N_INNER_FOLDS, return_search=False):
    """Final model for interpretation: the same tuning procedure (GridSearchCV with the inner
    CV) applied once to all patients, then refitted on all of them with the chosen values.

    The nested CV estimated how well this *procedure* generalises; this function applies it.
    """
    if not grid:
        out = (clone(pipeline).fit(X, y), {})
        return out + (None,) if return_search else out
    search = GridSearchCV(clone(pipeline), grid, scoring=scoring, cv=make_cv(task, n_inner),
                          n_jobs=-1, refit=True).fit(X, y)
    out = (search.best_estimator_, _clean_params(search.best_params_))
    return out + (search,) if return_search else out


def search_flatness(search, tolerance):
    """How flat is the inner-CV score surface? Number of configurations whose mean inner score is
    within `tolerance` of the best one, and the gap between the best and the 5th best."""
    scores = np.sort(search.cv_results_["mean_test_score"])[::-1]
    return {"Best inner score": scores[0],
            "5th best inner score": scores[min(4, len(scores) - 1)],
            f"Configurations within {tolerance} of the best": int(np.sum(scores >= scores[0] - tolerance)),
            "Configurations": len(scores)}


# ---------------------------------------------------------------------------
# Uncertainty and model comparison
# ---------------------------------------------------------------------------
def metric_cis(y, pred, metric_fn, stratify=None):
    """Bootstrap 95% CI of every metric of metric_fn, computed on the out-of-fold predictions.

    The same seed is used for every metric and every model, so all models are evaluated on the
    same bootstrap resamples of patients.
    """
    names = list(metric_fn(y, pred))
    out = {}
    for name in names:
        point, lo, hi = bootstrap_ci(lambda a, b, n=name: metric_fn(a, b)[n], y, pred, stratify=stratify)
        out[name] = (point, lo, hi)
    return out


def paired_difference_ci(y, pred_a, pred_b, metric, stratify=None):
    """Bootstrap CI of metric(model A) - metric(model B) on the same resampled patients.

    A paired comparison is more sensitive than checking whether two separate intervals overlap,
    because both models are evaluated on exactly the same patients in every resample.
    """
    return bootstrap_ci(lambda t, a, b: metric(t, a) - metric(t, b), y, pred_a, pred_b, stratify=stratify)


def best_and_ties(summary, metric, lower_is_better):
    """Model with the best point estimate and the models whose CI overlaps its CI.

    Rule used in this project: a model is not declared better than another if their 95%
    bootstrap intervals overlap.
    """
    s = summary.set_index("Model")
    best = s[metric].idxmin() if lower_is_better else s[metric].idxmax()
    lo_b, hi_b = s.loc[best, f"{metric} CI low"], s.loc[best, f"{metric} CI high"]
    ties = [m for m in s.index if m != best
            and not (s.loc[m, f"{metric} CI high"] < lo_b or s.loc[m, f"{metric} CI low"] > hi_b)]
    return best, ties
