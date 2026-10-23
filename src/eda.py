"""Statistics used in the exploratory data analysis (notebook 02).

Test choice:
- Age (continuous, not normal within class) -> Mann-Whitney U (Welch t-test reported for
  comparison), effect sizes Cohen's d and AUC = U / (n1 n0).
- Binary features (gender, mutations) -> Fisher's exact test on the 2x2 table, effect sizes
  odds ratio and phi coefficient.
- Race (4 nominal categories with very small counts) -> Fisher's exact test on white vs non-white.
- Multiple comparisons -> Bonferroni (family-wise error) and Benjamini-Hochberg (false discovery rate).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

from src import config
from src.evaluation import auc_score, bootstrap_ci


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def describe_by_class(df, col, target="Grade"):
    out = df.groupby(df[target].map(config.CLASS_LABELS))[col].describe()
    out["skewness"] = df.groupby(df[target].map(config.CLASS_LABELS))[col].skew()
    return out.loc[["LGG", "GBM"]]


def shapiro_by_class(df, col, target="Grade"):
    rows = []
    for code, label in config.CLASS_LABELS.items():
        x = df.loc[df[target] == code, col]
        w, p = stats.shapiro(x)
        rows.append({"Class": label, "n": len(x), "Shapiro-Wilk W": w, "p-value": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests between classes
# ---------------------------------------------------------------------------
def continuous_test(df, col, target="Grade"):
    """Welch t-test, Mann-Whitney U, Cohen's d and AUC for a continuous feature (GBM vs LGG)."""
    x1 = df.loc[df[target] == 1, col].values   # GBM
    x0 = df.loc[df[target] == 0, col].values   # LGG
    t = stats.ttest_ind(x1, x0, equal_var=False)
    u = stats.mannwhitneyu(x1, x0, alternative="two-sided")
    pooled_sd = np.sqrt(((len(x1) - 1) * x1.var(ddof=1) + (len(x0) - 1) * x0.var(ddof=1))
                        / (len(x1) + len(x0) - 2))
    return {
        "Feature": col, "Test": "Mann-Whitney U",
        "LGG": f"median {np.median(x0):.1f}", "GBM": f"median {np.median(x1):.1f}",
        "Effect size": f"d = {(x1.mean() - x0.mean()) / pooled_sd:.2f}",
        "AUC": u.statistic / (len(x1) * len(x0)),
        "p-value": u.pvalue,
        "Welch t p-value": t.pvalue,
        "Cohen d": (x1.mean() - x0.mean()) / pooled_sd,
    }


def _odds_ratio(table):
    """Odds ratio with Haldane-Anscombe correction (+0.5 to every cell) and 95% Woolf CI."""
    a, b, c, d = (table + 0.5).ravel()
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(0.975)
    return np.exp(log_or), np.exp(log_or - z * se), np.exp(log_or + z * se)


def binary_test(df, col, target="Grade", positive=1, label=None):
    """Fisher's exact test of a binary feature against the class.

    2x2 table rows: feature present / absent; columns: GBM / LGG.
    Odds ratio > 1 means that the feature is more frequent in GBM.
    """
    present = df[col] == positive
    y = df[target] == 1
    table = np.array([[np.sum(present & y), np.sum(present & ~y)],
                      [np.sum(~present & y), np.sum(~present & ~y)]])
    _, p = stats.fisher_exact(table)
    or_, lo, hi = _odds_ratio(table)
    phi = np.corrcoef(present.astype(int), y.astype(int))[0, 1]
    prev_lgg = present[~y].mean()
    prev_gbm = present[y].mean()
    return {
        "Feature": label or col, "Test": "Fisher exact",
        "LGG": f"{100 * prev_lgg:.1f}%", "GBM": f"{100 * prev_gbm:.1f}%",
        "Effect size": f"OR = {or_:.2f}",
        "AUC": auc_score(y.astype(int), present.astype(int)),
        "p-value": p,
        "Odds ratio": or_, "OR CI low": lo, "OR CI high": hi, "phi": phi,
        "Prevalence LGG": prev_lgg, "Prevalence GBM": prev_gbm,
    }


def adjust_pvalues(pvalues):
    """Bonferroni and Benjamini-Hochberg adjusted p-values."""
    pvalues = np.asarray(pvalues)
    bonf = multipletests(pvalues, method="bonferroni")[1]
    bh = multipletests(pvalues, method="fdr_bh")[1]
    return bonf, bh


def classification_tests(df, gene_cols, target="Grade"):
    rows = [continuous_test(df, "Age_at_diagnosis", target),
            binary_test(df, "Gender", target, positive=1, label="Gender (female)"),
            binary_test(df.assign(Race_nonwhite=(df["Race"] != 0).astype(int)), "Race_nonwhite",
                        target, positive=1, label="Race (non-white)")]
    rows += [binary_test(df, g, target) for g in gene_cols]
    table = pd.DataFrame(rows)
    table["p Bonferroni"], table["p BH"] = adjust_pvalues(table["p-value"])
    return table


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def iqr_outliers(df, col, target="Grade"):
    """Tukey's rule per class: below Q1 - 1.5 IQR or above Q3 + 1.5 IQR."""
    rows = []
    for code, label in config.CLASS_LABELS.items():
        x = df.loc[df[target] == code, col]
        q1, q3 = x.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        out = x[(x < lo) | (x > hi)]
        rows.append({"Class": label, "Lower fence": lo, "Upper fence": hi,
                     "n outliers": len(out), "Outlier values": ", ".join(f"{v:.1f}" for v in sorted(out))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Redundancy between features
# ---------------------------------------------------------------------------
def top_correlations(corr, n=10):
    """Largest absolute off-diagonal correlations, one row per pair."""
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().rename("r").reset_index()
    pairs.columns = ["Feature 1", "Feature 2", "r"]
    return pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)


def vif_table(X):
    """Variance inflation factor of each column: VIF_j = 1 / (1 - R_j^2).

    A constant column is added so that each auxiliary regression has an intercept.
    """
    Xc = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    vif = [variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])]
    return pd.DataFrame({"Feature": X.columns, "VIF": vif}).sort_values("VIF", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Relevance of each feature for the class
# ---------------------------------------------------------------------------
def feature_relevance(df, features, target="Grade", continuous=("Age_at_diagnosis",)):
    """Single-feature ROC-AUC with bootstrap CI, its orientation, and mutual information."""
    y = df[target].values
    discrete = np.array([f not in continuous for f in features])
    mi = mutual_info_classif(df[features].values, y, discrete_features=discrete, random_state=config.SEED)
    rows = []
    for f, m in zip(features, mi):
        auc, lo, hi = bootstrap_ci(auc_score, y, df[f].values, stratify=y)
        flip = auc < 0.5
        rows.append({
            "Feature": f,
            "Direction": "lower in GBM" if flip else "higher in GBM",
            "AUC": 1 - auc if flip else auc,
            "AUC CI low": 1 - hi if flip else lo,
            "AUC CI high": 1 - lo if flip else hi,
            "Mutual information (nats)": m,
        })
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def youden_threshold(x, y):
    """Threshold on x (predict GBM if x >= t) that maximises J = sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, x)
    j = tpr - fpr
    k = int(np.argmax(j))
    return {"threshold": float(thr[k]), "sensitivity": float(tpr[k]),
            "specificity": float(1 - fpr[k]), "youden_J": float(j[k])}


def conditional_probability_by_quantile(df, col, target="Grade", q=5):
    """P(GBM | quantile group of col) with 95% Wilson confidence intervals."""
    groups = pd.qcut(df[col], q=q)
    rows = []
    for interval, sub in df.groupby(groups, observed=True):
        k, n = int(sub[target].sum()), len(sub)
        lo, hi = proportion_confint(k, n, method="wilson")
        rows.append({"Group": str(interval), "Midpoint": sub[col].median(), "n": n,
                     "P(GBM)": k / n, "CI low": lo, "CI high": hi})
    return pd.DataFrame(rows)


def conditional_probability_binary(df, cols, target="Grade"):
    """P(GBM | feature = 1) and P(GBM | feature = 0) with 95% Wilson intervals."""
    rows = []
    for c in cols:
        row = {"Feature": c}
        for value, name in ((1, "mutated"), (0, "not mutated")):
            sub = df[df[c] == value]
            k, n = int(sub[target].sum()), len(sub)
            lo, hi = proportion_confint(k, n, method="wilson")
            row.update({f"n {name}": n, f"P(GBM | {name})": k / n,
                        f"CI low ({name})": lo, f"CI high ({name})": hi})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EDA for the regression target (age)
# ---------------------------------------------------------------------------
def age_by_binary(df, cols, age="Age_at_diagnosis"):
    """Mann-Whitney test of age between patients with and without each binary feature.

    Effect sizes: difference of medians (years) and rank-biserial correlation
    r = 2 P(age_with > age_without) - 1 (from -1 to 1, 0 = no difference).
    """
    rows = []
    for c in cols:
        a1 = df.loc[df[c] == 1, age].values
        a0 = df.loc[df[c] == 0, age].values
        u = stats.mannwhitneyu(a1, a0, alternative="two-sided")
        rows.append({"Feature": c, "n with": len(a1), "Median age with": np.median(a1),
                     "Median age without": np.median(a0),
                     "Median difference": np.median(a1) - np.median(a0),
                     "Rank-biserial r": 2 * u.statistic / (len(a1) * len(a0)) - 1,
                     "p-value": u.pvalue})
    table = pd.DataFrame(rows)
    table["p Bonferroni"], table["p BH"] = adjust_pvalues(table["p-value"])
    return table.sort_values("p-value").reset_index(drop=True)
