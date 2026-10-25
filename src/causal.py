"""Causal analysis (notebook 08): a causal diagram and a mediation decomposition with linear regression.

Question: how much of the association between an IDH mutation and a younger age at diagnosis
goes *through* the grade of the tumour?

Under the causal diagram of notebook 08 (IDH -> grade -> age at diagnosis, IDH -> age at
diagnosis, sex and race as possible common causes), three linear regressions estimate

    total effect   c  : age ~ IDH + covariates                 (grade NOT adjusted for)
    path IDH->grade a  : grade ~ IDH + covariates               (linear probability model)
    direct effect  c' : age ~ IDH + grade + covariates          (grade adjusted for)
    grade->age     b  : coefficient of grade in the last model

and, for ordinary least squares with the same patients and covariates, the total effect splits
exactly as c = c' + a * b (indirect effect a * b through the grade). Adjusting for the grade is
therefore not "better control": it removes the part of the effect that goes through the grade,
and answers a different question.

The decomposition is causal only under the assumptions of the diagram, in particular no
unmeasured common cause of the grade and the age at diagnosis; with observational data that
assumption cannot be checked.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config


def _ols_coefs(y, X):
    """OLS coefficients with an intercept (numpy only, fast enough for the bootstrap)."""
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0][1:]


def mediation_decomposition(df, exposure, mediator, outcome, covariates):
    """Total, direct and indirect effect of `exposure` on `outcome` through `mediator` (OLS).

    Returns a dict with c (total), a, b, c_prime (direct), indirect = a * b,
    proportion = indirect / total, and the check c - (c' + a b), which must be ~0.
    """
    y, m = df[outcome].to_numpy(float), df[mediator].to_numpy(float)
    e = df[[exposure]].to_numpy(float)
    C = df[covariates].to_numpy(float) if covariates else np.empty((len(df), 0))
    c = _ols_coefs(y, np.column_stack([e, C]))[0]
    a = _ols_coefs(m, np.column_stack([e, C]))[0]
    coef = _ols_coefs(y, np.column_stack([e, m, C]))
    c_prime, b = coef[0], coef[1]
    return {"total (c)": c, "IDH -> grade (a)": a, "grade -> age (b)": b, "direct (c')": c_prime,
            "indirect (a*b)": a * b, "proportion mediated": a * b / c, "check c - (c' + ab)": c - (c_prime + a * b)}


def bootstrap_mediation(df, exposure, mediator, outcome, covariates, n_boot=config.N_BOOTSTRAP,
                        seed=config.SEED, level=config.CI_LEVEL):
    """Percentile bootstrap CIs of every quantity of mediation_decomposition (patients resampled)."""
    rng = np.random.default_rng(seed)
    point = mediation_decomposition(df, exposure, mediator, outcome, covariates)
    keys = [k for k in point if not k.startswith("check")]
    draws = np.empty((n_boot, len(keys)))
    for i in range(n_boot):
        d = mediation_decomposition(df.iloc[rng.integers(0, len(df), len(df))], exposure, mediator, outcome, covariates)
        draws[i] = [d[k] for k in keys]
    alpha = (1 - level) / 2
    lo, hi = np.nanquantile(draws, [alpha, 1 - alpha], axis=0)
    return pd.DataFrame({"Estimate": [point[k] for k in keys], "CI low": lo, "CI high": hi}, index=keys)


def ols_table(df, outcome, predictors, robust="HC3"):
    """OLS fit with heteroscedasticity-robust standard errors (HC3), as a tidy table.

    The age residuals are not homoscedastic (notebook 04), so the classical standard errors would
    be wrong; HC3 standard errors do not assume equal variance.
    """
    fit = sm.OLS(df[outcome].astype(float), sm.add_constant(df[predictors].astype(float))).fit(cov_type=robust)
    ci = fit.conf_int()
    table = pd.DataFrame({"Coefficient": fit.params, "CI low": ci[0], "CI high": ci[1], "p-value": fit.pvalues})
    return fit, table


def _box_exit(center, target, box):
    """Point where the segment from `center` to `target` leaves the rectangle `box` (x0, y0, x1, y1)."""
    (cx, cy), (tx, ty) = center, target
    dx, dy = tx - cx, ty - cy
    ts = []
    for edge, d, c in ((box[0], dx, cx), (box[2], dx, cx)):
        if d:
            ts.append((edge - c) / d)
    for edge, d, c in ((box[1], dy, cy), (box[3], dy, cy)):
        if d:
            ts.append((edge - c) / d)
    t = min(t for t in ts if t > 0)
    return cx + t * dx, cy + t * dy


def draw_dag(ax, nodes, edges, dashed=()):
    """Draw a small causal diagram.

    nodes : {name: (x, y)} positions in axis units (0-1); a name starting with "U" is drawn dashed
            (unmeasured);
    edges : list of (from, to) arrows; `dashed` lists the edges drawn dashed (unmeasured paths).
    Arrows start and end at the border of each box, so no arrowhead is hidden behind a box.
    """
    from matplotlib.patches import FancyArrowPatch
    from src.plotting import INK, INK_SECONDARY
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    texts = {}
    for name, (x, y) in nodes.items():
        unobserved = name.startswith("U")
        texts[name] = ax.text(x, y, name, ha="center", va="center", fontsize=10,
                              color=INK_SECONDARY if unobserved else INK, zorder=3,
                              bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=INK_SECONDARY,
                                        ls="--" if unobserved else "-", lw=1.1))
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    boxes = {}
    for name, t in texts.items():
        (x0, y0), (x1, y1) = inv.transform(t.get_bbox_patch().get_window_extent().get_points())
        boxes[name] = (x0, y0, x1, y1)
    for a, b in edges:
        start = _box_exit(nodes[a], nodes[b], boxes[a])
        end = _box_exit(nodes[b], nodes[a], boxes[b])
        is_dashed = (a, b) in dashed
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, lw=1.3, zorder=2,
                                     color=INK_SECONDARY if is_dashed else INK, ls="--" if is_dashed else "-",
                                     shrinkA=3, shrinkB=3))
