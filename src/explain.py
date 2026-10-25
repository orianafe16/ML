"""Model explanations with SHAP values (notebook 07).

SHAP (SHapley Additive exPlanations) splits one prediction of a fitted model into one contribution
per input, such that

    prediction = base value + sum of the contributions of all inputs,

where the base value is the average prediction. Each contribution is the Shapley value of a
cooperative game in which the inputs are the players: the average change in the prediction when
that input is added to every possible subset of the other inputs.

- Tree models (random forest, XGBoost): TreeSHAP computes the values exactly from the trees
  (path-dependent version: the inputs that are "absent" are integrated out following the training
  patients that went down each branch). Random forest values are in probability units; XGBoost
  values are in log-odds (its raw output).
- Logistic regression: for a linear model with independent inputs the Shapley value has a closed
  form, coefficient x (value - mean value), in log-odds units (shap.LinearExplainer).

SHAP values explain the *model*, not the disease: correlated inputs share credit in a way that
depends on the algorithm, and a large value is not evidence of a causal effect.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src import config
from src.pipelines import readable_names
from matplotlib.colors import LinearSegmentedColormap

from src.plotting import INK_SECONDARY, NEUTRAL

# Colour of the input value in the summary plots: grey (low / 0) to dark violet (high / 1). The class
# colours (blue LGG, orange GBM) are not reused, because here the colour encodes the input, not the class.
VALUE_CMAP = LinearSegmentedColormap.from_list("input_value", ["#b4b2a9", "#8a7ee0", "#2d1f8f"])
SHORT_NAMES = {"Age_at_diagnosis": "Age", "Race_white": "Race: white", "Race_black": "Race: black",
               "Race_asian": "Race: asian", "Race_native": "Race: native"}


def transformed_inputs(pipe, X):
    """The inputs as the estimator sees them (after the pipeline's preprocessing), with readable names."""
    prep = pipe.named_steps["prep"]
    return pd.DataFrame(prep.transform(X), columns=readable_names(prep.get_feature_names_out()), index=X.index)


def shap_values(pipe, X):
    """SHAP values of a fitted Pipeline('prep', 'model') on the patients X.

    Returns a dict with
      values : n x p array (binary: contribution to the output for class 1) or n x p x K (multiclass);
      base   : base value(s);
      inputs : transformed inputs (DataFrame);
      units  : "probability" or "log-odds";
      output : the model output that base + sum(values) must reproduce (used to check additivity).
    """
    model = pipe.named_steps["model"]
    Xt = transformed_inputs(pipe, X)
    n_classes = len(model.classes_)
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, Xt)            # background: all patients
        units = "log-odds"
        output = model.decision_function(Xt.values)
    else:
        explainer = shap.TreeExplainer(model)
        if type(model).__name__.startswith("XGB"):
            units = "log-odds"
            output = model.predict(Xt.values, output_margin=True)
        else:
            units = "probability"
            output = model.predict_proba(Xt.values)
    values = np.asarray(explainer.shap_values(Xt.values))
    base = np.asarray(explainer.expected_value, dtype=float)

    if n_classes == 2:
        if values.ndim == 3:                       # (n, p, 2): keep the contribution to class 1 (GBM)
            values = values[:, :, 1]
            base = base.ravel()[-1]
        if np.ndim(output) == 2:
            output = output[:, 1]
        base = float(np.ravel(base)[-1])
    else:
        if values.shape[0] == n_classes and values.ndim == 3 and values.shape[1] == len(Xt):
            values = np.transpose(values, (1, 2, 0))   # older layout (K, n, p) -> (n, p, K)
    return {"values": values, "base": base, "inputs": Xt, "units": units, "output": np.asarray(output)}


def additivity_error(sv):
    """Largest |base + sum of contributions - model output| over patients (and classes): should be ~0."""
    total = sv["base"] + sv["values"].sum(axis=1)
    return float(np.max(np.abs(total - sv["output"])))


def global_importance(sv):
    """Mean |SHAP value| per input (summed over classes for multiclass), and its share of the total,
    so that models with outputs in different units can be compared by their *ranking*."""
    v = np.abs(sv["values"])
    mean_abs = v.mean(axis=0) if v.ndim == 2 else v.mean(axis=0).sum(axis=1)
    out = pd.DataFrame({"Mean |SHAP|": mean_abs}, index=sv["inputs"].columns)
    out["Share"] = out["Mean |SHAP|"] / out["Mean |SHAP|"].sum()
    return out.sort_values("Mean |SHAP|", ascending=False)


def per_class_importance(sv, class_labels):
    """Multiclass: mean |SHAP value| of every input for every class (inputs x classes)."""
    v = np.abs(sv["values"]).mean(axis=0)
    return pd.DataFrame(v, index=sv["inputs"].columns, columns=[class_labels[k] for k in range(v.shape[1])])


def beeswarm(ax, values, inputs, top=10, title="", xlabel="SHAP value"):
    """SHAP summary plot: one row per input (the `top` inputs by mean |SHAP|), one dot per patient at
    its contribution, coloured by the value of the input (grey = low or 0, dark violet = high or 1).
    The vertical spread is a deterministic jitter, only to make overlapping dots visible."""
    order = np.argsort(np.abs(values).mean(axis=0))[::-1][:top][::-1]
    rng = np.random.default_rng(config.SEED)
    for row, j in enumerate(order):
        x = inputs.iloc[:, j].values.astype(float)
        span = x.max() - x.min()
        colour = (x - x.min()) / span if span > 0 else np.zeros_like(x)
        jitter = rng.uniform(-0.32, 0.32, len(x))
        ax.scatter(values[:, j], row + jitter, c=colour, cmap=VALUE_CMAP, vmin=0, vmax=1, s=7,
                   alpha=0.75, linewidths=0, rasterized=True)
    ax.axvline(0, color=NEUTRAL, lw=1)
    ax.set_yticks(range(len(order)), [SHORT_NAMES.get(inputs.columns[j], inputs.columns[j]) for j in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(axis="y", colors=INK_SECONDARY)
    return order
