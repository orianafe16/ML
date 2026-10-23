"""Common plotting style and helpers, so that every figure of the project looks the same.

Colour rules:
- one fixed colour per class in every figure (LGG blue, GBM orange);
- a single-hue ramp (blue) for magnitudes and a blue-grey-red ramp for signed values
  such as correlations (grey = 0);
- the palette was checked with a colour-blindness validator before use.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config

# Class colours (fixed for the whole project)
LGG_COLOR = "#2a78d6"
GBM_COLOR = "#eb6834"
CLASS_COLORS = {"LGG": LGG_COLOR, "GBM": GBM_COLOR, 0: LGG_COLOR, 1: GBM_COLOR}

# Extra categorical colours (clusters, model families), always used in this order
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Cluster colours: deliberately different from the class colours, so that a cluster is never
# confused with a grade (validated for colour-blind readers; always used with distinct markers)
CLUSTER_COLORS = ["#008300", "#6250d6", "#eda100", "#e87ba4"]
CLUSTER_MARKERS = ["o", "s", "^", "D"]

NEUTRAL = "#898781"     # reference lines, "no effect", chance level
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32d2c"])


def set_style():
    """Apply the project style to matplotlib (call once at the start of each notebook)."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": config.FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def save_figure(fig, name):
    """Save a figure to results/figures/<name>.png and return the path."""
    path = config.FIGURES / f"{name}.{config.FIG_FORMAT}"
    fig.savefig(path)
    return path


def class_legend_handles():
    """Legend handles for the two classes (used when a plot has no automatic legend)."""
    from matplotlib.patches import Patch
    return [Patch(color=LGG_COLOR, label="LGG"), Patch(color=GBM_COLOR, label="GBM")]


def recolor_tree(annotations, tree, kind):
    """Repaint the node boxes drawn by sklearn.tree.plot_tree with the project colours.

    sklearn's default colours do not follow the project convention (for a classifier it paints
    class 0 in orange). Node ids are assigned in depth-first pre-order, which is also the order in
    which plot_tree returns its annotations, so annotation i belongs to node i.
    - kind="classifier": blue (LGG) or orange (GBM) for the majority class, more opaque when the
      node is purer.
    - kind="regressor": single-hue blue ramp, darker for a higher predicted value.
    """
    import numpy as np
    from matplotlib.colors import Normalize, to_rgb

    def tint(color, strength):
        # opaque mix of the colour with white (opaque, so that the arrows do not show through)
        c = np.array(to_rgb(color))
        return tuple(1 - strength * (1 - c))

    values = tree.tree_.value[:, 0, :]
    boxes = [a for a in annotations if a.get_bbox_patch() is not None]
    if kind == "classifier":
        p_gbm = values[:, 1] / values.sum(axis=1)
        for box, p in zip(boxes, p_gbm):
            base = GBM_COLOR if p >= 0.5 else LGG_COLOR
            box.get_bbox_patch().set_facecolor(tint(base, 0.15 + 0.6 * abs(p - 0.5) * 2))
    else:
        norm = Normalize(values[:, 0].min(), values[:, 0].max())
        for box, v in zip(boxes, values[:, 0]):
            box.get_bbox_patch().set_facecolor(tint(SEQUENTIAL(0.15 + 0.55 * norm(v)), 1.0))
