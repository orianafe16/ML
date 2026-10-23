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
