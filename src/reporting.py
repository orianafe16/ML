"""Writing results to disk: LaTeX tables and one JSON summary per stage.

The LaTeX files contain only the `tabular` environment (booktabs style). Captions and labels
are written in the report, so the same table can be used in the PRDL and the MLLB report.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src import config

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


_LATEX_SYMBOLS = {"±": r"$\pm$", "²": r"$^2$", "≥": r"$\geq$", "≤": r"$\leq$", "−": "-"}


def latex_escape(text):
    """Escape LaTeX special characters; a few Unicode symbols are mapped to math commands."""
    return "".join(_LATEX_SYMBOLS.get(ch, _LATEX_SPECIAL.get(ch, ch)) for ch in str(text))


def save_table_tex(df, name, float_format="{:.3f}", index=False, column_format=None):
    """Save a DataFrame as a booktabs `tabular` in results/tables/<name>.tex and return the path."""
    table = df.copy()
    if index:
        table = table.reset_index()

    def fmt(value):
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if value else "No"
        if isinstance(value, (float, np.floating)):
            return "--" if np.isnan(value) else float_format.format(value)
        return latex_escape(value)

    if column_format is None:
        column_format = "".join("r" if pd.api.types.is_numeric_dtype(table[c])
                                and not pd.api.types.is_bool_dtype(table[c]) else "l"
                                for c in table.columns)

    lines = [rf"\begin{{tabular}}{{{column_format}}}", r"\toprule",
             " & ".join(rf"\textbf{{{latex_escape(c)}}}" for c in table.columns) + r" \\",
             r"\midrule"]
    for _, row in table.iterrows():
        lines.append(" & ".join(fmt(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    path = config.TABLES / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_serialisable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return _to_serialisable(obj.to_dict())
    return obj


def save_json(summary, name):
    """Save the stage summary in results/logs/<name>.json (with a timestamp) and return the path."""
    payload = {"stage": name, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "seed": config.SEED, **_to_serialisable(summary)}
    path = config.LOGS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
