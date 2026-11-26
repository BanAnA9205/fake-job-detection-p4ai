"""Shared style and writers for the report figures and LaTeX tables saved by the notebooks."""

import os

import matplotlib.pyplot as plt

# Paths are relative to code/, where every notebook runs.
IMG = "../materials/report/img"
INCLUDES = "../materials/report/includes"

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8c5bc4"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

STYLE = {
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
}


def save_figure(fig, name):
    """Save a vector report figure and keep it displayed in the notebook."""
    os.makedirs(IMG, exist_ok=True)
    fig.savefig(os.path.join(IMG, name), bbox_inches="tight")
    print("saved report figure", name)


def tex(value):
    """Escape text used inside a LaTeX table cell."""
    value = str(value)
    for old, new in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                     ("#", "\\#"), ("_", "\\_")):
        value = value.replace(old, new)
    return value


def write_table_rows(name, rows):
    """Write LaTeX table body rows (lists of already formatted cells) for \\tabinput."""
    os.makedirs(INCLUDES, exist_ok=True)
    with open(os.path.join(INCLUDES, name), "w", encoding="utf-8") as f:
        f.write("\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n")
    print("saved report table", name)


def use_style():
    """Apply the shared report style to every later Matplotlib figure."""
    plt.rcParams.update(STYLE)
