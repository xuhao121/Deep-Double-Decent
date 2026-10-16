"""Matplotlib defaults sized for the two-column ISBI template, and a save helper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import ROOT  # noqa: E402

FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
COL_W = 3.4     # inches, one ISBI column
FULL_W = 7.0    # inches, full page width

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 3,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,   # keeps twin axes, colorbars and labels from overlapping
})


def save(fig, name: str) -> None:
    """Save as PDF (for LaTeX) and PNG (for a quick look) under figures/."""
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  figure  -> figures/{name}.pdf")


def save_table(df, name: str) -> None:
    TAB_DIR.mkdir(exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    print(f"  table   -> tables/{name}.csv")
    try:
        text = df.to_markdown(index=False, floatfmt=".4g")   # needs the optional package `tabulate`
    except ImportError:
        text = df.to_string(index=False)
    (TAB_DIR / f"{name}.md").write_text(text + "\n")
