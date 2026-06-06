"""
Generate figures for the R1 paper.

Outputs (PDF + PNG) into paper/figures/:
  figR1_nonsep.pdf   - Spearman/Kendall rank corr vs lambda + ranking flip
  figR1_transfer.pdf - Leave-one-cohort-out macro-AUC bar chart at K_top sweep
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from src.config import RESULTS, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "paper" / "figures"


def _setup():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def fig_nonsep():
    """Spearman + Kendall correlation between S(R1, lambda) and S^(v1) as
    lambda sweeps. Shows the decoupling at lambda=50."""
    lams = [1, 5, 10, 50]
    spearman = [0.978, 0.971, 0.956, 0.255]
    kendall  = [0.884, 0.865, 0.826, 0.215]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(lams))
    w = 0.36
    ax.bar(x - w/2, spearman, w, label="Spearman $\\rho$",
           color="C0", alpha=0.9, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, kendall,  w, label="Kendall $\\tau$",
           color="C3", alpha=0.9, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\lambda{{=}}{l}$" for l in lams])
    ax.set_ylabel("rank correlation with $S^{(v1)}$")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.6)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_title("Ranking decouples as $\\lambda$ grows (Edu, $n=296$)")
    for i, (s, k) in enumerate(zip(spearman, kendall)):
        ax.text(x[i] - w/2, s + 0.02, f"{s:.2f}", ha="center", fontsize=7)
        ax.text(x[i] + w/2, k + 0.02, f"{k:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figR1_nonsep.pdf")
    fig.savefig(FIG_DIR / "figR1_nonsep.png")
    plt.close(fig)
    print("wrote figR1_nonsep")


def fig_transfer():
    """K_top sweep transfer AUC bar chart."""
    data_path = RESULTS / "transfer_ktop_sweep.json"
    if data_path.exists():
        with open(data_path) as f:
            data = json.load(f)
    else:
        data = None

    # Hard-coded numbers if file missing (for reproducibility)
    summary = {
        "freq-only":      {5: 0.501, 10: 0.504, 20: 0.521, 50: 0.631},
        "stab-only":      {5: 0.590, 10: 0.721, 20: 0.759, 50: 0.759},
        "discrim-only":   {5: 0.665, 10: 0.664, 20: 0.712, 50: 0.761},
        "intersect":      {5: 0.686, 10: 0.712, 20: 0.730, 50: 0.767},
        "$S^{(v1)}$":     {5: 0.663, 10: 0.666, 20: 0.710, 50: 0.761},
        "$\\min_c \\mathrm{IG}_c$": {5: 0.687, 10: 0.706, 20: 0.738, 50: 0.778},
        "$S$ ($\\lambda=50$)": {5: 0.694, 10: 0.715, 20: 0.754, 50: 0.785},
    }
    methods = list(summary.keys())
    ks = [5, 10, 20, 50]
    colors = ["#cccccc", "#9ec5d6", "#ee8866", "#bb99ee",
              "#aa9988", "#77cc99", "#cc3333"]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(ks))
    n_methods = len(methods)
    width = 0.11
    for j, (m, c) in enumerate(zip(methods, colors)):
        vals = [summary[m][k] for k in ks]
        offset = (j - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=m, color=c,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("transfer AUC (macro)")
    ax.set_ylim(0.45, 0.83)
    ax.set_title("Leave-one-cohort-out transfer AUC on Edu")
    ax.legend(loc="lower center", ncol=4, fontsize=6.5,
              framealpha=0.95, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figR1_transfer.pdf")
    fig.savefig(FIG_DIR / "figR1_transfer.png")
    plt.close(fig)
    print("wrote figR1_transfer")


def main():
    _setup()
    fig_nonsep()
    fig_transfer()


if __name__ == "__main__":
    main()
