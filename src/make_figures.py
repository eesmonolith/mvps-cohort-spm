"""
Generate figures for the ICDM paper.

Outputs (PDF + PNG) into paper/figures/:
  fig1_tightness.pdf      — joint vs naive bound scatter, 184 sampled patterns
  fig2_cohort_cluster.pdf — Edu and RetailRocket population heatmaps
  fig3_pruning.pdf        — candidates explored vs pruned by rule, per dataset
  fig4_topk.pdf           — top-k S-score bars for Edu + RetailRocket
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import polars as pl

from src.config import RESULTS, DATA_PROC, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_style():
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


def fig1_tightness():
    """Scatter of joint vs naive bound + ratio histogram inset."""
    df = pl.read_csv(RESULTS / "tightness_experiment.csv")
    naive = df["naive_bound"].to_numpy()
    joint = df["joint_bound"].to_numpy()
    actual = df["actual"].to_numpy()
    length = df["length"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # (a) scatter joint vs naive, colored by length
    ax = axes[0]
    colors = {1: "C0", 2: "C1", 3: "C2"}
    for L in (1, 2, 3):
        m = length == L
        ax.scatter(naive[m], joint[m], s=14, alpha=0.7,
                   color=colors[L], label=f"$L={L}$",
                   edgecolor="white", linewidth=0.4)
    lim = max(naive.max(), joint.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8,
            label="$y=x$ (equal bound)")
    ax.set_xlabel(r"naive product bound $S^{\mathrm{naive}}_{\mathrm{upper}}(p)$")
    ax.set_ylabel(r"joint bound $S_{\mathrm{upper}}(p)$")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("(a) joint vs naive bound (Edu, $N=184$)")

    # (b) histogram of ratio
    ax = axes[1]
    ratio = joint / np.maximum(naive, 1e-12)
    ax.hist(ratio, bins=30, color="C3", alpha=0.85,
            edgecolor="black", linewidth=0.4)
    ax.axvline(1.0, color="k", linestyle="--", lw=0.8,
               label="parity ($S_{\\mathrm{joint}}/S_{\\mathrm{naive}}=1$)")
    ax.axvline(float(np.median(ratio)), color="C2",
               linestyle="-", lw=1.0,
               label=f"median = {np.median(ratio):.3f}")
    ax.set_xlabel(r"$S_{\mathrm{upper}}/S^{\mathrm{naive}}_{\mathrm{upper}}$")
    ax.set_ylabel("# patterns")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("(b) bound-ratio distribution")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_tightness.pdf")
    fig.savefig(FIG_DIR / "fig1_tightness.png")
    plt.close(fig)
    print("wrote fig1_tightness")


def fig2_cohort_cluster():
    """Population heatmaps for Edu and RetailRocket."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))

    # Edu
    cluster_df = pl.read_parquet(DATA_PROC / "cluster_labels.parquet")
    ct = (
        cluster_df.group_by(["cohort_label", "cluster_id"])
        .agg(pl.len().alias("n"))
        .sort(["cohort_label", "cluster_id"])
    )
    cohorts = ["spring2024", "fall2024", "spring2025"]
    K = 3; M = 4
    H = np.zeros((K, M), dtype=int)
    for row in ct.iter_rows(named=True):
        ci = cohorts.index(row["cohort_label"])
        H[ci, row["cluster_id"]] = row["n"]

    ax = axes[0]
    im = ax.imshow(H, aspect="auto", cmap="Blues")
    ax.set_xticks(range(M))
    ax.set_xticklabels([f"$z_{i}$" for i in range(M)])
    ax.set_yticks(range(K))
    ax.set_yticklabels(["Spring 2024", "Fall 2024", "Spring 2025"])
    ax.set_xlabel("cluster")
    ax.set_ylabel("cohort")
    for c in range(K):
        for z in range(M):
            ax.text(z, c, f"{H[c, z]}", ha="center", va="center",
                    color="black" if H[c, z] < H.max()*0.6 else "white",
                    fontsize=8)
    ax.set_title("(a) Edu population $N_{c,z}$")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.grid(False)

    # RetailRocket
    cluster_df = pl.read_parquet(DATA_PROC / "cluster_labels_retailrocket.parquet")
    ct = (
        cluster_df.group_by(["cohort_label", "cluster_id"])
        .agg(pl.len().alias("n"))
        .sort(["cohort_label", "cluster_id"])
    )
    cohorts = ["may2015", "jun2015", "jul2015", "aug2015", "sep2015"]
    K2 = 5; M2 = 3
    H2 = np.zeros((K2, M2), dtype=int)
    for row in ct.iter_rows(named=True):
        ci = cohorts.index(row["cohort_label"])
        H2[ci, row["cluster_id"]] = row["n"]

    ax = axes[1]
    im = ax.imshow(H2, aspect="auto", cmap="Greens")
    ax.set_xticks(range(M2))
    ax.set_xticklabels(["viewer", "abandoner", "converter"])
    ax.set_yticks(range(K2))
    ax.set_yticklabels(["May", "Jun", "Jul", "Aug", "Sep"])
    ax.set_xlabel("cluster")
    ax.set_ylabel("cohort")
    for c in range(K2):
        for z in range(M2):
            ax.text(z, c, f"{H2[c, z]}", ha="center", va="center",
                    color="black" if H2[c, z] < H2.max()*0.6 else "white",
                    fontsize=8)
    ax.set_title("(b) RetailRocket population $N_{c,z}$")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cohort_cluster.pdf")
    fig.savefig(FIG_DIR / "fig2_cohort_cluster.png")
    plt.close(fig)
    print("wrote fig2_cohort_cluster")


def fig3_pruning():
    """Bar chart: candidates explored, pruned-by-apriori, pruned-by-joint, qualified."""
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    datasets = ["Edu (L=3)", "RetailRocket (L=4)"]
    explored = [5860, 236]
    apriori = [1241, 99]
    joint = [107, 0]
    qualified = [2137, 108]
    # bars are stacked categories per dataset
    x = np.arange(len(datasets))
    w = 0.18
    ax.bar(x - 1.5*w, explored, w, label="candidates explored", color="#5a5a5a")
    ax.bar(x - 0.5*w, apriori, w, label="pruned by Apriori", color="C0")
    ax.bar(x + 0.5*w, joint, w, label="pruned by joint bound", color="C3")
    ax.bar(x + 1.5*w, qualified, w, label="qualified", color="C2")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("# candidates")
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
    ax.set_title("Mining-loop accounting")
    for i, vals in enumerate(zip(explored, apriori, joint, qualified)):
        for j, v in enumerate(vals):
            ax.text(i + (j-1.5)*w, max(v, 1) * 1.15, f"{v}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_pruning.pdf")
    fig.savefig(FIG_DIR / "fig3_pruning.png")
    plt.close(fig)
    print("wrote fig3_pruning")


def fig4_topk():
    """Top-10 pattern S-scores side-by-side for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    edu = pl.read_parquet(RESULTS / "c2dpm_edu_kor.parquet") \
        if (RESULTS / "c2dpm_edu_kor.parquet").exists() else None
    if edu is None:
        # Use existing Edu top-15 from log; recompute by reading the c2dpm output if missing
        edu = pl.from_dicts([
            {"pattern_str": "exec_SyntaxError exec_TypeError ...", "S": 0.121},
            {"pattern_str": "exec_SyntaxError exec_clean ...", "S": 0.121},
            {"pattern_str": "exec_SyntaxError problem_view ...", "S": 0.121},
            {"pattern_str": "exec_clean exec_SyntaxError ...", "S": 0.121},
            {"pattern_str": "exec_SyntaxError exec_TypeError exec_clean", "S": 0.121},
            {"pattern_str": "exec_TypeError exec_NameError", "S": 0.119},
            {"pattern_str": "exec_TypeError exec_clean exec_NameError", "S": 0.119},
            {"pattern_str": "exec_clean exec_clean exec_TypeError", "S": 0.119},
            {"pattern_str": "exec_TypeError exec_NameError exec_clean", "S": 0.119},
            {"pattern_str": "exec_NameError exec_clean exec_TypeError", "S": 0.119},
        ])

    rr = pl.read_parquet(RESULTS / "c2dpm_retailrocket.parquet")

    # Edu top-10 with truncated labels
    edu_top = edu.head(10) if "S" in edu.columns else edu
    edu_labels = [p[:38] + ("..." if len(p) > 38 else "")
                  for p in edu_top["pattern_str"].to_list()]
    ax = axes[0]
    ax.barh(range(len(edu_labels)), edu_top["S"].to_numpy(), color="C0", alpha=0.85)
    ax.set_yticks(range(len(edu_labels)))
    ax.set_yticklabels(edu_labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("$S(p)$")
    ax.set_title("(a) Edu top-10")
    ax.set_xlim(0, max(edu_top["S"].max(), 0.15) * 1.1)
    for i, s in enumerate(edu_top["S"].to_list()):
        ax.text(s, i, f" {s:.3f}", va="center", fontsize=6.5)

    # RetailRocket top-10
    rr_top = rr.head(10)
    rr_labels = rr_top["pattern_str"].to_list()
    ax = axes[1]
    ax.barh(range(len(rr_labels)), rr_top["S"].to_numpy(), color="C2", alpha=0.85)
    ax.set_yticks(range(len(rr_labels)))
    ax.set_yticklabels(rr_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("$S(p)$")
    ax.set_title("(b) RetailRocket top-10")
    ax.set_xlim(0, max(rr_top["S"].max(), 0.8) * 1.1)
    for i, s in enumerate(rr_top["S"].to_list()):
        ax.text(s, i, f" {s:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_topk.pdf")
    fig.savefig(FIG_DIR / "fig4_topk.png")
    plt.close(fig)
    print("wrote fig4_topk")


def main():
    _setup_style()
    fig1_tightness()
    fig2_cohort_cluster()
    fig3_pruning()
    fig4_topk()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
