"""
Render the full per-corpus, per-method matched-budget-70 detail table
that experiments.tex promises ("per-corpus MVPS/best@70/gap/p for all
six corpora is released with the code") into a standalone Markdown
file: docs/budget70_detail.md.

This is exactly the table that used to be Appendix Table XII
(tab:budget70) before it was cut from the PDF for the 10-page limit
(see camera-ready/CHANGES_PHASE3.md). The underlying numbers are
unchanged; this script only formats results/camera_ready/budget70_baselines.json
as Markdown.

Usage:
    python -m src.make_budget70_table
"""
from __future__ import annotations

import json

from src.config import RESULTS, PROJECT_ROOT

METHODS = [
    ("freq_only", "freq-only"),
    ("stab_only", "stab-only"),
    ("discrim_only", "discrim-only"),
    ("intersect", "intersect"),
    ("v1_pooled", "$S^{(v1)}$"),
    ("min_ig", "$\\min_c \\mathrm{IG}_c$"),
    ("r1_lam50", "$S$ ($\\lambda{=}50$)"),
]

CORPUS_ORDER = [
    ("edu_kor", "Edu"),
    ("codle_hashed", "EduB"),
    ("bpi2012", "BPI"),
    ("sepsis", "Sepsis"),
    ("oulad", "OULAD"),
    ("hospital_billing", "HB"),
]

OUT_PATH = PROJECT_ROOT / "docs" / "budget70_detail.md"


def main():
    with open(RESULTS / "camera_ready" / "budget70_baselines.json") as f:
        data = json.load(f)

    lines = []
    lines.append("# Matched-budget-70 baseline detail\n")
    lines.append(
        "Full per-corpus, per-method detail behind experiments.tex's "
        "\"Does a single view catch up at matched budget?\" paragraph "
        "(Section IV). Each single-view baseline is re-evaluated at "
        "top-K=70 patterns (matching MVPS's realised union size of "
        "50-70 patterns) instead of the paper's default top-50. "
        "Generated from `results/camera_ready/budget70_baselines.json` "
        "by `src/make_budget70_table.py` -- see "
        "`camera-ready/CHANGES_PHASE3.md` for why this table was moved "
        "here from the PDF appendix.\n"
    )
    lines.append(
        "Bold marks the best single view at top-70 per corpus. "
        "`gap` = MVPS mean AUC minus the best single view at top-70, "
        "in percentage points (positive = MVPS still ahead at matched "
        "budget). `p` is the one-sided paired Wilcoxon signed-rank "
        "p-value for MVPS > best-at-70 on the same LOCO folds.\n"
    )

    for corpus, disp in CORPUS_ORDER:
        if corpus not in data or "error" in data[corpus]:
            lines.append(f"## {disp} ({corpus})\n\n_no data (adapter not run / raw corpus unavailable)_\n")
            continue
        v = data[corpus]
        best_name = v["best_single_view_at_70"]["name"]
        lines.append(f"## {disp} ({corpus})\n")
        lines.append(f"K={v['K']}  M={v['M']}  N={v['N']}  folds={len(v['fold_ids'])}\n")
        lines.append("| Method | mean AUC@70 | std |")
        lines.append("|---|---:|---:|")
        for key, label in METHODS:
            if key not in v["baselines_at_70"]:
                continue
            b = v["baselines_at_70"][key]
            mark = "**" if key == best_name else ""
            lines.append(f"| {mark}{label}{mark} | {b['mean']:.4f} | {b['std']:.4f} |")
        lines.append(f"| **MVPS** (4-view union) | **{v['mvps']['mean']:.4f}** | {v['mvps']['std']:.4f} |")
        lines.append("")
        lines.append(
            f"Best single view @70: **{best_name}** "
            f"({v['best_single_view_at_70']['mean']:.4f}). "
            f"MVPS gap: **{v['gap_mvps_minus_best_single_view_at_70_pp']:+.2f}pp**. "
            f"Wilcoxon one-sided p (MVPS > best@70): "
            f"{v['wilcoxon_p_one_sided_mvps_gt_best_at_70']:.4f} "
            f"(n={len(v['fold_ids'])} LOCO folds).\n"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
