"""Sweep K_top in {5, 10, 20, 50} to understand how the head of the
ranking affects transfer AUC. Same setup as transfer_experiment.py.
"""
from __future__ import annotations

import json
import numpy as np

from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by, contains,
)
from src.c2dpm import load_dataset, C2DPMConfig
from src.config import RESULTS, VOCAB_SIZE, SEED


def main():
    cfg = C2DPMConfig(theta_sup=0.02, max_len=2)
    sequences, cohorts, clusters, N_cz = load_dataset("edu_kor")
    K_cohorts = int(N_cz.shape[0])
    M_clusters = int(N_cz.shape[1])
    tau_s = 0.7
    tau_d = 0.05

    K_top_grid = [5, 10, 20, 50]
    matrix = {}  # method -> K_top -> mean_auc

    for held_out in range(K_cohorts):
        mask_train = cohorts != held_out
        mask_test = cohorts == held_out
        train_seqs = [s for s, m in zip(sequences, mask_train) if m]
        train_z = clusters[mask_train]
        train_c = cohorts[mask_train]
        test_seqs = [s for s, m in zip(sequences, mask_test) if m]
        test_z = clusters[mask_test]

        K_train = len(np.unique(train_c))
        c_unique = np.unique(train_c)
        c_remap = {c: i for i, c in enumerate(c_unique)}
        train_c_remap = np.array([c_remap[c] for c in train_c])
        N_cz_train = np.zeros((K_train, M_clusters), dtype=np.int64)
        for c, z in zip(train_c_remap, train_z):
            N_cz_train[c, z] += 1

        rows = enumerate_patterns_restricted(
            train_seqs, train_c_remap, train_z, N_cz_train,
            K_train, M_clusters, max_len=cfg.max_len, theta_sup=cfg.theta_sup,
        )

        for K_top in K_top_grid:
            sets = {
                "freq_only": topk_by(rows, "support", K_top),
                "stab_only": (
                    [r["p"] for r in rows if r["stability"] >= tau_s][:K_top]
                    if any(r["stability"] >= tau_s for r in rows)
                    else topk_by(rows, "stability", K_top)
                ),
                "discrim_only": topk_by(rows, "discrim", K_top),
                "intersect": (
                    [r["p"] for r in rows if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:K_top]
                ),
                "v1_pooled": topk_by(rows, "S_v1", K_top),
                "r1_lam50": topk_by(rows, "S_R1_lam50", K_top),
                "min_ig": topk_by(rows, "min_ig", K_top),
            }
            for name, pats in sets.items():
                auc = evaluate_transfer(pats, train_seqs, train_z, test_seqs, test_z)
                matrix.setdefault(name, {}).setdefault(K_top, []).append(auc)
        print(f"fold {held_out} done")

    # Summarise as mean ± std per (method, K_top)
    print("\n" + "=" * 72)
    print(f"{'Method':<15s}  " + "  ".join(f"K={k:<3d}" for k in K_top_grid))
    print("=" * 72)
    summary = {}
    for name in ["freq_only", "stab_only", "discrim_only", "intersect",
                 "v1_pooled", "r1_lam50", "min_ig"]:
        row_vals = []
        summary[name] = {}
        for k in K_top_grid:
            vals = np.array(matrix[name][k])
            row_vals.append(f"{vals.mean():.3f}±{vals.std():.2f}")
            summary[name][k] = {"mean": float(vals.mean()),
                                "std": float(vals.std()),
                                "per_fold": vals.tolist()}
        print(f"{name:<15s}  " + "   ".join(row_vals))

    with open(RESULTS / "transfer_ktop_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {RESULTS / 'transfer_ktop_sweep.json'}")


if __name__ == "__main__":
    main()
