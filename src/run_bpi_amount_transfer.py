"""Run multi-lambda transfer on BPI amount-cohort dataset."""
from __future__ import annotations
import json
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.scoring_r1 import per_cohort_ig
from src.config import RESULTS


def main():
    spec = DATASET_REGISTRY["bpi2012_amount"]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"BPI amount: N={len(sequences)}  K={K}  M={M}")
    print("N_cz:\n", N_cz)

    tau_s = 0.7; tau_d = 0.05
    fold_data = []
    for held_out in range(K):
        mask_tr = cohorts != held_out
        train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
        train_z = clusters[mask_tr]
        train_c = cohorts[mask_tr]
        test_seqs = [s for s, m in zip(sequences, ~mask_tr) if m]
        test_z = clusters[~mask_tr]
        if len(test_seqs) < 10: continue
        c_unique = np.unique(train_c)
        c_remap = {c: i for i, c in enumerate(c_unique)}
        train_c_remap = np.array([c_remap[c] for c in train_c])
        K_tr = len(c_unique)
        N_cz_tr = np.zeros((K_tr, M), dtype=np.int64)
        for c, z in zip(train_c_remap, train_z): N_cz_tr[c, z] += 1
        rows = enumerate_patterns_restricted(
            train_seqs, train_c_remap, train_z, N_cz_tr,
            K_tr, M, max_len=2, theta_sup=0.02,
        )
        for r in rows: r["N_cz_train"] = N_cz_tr
        fold_data.append({
            "held": held_out, "rows": rows,
            "train_seqs": train_seqs, "train_z": train_z,
            "test_seqs": test_seqs, "test_z": test_z,
        })

    K_top = 50
    methods = ["freq_only", "stab_only", "discrim_only", "intersect",
               "v1_pooled", "min_ig", "r1_lam10", "r1_lam50", "r1_lam100", "r1_lam200"]
    agg = {m: [] for m in methods}

    for fd in fold_data:
        rows = fd["rows"]
        # baselines
        sets = {
            "freq_only": topk_by(rows, "support", K_top),
            "stab_only": ([r["p"] for r in rows if r["stability"] >= tau_s][:K_top]
                          or topk_by(rows, "stability", K_top)),
            "discrim_only": topk_by(rows, "discrim", K_top),
            "intersect": [r["p"] for r in rows
                          if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:K_top],
            "v1_pooled": topk_by(rows, "S_v1", K_top),
            "min_ig": topk_by(rows, "min_ig", K_top),
        }
        for lam in [10, 50, 100, 200]:
            ranked = []
            for r in rows:
                ig = per_cohort_ig(r["n_cz"], r["N_cz_train"])
                s = float(ig.mean() - lam * ig.var())
                ranked.append((r["p"], s))
            ranked.sort(key=lambda x: -x[1])
            sets[f"r1_lam{lam}"] = [p for p, _ in ranked[:K_top]]

        for m, pats in sets.items():
            auc = evaluate_transfer(pats, fd["train_seqs"], fd["train_z"],
                                    fd["test_seqs"], fd["test_z"])
            agg[m].append(auc)
        print(f"  fold {fd['held']}: freq={agg['freq_only'][-1]:.3f}  "
              f"stab={agg['stab_only'][-1]:.3f}  intersect={agg['intersect'][-1]:.3f}  "
              f"v1={agg['v1_pooled'][-1]:.3f}  r1@50={agg['r1_lam50'][-1]:.3f}  "
              f"r1@100={agg['r1_lam100'][-1]:.3f}")

    print("\nSummary (K_top=50):")
    summary = {}
    for m in methods:
        v = agg[m]
        summary[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_fold": v}
        print(f"  {m:14s} {np.mean(v):.3f} ± {np.std(v):.3f}")

    with open(RESULTS / "bpi_amount_transfer.json", "w") as f:
        json.dump({"K": K, "M": M, "N": len(sequences), "summary": summary,
                   "N_cz": N_cz.tolist()}, f, indent=2)
    print(f"\nwrote {RESULTS / 'bpi_amount_transfer.json'}")


if __name__ == "__main__":
    main()
