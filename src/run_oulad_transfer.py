"""OULAD LOCO transfer at K=50."""
from __future__ import annotations
import json
import time
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.scoring_r1 import per_cohort_ig
from src.config import RESULTS


def main():
    spec = DATASET_REGISTRY["oulad"]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"OULAD: N={len(sequences)}  K={K}  M={M}")
    print("N_cz:\n", N_cz)

    tau_s = 0.7; tau_d = 0.05
    methods = ["freq_only", "stab_only", "discrim_only", "intersect",
               "v1_pooled", "min_ig", "r1_lam10", "r1_lam50", "r1_lam100", "r1_lam200"]
    agg = {m: [] for m in methods}
    K_top = 50

    for held_out in range(K):
        t0 = time.time()
        mask_tr = cohorts != held_out
        train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
        train_z = clusters[mask_tr]
        train_c = cohorts[mask_tr]
        test_seqs = [s for s, m in zip(sequences, ~mask_tr) if m]
        test_z = clusters[~mask_tr]
        if len(test_seqs) < 10:
            print(f"  fold {held_out} skipped"); continue
        c_uniq = np.unique(train_c)
        c_remap = {c: i for i, c in enumerate(c_uniq)}
        train_c_remap = np.array([c_remap[c] for c in train_c])
        K_tr = len(c_uniq)
        N_cz_tr = np.zeros((K_tr, M), dtype=np.int64)
        for c, z in zip(train_c_remap, train_z): N_cz_tr[c, z] += 1
        rows = enumerate_patterns_restricted(
            train_seqs, train_c_remap, train_z, N_cz_tr,
            K_tr, M, max_len=2, theta_sup=0.02,
        )

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
                ig = per_cohort_ig(r["n_cz"], N_cz_tr)
                ranked.append((r["p"], float(ig.mean() - lam * ig.var())))
            ranked.sort(key=lambda x: -x[1])
            sets[f"r1_lam{lam}"] = [p for p, _ in ranked[:K_top]]

        for m, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z, test_seqs, test_z)
            agg[m].append(auc)
        el = time.time() - t0
        print(f"  fold {held_out} ({list(spec.cohort_keys)[held_out]}): "
              f"freq={agg['freq_only'][-1]:.3f}  stab={agg['stab_only'][-1]:.3f}  "
              f"v1={agg['v1_pooled'][-1]:.3f}  r1@10={agg['r1_lam10'][-1]:.3f}  "
              f"r1@50={agg['r1_lam50'][-1]:.3f}  r1@200={agg['r1_lam200'][-1]:.3f}  "
              f"t={el:.0f}s")

    print(f"\nSummary (K_top={K_top}):")
    summary = {}
    for m in methods:
        v = agg[m]
        if not v: continue
        summary[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                      "per_fold": v}
        print(f"  {m:14s} {np.mean(v):.3f} ± {np.std(v):.3f}  n_folds={len(v)}")

    with open(RESULTS / "oulad_transfer.json", "w") as f:
        json.dump({"K": K, "M": M, "N": len(sequences), "summary": summary,
                   "N_cz": N_cz.tolist()}, f, indent=2)
    print(f"\nwrote {RESULTS / 'oulad_transfer.json'}")


if __name__ == "__main__":
    main()
