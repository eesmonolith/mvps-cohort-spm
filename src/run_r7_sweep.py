"""Run R7 algorithm family across 5 datasets."""
from __future__ import annotations
import json
import time
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.scoring_r1 import per_cohort_ig
from src.scoring_r7 import (
    score_r7a_stab_gated, score_r7b_geomean,
    score_r7c_borda, score_r7d_cv_penalty, topk_pats,
)
from src.config import RESULTS


def run_dataset(name, K_top=50):
    spec = DATASET_REGISTRY[name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"\n=== {name}  N={len(sequences)}  K={K}  M={M} ===")
    tau_s = 0.7; tau_d = 0.05
    methods = ["freq", "stab", "discrim", "intersect", "v1", "min_ig", "r1_lam50",
               "R7a_gated_R1", "R7b_geomean", "R7c_borda", "R7d_cv"]
    agg = {m: [] for m in methods}

    for held_out in range(K):
        mask_tr = cohorts != held_out
        train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
        train_z = clusters[mask_tr]
        train_c = cohorts[mask_tr]
        test_seqs = [s for s, m in zip(sequences, ~mask_tr) if m]
        test_z = clusters[~mask_tr]
        if len(test_seqs) < 10: continue
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
        for r in rows:
            r["N_cz_train"] = N_cz_tr
            r["per_cohort_ig"] = per_cohort_ig(r["n_cz"], N_cz_tr)

        # baselines
        sets = {
            "freq": topk_by(rows, "support", K_top),
            "stab": ([r["p"] for r in rows if r["stability"] >= tau_s][:K_top]
                     or topk_by(rows, "stability", K_top)),
            "discrim": topk_by(rows, "discrim", K_top),
            "intersect": [r["p"] for r in rows
                          if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:K_top],
            "v1": topk_by(rows, "S_v1", K_top),
            "min_ig": topk_by(rows, "min_ig", K_top),
        }
        # R1 lam=50
        ranked = []
        for r in rows:
            ig = r["per_cohort_ig"]
            ranked.append((r["p"], float(ig.mean() - 50.0 * ig.var())))
        ranked.sort(key=lambda x: -x[1])
        sets["r1_lam50"] = [p for p, _ in ranked[:K_top]]

        # R7 family
        sets["R7a_gated_R1"] = topk_pats(score_r7a_stab_gated(rows, lam=50.0, tau_s=tau_s), K_top)
        sets["R7b_geomean"]  = topk_pats(score_r7b_geomean(rows), K_top)
        sets["R7c_borda"]    = topk_pats(score_r7c_borda(rows), K_top)
        sets["R7d_cv"]       = topk_pats(score_r7d_cv_penalty(rows, lam=50.0), K_top)

        for m, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z, test_seqs, test_z)
            agg[m].append(auc)

    print(f"\nSummary K_top={K_top}:")
    summary = {}
    for m in methods:
        v = [x for x in agg[m] if not np.isnan(x)]
        if not v: continue
        summary[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                      "per_fold": v}
        print(f"  {m:18s} {np.mean(v):.3f} ± {np.std(v):.3f}")
    return summary


def main():
    out = {}
    for ds in ["edu_kor", "edub_hashed", "bpi2012", "sepsis", "oulad"]:
        try:
            out[ds] = run_dataset(ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[ds] = {"error": str(e)}
    with open(RESULTS / "r7_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS / 'r7_sweep.json'}")

    # Cross-dataset comparison
    print("\n" + "=" * 80)
    print(f"{'Method':18s}", end="")
    for ds in ["edu_kor", "edub_hashed", "bpi2012", "sepsis", "oulad"]:
        print(f"{ds[:8]:>10s}", end="")
    print()
    print("-" * 80)
    methods = ["freq", "stab", "discrim", "intersect", "v1", "min_ig", "r1_lam50",
               "R7a_gated_R1", "R7b_geomean", "R7c_borda", "R7d_cv"]
    for m in methods:
        print(f"{m:18s}", end="")
        for ds in ["edu_kor", "edub_hashed", "bpi2012", "sepsis", "oulad"]:
            s = out[ds].get(m, {})
            mean = s.get("mean")
            print(f"{mean:>10.3f}" if mean is not None else f"{'--':>10s}", end="")
        print()


if __name__ == "__main__":
    main()
