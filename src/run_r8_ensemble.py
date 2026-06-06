"""R8 = ensemble union of top-K from multiple methods.

Take top-K from stab, R1, R7d (CV), discrim, intersect; dedupe;
use union as feature set. LogReg picks the best ones automatically.
"""
from __future__ import annotations
import json
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.scoring_r1 import per_cohort_ig
from src.scoring_r7 import score_r7d_cv_penalty, topk_pats
from src.config import RESULTS


def dedupe_patterns(pats_list):
    seen = set(); out = []
    for p in pats_list:
        key = tuple(p)
        if key not in seen:
            seen.add(key); out.append(p)
    return out


def run_dataset(name, K_top=50, K_per=20):
    """K_top per method, then union (capped or unbounded)."""
    spec = DATASET_REGISTRY[name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"\n=== {name}  N={len(sequences)}  K={K}  M={M} ===")
    tau_s = 0.7; tau_d = 0.05
    agg = {"r8_union_K50": [], "r8_union_K20each": [], "stab": [], "r1": [], "r7d": []}

    for held_out in range(K):
        mask_tr = cohorts != held_out
        train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
        train_z = clusters[mask_tr]; train_c = cohorts[mask_tr]
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
        stab_pats = ([r["p"] for r in rows if r["stability"] >= tau_s][:K_top]
                     or topk_by(rows, "stability", K_top))
        intersect_pats = [r["p"] for r in rows
                          if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:K_top]
        discrim_pats = topk_by(rows, "discrim", K_top)
        # R1
        ranked = [(r["p"], float(r["per_cohort_ig"].mean() -
                                  50.0 * r["per_cohort_ig"].var())) for r in rows]
        ranked.sort(key=lambda x: -x[1])
        r1_pats = [p for p, _ in ranked[:K_top]]
        r7d_pats = topk_pats(score_r7d_cv_penalty(rows, lam=50.0), K_top)

        # R8a: union of top-K_top from each method, capped to K_top (deduped)
        union_all = dedupe_patterns(stab_pats + r1_pats + r7d_pats +
                                     discrim_pats + intersect_pats)
        # R8b: union with K_per=20 each → up to 100 deduped patterns
        stab_20 = ([r["p"] for r in rows if r["stability"] >= tau_s][:K_per]
                   or topk_by(rows, "stability", K_per))
        r1_20 = r1_pats[:K_per]
        r7d_20 = r7d_pats[:K_per]
        discrim_20 = topk_by(rows, "discrim", K_per)
        union_20 = dedupe_patterns(stab_20 + r1_20 + r7d_20 + discrim_20)

        agg["stab"].append(evaluate_transfer(stab_pats, train_seqs, train_z, test_seqs, test_z))
        agg["r1"].append(evaluate_transfer(r1_pats, train_seqs, train_z, test_seqs, test_z))
        agg["r7d"].append(evaluate_transfer(r7d_pats, train_seqs, train_z, test_seqs, test_z))
        agg["r8_union_K50"].append(evaluate_transfer(union_all, train_seqs, train_z, test_seqs, test_z))
        agg["r8_union_K20each"].append(evaluate_transfer(union_20, train_seqs, train_z, test_seqs, test_z))

    print(f"\nSummary K_top={K_top}:")
    summary = {}
    for m, v in agg.items():
        v = [x for x in v if not np.isnan(x)]
        if not v: continue
        summary[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                      "per_fold": v, "n_folds": len(v)}
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
    with open(RESULTS / "r8_ensemble.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS / 'r8_ensemble.json'}")

    print("\n" + "=" * 80)
    print(f"{'Method':20s}", end="")
    for ds in ["edu_kor", "edub_hashed", "bpi2012", "sepsis", "oulad"]:
        print(f"{ds[:8]:>10s}", end="")
    print()
    print("-" * 80)
    for m in ["stab", "r1", "r7d", "r8_union_K50", "r8_union_K20each"]:
        print(f"{m:20s}", end="")
        for ds in ["edu_kor", "edub_hashed", "bpi2012", "sepsis", "oulad"]:
            s = out[ds].get(m, {})
            mean = s.get("mean")
            print(f"{mean:>10.3f}" if mean is not None else f"{'--':>10s}", end="")
        print()


if __name__ == "__main__":
    main()
