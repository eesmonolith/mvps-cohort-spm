"""Benchmark wall-clock R1 mining with facewise vs vertex bound on Edu."""
from __future__ import annotations
import time
import json
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.scoring_r1 import per_cohort_ig
from src.joint_bound_r1 import joint_upper_bound_r1
from src.joint_bound_facewise import joint_upper_bound_facewise


def count_pattern_per_cohort_cluster(seqs, cohorts, clusters, K, M, pattern):
    """Subsequence-with-gaps containment, count per (cohort, cluster)."""
    n_cz = np.zeros((K, M), dtype=np.int64)
    pl = pattern
    for s, c, z in zip(seqs, cohorts, clusters):
        i = 0
        for t in s:
            if i < len(pl) and t == pl[i]: i += 1
        if i == len(pl):
            n_cz[c, z] += 1
    return n_cz


def enumerate_with_bound(seqs, cohorts, clusters, K, M, vocab_size, lam,
                          theta_sup, max_len, bound_fn):
    """Apriori-style enum, joint bound prune."""
    N_cz = np.zeros((K, M), dtype=np.int64)
    for c, z in zip(cohorts, clusters): N_cz[c, z] += 1
    qualified = []
    explored = 0
    pruned_apriori = 0
    pruned_bound = 0
    bound_time = 0.0
    # Level 1: singletons
    L1 = []
    for v in range(vocab_size):
        explored += 1
        n_cz = count_pattern_per_cohort_cluster(seqs, cohorts, clusters, K, M, [v])
        sup = n_cz.sum() / len(seqs)
        if sup < theta_sup:
            pruned_apriori += 1; continue
        t0 = time.time()
        ub = bound_fn(n_cz, N_cz, lam)
        bound_time += time.time() - t0
        if ub < 0:
            pruned_bound += 1; continue
        L1.append([v])
        qualified.append({"p": [v], "n_cz": n_cz})
    # Level 2: pairs
    L2 = []
    for p in L1:
        for v in range(vocab_size):
            cand = p + [v]
            explored += 1
            n_cz = count_pattern_per_cohort_cluster(seqs, cohorts, clusters, K, M, cand)
            sup = n_cz.sum() / len(seqs)
            if sup < theta_sup:
                pruned_apriori += 1; continue
            if max_len >= 2:
                t0 = time.time()
                ub = bound_fn(n_cz, N_cz, lam)
                bound_time += time.time() - t0
                if ub < 0:
                    pruned_bound += 1; continue
                qualified.append({"p": cand, "n_cz": n_cz})
                L2.append(cand)
    return {"explored": explored, "pruned_apriori": pruned_apriori,
            "pruned_bound": pruned_bound, "qualified": len(qualified),
            "bound_time": bound_time}


def main():
    spec = DATASET_REGISTRY["edu_kor"]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    vocab_size = len(spec.vocab)
    print(f"Edu: N={len(sequences)}  K={K}  M={M}  V={vocab_size}")
    print(f"box dim KM={K*M}, vertex enum 2^{K*M}={2**(K*M)}, facewise 3^{K}={3**K}")

    out = {}
    from src.joint_bound_polyk import joint_upper_bound_polyk
    for name, fn in [("vertex_2^KM", joint_upper_bound_r1),
                     ("facewise_3^K", joint_upper_bound_facewise),
                     ("polyk_OKlogK", joint_upper_bound_polyk)]:
        print(f"\n=== {name} ===")
        t0 = time.time()
        stats = enumerate_with_bound(
            sequences, cohorts, clusters, K, M, vocab_size,
            lam=50.0, theta_sup=0.02, max_len=2, bound_fn=fn,
        )
        elapsed = time.time() - t0
        stats["wallclock_total"] = elapsed
        print(f"  explored: {stats['explored']}")
        print(f"  pruned (apriori): {stats['pruned_apriori']}")
        print(f"  pruned (bound):   {stats['pruned_bound']}")
        print(f"  qualified:        {stats['qualified']}")
        print(f"  bound time:       {stats['bound_time']:.2f}s")
        print(f"  total wall-clock: {elapsed:.2f}s")
        out[name] = stats

    print(f"\nspeedup: {out['vertex_2^KM']['wallclock_total'] / max(out['facewise_3^K']['wallclock_total'], 1e-6):.2f}x")
    print(f"bound-time speedup: {out['vertex_2^KM']['bound_time'] / max(out['facewise_3^K']['bound_time'], 1e-6):.2f}x")

    from src.config import RESULTS
    with open(RESULTS / "facewise_walltime.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS / 'facewise_walltime.json'}")


if __name__ == "__main__":
    main()
