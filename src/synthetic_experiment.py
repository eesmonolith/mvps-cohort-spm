"""
Synthetic two-axis discriminative SPM benchmark.

We generate sequence databases with controllable cohort-shift (the
\\sigma_C term in our stability score) and cluster-discriminability
(the IG term) and run the joint vs naive bound comparison on each.

Generator parameters:
  - K cohorts, M clusters
  - N entities, drawn uniformly across the (K, M) grid
  - alphabet size V
  - signature pattern p* of length L*
  - cluster-discriminability rho_d in [0, 1]: probability that an entity
        in cluster z=0 has p* injected (other clusters: 0)
  - cohort-shift rho_c in [0, 1]: cohort-c rate of p* in cluster 0 is
        rho_d * (1 - rho_c * (c / (K-1)))     i.e. cohort 0 = full,
        cohort K-1 = (1 - rho_c) * rho_d
  - base seq length L_base

For each (rho_c, rho_d) combination we generate a fresh DB and measure
the empirical bound-ratio on 100 sampled candidate patterns.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.config import RESULTS, SEED
from src.joint_bound import joint_upper_bound, naive_product_bound
from src.scoring import joint_score


def generate(K=3, M=4, N=2000, V=10, L_star=3, rho_d=0.7, rho_c=0.0,
             L_base=30, seed=0):
    rng = np.random.default_rng(seed)
    signature = tuple(rng.integers(0, V, size=L_star))
    sequences, cohorts, clusters = [], [], []
    for _ in range(N):
        c = int(rng.integers(0, K))
        z = int(rng.integers(0, M))
        seq = list(rng.integers(0, V, size=L_base).tolist())
        # Inject signature in (z=0) entities at a rate depending on cohort
        if z == 0:
            rate = rho_d * (1.0 - rho_c * (c / max(K - 1, 1)))
            if rng.random() < rate:
                # Inject signature at a random position
                pos = int(rng.integers(0, L_base))
                seq = seq[:pos] + list(signature) + seq[pos:]
        sequences.append(seq)
        cohorts.append(c)
        clusters.append(z)
    cohorts = np.asarray(cohorts, dtype=np.int32)
    clusters = np.asarray(clusters, dtype=np.int32)
    N_cz = np.zeros((K, M), dtype=np.int64)
    for c, z in zip(cohorts, clusters):
        N_cz[c, z] += 1
    return sequences, cohorts, clusters, N_cz, signature


def contains(seq, pat):
    j = 0
    for tok in seq:
        if tok == pat[j]:
            j += 1
            if j == len(pat):
                return True
    return False


def count_atomic(sequences, cohorts, clusters, pattern, K, M):
    n_cz = np.zeros((K, M), dtype=np.int64)
    for s, c, z in zip(sequences, cohorts, clusters):
        if contains(s, pattern):
            n_cz[c, z] += 1
    return n_cz


def sample_bound_ratio(sequences, cohorts, clusters, N_cz, V, n_samples=100,
                       max_len=3, seed=0):
    rng = np.random.default_rng(seed)
    K, M = N_cz.shape
    ratios = []
    for _ in range(n_samples):
        L = int(rng.integers(1, max_len + 1))
        p = tuple(rng.integers(0, V, size=L).tolist())
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if n_cz.sum() == 0:
            continue
        ub_joint = joint_upper_bound(n_cz, N_cz)
        ub_naive = naive_product_bound(n_cz, N_cz)
        if ub_naive > 0:
            ratios.append(ub_joint / ub_naive)
    return np.asarray(ratios)


def main():
    rho_c_grid = [0.0, 0.3, 0.6]
    rho_d_grid = [0.3, 0.5, 0.7]
    rows = []
    for rho_c, rho_d in itertools.product(rho_c_grid, rho_d_grid):
        t0 = time.time()
        sequences, cohorts, clusters, N_cz, _ = generate(
            rho_d=rho_d, rho_c=rho_c, seed=SEED
        )
        ratios = sample_bound_ratio(sequences, cohorts, clusters, N_cz, V=10,
                                    n_samples=100, seed=SEED + 1)
        rows.append({
            "rho_c": rho_c,
            "rho_d": rho_d,
            "n_valid": int(ratios.size),
            "ratio_median": float(np.median(ratios)) if ratios.size else None,
            "ratio_p10": float(np.percentile(ratios, 10)) if ratios.size else None,
            "ratio_p90": float(np.percentile(ratios, 90)) if ratios.size else None,
            "ratio_min": float(ratios.min()) if ratios.size else None,
            "ratio_max": float(ratios.max()) if ratios.size else None,
            "pct_strict_tighter": float((ratios < 1.0).mean() * 100)
                                    if ratios.size else None,
            "time_s": time.time() - t0,
        })
        print(f"rho_c={rho_c:.1f}  rho_d={rho_d:.1f}  "
              f"n={ratios.size}  median={np.median(ratios):.3f}  "
              f"strict<1: {100*(ratios<1.0).mean():.0f}%")

    out_csv = RESULTS / "synthetic_bound.csv"
    pl.from_dicts(rows).write_csv(out_csv)
    print(f"\nWrote {out_csv}")

    out_json = RESULTS / "synthetic_bound.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
