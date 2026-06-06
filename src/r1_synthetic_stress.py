"""
R1 synthetic stress test.

Sweep cohort-shift rho_c and cluster-discriminability rho_d on the
synthetic generator and measure the joint vs naive bound gap on the
R1 objective with lambda=50.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import polars as pl

from src.config import RESULTS, SEED
from src.synthetic_experiment import generate, contains
from src.joint_bound_r1 import joint_upper_bound_r1, naive_separable_bound_r1
from src.scoring_r1 import per_cohort_ig


def count_atomic(sequences, cohorts, clusters, pattern, K, M):
    n_cz = np.zeros((K, M), dtype=np.int64)
    for s, c, z in zip(sequences, cohorts, clusters):
        if contains(s, pattern):
            n_cz[c, z] += 1
    return n_cz


def sample_bound_gap(sequences, cohorts, clusters, N_cz, V, n_samples=100,
                    max_len=3, seed=0, lam=50.0):
    rng = np.random.default_rng(seed)
    K, M = N_cz.shape
    rows = []
    for _ in range(n_samples):
        L = int(rng.integers(1, max_len + 1))
        p = tuple(rng.integers(0, V, size=L).tolist())
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if n_cz.sum() == 0:
            continue
        u = n_cz / np.maximum(N_cz, 1)
        ig_actual = per_cohort_ig(n_cz, N_cz)
        actual = float(ig_actual.mean() - lam * ig_actual.var())
        naive = naive_separable_bound_r1(n_cz, N_cz, lam)
        joint = joint_upper_bound_r1(n_cz, N_cz, lam)
        rows.append({
            "actual": actual,
            "naive_bound": naive,
            "joint_bound": joint,
            "abs_gap": naive - joint,
            "rel_gap": (naive - joint) / max(abs(naive), 1e-9),
        })
    return rows


def main():
    rho_c_grid = [0.0, 0.3, 0.6]
    rho_d_grid = [0.3, 0.5, 0.7]
    summary = []
    for rho_c, rho_d in itertools.product(rho_c_grid, rho_d_grid):
        t0 = time.time()
        sequences, cohorts, clusters, N_cz, _ = generate(
            rho_d=rho_d, rho_c=rho_c, seed=SEED,
        )
        rows = sample_bound_gap(sequences, cohorts, clusters, N_cz, V=10,
                                n_samples=80, seed=SEED + 1, lam=50.0)
        abs_gaps = np.array([r["abs_gap"] for r in rows])
        rel_gaps = np.array([r["rel_gap"] for r in rows])
        strict_pct = float((abs_gaps > 0).mean() * 100)
        summary.append({
            "rho_c": rho_c,
            "rho_d": rho_d,
            "n_valid": int(len(rows)),
            "abs_gap_median": float(np.median(abs_gaps)),
            "abs_gap_p90": float(np.percentile(abs_gaps, 90)),
            "abs_gap_max": float(abs_gaps.max()),
            "rel_gap_median": float(np.median(rel_gaps)),
            "rel_gap_p90": float(np.percentile(rel_gaps, 90)),
            "pct_strict": strict_pct,
            "time_s": time.time() - t0,
        })
        print(f"rho_c={rho_c:.1f}  rho_d={rho_d:.1f}  "
              f"n={len(rows)}  abs_gap_med={np.median(abs_gaps):.3f}  "
              f"rel_med={np.median(rel_gaps)*100:.1f}%  "
              f"strict>0: {strict_pct:.0f}%")

    pl.from_dicts(summary).write_csv(RESULTS / "r1_synthetic_stress.csv")
    with open(RESULTS / "r1_synthetic_stress.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote results/r1_synthetic_stress.{{csv,json}}")


if __name__ == "__main__":
    main()
