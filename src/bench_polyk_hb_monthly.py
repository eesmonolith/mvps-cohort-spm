"""
Real-corpus high-K bound benchmark: Hospital Billing monthly (K=36).

Per-candidate bound cost:
- Vertex: 2^(K*M) = 2^(36*3) = 2^108 ≈ 3 * 10^32 ops (infeasible)
- Face-wise: 3^K = 3^36 ≈ 1.5 * 10^17 ops (infeasible)
- Polynomial: O(K log K + K * 2^M) = 36 log 36 + 36 * 8 ≈ 480 ops (feasible)

We mine real candidates from HB monthly and time polynomial bound.
"""
from __future__ import annotations
import json
import time
import numpy as np
import polars as pl

from src.config import DATA_PROC, RESULTS
from src.joint_bound_polyk import joint_upper_bound_polyk
from src.scoring_r1 import per_cohort_ig


def main():
    # Load HB monthly
    seq_df = pl.read_parquet(DATA_PROC / "sequences_hospital_billing_monthly.parquet")
    cl_df = pl.read_parquet(DATA_PROC / "cluster_labels_hospital_billing_monthly.parquet")
    joined = seq_df.join(cl_df, on=["user_id", "cohort_label"], how="inner")

    cohort_keys = sorted(joined["cohort_label"].unique().to_list())
    cohort_to_id = {c: i for i, c in enumerate(cohort_keys)}
    K = len(cohort_keys)
    M = 3  # 0=billed, 1=cancelled, 2=other
    print(f"HB monthly: N={joined.height}, K={K} months, M={M}")
    print(f"K*M = {K*M}, vertex 2^(K*M) = 2^{K*M} ≈ {2**(K*M):.2e}")
    print(f"3^K = 3^{K} ≈ {3**K:.2e}")
    print(f"polynomial K log K + K * 2^M ≈ {int(K * np.log2(K) + K * 2**M)} ops/call\n")

    # Sample candidate patterns and compute N_cz
    rng = np.random.default_rng(42)
    user_data = joined.to_dicts()
    cohorts = np.array([cohort_to_id[d["cohort_label"]] for d in user_data])
    clusters = np.array([d["cluster_id"] for d in user_data])
    sequences = [d["token_ids"] for d in user_data]
    N_cz = np.zeros((K, M), dtype=np.int64)
    for c, z in zip(cohorts, clusters): N_cz[c, z] += 1

    # Generate random length-1 patterns from vocab and count their per-cohort presence
    vocab = json.load(open(RESULTS / "hospital_billing_monthly_vocab.json"))
    V = vocab["n"]
    print(f"vocab size: {V}")

    # Sample patterns (single-token), compute n_cz
    candidates = []
    for v in range(V):
        n_cz = np.zeros((K, M), dtype=np.int64)
        for s, c, z in zip(sequences, cohorts, clusters):
            if v in s:
                n_cz[c, z] += 1
        candidates.append((v, n_cz))

    # Length-2 random samples
    rng = np.random.default_rng(42)
    for _ in range(20):
        v1, v2 = rng.choice(V, 2, replace=False)
        n_cz = np.zeros((K, M), dtype=np.int64)
        for s, c, z in zip(sequences, cohorts, clusters):
            # subsequence contains check
            i = 0
            target = [v1, v2]
            for t in s:
                if i < 2 and t == target[i]: i += 1
            if i == 2:
                n_cz[c, z] += 1
        if n_cz.sum() > 100:  # has support
            candidates.append((f"[{v1},{v2}]", n_cz))

    print(f"\nbenchmarking {len(candidates)} real candidates")

    # Polynomial bound timing
    n_calls = 50
    t0 = time.time()
    for _ in range(n_calls):
        for name, n_cz in candidates:
            joint_upper_bound_polyk(n_cz, N_cz, lam=50.0)
    elapsed = time.time() - t0
    per_call_us = elapsed / (n_calls * len(candidates)) * 1e6
    print(f"\npolynomial bound: {per_call_us:.1f} us per call")
    print(f"total {len(candidates)} candidates * {n_calls} reps = {elapsed*1000:.0f} ms")

    # Extrapolation: face-wise would take...
    face_us_per_op = 0.001  # rough estimate: 1 ns per evaluation
    face_total_seconds_per_call = 3**K * face_us_per_op / 1e6
    vertex_total_seconds_per_call = 2**(K*M) * face_us_per_op / 1e6
    print(f"\nExtrapolation per call:")
    print(f"  face-wise (3^{K}): ~{face_total_seconds_per_call:.2e} seconds")
    print(f"  vertex (2^{K*M}): ~{vertex_total_seconds_per_call:.2e} seconds")

    # Save
    summary = {
        "K": K, "M": M, "V": V, "N": joined.height,
        "n_candidates": len(candidates),
        "poly_us_per_call": per_call_us,
        "poly_total_seconds": elapsed,
        "vertex_2_KM": float(2**(K*M)),
        "facewise_3_K": float(3**K),
        "poly_ops_per_call": int(K * np.log2(K) + K * 2**M),
    }
    with open(RESULTS / "polyk_hb_monthly_bench.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS / 'polyk_hb_monthly_bench.json'}")


if __name__ == "__main__":
    main()
