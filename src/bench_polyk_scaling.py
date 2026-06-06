"""
Benchmark: poly-K bound vs face-wise bound wall-clock at varying K.

Uses synthetic generator (controllable K, M, V) to isolate the bound
computation bottleneck from sequence matching overhead.

K grid: 3, 5, 8, 12, 18, 24
M (clusters) = 3  (fixed, real Edu setting)
V (vocab)    = 10 (compact; L1+L2 candidate pool = V + V^2 = 110)
N sequences  = 1500 per K (kept small to cap matching time)

For each K we measure:
  - per-call bound time: median over 200 candidate evaluations
  - end-to-end mining wall-clock: apriori-style L=1+2 enum on synthetic DB
  - 3^K vs K+1  theoretical candidate count (bound itself)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.config import RESULTS

# ── synthetic generator ──────────────────────────────────────────────────────

def generate_synthetic(K: int, M: int = 3, V: int = 10, N: int = 1500,
                        L_base: int = 20, rho_d: float = 0.7, rho_c: float = 0.2,
                        seed: int = 42):
    """Return (sequences, cohorts, clusters, N_cz)."""
    rng = np.random.default_rng(seed)
    sig_len = 2
    signature = list(rng.integers(0, V, size=sig_len))

    sequences, cohorts, clusters = [], [], []
    for _ in range(N):
        c = int(rng.integers(0, K))
        z = int(rng.integers(0, M))
        seq = list(rng.integers(0, V, size=L_base).tolist())
        if z == 0:
            rate = rho_d * (1.0 - rho_c * (c / max(K - 1, 1)))
            if rng.random() < rate:
                pos = int(rng.integers(0, L_base))
                seq = seq[:pos] + signature + seq[pos:]
        sequences.append(seq)
        cohorts.append(c)
        clusters.append(z)

    cohorts = np.asarray(cohorts, dtype=np.int32)
    clusters = np.asarray(clusters, dtype=np.int32)
    N_cz = np.zeros((K, M), dtype=np.int64)
    for c, z in zip(cohorts, clusters):
        N_cz[c, z] += 1
    return sequences, cohorts, clusters, N_cz


# ── pattern counting ──────────────────────────────────────────────────────────

def count_pattern(seqs, cohorts, clusters, K, M, pattern):
    n_cz = np.zeros((K, M), dtype=np.int64)
    pl = pattern
    for s, c, z in zip(seqs, cohorts, clusters):
        i = 0
        for t in s:
            if i < len(pl) and t == pl[i]:
                i += 1
        if i == len(pl):
            n_cz[c, z] += 1
    return n_cz


# ── per-call timing: 200 random candidates ────────────────────────────────────

def time_per_call(sequences, cohorts, clusters, N_cz, V, bound_fn,
                  n_samples: int = 200, lam: float = 50.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    K, M = N_cz.shape
    times = []
    for _ in range(n_samples):
        length = int(rng.choice([1, 2]))
        pat = list(rng.integers(0, V, size=length))
        n_cz = count_pattern(sequences, cohorts, clusters, K, M, pat)
        t0 = time.perf_counter()
        bound_fn(n_cz, N_cz, lam=lam)
        times.append(time.perf_counter() - t0)
    return np.array(times)


# ── end-to-end mining (L=1+2) ────────────────────────────────────────────────

def mine_e2e(sequences, cohorts, clusters, N_cz, V, bound_fn,
             lam: float = 50.0, theta_sup: float = 0.02):
    K, M = N_cz.shape
    N = len(sequences)
    t_start = time.perf_counter()
    bound_total = 0.0
    explored = 0
    pruned_bound = 0
    pruned_sup = 0
    qualified = []

    # L1 singletons
    L1_pass = []
    for v in range(V):
        explored += 1
        n_cz = count_pattern(sequences, cohorts, clusters, K, M, [v])
        if n_cz.sum() / N < theta_sup:
            pruned_sup += 1
            continue
        tb = time.perf_counter()
        ub = bound_fn(n_cz, N_cz, lam=lam)
        bound_total += time.perf_counter() - tb
        if ub < 0:
            pruned_bound += 1
            continue
        L1_pass.append([v])
        qualified.append([v])

    # L2 pairs
    for p in L1_pass:
        for v in range(V):
            cand = p + [v]
            explored += 1
            n_cz = count_pattern(sequences, cohorts, clusters, K, M, cand)
            if n_cz.sum() / N < theta_sup:
                pruned_sup += 1
                continue
            tb = time.perf_counter()
            ub = bound_fn(n_cz, N_cz, lam=lam)
            bound_total += time.perf_counter() - tb
            if ub < 0:
                pruned_bound += 1
                continue
            qualified.append(cand)

    wall = time.perf_counter() - t_start
    return {
        "explored": explored,
        "pruned_sup": pruned_sup,
        "pruned_bound": pruned_bound,
        "qualified": len(qualified),
        "bound_time_s": bound_total,
        "wallclock_s": wall,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    from src.joint_bound_facewise import joint_upper_bound_facewise
    from src.joint_bound_polyk import joint_upper_bound_polyk

    K_GRID = [3, 5, 8, 12, 18, 24]
    M = 3
    V = 10
    N = 1500
    LAM = 50.0
    THETA_SUP = 0.02
    N_CALL_SAMPLES = 200

    rows_call = []   # per-call timing
    rows_e2e = []    # end-to-end mining

    print(f"{'K':>4}  {'3^K':>8}  {'K+1':>5}  "
          f"{'face_median_us':>15}  {'poly_median_us':>15}  "
          f"{'speedup_call':>13}")
    print("-" * 75)

    for K in K_GRID:
        seqs, cohs, clus, N_cz = generate_synthetic(K=K, M=M, V=V, N=N, seed=42)

        # --- per-call timing ---
        t_face = time_per_call(seqs, cohs, clus, N_cz, V,
                               joint_upper_bound_facewise,
                               n_samples=N_CALL_SAMPLES, lam=LAM)
        t_poly = time_per_call(seqs, cohs, clus, N_cz, V,
                               joint_upper_bound_polyk,
                               n_samples=N_CALL_SAMPLES, lam=LAM)

        face_med_us = float(np.median(t_face) * 1e6)
        poly_med_us = float(np.median(t_poly) * 1e6)
        speedup_call = face_med_us / max(poly_med_us, 1e-9)

        print(f"{K:>4}  {3**K:>8}  {K+1:>5}  "
              f"{face_med_us:>15.2f}  {poly_med_us:>15.2f}  "
              f"{speedup_call:>13.2f}x")

        rows_call.append({
            "K": K,
            "M": M,
            "V": V,
            "N": N,
            "face_candidates_3K": 3**K,
            "poly_candidates_Kp1": K + 1,
            "face_median_us": face_med_us,
            "face_p25_us": float(np.percentile(t_face, 25) * 1e6),
            "face_p75_us": float(np.percentile(t_face, 75) * 1e6),
            "face_mean_us": float(t_face.mean() * 1e6),
            "poly_median_us": poly_med_us,
            "poly_p25_us": float(np.percentile(t_poly, 25) * 1e6),
            "poly_p75_us": float(np.percentile(t_poly, 75) * 1e6),
            "poly_mean_us": float(t_poly.mean() * 1e6),
            "speedup_call": speedup_call,
        })

    print("\n--- End-to-end mining (L<=2) ---")
    print(f"{'K':>4}  {'face_wall_s':>12}  {'poly_wall_s':>12}  "
          f"{'face_bound_s':>13}  {'poly_bound_s':>13}  {'speedup_e2e':>12}")
    print("-" * 75)

    for K in K_GRID:
        seqs, cohs, clus, N_cz = generate_synthetic(K=K, M=M, V=V, N=N, seed=42)

        r_face = mine_e2e(seqs, cohs, clus, N_cz, V,
                          joint_upper_bound_facewise, lam=LAM, theta_sup=THETA_SUP)
        r_poly = mine_e2e(seqs, cohs, clus, N_cz, V,
                          joint_upper_bound_polyk, lam=LAM, theta_sup=THETA_SUP)

        speedup_e2e = r_face["wallclock_s"] / max(r_poly["wallclock_s"], 1e-9)
        speedup_bnd = r_face["bound_time_s"] / max(r_poly["bound_time_s"], 1e-9)

        print(f"{K:>4}  {r_face['wallclock_s']:>12.3f}  {r_poly['wallclock_s']:>12.3f}  "
              f"{r_face['bound_time_s']:>13.4f}  {r_poly['bound_time_s']:>13.4f}  "
              f"{speedup_e2e:>12.2f}x")

        rows_e2e.append({
            "K": K,
            "M": M,
            "V": V,
            "N": N,
            "face_explored": r_face["explored"],
            "face_pruned_bound": r_face["pruned_bound"],
            "face_qualified": r_face["qualified"],
            "face_wallclock_s": r_face["wallclock_s"],
            "face_bound_time_s": r_face["bound_time_s"],
            "poly_explored": r_poly["explored"],
            "poly_pruned_bound": r_poly["pruned_bound"],
            "poly_qualified": r_poly["qualified"],
            "poly_wallclock_s": r_poly["wallclock_s"],
            "poly_bound_time_s": r_poly["bound_time_s"],
            "speedup_e2e": speedup_e2e,
            "speedup_bound_only": speedup_bnd,
        })

    # ── write results ──────────────────────────────────────────────────────────
    out_call = RESULTS / "polyk_scaling_percall.json"
    out_e2e  = RESULTS / "polyk_scaling_e2e.json"

    with open(out_call, "w") as f:
        json.dump(rows_call, f, indent=2)
    with open(out_e2e, "w") as f:
        json.dump(rows_e2e, f, indent=2)

    # ── crossover analysis ────────────────────────────────────────────────────
    print("\n--- Crossover analysis ---")
    print("K where face-wise becomes >10x slower than poly-K:")
    for r in rows_call:
        if r["speedup_call"] >= 10.0:
            print(f"  K={r['K']}  speedup={r['speedup_call']:.1f}x  "
                  f"face={r['face_median_us']:.1f}us  poly={r['poly_median_us']:.1f}us")
            break
    else:
        # find max
        best = max(rows_call, key=lambda r: r["speedup_call"])
        print(f"  Max speedup at K={best['K']}: {best['speedup_call']:.1f}x")

    # ── combined table for paper ──────────────────────────────────────────────
    paper_rows = []
    call_by_K = {r["K"]: r for r in rows_call}
    e2e_by_K  = {r["K"]: r for r in rows_e2e}
    for K in K_GRID:
        c = call_by_K[K]
        e = e2e_by_K[K]
        paper_rows.append({
            "K": K,
            "face_internal_ops": 3**K,
            "poly_internal_ops": K + 1,
            "face_median_us": round(c["face_median_us"], 2),
            "poly_median_us": round(c["poly_median_us"], 2),
            "speedup_call": round(c["speedup_call"], 2),
            "face_e2e_s": round(e["face_wallclock_s"], 3),
            "poly_e2e_s": round(e["poly_wallclock_s"], 3),
            "speedup_e2e": round(e["speedup_e2e"], 2),
        })

    out_paper = RESULTS / "polyk_scaling_paper_table.json"
    with open(out_paper, "w") as f:
        json.dump(paper_rows, f, indent=2)

    print(f"\nWrote:")
    print(f"  {out_call}")
    print(f"  {out_e2e}")
    print(f"  {out_paper}")

    # ── figure-ready CSV ──────────────────────────────────────────────────────
    import csv
    out_csv = RESULTS / "polyk_scaling_paper_table.csv"
    if paper_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=paper_rows[0].keys())
            writer.writeheader()
            writer.writerows(paper_rows)
        print(f"  {out_csv}")


if __name__ == "__main__":
    main()
