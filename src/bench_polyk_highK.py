"""High-K bound benchmark: poly-K vs face-wise vs (analytic) vertex.

For K in {3, 5, 8, 12, 18, 24} measure per-call wall-clock of:
- joint_upper_bound_polyk
- joint_upper_bound_facewise
- vertex enum (only at small K to avoid 2^{KM} blowup)

Reports: ops count + time. Confirms polynomial advantage at high K.
"""
from __future__ import annotations
import json
import time
import numpy as np

from src.joint_bound_polyk import joint_upper_bound_polyk
from src.joint_bound_facewise import joint_upper_bound_facewise
from src.joint_bound_r1 import joint_upper_bound_r1


def gen_random_pattern(K, M, rng):
    N_cz = rng.integers(60, 200, size=(K, M))
    n_cz = (N_cz * rng.uniform(0.05, 0.5, size=(K, M))).astype(int)
    return n_cz, N_cz


def bench(K, M, n_patterns, n_repeats, lam=50.0):
    rng = np.random.default_rng(K * 100 + M)
    patterns = [gen_random_pattern(K, M, rng) for _ in range(n_patterns)]

    # Poly
    t0 = time.time()
    for _ in range(n_repeats):
        for n_cz, N_cz in patterns:
            joint_upper_bound_polyk(n_cz, N_cz, lam=lam)
    t_poly = (time.time() - t0) / (n_patterns * n_repeats) * 1e6  # us per call

    # Face-wise (only if 3^K reasonable)
    if K <= 12:
        t0 = time.time()
        for _ in range(n_repeats):
            for n_cz, N_cz in patterns:
                joint_upper_bound_facewise(n_cz, N_cz, lam=lam)
        t_face = (time.time() - t0) / (n_patterns * n_repeats) * 1e6
    else:
        t_face = float('nan')

    # Vertex (only if 2^{KM} reasonable, KM <= 18)
    if K * M <= 18:
        t0 = time.time()
        for _ in range(n_repeats):
            for n_cz, N_cz in patterns:
                joint_upper_bound_r1(n_cz, N_cz, lam=lam)
        t_vertex = (time.time() - t0) / (n_patterns * n_repeats) * 1e6
    else:
        t_vertex = float('nan')

    return {
        "K": K, "M": M,
        "poly_us_per_call": t_poly,
        "facewise_us_per_call": t_face,
        "vertex_us_per_call": t_vertex,
        "ops_face_3K": 3**K,
        "ops_vertex_2KM": 2**(K*M) if K*M <= 30 else float('inf'),
        "ops_poly_K1": K + 1,
        "speedup_poly_vs_face": t_face / t_poly if t_face == t_face else float('inf'),
        "speedup_poly_vs_vertex": t_vertex / t_poly if t_vertex == t_vertex else float('inf'),
    }


def main():
    M = 3
    rows = []
    print(f"{'K':>3s}  {'poly':>10s}  {'face':>12s}  {'vertex':>12s}  {'p-vs-f':>10s}  {'p-vs-v':>10s}")
    for K in [3, 5, 8, 12, 18, 24]:
        n_pat = 50 if K <= 8 else 20
        n_rep = 5 if K <= 12 else 2
        try:
            r = bench(K, M, n_pat, n_rep)
            poly_s = f"{r['poly_us_per_call']:>8.1f}us"
            face_s = f"{r['facewise_us_per_call']:>10.1f}us" if r['facewise_us_per_call']==r['facewise_us_per_call'] else f"{'N/A':>12s}"
            vert_s = f"{r['vertex_us_per_call']:>10.1f}us" if r['vertex_us_per_call']==r['vertex_us_per_call'] else f"{'N/A':>12s}"
            pf = f"{r['speedup_poly_vs_face']:>8.1f}x" if r['speedup_poly_vs_face']==r['speedup_poly_vs_face'] else f"{'inf':>10s}"
            pv = f"{r['speedup_poly_vs_vertex']:>8.1f}x" if r['speedup_poly_vs_vertex']==r['speedup_poly_vs_vertex'] else f"{'inf':>10s}"
            print(f"{K:>3d}  {poly_s}  {face_s}  {vert_s}  {pf}  {pv}")
            rows.append(r)
        except Exception as e:
            print(f"K={K}: ERROR {e}")

    from src.config import RESULTS
    with open(RESULTS / "bound_walltime_highK.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {RESULTS / 'bound_walltime_highK.json'}")


if __name__ == "__main__":
    main()
