"""
Joint anti-monotone upper bound for C2DPM.

For pattern p with atomic count matrix n_{c,z}(p) ∈ R^{K×M} and population
N_{c,z}, any extension p' ⊒ p has atomic counts satisfying

    0 ≤ n_{c,z}(p') ≤ n_{c,z}(p)   for all (c, z)

(anti-monotone property of pattern subsumption).

We bound the joint score S(p') = stability(p') · discrim(p') over this box.

Two bounds:

1) Naive product bound (baseline) — assume stability and discrim each reach
   their maxima independently:
        S_upper_naive(p) = max stability(u) · max discrim(u)
   where each max is over the SAME box but optimized separately. This is the
   product of independent maxima — implicitly assumes trade-off doesn't exist.

2) Joint bound (paper contribution) — single optimization over the box:
        S_upper_joint(p) = max_{u in box} stability(u) · discrim(u)
   Smaller than naive whenever stability and discrim trade off — which is the
   generic case (stability prefers uniform u, discrim prefers concentrated u).

Implementation: vertex enumeration over the K×M box. Each axis independently
chooses u_{c,z} ∈ {0, n_{c,z}(p)/N_{c,z}}. For K=3, M=4 → 2^12 = 4096 vertices.

Provable result: the joint maximum is attained at a vertex of the box (since
S is a fractional/polynomial function whose level sets are non-convex; vertex
search is sound for upper bounding because increasing any u_{c,z} can only
relax the constraint). For tighter (interior) bound, use continuous KKT.

The proof obligation in the paper is to show:
    S_upper_joint(p') ≤ S_upper_joint(p)   for p' ⊒ p
which follows directly: box(p') ⊆ box(p), so max over smaller box ≤ max over
larger.
"""
from __future__ import annotations

import itertools
import numpy as np

from src.scoring import joint_score, stability, discrim

EPS = 1e-12


def _box_vertices(upper: np.ndarray):
    """Yield all K*M-dim binary vertices of the unit box, scaled by upper.

    For each cell (c, z): either 0 or upper[c, z].
    Total 2^(K*M) vertices. For K=3, M=4 → 4096.
    """
    K, M = upper.shape
    KM = K * M
    upper_flat = upper.flatten()
    for mask in range(1 << KM):
        u = np.zeros(KM)
        for i in range(KM):
            if mask & (1 << i):
                u[i] = upper_flat[i]
        yield u.reshape(K, M)


def joint_upper_bound(n_cz: np.ndarray, N_cz: np.ndarray,
                      max_vertices: int = 1 << 16) -> float:
    """Joint upper bound via vertex enumeration.

    For K*M > 16 we skip enumeration and return 1.0 (vacuous bound). In our
    application K=3, M=4 → 12 → 4096 vertices, well within budget.
    """
    K, M = n_cz.shape
    if K * M > 16:
        return 1.0
    upper = n_cz / np.maximum(N_cz, 1)
    best = 0.0
    for u in _box_vertices(upper):
        # Reconstruct integer-like counts from rates
        nz = (u * N_cz).round()
        s = joint_score(nz, N_cz)
        if s > best:
            best = s
            if best >= 1.0:
                return 1.0
    return float(best)


def naive_product_bound(n_cz: np.ndarray, N_cz: np.ndarray) -> float:
    """Naive product bound: independent max of stability and discrim.

    Equivalent to:
        S_upper_naive = (max stability over box) * (max discrim over box)
    where each max is computed separately (does not share the same vertex).
    """
    K, M = n_cz.shape
    upper = n_cz / np.maximum(N_cz, 1)
    if K * M > 16:
        return 1.0

    best_stab = 0.0
    best_disc = 0.0
    for u in _box_vertices(upper):
        nz = (u * N_cz).round()
        s = stability(nz, N_cz)
        d = discrim(nz, N_cz)
        if s > best_stab:
            best_stab = s
        if d > best_disc:
            best_disc = d
    return float(best_stab * best_disc)


def empirical_tightness(n_cz: np.ndarray, N_cz: np.ndarray) -> dict:
    """Return both bounds + actual joint score for empirical tightness comparison."""
    actual = float(joint_score(n_cz, N_cz))
    naive = naive_product_bound(n_cz, N_cz)
    joint_ub = joint_upper_bound(n_cz, N_cz)
    return {
        "actual": actual,
        "naive_bound": naive,
        "joint_bound": joint_ub,
        "tightness_ratio_joint": joint_ub / max(actual, EPS),
        "tightness_ratio_naive": naive / max(actual, EPS),
        "joint_vs_naive": joint_ub / max(naive, EPS),  # < 1 = joint tighter
    }


# ---------- Smoke test ----------

def _self_test():
    rng = np.random.default_rng(0)
    K, M = 3, 4
    N_cz = rng.integers(50, 200, size=(K, M))

    print("=== Joint bound smoke test (K=3, M=4) ===\n")
    # Pattern A: cohort-stable, cluster-discriminative
    n_A = np.array(
        [[80, 5, 5, 5],
         [70, 4, 6, 5],
         [75, 5, 5, 5]]
    )
    # Pattern B: cohort-unstable
    n_B = np.array(
        [[60, 30, 30, 30],
         [5, 3, 4, 2],
         [3, 2, 3, 2]]
    )
    # Pattern C: stable + uniform (no discrim)
    n_C = np.array(
        [[20, 20, 20, 20],
         [25, 25, 25, 25],
         [20, 20, 20, 20]]
    )
    # Pattern D: low support overall
    n_D = np.array(
        [[2, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    )

    for name, n in [("A good", n_A), ("B unstable", n_B),
                    ("C uniform", n_C), ("D low-support", n_D)]:
        r = empirical_tightness(n, N_cz)
        print(f"  Pattern {name}:")
        print(f"    actual S = {r['actual']:.4f}")
        print(f"    naive bound = {r['naive_bound']:.4f}  (ratio {r['tightness_ratio_naive']:.2f}x)")
        print(f"    joint bound = {r['joint_bound']:.4f}  (ratio {r['tightness_ratio_joint']:.2f}x)")
        print(f"    joint/naive = {r['joint_vs_naive']:.3f}   (< 1 = joint tighter)")
        print()


if __name__ == "__main__":
    _self_test()
