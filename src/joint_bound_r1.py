"""
Joint anti-monotone upper bound for the R1 objective.

S_R1(p) = mean_c IG_c(p) - lambda * Var_c IG_c(p).

Naive separable bound: max mean_c IG_c (computed via the per-cohort
analogue of the Cheng et al. (2007) IG upper bound) MINUS lambda *
0 (the minimum attainable variance). The variance-zero hypothesis is
never tight in practice, so the naive bound massively overshoots.

Joint bound: max over the box B(p) of [mean - lambda * var], evaluated
by vertex enumeration of B(p). Variance and mean share the same atomic
rate vector u, so they trade off automatically.

Both bounds are anti-monotone (B(p') ⊆ B(p)).
"""
from __future__ import annotations

import itertools
import numpy as np

from src.scoring_r1 import per_cohort_ig

EPS = 1e-12


def _evaluate(u_cz: np.ndarray, N_cz: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Compute S_R1 evaluated at rate vector u (interpreted as
    n_{c,z}(p) / N_{c,z}). Returns (S_R1, per-cohort IG)."""
    n_cz = (u_cz * N_cz).round()  # convert back to counts
    ig_c = per_cohort_ig(n_cz, N_cz)
    mean_ = float(ig_c.mean())
    var_ = float(ig_c.var())
    return mean_ - lam * var_, ig_c


def _vertices(n_cz: np.ndarray, N_cz: np.ndarray):
    K, M = n_cz.shape
    upper = n_cz / np.maximum(N_cz, 1)
    KM = K * M
    flat = upper.flatten()
    for mask in range(1 << KM):
        u = np.zeros(KM)
        for i in range(KM):
            if mask & (1 << i):
                u[i] = flat[i]
        yield u.reshape(K, M)


def joint_upper_bound_r1(n_cz: np.ndarray, N_cz: np.ndarray, lam: float = 1.0) -> float:
    """Joint upper bound via vertex enumeration on B(p) (slow, O(2^{KM}))."""
    K, M = n_cz.shape
    if K * M > 16:
        return 1.0
    best = -np.inf
    for u in _vertices(n_cz, N_cz):
        S, _ = _evaluate(u, N_cz, lam)
        if S > best:
            best = S
    return float(best)


def joint_upper_bound_r1_corner(n_cz: np.ndarray, N_cz: np.ndarray,
                                 lam: float = 1.0) -> float:
    """Joint upper bound via 2^K corner enumeration after per-cohort
    decoupling (Appendix A).

    Decoupling: B(p) = prod_c B_c(p), each IG_c depends only on u_c.
    Score(u) = mean_c IG_c(u_c) - lam * Var_c IG_c(u_c).

    Step 1: per-cohort, find y_c^max = max_{u_c in B_c(p)} IG_c(u_c)
            by enumerating 2^M vertices of the M-dim sub-box.
    Step 2: outer-relax to y in prod_c [0, y_c^max]. By Appendix A
            the K-cube has no interior critical point; maximum at
            corner. Enumerate 2^K corners.

    Total cost: K * 2^M + 2^K, vs 2^{K M} for full vertex enum.
    For K=3, M=4: 3*16 + 8 = 56 vs 4096 (~73x speedup).
    """
    K, M = n_cz.shape
    upper = n_cz / np.maximum(N_cz, 1)

    # Step 1: per-cohort y_c^max via M-dim vertex enumeration
    y_max = np.zeros(K)
    for c in range(K):
        best_c = 0.0
        for mask in range(1 << M):
            u_c = np.zeros(M)
            for j in range(M):
                if mask & (1 << j):
                    u_c[j] = upper[c, j]
            # build synthetic n with only cohort c populated; per_cohort_ig
            # restricted to cohort c only depends on n[c, :] vs N[c, :]
            n_synth = np.zeros_like(n_cz, dtype=np.float64)
            n_synth[c] = u_c * N_cz[c]
            ig_c = per_cohort_ig(n_synth, N_cz)[c]
            if ig_c > best_c:
                best_c = ig_c
        y_max[c] = best_c

    # Step 2: 2^K corner enumeration over [0, y_max[c]] cube
    best = -np.inf
    for mask in range(1 << K):
        y = np.zeros(K)
        for c in range(K):
            if mask & (1 << c):
                y[c] = y_max[c]
        score = float(y.mean() - lam * y.var())
        if score > best:
            best = score
    return float(best)


def naive_separable_bound_r1(n_cz: np.ndarray, N_cz: np.ndarray, lam: float = 1.0) -> float:
    """Naive separable bound: max mean independently, min variance independently.

    Min variance = 0 (achievable in principle by setting all per-cohort
    IG equal). Max mean is the largest per-cohort IG over the box, which
    is bounded above by the cohort-c IG upper bound at the vertex
    u_{c,z} = n_{c,z}(p)/N_{c,z} (i.e. taking all mass and IG-maximising
    within cohort c). Computed by per-cohort max independently.
    """
    K, M = n_cz.shape
    if K * M > 16:
        return 1.0
    upper = n_cz / np.maximum(N_cz, 1)
    # For each cohort independently, find max IG_c(u_c) over u_c in
    # [0, upper[c,:]]. Vertex enumeration over the M-dim subcube.
    max_per_cohort = np.zeros(K)
    for c in range(K):
        # Enumerate vertices of cohort-c subcube
        for mask in range(1 << M):
            u_c = np.zeros(M)
            for j in range(M):
                if mask & (1 << j):
                    u_c[j] = upper[c, j]
            # Build a synthetic n_cz that only has cohort c populated
            n_synth = np.zeros_like(n_cz, dtype=np.float64)
            n_synth[c] = u_c * N_cz[c]
            ig = per_cohort_ig(n_synth, N_cz)[c]
            if ig > max_per_cohort[c]:
                max_per_cohort[c] = ig
    mean_upper = max_per_cohort.mean()  # if all cohorts hit their own max
    var_lower = 0.0                      # min attainable variance
    return float(mean_upper - lam * var_lower)


def empirical_tightness_r1(n_cz: np.ndarray, N_cz: np.ndarray, lam: float = 1.0) -> dict:
    actual = float(_evaluate(n_cz / np.maximum(N_cz, 1), N_cz, lam)[0])
    naive = naive_separable_bound_r1(n_cz, N_cz, lam)
    joint = joint_upper_bound_r1(n_cz, N_cz, lam)
    return {
        "actual": actual,
        "naive_bound": naive,
        "joint_bound": joint,
        "tightness_joint": joint / max(abs(actual), EPS),
        "joint_vs_naive": joint / max(abs(naive), EPS),
        "naive_minus_joint_abs": naive - joint,
    }


def _self_test():
    rng = np.random.default_rng(0)
    K, M = 3, 4
    N_cz = rng.integers(60, 200, size=(K, M))

    print("=== R1 joint bound smoke test (K=3, M=4) ===\n")
    # Same fixtures as scoring smoke
    patterns = {
        "A pooled-only": np.array([[90, 5, 5, 5], [3, 1, 1, 1], [4, 2, 2, 2]]),
        "B uniform": np.array([[20, 20, 20, 20], [22, 22, 22, 22], [18, 18, 18, 18]]),
        "C good-everywhere": np.array([[80, 5, 5, 5], [70, 4, 6, 5], [75, 5, 5, 5]]),
        "D cohort-specific-aligned": np.array([[60, 5, 5, 5], [5, 50, 5, 5], [5, 5, 55, 5]]),
    }
    for name, n in patterns.items():
        r = empirical_tightness_r1(n, N_cz)
        print(f"Pattern {name}:")
        print(f"  actual S_R1 = {r['actual']:+.4f}")
        print(f"  naive bound = {r['naive_bound']:.4f}")
        print(f"  joint bound = {r['joint_bound']:.4f}")
        print(f"  joint-naive abs gap = {r['naive_minus_joint_abs']:.4f}")
        print()


if __name__ == "__main__":
    _self_test()
