"""
Polynomial-K closed-form joint upper bound.

By KKT analysis: at optimum of f(y) = mean(y) - lambda*Var(y) on
box B = prod_c [0, y_c^max],
  - L (y_c = 0) condition requires y_bar <= -1/(2*lambda), which is
    impossible since y_c >= 0 implies y_bar >= 0 > -1/(2*lambda).
    So L is always empty.
  - U (y_c = y_c^max) condition requires y_c^max <= v where
    v = y_bar + 1/(2*lambda).
  - F (free, y_c = v) condition: all free coords equal v.

So coords are either UPPER (M_c <= v) or FREE (M_c > v, set to v).
Sorting M's in ascending order, U = bottom-(K-j), F = top-j, for some
threshold j in {0, 1, ..., K-1}; the corner case is j=0 (all upper).

Algorithm:
  1. Compute per-cohort y_c^max via 2^M sub-vertex enum (O(K * 2^M)).
  2. Sort y_max ascending. M_(1) <= M_(2) <= ... <= M_(K).
  3. For each j in {0, 1, ..., K-1}:
       U = K-j bottom; F = j top.
       v_j = (sum_U M + K / (2*lam)) / (K - j) if K-j > 0
       Skip if K-j == 0 (all free, undefined).
       Feasibility: M_(K-j) <= v_j < M_(K-j+1) (or v_j unbounded above).
       Evaluate f at y = (M_(1),...,M_(K-j), v, v,..., v).
  4. Add all-upper corner: y = M, f(M).
  5. Return max over all candidates.

Total: O(K * 2^M) for y_max + O(K log K) for sort + O(K) for K+1
candidate evaluations.
At K=3, M=4: 48 + 3 + 4 = 55 ops vs 3^K=27 face-wise vs 2^{KM}=4096
vertex. Crucially scales as O(K) for K=10, 100, etc., escaping the
3^K growth.
"""
from __future__ import annotations

import numpy as np

from src.scoring_r1 import per_cohort_ig

EPS = 1e-12


def _per_cohort_y_max(n_cz: np.ndarray, N_cz: np.ndarray) -> np.ndarray:
    K, M = n_cz.shape
    upper = n_cz / np.maximum(N_cz, 1)
    y_max = np.zeros(K)
    for c in range(K):
        best = 0.0
        for mask in range(1 << M):
            u_c = np.zeros(M)
            for j in range(M):
                if mask & (1 << j):
                    u_c[j] = upper[c, j]
            n_synth = np.zeros_like(n_cz, dtype=np.float64)
            n_synth[c] = u_c * N_cz[c]
            ig_c = per_cohort_ig(n_synth, N_cz)[c]
            if ig_c > best:
                best = ig_c
        y_max[c] = best
    return y_max


def _score(y: np.ndarray, lam: float) -> float:
    return float(y.mean() - lam * y.var())


def joint_upper_bound_polyk(n_cz: np.ndarray, N_cz: np.ndarray,
                             lam: float = 50.0) -> float:
    """Polynomial-K closed-form joint upper bound.

    Returns sound upper bound on
    max_{u in B(p)} (mean_c IG_c(u) - lam * Var_c IG_c(u))
    in O(K log K + K * 2^M) time, vs O(3^K) for face-wise or
    O(2^{KM}) for vertex enumeration.
    """
    K, _ = n_cz.shape
    if K == 0:
        return 0.0
    if lam <= 0:
        # gradient is constant 1/K > 0, maximum at all-upper corner
        y = _per_cohort_y_max(n_cz, N_cz)
        return _score(y, lam)
    y_max = _per_cohort_y_max(n_cz, N_cz)

    # Sort ascending
    order = np.argsort(y_max)  # smallest first
    M_sorted = y_max[order]

    best = -np.inf

    # All-upper corner (j=0): y = M, no free
    best = max(best, _score(y_max, lam))

    # j free coords (j = 1, 2, ..., K)
    # U = bottom (K-j), F = top j
    cum_M = np.cumsum(M_sorted)  # cum_M[i] = sum of bottom i+1
    for j in range(1, K + 1):
        nU = K - j  # number of upper coords
        if nU == 0:
            # all free: no interior critical (gradient zero condition contradictory)
            continue
        sum_U = cum_M[nU - 1] if nU > 0 else 0.0
        v = (sum_U + K / (2.0 * lam)) / nU
        # Feasibility (KKT):
        #   For U: M_c <= v for all c in U  <=>  M_(nU) <= v (largest in U)
        #   For F: 0 <= v <= M_c for all c in F  <=>  v <= M_(nU+1) (smallest in F)
        if nU > 0 and M_sorted[nU - 1] > v + EPS:
            continue  # infeasible: largest-U exceeds v
        if v < -EPS:
            continue
        # F values <= their M? F coords have M >= M_(nU), and v should be <= M_c for c in F.
        # Smallest M in F is M_(nU). v <= M_(nU)? Actually v >= M_(nU-1) (from U side),
        # and we want v <= M_(nU) (smallest in F).
        if nU < K and v > M_sorted[nU] + EPS:
            continue

        y = np.empty(K)
        y[order[:nU]] = M_sorted[:nU]  # upper coords
        y[order[nU:]] = v               # free coords
        best = max(best, _score(y, lam))

    return float(best)


def _self_test():
    """Compare polynomial bound to vertex enum and face-wise on random patterns."""
    from src.joint_bound_r1 import joint_upper_bound_r1
    from src.joint_bound_facewise import joint_upper_bound_facewise

    rng = np.random.default_rng(0)
    K, M = 3, 4
    N_cz = rng.integers(60, 200, size=(K, M))

    patterns = {
        "A pooled-only": np.array([[90, 5, 5, 5], [3, 1, 1, 1], [4, 2, 2, 2]]),
        "B uniform": np.array([[20, 20, 20, 20], [22, 22, 22, 22], [18, 18, 18, 18]]),
        "C good-everywhere": np.array([[80, 5, 5, 5], [70, 4, 6, 5], [75, 5, 5, 5]]),
        "D cohort-specific": np.array([[60, 5, 5, 5], [5, 50, 5, 5], [5, 5, 55, 5]]),
    }
    for lam in [10.0, 50.0, 100.0, 200.0]:
        print(f"\n=== lambda={lam} ===")
        for name, n in patterns.items():
            poly = joint_upper_bound_polyk(n, N_cz, lam=lam)
            face = joint_upper_bound_facewise(n, N_cz, lam=lam)
            vertex = joint_upper_bound_r1(n, N_cz, lam=lam)
            ok = "OK" if (poly >= vertex - 1e-6 and poly >= face - 1e-6) else "BUG"
            print(f"  {name:30s}  poly={poly:+.4f}  face={face:+.4f}  vertex={vertex:+.4f}  poly-face={poly-face:+.4f}  {ok}")


def _scaling_test():
    """Check K-scaling: time per call as K grows."""
    import time
    rng = np.random.default_rng(0)
    for K in [3, 5, 8, 10, 15, 20, 30]:
        M = 3
        N_cz = rng.integers(60, 200, size=(K, M))
        n_cz = (N_cz * rng.uniform(0.1, 0.8, size=(K, M))).astype(int)
        t0 = time.time()
        for _ in range(100):
            joint_upper_bound_polyk(n_cz, N_cz, lam=50.0)
        elapsed = (time.time() - t0) * 10  # microsec per call
        print(f"  K={K:3d}  M={M}  100 calls = {elapsed:.1f}us per call")


if __name__ == "__main__":
    _self_test()
    print("\n=== Scaling test ===")
    _scaling_test()
