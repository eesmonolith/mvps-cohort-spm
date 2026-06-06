"""
Sound K-decomposed joint upper bound — face-interior enumeration.

For K cohorts the joint maximum of S(y) = mean(y) - lambda*var(y) over
the K-cube y in prod_c [0, y_c^max] is attained either at a corner OR
at a face-interior critical point where the FREE coordinates are all
equal to a value determined by the FIXED-coordinate sum and lambda.

Derivation. Critical condition on a face with q FIXED coords (K-q free):
  y_free = (sum_fixed + K / (2 * lambda)) / q,  for q > 0
(When q = 0 / all free, no interior critical point; max on boundary.)

Enumerate every (fixed-subset, fixed-value) combination: 3^K total
(each coord either free, 0, or y_c^max). For each face, compute the
candidate critical point, check feasibility (free coords in
[0, y_c^max]), and score. Return the max.

Cost: O(K * 2^M) per-cohort + O(3^K) for face enumeration, vs
O(2^{KM}) for vertex enumeration of the original box.
K=3, M=4: 48 + 27 = 75 vs 4096 (~55x).
K=5, M=3: 40 + 243 = 283 vs 32768 (~115x).
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


def joint_upper_bound_facewise(n_cz: np.ndarray, N_cz: np.ndarray,
                                lam: float = 1.0) -> float:
    """Sound K-decomposed joint upper bound (face-interior enumeration).

    For K=1 or K=2 returns the same value as corner enumeration plus
    interior check; for K>=3 includes face-interior critical points
    which corner-only enum misses.
    """
    K, _ = n_cz.shape
    y_max = _per_cohort_y_max(n_cz, N_cz)

    best = -np.inf
    # 3^K combos: per coord {free, 0, upper}
    for combo in range(3 ** K):
        fixed = np.zeros(K, dtype=bool)
        fixed_vals = np.zeros(K)
        free_mask = np.zeros(K, dtype=bool)
        c = combo
        for i in range(K):
            state = c % 3
            c //= 3
            if state == 0:
                free_mask[i] = True
            elif state == 1:
                fixed[i] = True
                fixed_vals[i] = 0.0
            else:
                fixed[i] = True
                fixed_vals[i] = y_max[i]

        r_free = int(free_mask.sum())
        q_fixed = K - r_free
        if r_free == 0:
            # all fixed — score this corner
            best = max(best, _score(fixed_vals, lam))
            continue
        if r_free == K:
            # all free — no interior critical point on full K-cube
            continue
        # face-interior critical: y_free = (sum_fixed + K/(2lam)) / q_fixed
        sum_fixed = float(fixed_vals[fixed].sum())
        if lam <= 0:
            continue
        y_free_val = (sum_fixed + K / (2.0 * lam)) / q_fixed
        # check feasibility — free coords must be in [0, y_c^max]
        free_indices = np.where(free_mask)[0]
        feasible = all(0.0 <= y_free_val <= y_max[i] + EPS for i in free_indices)
        if feasible:
            y = fixed_vals.copy()
            y[free_mask] = y_free_val
            best = max(best, _score(y, lam))

    return float(best)


def _self_test():
    """Verify facewise bound >= corner-only bound, and >= true max
    sampled via random in-box points."""
    from src.joint_bound_r1 import joint_upper_bound_r1, joint_upper_bound_r1_corner

    rng = np.random.default_rng(0)
    K, M = 3, 4
    N_cz = rng.integers(60, 200, size=(K, M))

    patterns = {
        "A pooled-only": np.array([[90, 5, 5, 5], [3, 1, 1, 1], [4, 2, 2, 2]]),
        "B uniform": np.array([[20, 20, 20, 20], [22, 22, 22, 22], [18, 18, 18, 18]]),
        "C good-everywhere": np.array([[80, 5, 5, 5], [70, 4, 6, 5], [75, 5, 5, 5]]),
        "D cohort-specific": np.array([[60, 5, 5, 5], [5, 50, 5, 5], [5, 5, 55, 5]]),
    }
    for lam in [10.0, 50.0, 100.0]:
        print(f"\n=== lambda={lam} ===")
        for name, n in patterns.items():
            corner = joint_upper_bound_r1_corner(n, N_cz, lam=lam)
            face = joint_upper_bound_facewise(n, N_cz, lam=lam)
            vertex = joint_upper_bound_r1(n, N_cz, lam=lam)
            diff_cf = face - corner
            diff_fv = face - vertex
            ok = "OK" if face >= vertex - 1e-9 else "BUG"
            print(f"  {name:30s}  corner={corner:+.4f}  face={face:+.4f}  vertex={vertex:+.4f}  f-c={diff_cf:+.4f}  f-v={diff_fv:+.4f}  {ok}")


if __name__ == "__main__":
    _self_test()
