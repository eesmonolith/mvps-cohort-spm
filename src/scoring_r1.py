"""
R1 scoring: cohort-conditional information gain (non-separable).

For each cohort c, compute IG_c(p) = mutual information between pattern
occurrence and cluster label, computed using only entities in cohort c.

The R1 joint score is

    S_R1(p) = mean_c IG_c(p) - lambda * Var_c IG_c(p)        (eq R1)

normalised by H_max(Z) so it lives in roughly [-1, 1].

Non-separability claim
======================
S_R1 cannot be written as F(g(stability(p)), h(discrim(p))) for any
functions g, h where stability is the cohort-pooled support variance
and discrim is the cohort-pooled IG. Reason: Var_c IG_c depends on
the joint atomic distribution {n_{c,z}(p)}; the two pooled marginals
discard this information. The counterexample search in
`mine_counterexamples` makes this constructive.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# ----- cohort-conditional IG -----

def _entropy(p):
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def per_cohort_ig(n_cz: np.ndarray, N_cz: np.ndarray) -> np.ndarray:
    """Return IG_c(p) for c=0..K-1, each as normalised mutual information
    between pattern occurrence and cluster label, evaluated using only
    cohort-c entities.

    IG_c(p) = H(Z | C=c) - sup_c(p) * H(Z | X_p=1, C=c)
                          - (1 - sup_c(p)) * H(Z | X_p=0, C=c)
    """
    K, M = n_cz.shape
    out = np.zeros(K)
    for c in range(K):
        nz_c = n_cz[c, :].astype(np.float64)        # occurrences per cluster within cohort c
        Nz_c = N_cz[c, :].astype(np.float64)        # population per cluster within cohort c
        n_total = Nz_c.sum()
        if n_total <= 0:
            continue

        # P(Z | C=c)
        pz = Nz_c / n_total
        Hz = _entropy(pz)
        if Hz <= 0:
            continue

        # P(X_p=1 | C=c)
        sup_c = nz_c.sum() / n_total
        if sup_c <= 0 or sup_c >= 1:
            # vacuous: pattern absent or saturates within cohort
            out[c] = 0.0
            continue

        # P(Z | X_p=1, C=c) -- entities matching pattern, by cluster
        denom_pos = nz_c.sum()
        if denom_pos > 0:
            ppos = nz_c / denom_pos
            Hpos = _entropy(ppos)
        else:
            Hpos = 0.0

        # P(Z | X_p=0, C=c) -- entities NOT matching pattern, by cluster
        not_nz = Nz_c - nz_c
        denom_neg = not_nz.sum()
        if denom_neg > 0:
            pneg = not_nz / denom_neg
            Hneg = _entropy(pneg)
        else:
            Hneg = 0.0

        ig = Hz - (sup_c * Hpos + (1.0 - sup_c) * Hneg)
        # Normalise by max attainable entropy in cohort c
        out[c] = max(0.0, ig / Hz)
    return out


def r1_components(n_cz: np.ndarray, N_cz: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (mean_c IG_c, var_c IG_c, IG_c vector)."""
    ig_c = per_cohort_ig(n_cz, N_cz)
    return float(ig_c.mean()), float(ig_c.var()), ig_c


def s_r1(n_cz: np.ndarray, N_cz: np.ndarray, lam: float = 1.0) -> float:
    """S_R1(p) = mean_c IG_c(p) - lam * Var_c IG_c(p)."""
    mean_, var_, _ = r1_components(n_cz, N_cz)
    return mean_ - lam * var_


# ----- separable approximations (the naive baseline to beat) -----

def s_pooled(n_cz: np.ndarray, N_cz: np.ndarray) -> tuple[float, float, float]:
    """Cohort-pooled IG and cohort-pooled support stability,
    matching our original v1 paper formulation.

    Returns (stability_pooled, discrim_pooled, S_v1 = stab * disc).
    """
    from src.scoring import stability, discrim, joint_score
    return stability(n_cz, N_cz), discrim(n_cz, N_cz), joint_score(n_cz, N_cz)


# ----- smoke test -----

def _self_test():
    print("=== R1 scoring smoke test ===\n")
    rng = np.random.default_rng(0)
    K, M = 3, 4
    N_cz = rng.integers(60, 200, size=(K, M))

    # Pattern A: pooled IG looks good (concentrated in cluster 0 overall)
    # but only cohort-0 drives it -> S_R1 should kill, pooled survives.
    n_A = np.array(
        [[90, 5, 5, 5],
         [3, 1, 1, 1],
         [4, 2, 2, 2]]
    )
    # Pattern B: every cohort has uniform support -> pooled stability=1
    # but IG_c=0 in every cohort -> S_R1 = 0
    n_B = np.array(
        [[20, 20, 20, 20],
         [22, 22, 22, 22],
         [18, 18, 18, 18]]
    )
    # Pattern C: every cohort has high IG with same direction
    # -> mean high, var low, S_R1 high; pooled also good
    n_C = np.array(
        [[80, 5, 5, 5],
         [70, 4, 6, 5],
         [75, 5, 5, 5]]
    )
    # Pattern D: every cohort has high IG but in *different* clusters
    # (e.g. cohort-0 -> cluster 0, cohort-1 -> cluster 1, etc.)
    # -> mean high, var low, S_R1 high; pooled IG is *killed* by mixing
    n_D = np.array(
        [[60, 5, 5, 5],
         [5, 50, 5, 5],
         [5, 5, 55, 5]]
    )

    for name, n in [("A pooled-only", n_A),
                    ("B uniform-stable", n_B),
                    ("C good-everywhere", n_C),
                    ("D cohort-specific-but-aligned", n_D)]:
        stab, disc, S_v1 = s_pooled(n, N_cz)
        S = s_r1(n, N_cz, lam=1.0)
        mean_, var_, ig_c = r1_components(n, N_cz)
        print(f"Pattern {name}:")
        print(f"  pooled: stability={stab:.3f}  discrim={disc:.3f}  S_v1={S_v1:.4f}")
        print(f"  per-cohort IG_c: {np.round(ig_c, 3).tolist()}")
        print(f"  mean_IG={mean_:.3f}  var_IG={var_:.4f}  S_R1={S:.4f}")
        print()


if __name__ == "__main__":
    _self_test()
