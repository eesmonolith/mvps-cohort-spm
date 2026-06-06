"""
C2DPM scoring primitives.

Given a candidate pattern p and a `Counts` table with per-(cohort, cluster)
occurrence counts, compute:
  - sup_{c,z}(p)        atomic support
  - sup_c(p)            cohort-conditional support
  - sup_z(p)            cluster-conditional support
  - sup(p)              total support
  - sup_min(p)          min cohort support (Apriori axis)
  - stability(p)        1 - normalized variance of {sup_c}
  - discrim(p)          normalized information gain w.r.t. cluster label
  - S(p) = stability(p) * discrim(p)

All functions are pure (numpy-only) so they unit-test easily.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def cohort_support(n_cz: np.ndarray, N_cz: np.ndarray) -> np.ndarray:
    """sup_c(p) = sum_z n_{c,z}(p) / sum_z N_{c,z}, per cohort.

    Args:
        n_cz: shape (K, M) count of pattern in (cohort, cluster)
        N_cz: shape (K, M) total population per (cohort, cluster)
    Returns:
        shape (K,) per-cohort support
    """
    num = n_cz.sum(axis=1)
    den = N_cz.sum(axis=1)
    return num / np.maximum(den, 1)


def cluster_support(n_cz: np.ndarray, N_cz: np.ndarray) -> np.ndarray:
    """sup_z(p) = sum_c n_{c,z}(p) / sum_c N_{c,z}, per cluster."""
    num = n_cz.sum(axis=0)
    den = N_cz.sum(axis=0)
    return num / np.maximum(den, 1)


def total_support(n_cz: np.ndarray, N_cz: np.ndarray) -> float:
    return float(n_cz.sum() / max(N_cz.sum(), 1))


def cohort_min_support(n_cz: np.ndarray, N_cz: np.ndarray) -> float:
    """sup_min(p) = min_c sup_c(p). Anti-monotone scoring axis (Apriori)."""
    return float(cohort_support(n_cz, N_cz).min())


# ---------- Stability (cohort-axis) ----------

def stability(n_cz: np.ndarray, N_cz: np.ndarray) -> float:
    """1 - sigma^2_C(p) / [mu_C(p) * (1 - mu_C(p)) + eps].

    Returns in [0, 1]. Higher = pattern occurs at similar rates across cohorts.
    """
    sup_c = cohort_support(n_cz, N_cz)
    mu = float(sup_c.mean())
    var = float(((sup_c - mu) ** 2).mean())
    denom = mu * (1.0 - mu) + EPS
    s = 1.0 - var / denom
    return float(np.clip(s, 0.0, 1.0))


# ---------- Discriminativeness (cluster-axis via IG) ----------

def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def cluster_label_entropy(N_cz: np.ndarray) -> float:
    """H(Z) = entropy of cluster label distribution in population."""
    Nz = N_cz.sum(axis=0).astype(np.float64)
    total = Nz.sum()
    if total <= 0:
        return 0.0
    return _entropy(Nz / total)


def conditional_entropy_on_pattern(n_cz: np.ndarray, N_cz: np.ndarray) -> tuple[float, float]:
    """Return (H(Z | X_p=1), H(Z | X_p=0))."""
    nz = n_cz.sum(axis=0).astype(np.float64)
    n_total = nz.sum()
    if n_total > 0:
        H_pos = _entropy(nz / n_total)
    else:
        H_pos = 0.0

    Nz = N_cz.sum(axis=0).astype(np.float64)
    not_nz = Nz - nz
    not_total = not_nz.sum()
    if not_total > 0:
        H_neg = _entropy(not_nz / not_total)
    else:
        H_neg = 0.0
    return H_pos, H_neg


def information_gain(n_cz: np.ndarray, N_cz: np.ndarray) -> float:
    """IG(p) = H(Z) - [sup(p)*H(Z|X_p=1) + (1-sup(p))*H(Z|X_p=0)]."""
    HZ = cluster_label_entropy(N_cz)
    if HZ <= 0:
        return 0.0
    s = total_support(n_cz, N_cz)
    Hp, Hn = conditional_entropy_on_pattern(n_cz, N_cz)
    return float(HZ - (s * Hp + (1.0 - s) * Hn))


def discrim(n_cz: np.ndarray, N_cz: np.ndarray) -> float:
    """Normalized IG in [0, 1]."""
    HZ = cluster_label_entropy(N_cz)
    if HZ <= 0:
        return 0.0
    return float(np.clip(information_gain(n_cz, N_cz) / HZ, 0.0, 1.0))


# ---------- Joint score ----------

def joint_score(n_cz: np.ndarray, N_cz: np.ndarray) -> float:
    """S(p) = stability(p) * discrim(p). Both in [0, 1]."""
    return stability(n_cz, N_cz) * discrim(n_cz, N_cz)


# ---------- Quick smoke test ----------

def _self_test():
    # Synthetic: 3 cohorts, 4 clusters, K x M counts
    rng = np.random.default_rng(0)
    K, M = 3, 4
    # Population sizes (cohort x cluster)
    N_cz = rng.integers(50, 200, size=(K, M))

    # Pattern A: cohort-stable, cluster-discriminative (mostly in cluster 0)
    n_A = np.array(
        [[80, 5, 5, 5],
         [70, 4, 6, 5],
         [75, 5, 5, 5]]
    )
    # Pattern B: cohort-unstable (cohort 0 dominates)
    n_B = np.array(
        [[60, 30, 30, 30],
         [5, 3, 4, 2],
         [3, 2, 3, 2]]
    )
    # Pattern C: stable but cluster-uniform (no discrim)
    n_C = np.array(
        [[20, 20, 20, 20],
         [25, 25, 25, 25],
         [20, 20, 20, 20]]
    )
    for name, n in [("A (good)", n_A), ("B (unstable)", n_B), ("C (uniform)", n_C)]:
        s = stability(n, N_cz)
        d = discrim(n, N_cz)
        S = joint_score(n, N_cz)
        print(f"  Pattern {name}:  stability={s:.3f}  discrim={d:.3f}  S={S:.3f}")


if __name__ == "__main__":
    print("=== Scoring smoke test ===")
    _self_test()
