"""
R7 scoring family — new candidate objectives for cross-dataset robustness.

Diagnosis of R1 failure modes:
- BPI: stab_only filters stab>=0.7 then sorts by support; wins via filter.
  R1 has no stability filter so picks high-mean low-stab patterns.
- Codle: similar — stable patterns transfer better.
- OULAD: weak base signal; no method can amplify what isn't there.
- Edu: high cohort heterogeneity + high mean spread; R1's variance
  penalty is the right inductive bias.

R7 family:
  R7a: Stability-gated R1 — hard stab filter + R1 ranking.
  R7b: Geometric mean of per-cohort IG.
  R7c: Borda rank fusion across 4 criteria.
  R7d: CV-penalized: mean - lam * var / (mean + eps).
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def score_r7a_stab_gated(rows, lam: float = 50.0, tau_s: float = 0.7):
    """Stability-gated R1. Patterns failing stab filter get -inf."""
    out = []
    for r in rows:
        if r["stability"] < tau_s:
            out.append((r["p"], -np.inf))
            continue
        # use precomputed per_cohort_ig if available
        if "per_cohort_ig" in r:
            ig = r["per_cohort_ig"]
        else:
            from src.scoring_r1 import per_cohort_ig
            ig = per_cohort_ig(r["n_cz"], r["N_cz_train"])
        s = float(ig.mean() - lam * ig.var())
        out.append((r["p"], s))
    out.sort(key=lambda x: -x[1])
    return out


def score_r7b_geomean(rows):
    """Geometric mean of per-cohort IG."""
    out = []
    for r in rows:
        if "per_cohort_ig" in r:
            ig = r["per_cohort_ig"]
        else:
            from src.scoring_r1 import per_cohort_ig
            ig = per_cohort_ig(r["n_cz"], r["N_cz_train"])
        ig_clip = np.maximum(ig, 0)
        # geometric mean with epsilon shift for stability
        s = float(np.exp(np.mean(np.log(ig_clip + EPS))))
        out.append((r["p"], s))
    out.sort(key=lambda x: -x[1])
    return out


def score_r7c_borda(rows):
    """Borda rank fusion: lower aggregate rank = better."""
    n = len(rows)
    # Compute ranks for each criterion (high value = rank 1)
    def get_ig(r):
        if "per_cohort_ig" in r: return r["per_cohort_ig"]
        from src.scoring_r1 import per_cohort_ig
        return per_cohort_ig(r["n_cz"], r["N_cz_train"])

    stab_vals = np.array([r["stability"] for r in rows])
    discrim_vals = np.array([r["discrim"] for r in rows])
    mean_ig_vals = np.array([get_ig(r).mean() for r in rows])
    min_ig_vals = np.array([get_ig(r).min() for r in rows])

    def rank_desc(vals):
        # rank 1 = largest value
        order = np.argsort(-vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        return ranks

    r_stab = rank_desc(stab_vals)
    r_discrim = rank_desc(discrim_vals)
    r_mean = rank_desc(mean_ig_vals)
    r_min = rank_desc(min_ig_vals)

    borda = r_stab + r_discrim + r_mean + r_min
    out = [(r["p"], -float(borda[i])) for i, r in enumerate(rows)]
    out.sort(key=lambda x: -x[1])
    return out


def score_r7d_cv_penalty(rows, lam: float = 50.0):
    """CV-penalized: mean - lam * var / (mean + eps)."""
    out = []
    for r in rows:
        if "per_cohort_ig" in r:
            ig = r["per_cohort_ig"]
        else:
            from src.scoring_r1 import per_cohort_ig
            ig = per_cohort_ig(r["n_cz"], r["N_cz_train"])
        m = float(ig.mean())
        v = float(ig.var())
        s = m - lam * v / (m + EPS)
        out.append((r["p"], s))
    out.sort(key=lambda x: -x[1])
    return out


def topk_pats(scored_list, K_top: int):
    return [p for p, _ in scored_list[:K_top]]
