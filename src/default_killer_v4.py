"""
Default-killer v4: stability MUST die.

Strategy:
- 4 cohorts, each with its OWN unique outcome-predictive pattern
- TRUE patterns p_c: high support only in cohort c, zero elsewhere
  → variance across cohorts MAXIMAL → stability filter REJECTS them
- 8 noise patterns p_N1..p_N8: uniform high support across all cohorts,
  ZERO correlation with outcome
  → stability filter LOVES them (high support, near-zero variance)
  → mining top-K_stab fills up with noise

K_top set to fit noise quota only (= 8). True patterns absent from stab top-K.
LogReg trained on noise features cannot predict outcome → stab fails.

Discrim filter: pooled IG of true patterns is high (each predicts class 1 in
its cohort), noise has IG=0. Discrim picks true patterns → wins.
MVPS union grabs true patterns via discrim view → wins.
"""
from __future__ import annotations
import json
import numpy as np

from src.config import RESULTS

K = 4
M = 3
V = 30  # enough for 4 true + 8 noise + filler tokens
N_PER_COHORT = 600
SEQ_LEN = 16

# Two cohort types: S uses p_A, D uses p_B
# Cohorts [0, 2] = S-type; [1, 3] = D-type
COHORT_TYPE = ['S', 'D', 'S', 'D']
P_A = [0, 1]   # S-type outcome driver
P_B = [2, 3]   # D-type outcome driver
PATTERN_FOR_TYPE = {'S': P_A, 'D': P_B}

# Noise patterns (length 2): universally frequent, no outcome signal
P_NOISE = [[10, 11], [12, 13], [14, 15], [16, 17],
           [18, 19], [20, 21], [22, 23], [24, 25]]

# Filler vocabulary: 26-29


def gen_seq(rng, true_inject, noise_injects):
    """SEQ_LEN tokens drawn from filler 26-29, with optional pattern injections."""
    seq = list(rng.integers(26, V, size=SEQ_LEN))
    # Inject true pattern (length 2 subsequence with gaps)
    if true_inject is not None:
        positions = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, t in zip(positions, true_inject): seq[i] = t
    # Inject each noise pattern with prob configurable
    for noise_pat in noise_injects:
        positions = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, t in zip(positions, noise_pat): seq[i] = t
    return seq


def gen_corpus(seed=42):
    rng = np.random.default_rng(seed)
    sequences, cohorts, clusters = [], [], []

    for c in range(K):
        ctype = COHORT_TYPE[c]
        type_pattern = PATTERN_FOR_TYPE[ctype]
        for _ in range(N_PER_COHORT):
            has_true = rng.random() < 0.6
            true_pat = type_pattern if has_true else None
            n_noise = rng.integers(4, 7)
            noise_idx = rng.choice(len(P_NOISE), size=n_noise, replace=False)
            noise_injects = [P_NOISE[i] for i in noise_idx]

            seq = gen_seq(rng, true_pat, noise_injects)

            if has_true:
                outcome = 1 if rng.random() < 0.90 else 0
            else:
                outcome = 0 if rng.random() < 0.85 else 2

            sequences.append(seq)
            cohorts.append(c)
            clusters.append(outcome)

    cohorts = np.array(cohorts, dtype=np.int64)
    clusters = np.array(clusters, dtype=np.int64)
    N_cz = np.zeros((K, M), dtype=np.int64)
    for c, z in zip(cohorts, clusters): N_cz[c, z] += 1
    return sequences, cohorts, clusters, N_cz


def main():
    from src.transfer_experiment import (
        enumerate_patterns_restricted, evaluate_transfer, topk_by,
    )
    from src.scoring_r1 import per_cohort_ig
    from src.scoring_r7 import score_r7d_cv_penalty, topk_pats

    def dedupe(pl):
        seen, out = set(), []
        for p in pl:
            t = tuple(p)
            if t not in seen: seen.add(t); out.append(p)
        return out

    print("=== Default-killer v4 (kill stability) ===")
    sequences, cohorts, clusters, N_cz = gen_corpus(seed=42)
    print(f"N={len(sequences)}  K={K}  M={M}  V={V}")
    print(f"COHORT_TYPE: {COHORT_TYPE}, p_A={P_A}, p_B={P_B}")
    print(f"P_NOISE count: {len(P_NOISE)} universally frequent\n")
    print(f"N_cz:\n{N_cz}\n")

    # K_top = 10 (small budget — noise dominates stab top-K, no room for true)
    K_top = 10; K_per = 10
    tau_s = 0.7; tau_d = 0.05
    methods = ['freq_only', 'stab_only', 'discrim_only', 'v1_pooled',
               'r1_lam50', 'min_ig', 'mvps']
    agg = {m: [] for m in methods}

    for held in range(K):
        mask_tr = cohorts != held
        train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
        train_z = clusters[mask_tr]; train_c = cohorts[mask_tr]
        test_seqs = [s for s, m in zip(sequences, ~mask_tr) if m]
        test_z = clusters[~mask_tr]
        c_uniq = np.unique(train_c)
        c_remap = {c: i for i, c in enumerate(c_uniq)}
        train_c_r = np.array([c_remap[c] for c in train_c])
        K_tr = len(c_uniq)
        N_cz_tr = np.zeros((K_tr, M), dtype=np.int64)
        for c, z in zip(train_c_r, train_z): N_cz_tr[c, z] += 1
        rows = enumerate_patterns_restricted(
            train_seqs, train_c_r, train_z, N_cz_tr,
            K_tr, M, max_len=2, theta_sup=0.03,
        )
        for r in rows:
            r['N_cz_train'] = N_cz_tr
            r['per_cohort_ig'] = per_cohort_ig(r['n_cz'], N_cz_tr)

        freq_p = topk_by(rows, 'support', K_top)
        stab_p = ([r['p'] for r in rows if r['stability'] >= tau_s][:K_top]
                  or topk_by(rows, 'stability', K_top))
        discrim_p = topk_by(rows, 'discrim', K_top)
        v1_p = topk_by(rows, 'S_v1', K_top)
        ranked = sorted([(r['p'], float(r['per_cohort_ig'].mean() - 50*r['per_cohort_ig'].var())) for r in rows], key=lambda x:-x[1])
        r1_p = [p for p, _ in ranked[:K_top]]
        min_p = topk_by(rows, 'min_ig', K_top)

        stab_per = ([r['p'] for r in rows if r['stability'] >= tau_s][:K_per]
                    or topk_by(rows, 'stability', K_per))
        r1_per = r1_p[:K_per]
        cv_per = topk_pats(score_r7d_cv_penalty(rows, lam=50), K_per)
        discrim_per = topk_by(rows, 'discrim', K_per)
        mvps_p = dedupe(stab_per + r1_per + cv_per + discrim_per)

        sets = {'freq_only': freq_p, 'stab_only': stab_p,
                'discrim_only': discrim_p, 'v1_pooled': v1_p,
                'r1_lam50': r1_p, 'min_ig': min_p, 'mvps': mvps_p}
        for m, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z,
                                    test_seqs, test_z)
            agg[m].append(auc)
        print(f'  held {held}: '
              f'freq={agg["freq_only"][-1]:.3f} '
              f'stab={agg["stab_only"][-1]:.3f} '
              f'discrim={agg["discrim_only"][-1]:.3f} '
              f'r1={agg["r1_lam50"][-1]:.3f} '
              f'min_ig={agg["min_ig"][-1]:.3f} '
              f'MVPS={agg["mvps"][-1]:.3f}')

    print('\n=== Per-method analysis ===')
    print(f'{"Method":15s} {"mean":>6s} {"worst":>6s} {"drop_vs_MVPS_worst":>20s}')
    mvps_arr = np.array(agg['mvps'])
    summary = {}
    for m in methods:
        arr = np.array(agg[m])
        worst = arr.min()
        drop = max(0.0, (mvps_arr - arr).max())
        summary[m] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                      'worst_fold': float(worst), 'max_drop_vs_mvps': float(drop),
                      'per_fold': arr.tolist()}
        print(f'{m:15s} {arr.mean():.3f}  {worst:.3f}  {drop*100:>+18.2f}pp')

    with open(RESULTS / 'default_killer_v4.json', 'w') as f:
        json.dump({'K': K, 'M': M, 'N_per_cohort': N_PER_COHORT,
                   'K_top': K_top, 'K_per': K_per, 'summary': summary,
                   'description': 'Stability killed by universal-noise crowding: 8 noise patterns dominate top-K_stab, true patterns absent'},
                  f, indent=2)


if __name__ == "__main__":
    main()
