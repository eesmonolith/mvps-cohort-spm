"""
Default-killer v5: strict crowd-out of stability with 8 noise + small K_top.

Design:
- K=4 cohorts, types AABB
- Type A: outcome 1 if p_A present (else 0)
- Type B: outcome 1 if p_B present (else 0)
- Both p_A and p_B have support 0.65 in their type cohorts, 0 elsewhere
  → variance MAXIMAL → stability filter rejects both
- 8 noise patterns p_N1..p_N8: support 0.90 in EVERY cohort, no outcome
  correlation → variance MINIMAL → stability filter selects ALL of them
- K_top = 8 → stab top-K filled entirely with noise patterns
- Discrim picks p_A and p_B (high pooled IG over train cohorts)
- MVPS picks both via discrim view

LOCO held=c: train cohorts != c. Both p_A and p_B exist in train (since
2 of each type). Discrim's pooled IG ranking finds them. Stab's stability
ranking finds only noise. Stab fails on held-out, discrim and MVPS win.
"""
from __future__ import annotations
import json
import numpy as np

from src.config import RESULTS

K = 6
M = 3
V = 30
N_PER_COHORT = 1200
SEQ_LEN = 14

COHORT_TYPE = ['A', 'B', 'A', 'B', 'A', 'B']
P_A = [0, 1]
P_B = [2, 3]
PATTERN_FOR_TYPE = {'A': P_A, 'B': P_B}

# 8 noise patterns (length 2)
P_NOISE = [[10, 11], [12, 13], [14, 15], [16, 17],
           [18, 19], [20, 21], [22, 23], [24, 25]]


def gen_seq(rng, type_inject, all_noise=True):
    seq = list(rng.integers(26, V, size=SEQ_LEN))
    if type_inject is not None:
        pos = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, t in zip(pos, type_inject): seq[i] = t
    # Inject ALL 8 noise patterns at random positions (universal)
    if all_noise:
        # Each noise pattern gets injected with prob 0.9 → support 0.9 per pattern
        for noise_pat in P_NOISE:
            if rng.random() < 0.9:
                pos = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
                for i, t in zip(pos, noise_pat): seq[i] = t
    return seq


def gen_corpus(seed=42):
    rng = np.random.default_rng(seed)
    sequences, cohorts, clusters = [], [], []
    for c in range(K):
        ctype = COHORT_TYPE[c]
        type_pat = PATTERN_FOR_TYPE[ctype]
        other_pat = P_B if ctype == 'A' else P_A
        for _ in range(N_PER_COHORT):
            # 15% baseline class 2 uniform
            if rng.random() < 0.15:
                outcome = 2; true_pat = None; extra_pat = None
            else:
                has_true = rng.random() < 0.65
                true_pat = type_pat if has_true else None
                outcome = 1 if has_true else 0
                if rng.random() < 0.10: outcome = 1 - outcome
                extra_pat = None
            # Background: inject other-type pattern at 0.15 (no outcome correlation)
            # ensures both p_A and p_B have support > theta_sup in every cohort
            inject_other = rng.random() < 0.15
            seq = list(rng.integers(26, V, size=SEQ_LEN))
            if true_pat is not None:
                pos = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
                for i, t in zip(pos, true_pat): seq[i] = t
            if inject_other:
                pos = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
                for i, t in zip(pos, other_pat): seq[i] = t
            # Noise patterns
            for noise_pat in P_NOISE:
                if rng.random() < 0.9:
                    pos = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
                    for i, t in zip(pos, noise_pat): seq[i] = t
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

    print("=== Default-killer v5 (kill stability via noise crowd-out) ===")
    sequences, cohorts, clusters, N_cz = gen_corpus(seed=42)
    print(f"N={len(sequences)} K={K} M={M} V={V}")
    print(f"Types: {COHORT_TYPE}, p_A={P_A}, p_B={P_B}")
    print(f"NOISE: {len(P_NOISE)} patterns each at 0.9 support uniformly")
    print(f"N_cz:\n{N_cz}\n")

    K_top = 8   # strict: only 8 patterns per view
    K_per = 8
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
            K_tr, M, max_len=2, theta_sup=0.05,
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
        print(f'  held {held} ({COHORT_TYPE[held]}): '
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

    with open(RESULTS / 'default_killer_v5.json', 'w') as f:
        json.dump({'K': K, 'M': M, 'K_top': K_top,
                   'summary': summary,
                   'description': 'Noise crowd-out: 8 universal noise patterns + small K_top forces stab into pure noise selection'},
                  f, indent=2)


if __name__ == "__main__":
    main()
