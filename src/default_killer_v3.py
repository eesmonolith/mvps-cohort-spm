"""
Default-killer v3: regime where EVERY single-view default fails on
at least one fold by >5pp vs MVPS.

Key change from v2: inject a STABLE NOISE pattern p_N that has uniform
high support across all cohorts but is UNCORRELATED with outcome.
Stability filter picks p_N (high stability), then fails on test
because p_N has no predictive signal.

Outcome-predictive patterns p_A (S-cohorts), p_B (D-cohorts) remain.

Result hypothesis:
- freq default: picks p_N (universally frequent) → fails (no signal)
- stab default: picks p_N (universally stable) → fails (no signal)
- discrim default: picks p_A and p_B (high pooled IG) → fine on most folds
                   but fails on cohorts whose dominant pattern wasn't trained
- r1 default: picks low-variance patterns (= p_N) → fails
- MVPS: union grabs p_A and p_B via discrim view + still includes p_N
        → discrim view dominates, predicts test better.
"""
from __future__ import annotations
import json
import numpy as np

from src.config import RESULTS

K = 4
M = 3
V = 12
N_PER_COHORT = 800
SEQ_LEN = 20

TYPES = ['S', 'D', 'S', 'D']

P_A = [0, 1]  # dominant in S
P_B = [3, 4]  # dominant in D
P_N = [6, 7]  # noise: universally frequent but uncorrelated with outcome


def gen_seq(rng, inject_A, inject_B, inject_N):
    seq = list(rng.integers(8, V, size=SEQ_LEN))  # noise tokens 8-11
    if inject_A:
        positions = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, p in zip(positions, P_A): seq[i] = p
    if inject_B:
        positions = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, p in zip(positions, P_B): seq[i] = p
    if inject_N:
        positions = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, p in zip(positions, P_N): seq[i] = p
    return seq


def gen_corpus(seed=42):
    rng = np.random.default_rng(seed)
    sequences, cohorts, clusters = [], [], []

    for c in range(K):
        ctype = TYPES[c]
        for _ in range(N_PER_COHORT):
            # NOISE: universally present at 80% across cohorts, uncorrelated
            has_N = rng.random() < 0.80
            r0 = rng.random()
            if r0 < 0.10:
                outcome = 1; has_A, has_B = False, False
            elif r0 < 0.20:
                outcome = 2; has_A, has_B = False, False
            elif ctype == 'S':
                has_A = rng.random() < 0.65
                has_B = rng.random() < 0.1
                outcome = 1 if has_A else 0
                if rng.random() < 0.05: outcome = (outcome + 1) % 2
            else:  # D
                has_A = rng.random() < 0.1
                has_B = rng.random() < 0.65
                outcome = 2 if has_B else 0
                if rng.random() < 0.05: outcome = 0 if outcome == 2 else 2
            sequences.append(gen_seq(rng, has_A, has_B, has_N))
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

    print("=== Default-killer v3 (noise-pattern adversarial) ===")
    sequences, cohorts, clusters, N_cz = gen_corpus(seed=42)
    print(f"N={len(sequences)}  K={K}  M={M}")
    print(f"N_cz:\n{N_cz}\n")

    K_top = 40; K_per = 20
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
            K_tr, M, max_len=3, theta_sup=0.02,
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
        print(f'  held {held} ({TYPES[held]}): '
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

    with open(RESULTS / 'default_killer_v3.json', 'w') as f:
        json.dump({'cohort_types': TYPES, 'K': K, 'M': M,
                   'N_per_cohort': N_PER_COHORT, 'summary': summary,
                   'description': 'Noise-pattern adversarial; p_N universally stable but uncorrelated with outcome to fool stability filter'},
                  f, indent=2)
    print(f'\nwrote {RESULTS / "default_killer_v3.json"}')


if __name__ == "__main__":
    main()
