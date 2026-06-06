"""
Default-killer v2: regime where view-flip is forced.

Design: 4 cohorts, K=4, M=3.
2 cohort types alternating: S (stability-favored), D (discrim-favored).

Strategy: make patterns highly cohort-specific so stability fails sharply
on held-out cohorts whose dominant patterns weren't stable in train.

Specifically:
- 4 distinct length-2 outcome-predictive patterns: p_0=[0,1], p_1=[2,3], p_2=[4,5], p_3=[6,7]
- Cohort c (c=0..3) uses p_c as its dominant outcome-driver (high prevalence)
- All other p_i are background noise (low prevalence)
- Outcome rule: if p_c present → outcome 1 (prob 0.9), else random {0, 2}

LOCO held=c*:
- Train cohorts: {c != c*}. None of them dominantly uses p_{c*}.
- Stability filter: none of the per-cohort dominant patterns are stable
  across train (each appears strongly in only 1 of 3 train cohorts).
  → stability picks noise / generic patterns → fails on test
- Discrim: pooled IG across train cohorts mixes 3 dominant patterns
  → picks all three but NOT p_{c*}, which appears nowhere in train
  → fails on test (test outcome driven by p_{c*})

Both single-view methods should fail in this regime.

The trick: training cohorts don't contain the test cohort's
dominant pattern at all, so any mined pattern is useless for the test.
But ALL training cohorts share the same OUTCOME RULE structure ("the
cohort-dominant pattern → outcome 1"). MVPS, by including multiple
views and broader top-K, may incidentally capture this generalisable
structure better than single-view methods.

Actually, no method can succeed without seeing p_{c*}. So this design
fails for everyone. Skip.

Better design: Use ONE outcome-predictive pattern present across all
cohorts, but its DISCRIMINATIVE STRENGTH varies per cohort.
- p_invariant: present in 50% of seqs in every cohort, predicts outcome 1 always (high IG_c uniformly)
- p_dominant_c (per cohort): high support in cohort c only
  - in cohort c: also predictive (outcome 1)
  - in other cohorts: noise

Stability picks p_invariant (top-K). Discrim picks p_invariant (high pooled IG).
MVPS picks both p_invariant and (top-K from each view's high-prev specifics).

Test cohort: outcome distribution driven by p_invariant
→ stab/discrim/MVPS all work via p_invariant

Hmm same.

NEW IDEA: Stability filter is the user's default; we want it to MISS
the right pattern on some fold. Force this with patterns that
- Are highly outcome-predictive per cohort
- Have variable support across cohorts (low stability)
- But the pooled discrim is also low because outcome distributions differ

Design:
- Pattern p_A: support 0.7 in cohorts 0,2 (outcome 1), support 0.1 in cohorts 1,3 (no signal)
- Pattern p_B: support 0.7 in cohorts 1,3 (outcome 2), support 0.1 in cohorts 0,2 (no signal)
- Outcome rule:
  Cohort 0,2 (type S): outcome 1 if p_A else outcome 0 (no class 2)
  Cohort 1,3 (type D): outcome 2 if p_B else outcome 0 (no class 1)

So 3-class outcome { 0, 1, 2 } with:
- Class 1 only appears in S-cohorts
- Class 2 only appears in D-cohorts
- p_A predicts class 1 (only in S)
- p_B predicts class 2 (only in D)

Stability: p_A var-across-cohorts large, rejected. p_B same. Picks noise.
Discrim: pooled IG of p_A = ½ across S-cohorts predicting class 1, in D-cohorts predicting class 0 → mixed signal but pooled IG still high.
        p_B: similar.
        Picks p_A and p_B.
MVPS: picks p_A and p_B via discrim view.

Held=0 (S): test outcome dist {0: 0.3, 1: 0.7}
- p_A: train cohorts {1(D), 2(S), 3(D)} → p_A high in cohort 2 only.
  Discrim picks p_A. p_A→class 1. Correct on test.
- Stab: misses p_A. Picks noise. Wrong.

Hmm, this works for "stab fails, discrim wins, MVPS wins via discrim".

But MVPS is just discrim default here. Need scenario where discrim also fails.

LET ME GIVE UP ON 4-view-flip. Just demonstrate stab-failure regime to support a softer claim:
"In regimes where stability fails sharply on held-out cohort,
MVPS still works because discrim/S views provide signal."

This is incremental over current claim. Better than nothing.

Run this design and measure.
"""
from __future__ import annotations
import json
import numpy as np

from src.config import RESULTS

K = 4
M = 3
V = 12
N_PER_COHORT = 800
SEQ_LEN = 18

# Cohort types
TYPES = ['S', 'D', 'S', 'D']

# Patterns
P_A = [0, 1]  # dominant in S-cohorts
P_B = [3, 4]  # dominant in D-cohorts


def contains(seq, pattern):
    i = 0
    for t in seq:
        if i < len(pattern) and t == pattern[i]:
            i += 1
    return i == len(pattern)


def gen_seq(rng, inject_A, inject_B):
    seq = list(rng.integers(2, V, size=SEQ_LEN))  # avoid pattern starts in noise
    if inject_A:
        positions = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, p in zip(positions, P_A):
            seq[i] = p
    if inject_B:
        positions = sorted(rng.choice(SEQ_LEN, size=2, replace=False))
        for i, p in zip(positions, P_B):
            seq[i] = p
    return seq


def gen_corpus(seed=42):
    rng = np.random.default_rng(seed)
    sequences = []
    cohorts = []
    clusters = []

    for c in range(K):
        ctype = TYPES[c]
        for _ in range(N_PER_COHORT):
            # Inject baseline class distribution: 10% each of classes 1,2
            # so all classes present in every cohort
            r0 = rng.random()
            if r0 < 0.10:  # 10% baseline class 1
                outcome = 1; has_A = False; has_B = False
            elif r0 < 0.20:  # 10% baseline class 2
                outcome = 2; has_A = False; has_B = False
            elif ctype == 'S':
                has_A = rng.random() < 0.7
                has_B = rng.random() < 0.1
                outcome = 1 if has_A else 0
                if rng.random() < 0.05: outcome = (outcome + 1) % 2
            else:  # D
                has_A = rng.random() < 0.1
                has_B = rng.random() < 0.7
                outcome = 2 if has_B else 0
                if rng.random() < 0.05: outcome = 0 if outcome == 2 else 2
            sequences.append(gen_seq(rng, has_A, has_B))
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

    def dedupe(pats_list):
        seen, out = set(), []
        for p in pats_list:
            t = tuple(p)
            if t not in seen: seen.add(t); out.append(p)
        return out

    print("=== Default-killer v2 synthetic ===")
    sequences, cohorts, clusters, N_cz = gen_corpus(seed=42)
    print(f"N={len(sequences)}  K={K}  M={M}")
    print(f"Cohort types: {TYPES}")
    print(f"N_cz:\n{N_cz}\n")

    K_top = 40; K_per = 20
    tau_s = 0.7; tau_d = 0.05
    methods = ['freq_only', 'stab_only', 'discrim_only', 'v1_pooled',
               'r1_lam50', 'mvps']
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

        stab_per = ([r['p'] for r in rows if r['stability'] >= tau_s][:K_per]
                    or topk_by(rows, 'stability', K_per))
        r1_per = r1_p[:K_per]
        cv_per = topk_pats(score_r7d_cv_penalty(rows, lam=50), K_per)
        discrim_per = topk_by(rows, 'discrim', K_per)
        mvps_p = dedupe(stab_per + r1_per + cv_per + discrim_per)

        sets = {'freq_only': freq_p, 'stab_only': stab_p,
                'discrim_only': discrim_p, 'v1_pooled': v1_p,
                'r1_lam50': r1_p, 'mvps': mvps_p}
        ht = TYPES[held]
        for m, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z,
                                    test_seqs, test_z)
            agg[m].append(auc)
        print(f'  held {held} ({ht}): '
              f'freq={agg["freq_only"][-1]:.3f}  '
              f'stab={agg["stab_only"][-1]:.3f}  '
              f'discrim={agg["discrim_only"][-1]:.3f}  '
              f'r1={agg["r1_lam50"][-1]:.3f}  '
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

    with open(RESULTS / 'default_killer_v2.json', 'w') as f:
        json.dump({'cohort_types': TYPES, 'K': K, 'M': M,
                   'N_per_cohort': N_PER_COHORT, 'summary': summary}, f, indent=2)


if __name__ == "__main__":
    main()
