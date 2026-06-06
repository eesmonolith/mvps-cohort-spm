"""
Default-killer synthetic experiment.

Constructs a synthetic regime where dominant view flips per cohort,
so any fixed single-view default catastrophically fails on at least
one held-out cohort while MVPS does not.

Setup:
- K=4 cohorts, M=2 outcomes (binary cluster)
- 2 cohort types alternating:
    Type S (stability-favored): outcome correlates with PRESENCE of
      a high-frequency pattern p_freq. Stability filter wins because
      p_freq is consistently frequent across S-cohorts.
    Type D (discrim-favored): outcome correlates with PRESENCE of
      a RARE pattern p_rare. Stability filter rejects p_rare; pooled
      discrimination wins.
- Cohorts 0,2 are Type S; cohorts 1,3 are Type D.

LOCO transfer:
- Hold out cohort c*; mine on K-1=3 train cohorts.
- 4 single-view methods (freq/stab/discrim/r1_lam50) plus MVPS.
- Held-out is alternately S or D, so any fixed single-view picks
  patterns appropriate for one type but mistuned for the other.
"""
from __future__ import annotations
import json
import numpy as np

from src.config import RESULTS, SEED

# 4 cohorts, alternating S/D type
COHORT_TYPES = ['S', 'D', 'S', 'D']
K = len(COHORT_TYPES)
M = 3  # 3 outcome classes
V = 8  # vocab size

# Pattern definitions
P_FREQ = [0, 1]  # high-frequency pattern - drives outcome in S-cohorts
P_RARE = [4, 5, 6]  # rare pattern - drives outcome in D-cohorts

# Sequence generation params
N_PER_COHORT = 500  # 500 entities per cohort
N_TOTAL = N_PER_COHORT * K
SEQ_LEN_BASE = 15

# Outcome probability conditional on pattern presence
# Type S: outcome=1 if p_freq present (probability HIGH, simulates stable signal)
# Type D: outcome=1 if p_rare present (probability HIGH but RARE in population)
P_FREQ_BASE_RATE_S = 0.7   # 70% of S-cohort sequences carry p_freq
P_FREQ_BASE_RATE_D = 0.7   # also 70% in D-cohort (frequency stays similar)
P_RARE_BASE_RATE_S = 0.10  # 10% of S-cohort carries p_rare
P_RARE_BASE_RATE_D = 0.35  # 35% of D-cohort carries p_rare

# Outcome rules
P_OUT_GIVEN_FREQ_S = 0.85  # S-cohort: p_freq strongly predicts outcome=1
P_OUT_GIVEN_NOFREQ_S = 0.15
P_OUT_GIVEN_RARE_D = 0.85  # D-cohort: p_rare strongly predicts outcome=1
P_OUT_GIVEN_NORARE_D = 0.20


def contains(seq, pattern):
    i = 0
    for t in seq:
        if i < len(pattern) and t == pattern[i]:
            i += 1
    return i == len(pattern)


def gen_sequence(rng, has_freq, has_rare):
    """Generate sequence of length ~SEQ_LEN_BASE with embedded patterns."""
    # Base random tokens (avoiding pattern start tokens)
    L = SEQ_LEN_BASE + rng.integers(-3, 4)
    seq = list(rng.integers(0, V, size=L))
    if has_freq:
        # Inject p_freq at random position (subseq with gaps)
        pos = sorted(rng.choice(L, size=len(P_FREQ), replace=False))
        for i, p in zip(pos, P_FREQ):
            seq[i] = p
    if has_rare:
        pos = sorted(rng.choice(L, size=len(P_RARE), replace=False))
        for i, p in zip(pos, P_RARE):
            seq[i] = p
    return seq


def gen_corpus(seed=42):
    rng = np.random.default_rng(seed)
    sequences = []
    cohorts = np.zeros(N_TOTAL, dtype=np.int64)
    clusters = np.zeros(N_TOTAL, dtype=np.int64)

    idx = 0
    for c in range(K):
        ctype = COHORT_TYPES[c]
        for _ in range(N_PER_COHORT):
            if ctype == 'S':
                has_freq = rng.random() < P_FREQ_BASE_RATE_S
                has_rare = rng.random() < P_RARE_BASE_RATE_S
                # 3-class outcome: 0=neg, 1=pos (driven by p_freq in S), 2=other
                if has_freq:
                    outcome = 1 if rng.random() < 0.85 else (0 if rng.random() < 0.5 else 2)
                else:
                    outcome = 0 if rng.random() < 0.6 else 2
            else:  # D
                has_freq = rng.random() < P_FREQ_BASE_RATE_D
                has_rare = rng.random() < P_RARE_BASE_RATE_D
                # 3-class: 0=neg, 1=pos (driven by p_rare in D), 2=other
                if has_rare:
                    outcome = 1 if rng.random() < 0.85 else (0 if rng.random() < 0.5 else 2)
                else:
                    outcome = 0 if rng.random() < 0.7 else 2

            seq = gen_sequence(rng, has_freq, has_rare)
            sequences.append(seq)
            cohorts[idx] = c
            clusters[idx] = outcome
            idx += 1

    N_cz = np.zeros((K, M), dtype=np.int64)
    for c, z in zip(cohorts, clusters):
        N_cz[c, z] += 1
    return sequences, cohorts, clusters, N_cz


def evaluate_transfer(sequences, cohorts, clusters):
    """LOCO transfer for each held-out cohort with 5 methods + MVPS."""
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

    K_top = 30; K_per = 15  # smaller for this small synthetic
    tau_s = 0.7; tau_d = 0.05
    methods = ['freq_only', 'stab_only', 'discrim_only', 'v1_pooled', 'r1_lam50', 'mvps']
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

        # views
        freq_p = topk_by(rows, 'support', K_top)
        stab_p = ([r['p'] for r in rows if r['stability'] >= tau_s][:K_top]
                  or topk_by(rows, 'stability', K_top))
        discrim_p = topk_by(rows, 'discrim', K_top)
        v1_p = topk_by(rows, 'S_v1', K_top)
        # R1 lam=50
        ranked = sorted([(r['p'], float(r['per_cohort_ig'].mean() - 50*r['per_cohort_ig'].var())) for r in rows], key=lambda x:-x[1])
        r1_p = [p for p,_ in ranked[:K_top]]
        # MVPS K_per each
        stab_per = ([r['p'] for r in rows if r['stability'] >= tau_s][:K_per]
                    or topk_by(rows, 'stability', K_per))
        r1_per = r1_p[:K_per]
        cv_per = topk_pats(score_r7d_cv_penalty(rows, lam=50), K_per)
        discrim_per = topk_by(rows, 'discrim', K_per)
        mvps_p = dedupe(stab_per + r1_per + cv_per + discrim_per)

        sets = {
            'freq_only': freq_p, 'stab_only': stab_p,
            'discrim_only': discrim_p, 'v1_pooled': v1_p,
            'r1_lam50': r1_p, 'mvps': mvps_p,
        }
        held_type = COHORT_TYPES[held]
        for m, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z, test_seqs, test_z)
            agg[m].append(auc)
        print(f'  held {held} ({held_type}-type): freq={agg["freq_only"][-1]:.3f} '
              f'stab={agg["stab_only"][-1]:.3f} discrim={agg["discrim_only"][-1]:.3f} '
              f'v1={agg["v1_pooled"][-1]:.3f} r1={agg["r1_lam50"][-1]:.3f} '
              f'MVPS={agg["mvps"][-1]:.3f}')

    return agg


def main():
    print("Generating default-killer synthetic corpus...")
    sequences, cohorts, clusters, N_cz = gen_corpus(seed=42)
    print(f"  N={len(sequences)}, K={K}, M={M}, V={V}")
    print(f"  Cohort types: {COHORT_TYPES}")
    print(f"  N_cz:\n{N_cz}\n")

    print("Running LOCO transfer:")
    agg = evaluate_transfer(sequences, cohorts, clusters)

    # Per-method drops if you commit to that view
    print("\n=== Default-killer analysis ===")
    print(f'{"Method":15s}  {"mean":>6s}  {"worst-fold":>11s}  {"worst-drop-vs-MVPS":>20s}')
    mvps_per_fold = np.array(agg['mvps'])
    summary = {}
    for m in ['freq_only', 'stab_only', 'discrim_only', 'v1_pooled', 'r1_lam50', 'mvps']:
        arr = np.array(agg[m])
        worst = arr.min()
        drop_vs_mvps = (mvps_per_fold - arr).max()
        summary[m] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                      'worst_fold': float(worst), 'max_drop_vs_mvps': float(drop_vs_mvps),
                      'per_fold': arr.tolist()}
        print(f'{m:15s}  {arr.mean():.3f}  {worst:.3f}     {drop_vs_mvps*100:+.2f}pp')

    with open(RESULTS / 'default_killer_synthetic.json', 'w') as f:
        json.dump({'cohort_types': COHORT_TYPES, 'K': K, 'M': M,
                   'N_per_cohort': N_PER_COHORT, 'summary': summary}, f, indent=2)
    print(f"\nwrote {RESULTS / 'default_killer_synthetic.json'}")


if __name__ == "__main__":
    main()
