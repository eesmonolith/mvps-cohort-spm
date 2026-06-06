"""V-REx-style strong-penalty baseline: S(λ=500) across 5 datasets.

V-REx [Krueger 2021] uses very large variance-penalty coefficients
(often λ∈[100, 10000]) to enforce environment invariance strongly.
This serves as a distinct baseline from our default S(λ=50).
"""
from __future__ import annotations
import json
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.scoring_r1 import per_cohort_ig
from src.config import RESULTS


def run_one(ds_name, lam=500.0, K_top=50, max_len=2, theta_sup=0.02):
    spec = DATASET_REGISTRY[ds_name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"\n{ds_name}: N={len(sequences)} K={K} M={M}")
    fold_aucs = []
    for held in range(K):
        mask_tr = cohorts != held
        train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
        train_z = clusters[mask_tr]; train_c = cohorts[mask_tr]
        test_seqs = [s for s, m in zip(sequences, ~mask_tr) if m]
        test_z = clusters[~mask_tr]
        if len(test_seqs) < 10: continue
        c_uniq = np.unique(train_c)
        c_remap = {c: i for i, c in enumerate(c_uniq)}
        train_c_r = np.array([c_remap[c] for c in train_c])
        K_tr = len(c_uniq)
        N_cz_tr = np.zeros((K_tr, M), dtype=np.int64)
        for c, z in zip(train_c_r, train_z): N_cz_tr[c, z] += 1
        rows = enumerate_patterns_restricted(
            train_seqs, train_c_r, train_z, N_cz_tr,
            K_tr, M, max_len=max_len, theta_sup=theta_sup,
        )
        ranked = []
        for r in rows:
            ig = per_cohort_ig(r['n_cz'], N_cz_tr)
            ranked.append((r['p'], float(ig.mean() - lam * ig.var())))
        ranked.sort(key=lambda x: -x[1])
        pats = [p for p, _ in ranked[:K_top]]
        auc = evaluate_transfer(pats, train_seqs, train_z, test_seqs, test_z)
        fold_aucs.append(auc)
        print(f"  fold {held}: AUC={auc:.3f}")
    return fold_aucs


def main():
    out = {}
    for ds in ['edu_kor', 'edub_hashed', 'bpi2012', 'sepsis', 'oulad']:
        try:
            fold_aucs = run_one(ds, lam=500.0)
            out[ds] = {'lam': 500.0, 'per_fold': fold_aucs,
                       'mean': float(np.mean(fold_aucs)),
                       'std': float(np.std(fold_aucs))}
            print(f"  summary: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
        except Exception as e:
            print(f"ERROR {ds}: {e}")
            out[ds] = {'error': str(e)}
    with open(RESULTS / 'vrex_baseline_lam500.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS / 'vrex_baseline_lam500.json'}")


if __name__ == "__main__":
    main()
