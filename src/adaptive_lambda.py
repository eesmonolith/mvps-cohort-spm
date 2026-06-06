"""
Adaptive lambda per dataset via nested leave-one-cohort-out.

For each outer test fold c*:
  - inner: for each c' in {non-c*}, hold c' out; choose lambda* maximising
    mean inner transfer AUC over c'.
  - outer: train rank on remaining K-1 cohorts with lambda*, transfer to c*.

Reports nested-CV mean ± std and selected lambda* per outer fold.
"""
from __future__ import annotations
import json
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer,
)
from src.scoring_r1 import per_cohort_ig
from src.config import RESULTS

LAMBDAS = [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]


def rank_topk(rows, lam, K_top):
    ranked = []
    for r in rows:
        ig = per_cohort_ig(r["n_cz"], r["N_cz_tr"])
        s = float(ig.mean() - lam * ig.var())
        ranked.append((r["p"], s))
    ranked.sort(key=lambda x: -x[1])
    return [p for p, _ in ranked[:K_top]]


def mine_fold(sequences, cohorts, clusters, K_global, M, held_out, max_len=2,
              theta_sup=0.02):
    mask_tr = cohorts != held_out
    train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
    train_z = clusters[mask_tr]
    train_c = cohorts[mask_tr]
    test_seqs = [s for s, m in zip(sequences, ~mask_tr) if m]
    test_z = clusters[~mask_tr]
    if len(test_seqs) < 10: return None
    c_unique = np.unique(train_c)
    c_remap = {c: i for i, c in enumerate(c_unique)}
    train_c_remap = np.array([c_remap[c] for c in train_c])
    K_tr = len(c_unique)
    N_cz_tr = np.zeros((K_tr, M), dtype=np.int64)
    for c, z in zip(train_c_remap, train_z): N_cz_tr[c, z] += 1
    rows = enumerate_patterns_restricted(
        train_seqs, train_c_remap, train_z, N_cz_tr,
        K_tr, M, max_len=max_len, theta_sup=theta_sup,
    )
    for r in rows: r["N_cz_tr"] = N_cz_tr
    return {
        "rows": rows, "train_seqs": train_seqs, "train_z": train_z,
        "train_c": train_c, "train_c_remap": train_c_remap,
        "test_seqs": test_seqs, "test_z": test_z, "K_tr": K_tr, "held": held_out,
    }


def run_dataset(dataset_name, K_top=50):
    print("=" * 70)
    print(f"Dataset: {dataset_name}  (nested LOCO, adaptive lambda)")
    print("=" * 70)
    spec = DATASET_REGISTRY[dataset_name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"  N={len(sequences)}  K={K}  M={M}")

    outer_aucs = []
    selected_lams = []
    for outer in range(K):
        outer_fd = mine_fold(sequences, cohorts, clusters, K, M, outer)
        if outer_fd is None: continue

        # inner: re-split outer's train set further
        train_mask_idx = np.where(cohorts != outer)[0]
        train_c_global = cohorts[train_mask_idx]
        inner_ks = np.unique(train_c_global).tolist()
        # inner fold: hold one cohort out, mine on remaining, eval on inner-held
        lam_inner_means = {lam: [] for lam in LAMBDAS}
        for inner_c in inner_ks:
            # Build sub-dataset cohorts != outer and != inner_c
            mask_inner_tr = (cohorts != outer) & (cohorts != inner_c)
            tr_seqs = [s for s, m in zip(sequences, mask_inner_tr) if m]
            tr_z = clusters[mask_inner_tr]
            tr_c = cohorts[mask_inner_tr]
            mask_inner_te = cohorts == inner_c
            te_seqs = [s for s, m in zip(sequences, mask_inner_te) if m]
            te_z = clusters[mask_inner_te]
            if len(te_seqs) < 10: continue
            c_uniq = np.unique(tr_c)
            c_remap = {c: i for i, c in enumerate(c_uniq)}
            tr_c_r = np.array([c_remap[c] for c in tr_c])
            K_inner = len(c_uniq)
            N_cz_inner = np.zeros((K_inner, M), dtype=np.int64)
            for c, z in zip(tr_c_r, tr_z): N_cz_inner[c, z] += 1
            rows = enumerate_patterns_restricted(
                tr_seqs, tr_c_r, tr_z, N_cz_inner, K_inner, M,
                max_len=2, theta_sup=0.02,
            )
            for r in rows: r["N_cz_tr"] = N_cz_inner
            for lam in LAMBDAS:
                pats = rank_topk(rows, lam, K_top)
                auc = evaluate_transfer(pats, tr_seqs, tr_z, te_seqs, te_z)
                lam_inner_means[lam].append(auc)

        # Choose lambda*
        lam_scores = {lam: float(np.mean(v)) if v else 0.0
                      for lam, v in lam_inner_means.items()}
        lam_star = max(lam_scores.items(), key=lambda x: x[1])[0]
        print(f"\n  outer {outer}: inner scores = {[f'{l}:{s:.3f}' for l, s in lam_scores.items()]}")
        print(f"  outer {outer}: lambda* = {lam_star}")

        # Outer evaluate
        pats_outer = rank_topk(outer_fd["rows"], lam_star, K_top)
        auc_outer = evaluate_transfer(
            pats_outer, outer_fd["train_seqs"], outer_fd["train_z"],
            outer_fd["test_seqs"], outer_fd["test_z"]
        )
        outer_aucs.append(auc_outer)
        selected_lams.append(lam_star)
        print(f"  outer {outer}: lambda*={lam_star}  AUC={auc_outer:.3f}")

    summary = {
        "outer_aucs": outer_aucs,
        "selected_lams": selected_lams,
        "mean": float(np.mean(outer_aucs)) if outer_aucs else float("nan"),
        "std": float(np.std(outer_aucs)) if outer_aucs else float("nan"),
    }
    print(f"\nAdaptive {dataset_name}: {summary['mean']:.3f} ± {summary['std']:.3f}  "
          f"lambdas={selected_lams}")
    return summary


def main():
    out = {}
    for ds in ["edu_kor", "codle_hashed", "bpi2012", "sepsis"]:
        try:
            out[ds] = run_dataset(ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[ds] = {"error": str(e)}
        print()
    with open(RESULTS / "adaptive_lambda.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS / 'adaptive_lambda.json'}")


if __name__ == "__main__":
    main()
