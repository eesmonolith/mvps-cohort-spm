"""
K_top sweep + lambda sweep on each dataset.

For each dataset find optimal R1 lambda and report transfer AUC at
K_top in {5, 10, 20, 50}.
"""
from __future__ import annotations
import json
import time
import numpy as np
import polars as pl

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.scoring_r1 import per_cohort_ig
from src.config import RESULTS, SEED


def score_with_lambda(rows, lam):
    """Recompute R1 score for each row with arbitrary lambda."""
    out = []
    for r in rows:
        # need cohort-conditional IG vector. Earlier we stored only
        # mean/var; recompute from n_cz.
        n_cz = r["n_cz"]
        N_cz = r.get("N_cz")
        if N_cz is None:
            continue
        ig = per_cohort_ig(n_cz, N_cz)
        s = float(ig.mean() - lam * ig.var())
        out.append((r["p"], s))
    out.sort(key=lambda x: -x[1])
    return out


def run_dataset(dataset_name, lambdas=(1.0, 10.0, 50.0, 100.0, 200.0),
                K_top_grid=(5, 10, 20, 50), max_len=2, theta_sup=0.02):
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)
    spec = DATASET_REGISTRY[dataset_name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"  N={len(sequences)}  K={K}  M={M}")

    tau_s = 0.7; tau_d = 0.05

    fold_pattern_lists = []
    for held_out in range(K):
        mask_train = cohorts != held_out
        train_seqs = [s for s, m in zip(sequences, mask_train) if m]
        train_z = clusters[mask_train]
        train_c = cohorts[mask_train]
        test_seqs = [s for s, m in zip(sequences, mask_train == False) if m]
        test_z = clusters[mask_train == False]
        if len(test_seqs) < 10:
            continue
        c_unique = np.unique(train_c)
        c_remap = {c: i for i, c in enumerate(c_unique)}
        train_c_remap = np.array([c_remap[c] for c in train_c])
        K_train = len(c_unique)
        N_cz_train = np.zeros((K_train, M), dtype=np.int64)
        for c, z in zip(train_c_remap, train_z):
            N_cz_train[c, z] += 1
        rows = enumerate_patterns_restricted(
            train_seqs, train_c_remap, train_z, N_cz_train,
            K_train, M, max_len=max_len, theta_sup=theta_sup,
        )
        # store N_cz_train for later lambda recompute
        for r in rows:
            r["N_cz"] = N_cz_train
        fold_pattern_lists.append({
            "fold": held_out,
            "rows": rows,
            "train_seqs": train_seqs,
            "train_z": train_z,
            "test_seqs": test_seqs,
            "test_z": test_z,
        })

    # For each (lambda, K_top), compute transfer per fold then aggregate
    results = {}
    for lam in lambdas:
        for K_top in K_top_grid:
            aucs = []
            for fold in fold_pattern_lists:
                # rank rows by S_R1(lam)
                ranked = []
                for r in fold["rows"]:
                    ig = per_cohort_ig(r["n_cz"], r["N_cz"])
                    s = float(ig.mean() - lam * ig.var())
                    ranked.append((r["p"], s))
                ranked.sort(key=lambda x: -x[1])
                pats = [p for p, _ in ranked[:K_top]]
                auc = evaluate_transfer(pats, fold["train_seqs"],
                                        fold["train_z"], fold["test_seqs"],
                                        fold["test_z"])
                aucs.append(auc)
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs))
            results[f"lam={lam}_k={K_top}"] = {
                "mean": mean_auc, "std": std_auc, "per_fold": aucs,
            }

    # Print summary
    print(f"\n{'lambda':>8}  " + "  ".join(f"k={k:<3}" for k in K_top_grid))
    for lam in lambdas:
        row = [f"{lam:>8.1f}"]
        for K_top in K_top_grid:
            key = f"lam={lam}_k={K_top}"
            r = results[key]
            row.append(f"{r['mean']:.3f}±{r['std']:.02f}")
        print("  ".join(row))

    # Pick best (lam, K_top) per dataset
    best = max(results.items(), key=lambda x: x[1]["mean"])
    print(f"\n  best: {best[0]}  AUC={best[1]['mean']:.3f}")

    return {
        "dataset": dataset_name, "K": K, "M": M, "N": len(sequences),
        "grid": results, "best": best[0],
    }


def main():
    out = {}
    for ds in ["edu_kor", "codle_hashed", "bpi2012"]:
        try:
            out[ds] = run_dataset(ds)
        except Exception as e:
            import traceback
            print(f"ERROR {ds}: {e}")
            traceback.print_exc()
            out[ds] = {"error": str(e)}
        print()
    with open(RESULTS / "ktop_lambda_sweep.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {RESULTS / 'ktop_lambda_sweep.json'}")


if __name__ == "__main__":
    main()
