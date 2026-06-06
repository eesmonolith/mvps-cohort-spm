"""
Multi-dataset leave-one-cohort-out transfer experiment.

Runs the same R1 / v1 / baselines pipeline across multiple datasets
and reports per-dataset transfer AUC plus aggregate Wilcoxon test.
"""
from __future__ import annotations

import json
import time
import numpy as np

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.config import RESULTS, SEED


def run_dataset(dataset_name: str, K_top: int = 50, max_len: int = 2,
                theta_sup: float = 0.02):
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)
    spec = DATASET_REGISTRY[dataset_name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"  N={len(sequences)}  K={K}  M={M}  vocab={len(spec.vocab)}")
    print(f"  N_cz=\n{N_cz}")

    tau_s = 0.7
    tau_d = 0.05

    fold_results = []
    for held_out in range(K):
        t0 = time.time()
        mask_train = cohorts != held_out
        mask_test = cohorts == held_out
        train_seqs = [s for s, m in zip(sequences, mask_train) if m]
        train_z = clusters[mask_train]
        train_c = cohorts[mask_train]
        test_seqs = [s for s, m in zip(sequences, mask_test) if m]
        test_z = clusters[mask_test]
        if len(test_seqs) < 10:
            print(f"  fold {held_out} skipped (test size {len(test_seqs)})")
            continue

        # remap train_c to 0..K_train-1
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

        sets = {
            "freq_only": topk_by(rows, "support", K_top),
            "stab_only": (
                [r["p"] for r in rows if r["stability"] >= tau_s][:K_top]
                if any(r["stability"] >= tau_s for r in rows)
                else topk_by(rows, "stability", K_top)
            ),
            "discrim_only": topk_by(rows, "discrim", K_top),
            "intersect": (
                [r["p"] for r in rows
                 if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:K_top]
            ),
            "v1_pooled": topk_by(rows, "S_v1", K_top),
            "r1_lam50": topk_by(rows, "S_R1_lam50", K_top),
            "min_ig": topk_by(rows, "min_ig", K_top),
        }
        fold = {"held_out": held_out, "n_train": len(train_seqs),
                "n_test": len(test_seqs)}
        for name, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z, test_seqs, test_z)
            fold[name] = auc
        elapsed = time.time() - t0
        fold["time_s"] = elapsed
        print(f"  fold {held_out}: r1={fold['r1_lam50']:.3f}  "
              f"v1={fold['v1_pooled']:.3f}  intersect={fold['intersect']:.3f}  "
              f"t={elapsed:.0f}s")
        fold_results.append(fold)

    # Aggregate
    summary = {}
    keys = ["freq_only", "stab_only", "discrim_only", "intersect",
            "v1_pooled", "r1_lam50", "min_ig"]
    print("\nSummary:")
    for k in keys:
        vals = [f[k] for f in fold_results if not np.isnan(f.get(k, np.nan))]
        if not vals: continue
        summary[k] = {"mean": float(np.mean(vals)),
                      "std": float(np.std(vals)),
                      "per_fold": vals}
        print(f"  {k:14s}  {np.mean(vals):.3f} ± {np.std(vals):.3f}  n_folds={len(vals)}")
    return {"dataset": dataset_name, "K": K, "M": M, "N": len(sequences),
            "folds": fold_results, "summary": summary}


def main():
    out = {}
    for ds in ["edu_kor", "edub_hashed", "bpi2012", "sepsis"]:
        try:
            out[ds] = run_dataset(ds)
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            import traceback; traceback.print_exc()
            out[ds] = {"error": str(e)}
        print()
    with open(RESULTS / "multi_dataset_transfer.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS / 'multi_dataset_transfer.json'}")


if __name__ == "__main__":
    main()
