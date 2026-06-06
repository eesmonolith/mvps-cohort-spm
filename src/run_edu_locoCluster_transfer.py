"""Edu LOCO transfer with train-only KMeans (breaks cluster circularity).

For each held-out cohort fold:
  1. Refit KMeans on TRAIN-cohort behavioural features only.
  2. Project TRAIN students to refit centroids → train_z.
  3. Project HELD-OUT students to refit centroids → test_z.
  4. Mine + transfer as usual.

This isolates the question: does S beat baselines when the cluster
label is assigned to held-out cohort by an out-of-sample projection
rather than a pooled fit?
"""
from __future__ import annotations
import json
import time
import numpy as np
import polars as pl

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.cluster_labels import build_features, FEATURE_NAMES, N_CLUSTERS
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.scoring_r1 import per_cohort_ig
from src.config import DATA_PROC, RESULTS, COHORTS, SEED


def main():
    spec = DATASET_REGISTRY["edu_kor"]()
    cohort_labels = [c["label"] for c in COHORTS.values()]
    # Load per-cohort sequences + features
    all_seqs = {}
    all_feats = {}
    for cl in cohort_labels:
        seq = pl.read_parquet(DATA_PROC / f"sequences_{cl}.parquet")
        feats = build_features(seq)
        all_seqs[cl] = seq
        all_feats[cl] = feats
    sequences_all, cohorts_all, _, _ = load_dataset(spec)
    print(f"Edu: N={len(sequences_all)}  K={len(cohort_labels)}  M={N_CLUSTERS}")

    methods = ["freq_only", "stab_only", "discrim_only", "intersect",
               "v1_pooled", "min_ig", "r1_lam50", "r1_lam100"]
    agg = {m: [] for m in methods}
    K_top = 50
    M = N_CLUSTERS
    K = len(cohort_labels)
    tau_s = 0.7; tau_d = 0.05

    for held_idx, held_label in enumerate(cohort_labels):
        train_labels = [c for c in cohort_labels if c != held_label]
        # Build train features + refit KMeans
        train_feat_df = pl.concat([all_feats[c] for c in train_labels],
                                  how="vertical")
        X_train_raw = train_feat_df.select(FEATURE_NAMES).to_numpy()
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        km = KMeans(n_clusters=M, random_state=SEED, n_init=20)
        train_z_loco = km.fit_predict(X_train).astype(np.int64)

        # Project test
        test_feat_df = all_feats[held_label]
        X_test = scaler.transform(test_feat_df.select(FEATURE_NAMES).to_numpy())
        test_z_loco = km.predict(X_test).astype(np.int64)

        # Build sequence lists aligned to train_feat_df / test_feat_df
        train_seq_df = pl.concat([all_seqs[c] for c in train_labels],
                                 how="vertical")
        train_seqs = train_seq_df["token_ids"].to_list()
        train_c = np.array([train_labels.index(c) for c in
                            train_seq_df["cohort_label"].to_list()])
        test_seqs = all_seqs[held_label]["token_ids"].to_list()

        # length match check
        assert len(train_seqs) == len(train_z_loco), \
            f"mismatch {len(train_seqs)} vs {len(train_z_loco)}"
        assert len(test_seqs) == len(test_z_loco), \
            f"mismatch {len(test_seqs)} vs {len(test_z_loco)}"

        K_tr = len(train_labels)
        N_cz_tr = np.zeros((K_tr, M), dtype=np.int64)
        for c, z in zip(train_c, train_z_loco): N_cz_tr[c, z] += 1
        print(f"\nfold {held_idx} (held={held_label}):  N_cz_train=")
        print(N_cz_tr)

        t0 = time.time()
        rows = enumerate_patterns_restricted(
            train_seqs, train_c, train_z_loco, N_cz_tr,
            K_tr, M, max_len=2, theta_sup=0.02,
        )

        sets = {
            "freq_only": topk_by(rows, "support", K_top),
            "stab_only": ([r["p"] for r in rows if r["stability"] >= tau_s][:K_top]
                          or topk_by(rows, "stability", K_top)),
            "discrim_only": topk_by(rows, "discrim", K_top),
            "intersect": [r["p"] for r in rows
                          if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:K_top],
            "v1_pooled": topk_by(rows, "S_v1", K_top),
            "min_ig": topk_by(rows, "min_ig", K_top),
        }
        for lam in [50, 100]:
            ranked = []
            for r in rows:
                ig = per_cohort_ig(r["n_cz"], N_cz_tr)
                ranked.append((r["p"], float(ig.mean() - lam * ig.var())))
            ranked.sort(key=lambda x: -x[1])
            sets[f"r1_lam{lam}"] = [p for p, _ in ranked[:K_top]]

        for m, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z_loco,
                                    test_seqs, test_z_loco)
            agg[m].append(auc)
        el = time.time() - t0
        print(f"  freq={agg['freq_only'][-1]:.3f}  stab={agg['stab_only'][-1]:.3f}  "
              f"v1={agg['v1_pooled'][-1]:.3f}  r1@50={agg['r1_lam50'][-1]:.3f}  "
              f"r1@100={agg['r1_lam100'][-1]:.3f}  t={el:.0f}s")

    print(f"\nEdu LOCO-cluster summary (K_top={K_top}):")
    summary = {}
    for m in methods:
        v = agg[m]
        summary[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                      "per_fold": v}
        print(f"  {m:14s} {np.mean(v):.3f} ± {np.std(v):.3f}")

    with open(RESULTS / "edu_loco_cluster_transfer.json", "w") as f:
        json.dump({"K": K, "M": M, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
