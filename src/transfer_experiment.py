"""
E2 — Leave-one-cohort-out transfer of discovered patterns.

For each held-out cohort c*:
  1. Restrict the dataset to cohorts != c*.
  2. Mine top-k patterns under each scorer (R1, v1-pooled, discrim-only,
     stability-only, intersect, frequency-only).
  3. Train a multinomial logistic regression classifier with one-hot
     pattern-occurrence features on cohorts != c*.
  4. Test classifier on held-out cohort c*; record macro-AUC (one-vs-rest).

Report: AUC per held-out fold + macro across folds.

The headline claim is that R1 patterns transfer better than pooled
v1 / single-axis / intersect, because R1 selects patterns that are
informative in every cohort (low variance of per-cohort IG).
"""
from __future__ import annotations

import time
import json
from itertools import combinations
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.c2dpm import load_dataset, count_atomic, C2DPMConfig
from src.config import RESULTS, COHORTS, VOCAB_SIZE, ID_TO_TOKEN, SEED
from src.scoring import (
    cohort_min_support, stability, discrim, joint_score,
)
from src.scoring_r1 import per_cohort_ig


def enumerate_patterns_restricted(
    sequences, cohorts, clusters, N_cz, K, M, max_len=2, theta_sup=0.02
):
    """Enumerate all patterns up to max_len with sup_min >= theta_sup."""
    rows = []
    frontier = []
    for tok in range(VOCAB_SIZE):
        p = (tok,)
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if cohort_min_support(n_cz, N_cz) < theta_sup:
            continue
        s = stability(n_cz, N_cz)
        d = discrim(n_cz, N_cz)
        ig_c = per_cohort_ig(n_cz, N_cz)
        rows.append({
            "p": p,
            "n_cz": n_cz,
            "stability": s,
            "discrim": d,
            "S_v1": s * d,
            "S_R1_lam50": float(ig_c.mean() - 50.0 * ig_c.var()),
            "mean_ig": float(ig_c.mean()),
            "var_ig": float(ig_c.var()),
            "min_ig": float(ig_c.min()),
            "support": float(n_cz.sum() / max(N_cz.sum(), 1)),
        })
        frontier.append((p, n_cz))
    for L in range(2, max_len + 1):
        next_f = []
        for prev, _ in frontier:
            for tok in range(VOCAB_SIZE):
                p = prev + (tok,)
                n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
                if cohort_min_support(n_cz, N_cz) < theta_sup:
                    continue
                s = stability(n_cz, N_cz)
                d = discrim(n_cz, N_cz)
                ig_c = per_cohort_ig(n_cz, N_cz)
                rows.append({
                    "p": p,
                    "n_cz": n_cz,
                    "stability": s,
                    "discrim": d,
                    "S_v1": s * d,
                    "S_R1_lam50": float(ig_c.mean() - 50.0 * ig_c.var()),
                    "mean_ig": float(ig_c.mean()),
                    "var_ig": float(ig_c.var()),
                    "min_ig": float(ig_c.min()),
                    "support": float(n_cz.sum() / max(N_cz.sum(), 1)),
                })
                next_f.append((p, n_cz))
        if not next_f:
            break
        frontier = next_f
    return rows


def topk_by(rows, key, k=20):
    sorted_rows = sorted(rows, key=lambda r: -r[key])
    return [r["p"] for r in sorted_rows[:k]]


def contains(seq, pat):
    j = 0
    for tok in seq:
        if tok == pat[j]:
            j += 1
            if j == len(pat):
                return True
    return False


def make_features(sequences, patterns):
    """Return one-hot occurrence matrix (N, |patterns|)."""
    N = len(sequences)
    K = len(patterns)
    X = np.zeros((N, K), dtype=np.float32)
    for j, pat in enumerate(patterns):
        for i, s in enumerate(sequences):
            if contains(s, pat):
                X[i, j] = 1.0
    return X


def evaluate_transfer(patterns, train_seqs, train_y, test_seqs, test_y,
                      seed=SEED):
    """Train logistic regression on train, evaluate on test. Return macro AUC."""
    if not patterns:
        # Random baseline: predict uniform
        n_classes = len(np.unique(np.concatenate([train_y, test_y])))
        y_score = np.full((len(test_y), n_classes), 1.0 / n_classes)
        try:
            return float(roc_auc_score(
                test_y, y_score, multi_class="ovr", average="macro"
            ))
        except ValueError:
            return float("nan")

    X_train = make_features(train_seqs, patterns)
    X_test = make_features(test_seqs, patterns)
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                             random_state=seed, C=1.0)
    clf.fit(X_train_s, train_y)
    proba = clf.predict_proba(X_test_s)
    # Align probabilities to all classes in test_y if any unseen
    classes_seen = clf.classes_
    classes_all = np.unique(np.concatenate([train_y, test_y]))
    if not np.array_equal(classes_seen, classes_all):
        proba_aligned = np.zeros((len(test_y), len(classes_all)))
        for j, c in enumerate(classes_seen):
            proba_aligned[:, np.where(classes_all == c)[0][0]] = proba[:, j]
        proba = proba_aligned
    try:
        return float(roc_auc_score(
            test_y, proba, multi_class="ovr", average="macro",
            labels=classes_all,
        ))
    except ValueError as e:
        print(f"AUC compute error: {e}")
        return float("nan")


def main(K_top: int = 20, max_len: int = 2, theta_sup: float = 0.02):
    cfg = C2DPMConfig(theta_sup=theta_sup, max_len=max_len)
    sequences, cohorts, clusters, N_cz = load_dataset("edu_kor")
    K_cohorts = int(N_cz.shape[0])
    M_clusters = int(N_cz.shape[1])
    print(f"loaded {len(sequences)} entities, K={K_cohorts}, M={M_clusters}")

    # Stability filter threshold and discrim threshold for single-axis
    tau_s = 0.7
    tau_d = 0.05

    folds = []
    for held_out in range(K_cohorts):
        print(f"\n=== leave-out cohort {held_out} ===")
        mask_train = cohorts != held_out
        mask_test = cohorts == held_out
        train_seqs = [s for s, m in zip(sequences, mask_train) if m]
        train_z = clusters[mask_train]
        train_c = cohorts[mask_train]
        test_seqs = [s for s, m in zip(sequences, mask_test) if m]
        test_z = clusters[mask_test]

        # Build N_cz_train (restricted)
        K_train = len(np.unique(train_c))
        # remap train_c to 0..K_train-1
        c_unique = np.unique(train_c)
        c_remap = {c: i for i, c in enumerate(c_unique)}
        train_c_remap = np.array([c_remap[c] for c in train_c])
        N_cz_train = np.zeros((K_train, M_clusters), dtype=np.int64)
        for c, z in zip(train_c_remap, train_z):
            N_cz_train[c, z] += 1

        # Enumerate patterns on the TRAIN subset
        t0 = time.time()
        rows = enumerate_patterns_restricted(
            train_seqs, train_c_remap, train_z, N_cz_train,
            K_train, M_clusters, max_len=max_len, theta_sup=theta_sup,
        )
        print(f"  enumerated {len(rows)} patterns in {time.time()-t0:.1f}s")

        # Build candidate sets per scorer
        sets = {
            "freq_only": topk_by(rows, "support", K_top),
            "stab_only": [r["p"] for r in rows if r["stability"] >= tau_s][:K_top]
                          if any(r["stability"] >= tau_s for r in rows)
                          else topk_by(rows, "stability", K_top),
            "discrim_only": topk_by(rows, "discrim", K_top),
            "intersect": (
                [r["p"] for r in rows if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:K_top]
            ),
            "v1_pooled": topk_by(rows, "S_v1", K_top),
            "r1_lam50": topk_by(rows, "S_R1_lam50", K_top),
            "min_ig": topk_by(rows, "min_ig", K_top),
        }

        fold_result = {"fold_held_out": int(held_out)}
        for name, pats in sets.items():
            auc = evaluate_transfer(pats, train_seqs, train_z, test_seqs, test_z)
            print(f"  {name:15s}  n_patterns={len(pats):3d}  transfer AUC={auc:.3f}")
            fold_result[name] = auc
            fold_result[f"{name}_n_patterns"] = len(pats)
        folds.append(fold_result)

    # Summarise
    print("\n" + "=" * 70)
    print("Transfer AUC summary (mean ± std across folds)")
    print("=" * 70)
    summary = {}
    keys = ["freq_only", "stab_only", "discrim_only", "intersect",
            "v1_pooled", "r1_lam50", "min_ig"]
    for k in keys:
        vals = np.array([f[k] for f in folds if not np.isnan(f[k])])
        if vals.size:
            summary[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "per_fold": vals.tolist(),
            }
            print(f"  {k:15s}  {vals.mean():.3f} ± {vals.std():.3f}  "
                  f"per-fold: {[round(v, 3) for v in vals.tolist()]}")
    out = {"folds": folds, "summary": summary,
           "K_top": K_top, "max_len": max_len, "theta_sup": theta_sup}
    with open(RESULTS / "transfer_e2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS / 'transfer_e2.json'}")


if __name__ == "__main__":
    main(K_top=20, max_len=2, theta_sup=0.02)
