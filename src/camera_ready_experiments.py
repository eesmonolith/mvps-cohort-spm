"""
Camera-ready experiments for DM1622 "Multi-View Pattern Selection for
Cohort-Robust Sequential Pattern Mining" (MVPS), requested by Reviewer #4.

Experiment A — MVPS-minus-S ablation
    Full MVPS (4 views: stab, S, discrim, S_CV) vs MVPS-minus-S
    (3 views: stab, discrim, S_CV — S excluded, S_CV kept per the
    reviewer's exact wording "S removed, other three unioned").
    Tests the paper's claim that Edu's gain comes from S while EduB's
    gain comes from the union mechanism itself.

Experiment B — matched-budget-70 single-view baselines
    Re-evaluates the seven single-view baselines of Table
    tab:multi-dataset at top-K=70 (MVPS's union lands at 50-70
    patterns) instead of the original top-50, to rule out "MVPS wins
    because it uses more patterns."

Both experiments reuse the existing LOCO transfer protocol
(enumerate_patterns_restricted / evaluate_transfer, max_len=2,
theta_sup=0.02) exactly as in transfer_experiment.py / run_r7_sweep.py /
run_r8_ensemble.py / run_hb_transfer.py / run_oulad_transfer.py. No
re-mining: pattern candidates are re-enumerated per LOCO fold (as all
prior experiments in this repo do — this is the existing "enumerated
pool", not the expensive C2DPM Apriori+joint-bound miner) and only the
selection/ranking/union step is new.

Experiment A2 — MVPS-minus-S-family ablation
    Follow-up to Experiment A. Removing S alone left S_CV (built from
    the same mean_ig/var_ig ingredients) as a near-perfect stand-in, so
    Experiment A showed almost no drop — that only proves S is
    redundant given S_CV, not that the S-derived (M2) signal is
    unimportant. A2 removes the whole S-family (both S and S_CV) and
    unions only the two remaining, non-S-derived views: stab (M1) and
    pooled discrim (M3). Compares this 2-view union against full MVPS
    (4 views) to test whether the M2 signal contributes anything MVPS
    actually needs, independent of which of {S, S_CV} carries it.

Six real corpora (the paper's "six real corpora", excluding the
RetailRocket degenerate control): Edu (edu_kor), EduB (codle_hashed),
BPI (bpi2012), Sepsis (sepsis), OULAD (oulad), HB (hospital_billing).

Outputs:
    results/camera_ready/ablation_minus_s.json
    results/camera_ready/budget70_baselines.json
    results/camera_ready/ablation_minus_s_family.json
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import wilcoxon

from src.c2dpm import load_dataset, DATASET_REGISTRY
from src.transfer_experiment import (
    enumerate_patterns_restricted, evaluate_transfer, topk_by,
)
from src.config import RESULTS, SEED

EPS = 1e-12
TAU_S = 0.7
TAU_D = 0.05
LAM = 50.0
K_PER = 20          # per-view budget for MVPS union (matches method.tex)
K_TOP_BASELINE_ORIG = 50
K_TOP_BASELINE_B = 70   # matched-budget for Experiment B

CORPORA = [
    ("edu_kor", "Edu"),
    ("codle_hashed", "EduB"),
    ("bpi2012", "BPI"),
    ("sepsis", "Sepsis"),
    ("oulad", "OULAD"),
    ("hospital_billing", "HB"),
]

CAMERA_READY_DIR = RESULTS / "camera_ready"
CAMERA_READY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# View scores (reuse fields already computed by
# enumerate_patterns_restricted: stability, discrim, mean_ig, var_ig)
# ============================================================

def s_score(r, lam=LAM):
    """M2: S(p) = mean_c IG_c - lambda * var_c IG_c."""
    return r["mean_ig"] - lam * r["var_ig"]


def s_cv_score(r, lam=LAM):
    """M2-M4 hybrid: S_CV(p) = mean_c IG_c - lambda * var_c IG_c / (mean_c IG_c + eps)."""
    return r["mean_ig"] - lam * r["var_ig"] / (r["mean_ig"] + EPS)


def dedupe(pats_list):
    seen = set()
    out = []
    for p in pats_list:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def stab_topk(rows, k, tau_s=TAU_S):
    """Matches the convention used throughout this repo
    (transfer_experiment.py, run_r7_sweep.py, run_r8_ensemble.py,
    run_hb_transfer.py, run_oulad_transfer.py): filter by tau_s first,
    truncate to k in enumeration order; fall back to top-k by raw
    stability value only if the filter yields nothing."""
    filtered = [r["p"] for r in rows if r["stability"] >= tau_s][:k]
    if filtered:
        return filtered
    return topk_by(rows, "stability", k)


def topk_by_custom(rows, score_fn, k):
    ranked = sorted(rows, key=lambda r: -score_fn(r))
    return [r["p"] for r in ranked[:k]]


# ============================================================
# Per-fold row enumeration (shared by both experiments)
# ============================================================

def _fold_rows(sequences, cohorts, clusters, N_cz, held_out, max_len=2,
               theta_sup=0.02):
    K, M = N_cz.shape
    mask_tr = cohorts != held_out
    mask_te = cohorts == held_out
    train_seqs = [s for s, m in zip(sequences, mask_tr) if m]
    train_z = clusters[mask_tr]
    train_c = cohorts[mask_tr]
    test_seqs = [s for s, m in zip(sequences, mask_te) if m]
    test_z = clusters[mask_te]
    if len(test_seqs) < 10:
        return None
    c_uniq = np.unique(train_c)
    c_remap = {c: i for i, c in enumerate(c_uniq)}
    train_c_remap = np.array([c_remap[c] for c in train_c])
    K_tr = len(c_uniq)
    N_cz_tr = np.zeros((K_tr, M), dtype=np.int64)
    for c, z in zip(train_c_remap, train_z):
        N_cz_tr[c, z] += 1
    rows = enumerate_patterns_restricted(
        train_seqs, train_c_remap, train_z, N_cz_tr,
        K_tr, M, max_len=max_len, theta_sup=theta_sup,
    )
    return {
        "rows": rows,
        "train_seqs": train_seqs, "train_z": train_z,
        "test_seqs": test_seqs, "test_z": test_z,
        "K_tr": K_tr, "n_train": len(train_seqs), "n_test": len(test_seqs),
    }


# ============================================================
# Experiment A: MVPS-minus-S ablation
# ============================================================

def run_ablation_minus_s(dataset_name: str, display_name: str,
                          K_per: int = K_PER):
    spec = DATASET_REGISTRY[dataset_name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"\n=== [A] {display_name} ({dataset_name})  N={len(sequences)}  K={K}  M={M} ===")

    full_aucs, minus_s_aucs = [], []
    full_sizes, minus_s_sizes = [], []
    fold_ids = []

    for held_out in range(K):
        t0 = time.time()
        fd = _fold_rows(sequences, cohorts, clusters, N_cz, held_out)
        if fd is None:
            print(f"  fold {held_out}: skipped (test too small)")
            continue
        rows = fd["rows"]

        stab_20 = stab_topk(rows, K_per)
        s_20 = topk_by_custom(rows, s_score, K_per)
        discrim_20 = topk_by(rows, "discrim", K_per)
        scv_20 = topk_by_custom(rows, s_cv_score, K_per)

        full_mvps = dedupe(stab_20 + s_20 + discrim_20 + scv_20)
        minus_s_mvps = dedupe(stab_20 + discrim_20 + scv_20)

        auc_full = evaluate_transfer(full_mvps, fd["train_seqs"], fd["train_z"],
                                      fd["test_seqs"], fd["test_z"])
        auc_minus = evaluate_transfer(minus_s_mvps, fd["train_seqs"], fd["train_z"],
                                       fd["test_seqs"], fd["test_z"])

        full_aucs.append(auc_full)
        minus_s_aucs.append(auc_minus)
        full_sizes.append(len(full_mvps))
        minus_s_sizes.append(len(minus_s_mvps))
        fold_ids.append(int(held_out))
        el = time.time() - t0
        print(f"  fold {held_out}: full_mvps={auc_full:.4f} (n={len(full_mvps)})  "
              f"minus_s={auc_minus:.4f} (n={len(minus_s_mvps)})  "
              f"diff={100*(auc_full-auc_minus):+.2f}pp  t={el:.0f}s")

    full_aucs = np.array(full_aucs, dtype=float)
    minus_s_aucs = np.array(minus_s_aucs, dtype=float)
    valid = ~(np.isnan(full_aucs) | np.isnan(minus_s_aucs))
    diffs = (full_aucs - minus_s_aucs)[valid]

    wilcoxon_p = None
    wilcoxon_stat = None
    if diffs.size >= 2 and not np.allclose(diffs, 0):
        try:
            stat, p = wilcoxon(diffs, alternative="greater")
            wilcoxon_stat, wilcoxon_p = float(stat), float(p)
        except ValueError:
            pass

    result = {
        "dataset": dataset_name,
        "display_name": display_name,
        "K": int(K), "M": int(M), "N": len(sequences),
        "n_folds": int(valid.sum()),
        "fold_ids": fold_ids,
        "full_mvps": {
            "mean": float(np.nanmean(full_aucs)),
            "std": float(np.nanstd(full_aucs)),
            "per_fold": full_aucs.tolist(),
            "union_size_per_fold": full_sizes,
            "union_size_mean": float(np.mean(full_sizes)) if full_sizes else None,
        },
        "mvps_minus_s": {
            "mean": float(np.nanmean(minus_s_aucs)),
            "std": float(np.nanstd(minus_s_aucs)),
            "per_fold": minus_s_aucs.tolist(),
            "union_size_per_fold": minus_s_sizes,
            "union_size_mean": float(np.mean(minus_s_sizes)) if minus_s_sizes else None,
        },
        "per_fold_diff_pp": (diffs * 100).tolist(),
        "mean_diff_pp": float(diffs.mean() * 100) if diffs.size else None,
        "full_gt_minus_s_all_folds": bool((diffs > 0).all()) if diffs.size else None,
        "n_folds_full_gt_minus_s": int((diffs > 0).sum()) if diffs.size else None,
        "wilcoxon_stat_one_sided_full_gt_minus_s": wilcoxon_stat,
        "wilcoxon_p_one_sided_full_gt_minus_s": wilcoxon_p,
    }
    print(f"  >> {display_name}: full_mvps={result['full_mvps']['mean']:.4f}  "
          f"minus_s={result['mvps_minus_s']['mean']:.4f}  "
          f"mean_diff={result['mean_diff_pp']:+.2f}pp")
    return result


# ============================================================
# Experiment A2: MVPS-minus-S-family ablation (S and S_CV both removed)
# ============================================================

def run_ablation_minus_s_family(dataset_name: str, display_name: str,
                                 K_per: int = K_PER):
    """Full MVPS (4 views) vs a 2-view union of stab + discrim only,
    i.e. both S and S_CV (the whole M2 / mean_ig-var_ig family) removed.

    Experiment A showed that removing S alone barely changes anything
    because S_CV (built from the same mean_ig/var_ig) stands in for it.
    This tests the stronger claim: does the S-*family* signal (M2),
    regardless of which member carries it, contribute anything MVPS
    needs beyond stab (M1) + pooled discrim (M3)?
    """
    spec = DATASET_REGISTRY[dataset_name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"\n=== [A2] {display_name} ({dataset_name})  N={len(sequences)}  K={K}  M={M} ===")

    full_aucs, minus_family_aucs = [], []
    full_sizes, minus_family_sizes = [], []
    fold_ids = []

    for held_out in range(K):
        t0 = time.time()
        fd = _fold_rows(sequences, cohorts, clusters, N_cz, held_out)
        if fd is None:
            print(f"  fold {held_out}: skipped (test too small)")
            continue
        rows = fd["rows"]

        stab_20 = stab_topk(rows, K_per)
        s_20 = topk_by_custom(rows, s_score, K_per)
        discrim_20 = topk_by(rows, "discrim", K_per)
        scv_20 = topk_by_custom(rows, s_cv_score, K_per)

        full_mvps = dedupe(stab_20 + s_20 + discrim_20 + scv_20)
        minus_family = dedupe(stab_20 + discrim_20)   # only M1 + M3

        auc_full = evaluate_transfer(full_mvps, fd["train_seqs"], fd["train_z"],
                                      fd["test_seqs"], fd["test_z"])
        auc_minus = evaluate_transfer(minus_family, fd["train_seqs"], fd["train_z"],
                                       fd["test_seqs"], fd["test_z"])

        full_aucs.append(auc_full)
        minus_family_aucs.append(auc_minus)
        full_sizes.append(len(full_mvps))
        minus_family_sizes.append(len(minus_family))
        fold_ids.append(int(held_out))
        el = time.time() - t0
        print(f"  fold {held_out}: full_mvps={auc_full:.4f} (n={len(full_mvps)})  "
              f"minus_s_family={auc_minus:.4f} (n={len(minus_family)})  "
              f"diff={100*(auc_full-auc_minus):+.2f}pp  t={el:.0f}s")

    full_aucs = np.array(full_aucs, dtype=float)
    minus_family_aucs = np.array(minus_family_aucs, dtype=float)
    valid = ~(np.isnan(full_aucs) | np.isnan(minus_family_aucs))
    diffs = (full_aucs - minus_family_aucs)[valid]

    wilcoxon_p = None
    wilcoxon_stat = None
    if diffs.size >= 2 and not np.allclose(diffs, 0):
        try:
            stat, p = wilcoxon(diffs, alternative="greater")
            wilcoxon_stat, wilcoxon_p = float(stat), float(p)
        except ValueError:
            pass

    result = {
        "dataset": dataset_name,
        "display_name": display_name,
        "K": int(K), "M": int(M), "N": len(sequences),
        "n_folds": int(valid.sum()),
        "fold_ids": fold_ids,
        "full_mvps": {
            "mean": float(np.nanmean(full_aucs)),
            "std": float(np.nanstd(full_aucs)),
            "per_fold": full_aucs.tolist(),
            "union_size_per_fold": full_sizes,
            "union_size_mean": float(np.mean(full_sizes)) if full_sizes else None,
        },
        "mvps_minus_s_family": {
            "mean": float(np.nanmean(minus_family_aucs)),
            "std": float(np.nanstd(minus_family_aucs)),
            "per_fold": minus_family_aucs.tolist(),
            "union_size_per_fold": minus_family_sizes,
            "union_size_mean": float(np.mean(minus_family_sizes)) if minus_family_sizes else None,
        },
        "per_fold_diff_pp": (diffs * 100).tolist(),
        "mean_diff_pp": float(diffs.mean() * 100) if diffs.size else None,
        "full_gt_minus_family_all_folds": bool((diffs > 0).all()) if diffs.size else None,
        "n_folds_full_gt_minus_family": int((diffs > 0).sum()) if diffs.size else None,
        "wilcoxon_stat_one_sided_full_gt_minus_family": wilcoxon_stat,
        "wilcoxon_p_one_sided_full_gt_minus_family": wilcoxon_p,
    }
    print(f"  >> {display_name}: full_mvps={result['full_mvps']['mean']:.4f}  "
          f"minus_s_family={result['mvps_minus_s_family']['mean']:.4f}  "
          f"mean_diff={result['mean_diff_pp']:+.2f}pp")
    return result


# ============================================================
# Experiment B: matched-budget-70 single-view baselines
# ============================================================

SINGLE_VIEW_METHODS = [
    "freq_only", "stab_only", "discrim_only", "intersect",
    "v1_pooled", "min_ig", "r1_lam50",
]


def _single_view_sets(rows, k, tau_s=TAU_S, tau_d=TAU_D):
    return {
        "freq_only": topk_by(rows, "support", k),
        "stab_only": stab_topk(rows, k, tau_s=tau_s),
        "discrim_only": topk_by(rows, "discrim", k),
        "intersect": [r["p"] for r in rows
                      if (r["stability"] >= tau_s and r["discrim"] >= tau_d)][:k],
        "v1_pooled": topk_by(rows, "S_v1", k),
        "min_ig": topk_by(rows, "min_ig", k),
        "r1_lam50": topk_by_custom(rows, s_score, k),
    }


def run_budget70(dataset_name: str, display_name: str,
                  K_top_new: int = K_TOP_BASELINE_B, K_per: int = K_PER):
    spec = DATASET_REGISTRY[dataset_name]()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    print(f"\n=== [B] {display_name} ({dataset_name})  N={len(sequences)}  K={K}  M={M} ===")

    agg = {m: [] for m in SINGLE_VIEW_METHODS}
    mvps_aucs = []
    mvps_sizes = []
    fold_ids = []

    for held_out in range(K):
        t0 = time.time()
        fd = _fold_rows(sequences, cohorts, clusters, N_cz, held_out)
        if fd is None:
            print(f"  fold {held_out}: skipped (test too small)")
            continue
        rows = fd["rows"]

        sets70 = _single_view_sets(rows, K_top_new)
        for m, pats in sets70.items():
            auc = evaluate_transfer(pats, fd["train_seqs"], fd["train_z"],
                                     fd["test_seqs"], fd["test_z"])
            agg[m].append(auc)

        # Full MVPS (4-view union, K_per=20 each) for comparison at the
        # same fold split.
        stab_20 = stab_topk(rows, K_per)
        s_20 = topk_by_custom(rows, s_score, K_per)
        discrim_20 = topk_by(rows, "discrim", K_per)
        scv_20 = topk_by_custom(rows, s_cv_score, K_per)
        full_mvps = dedupe(stab_20 + s_20 + discrim_20 + scv_20)
        auc_mvps = evaluate_transfer(full_mvps, fd["train_seqs"], fd["train_z"],
                                      fd["test_seqs"], fd["test_z"])
        mvps_aucs.append(auc_mvps)
        mvps_sizes.append(len(full_mvps))
        fold_ids.append(int(held_out))

        el = time.time() - t0
        best_m = max(sets70, key=lambda m: (agg[m][-1] if not np.isnan(agg[m][-1]) else -1))
        print(f"  fold {held_out}: mvps={auc_mvps:.4f} (n={len(full_mvps)})  "
              f"best@70={best_m}={agg[best_m][-1]:.4f}  t={el:.0f}s")

    baselines_summary = {}
    for m in SINGLE_VIEW_METHODS:
        v = np.array(agg[m], dtype=float)
        v_valid = v[~np.isnan(v)]
        if v_valid.size == 0:
            continue
        baselines_summary[m] = {
            "mean": float(v_valid.mean()),
            "std": float(v_valid.std()),
            "per_fold": v.tolist(),
        }

    mvps_arr = np.array(mvps_aucs, dtype=float)
    mvps_valid = mvps_arr[~np.isnan(mvps_arr)]

    best_name, best_mean = None, -np.inf
    for m, s in baselines_summary.items():
        if s["mean"] > best_mean:
            best_mean, best_name = s["mean"], m

    gap_pp = None
    wilcoxon_p_vs_best = None
    if best_name is not None and mvps_valid.size:
        gap_pp = float((mvps_valid.mean() - best_mean) * 100)
        # paired test vs the best-at-70 baseline, matched fold-by-fold
        best_arr = np.array(agg[best_name], dtype=float)
        pair_valid = ~(np.isnan(mvps_arr) | np.isnan(best_arr))
        pdiff = (mvps_arr - best_arr)[pair_valid]
        if pdiff.size >= 2 and not np.allclose(pdiff, 0):
            try:
                _, p = wilcoxon(pdiff, alternative="greater")
                wilcoxon_p_vs_best = float(p)
            except ValueError:
                pass

    result = {
        "dataset": dataset_name,
        "display_name": display_name,
        "K": int(K), "M": int(M), "N": len(sequences),
        "K_top_baseline_new": K_top_new,
        "K_top_baseline_orig": K_TOP_BASELINE_ORIG,
        "fold_ids": fold_ids,
        "baselines_at_70": baselines_summary,
        "mvps": {
            "mean": float(mvps_valid.mean()) if mvps_valid.size else None,
            "std": float(mvps_valid.std()) if mvps_valid.size else None,
            "per_fold": mvps_arr.tolist(),
            "union_size_per_fold": mvps_sizes,
            "union_size_mean": float(np.mean(mvps_sizes)) if mvps_sizes else None,
        },
        "best_single_view_at_70": {"name": best_name, "mean": best_mean},
        "gap_mvps_minus_best_single_view_at_70_pp": gap_pp,
        "wilcoxon_p_one_sided_mvps_gt_best_at_70": wilcoxon_p_vs_best,
    }
    print(f"  >> {display_name}: mvps={result['mvps']['mean']:.4f}  "
          f"best@70={best_name}={best_mean:.4f}  gap={gap_pp:+.2f}pp"
          if gap_pp is not None else f"  >> {display_name}: incomplete")
    return result


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)

    print("#" * 70)
    print("Experiment A: MVPS-minus-S ablation (Reviewer #4, weakness 3)")
    print("#" * 70)
    ablation_out = {}
    all_diffs_pooled = []
    for ds, disp in CORPORA:
        try:
            r = run_ablation_minus_s(ds, disp)
            ablation_out[ds] = r
            all_diffs_pooled.extend(r["per_fold_diff_pp"])
        except Exception as e:
            import traceback; traceback.print_exc()
            ablation_out[ds] = {"error": str(e)}

    # Pooled test across all (corpus, fold) pairs
    pooled = np.array(all_diffs_pooled, dtype=float) / 100.0
    pooled_wilcoxon = None
    if pooled.size >= 2 and not np.allclose(pooled, 0):
        try:
            stat, p = wilcoxon(pooled, alternative="greater")
            pooled_wilcoxon = {
                "n": int(pooled.size),
                "median_diff_pp": float(np.median(pooled) * 100),
                "wilcoxon_stat": float(stat),
                "wilcoxon_p_one_sided": float(p),
                "n_positive": int((pooled > 0).sum()),
            }
        except ValueError:
            pass
    ablation_out["_pooled_across_corpora"] = pooled_wilcoxon

    with open(CAMERA_READY_DIR / "ablation_minus_s.json", "w") as f:
        json.dump(ablation_out, f, indent=2)
    print(f"\nwrote {CAMERA_READY_DIR / 'ablation_minus_s.json'}")

    print("\n" + "#" * 70)
    print("Experiment B: matched-budget-70 single-view baselines (Reviewer #4, weakness 4)")
    print("#" * 70)
    budget_out = {}
    for ds, disp in CORPORA:
        try:
            r = run_budget70(ds, disp)
            budget_out[ds] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            budget_out[ds] = {"error": str(e)}

    with open(CAMERA_READY_DIR / "budget70_baselines.json", "w") as f:
        json.dump(budget_out, f, indent=2)
    print(f"\nwrote {CAMERA_READY_DIR / 'budget70_baselines.json'}")

    print("\n" + "#" * 70)
    print("Experiment A2: MVPS-minus-S-family ablation (S and S_CV both removed)")
    print("#" * 70)
    family_out = {}
    all_diffs_pooled_family = []
    for ds, disp in CORPORA:
        try:
            r = run_ablation_minus_s_family(ds, disp)
            family_out[ds] = r
            all_diffs_pooled_family.extend(r["per_fold_diff_pp"])
        except Exception as e:
            import traceback; traceback.print_exc()
            family_out[ds] = {"error": str(e)}

    pooled_family = np.array(all_diffs_pooled_family, dtype=float) / 100.0
    pooled_family_wilcoxon = None
    if pooled_family.size >= 2 and not np.allclose(pooled_family, 0):
        try:
            stat, p = wilcoxon(pooled_family, alternative="greater")
            pooled_family_wilcoxon = {
                "n": int(pooled_family.size),
                "median_diff_pp": float(np.median(pooled_family) * 100),
                "wilcoxon_stat": float(stat),
                "wilcoxon_p_one_sided": float(p),
                "n_positive": int((pooled_family > 0).sum()),
            }
        except ValueError:
            pass
    family_out["_pooled_across_corpora"] = pooled_family_wilcoxon

    with open(CAMERA_READY_DIR / "ablation_minus_s_family.json", "w") as f:
        json.dump(family_out, f, indent=2)
    print(f"\nwrote {CAMERA_READY_DIR / 'ablation_minus_s_family.json'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Corpus':<10s} {'full MVPS':>10s} {'minus-S':>10s} {'diff(pp)':>9s}  | "
          f"{'MVPS@~60':>9s} {'best@70':>9s} {'gap(pp)':>8s}")
    for ds, disp in CORPORA:
        a = ablation_out.get(ds, {})
        b = budget_out.get(ds, {})
        if "full_mvps" in a and "mvps" in b:
            print(f"{disp:<10s} {a['full_mvps']['mean']:>10.4f} "
                  f"{a['mvps_minus_s']['mean']:>10.4f} {a['mean_diff_pp']:>9.2f}  | "
                  f"{b['mvps']['mean']:>9.4f} {b['best_single_view_at_70']['mean']:>9.4f} "
                  f"{b['gap_mvps_minus_best_single_view_at_70_pp']:>8.2f}")


if __name__ == "__main__":
    main()
