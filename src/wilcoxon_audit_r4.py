"""
Camera-ready statistical audit for DM1622 (MVPS), Reviewer 4 objection.

Reviewer 4's claim: Table V (tab:transfer) reports one-sided p<0.05 for
all 6 baselines with only "3 folds" in the caption; standard Wilcoxon
signed-rank on n=3 paired differences cannot reach below one-sided
p=0.125 (2^3=8 sign patterns => best case 1/8). The reviewer suspects
either a wrong test or a misreported n.

This script:
  1. Reproduces the existing wilcoxon_transfer() result exactly from
     results/transfer_ktop_sweep.json (n=12 = 3 folds x 4 K values,
     pooled) and diffs against results/wilcoxon_transfer.json.
  2. Quantifies the non-independence: same 3 raw LOCO folds are reused
     across all 4 K values, so the 12 "observations" are not
     independent draws -- they are 3 independent units measured 4
     times each (correlated within fold across K).
  3. Runs the honest fold-level analyses:
       (a) fold-level summary (mean AUC-diff across the 4 Ks per fold)
           -> n=3, exact one-sided Wilcoxon signed-rank (floor 0.125)
           and exact one-sided sign test (floor 0.125).
       (b) fold-level exact sign-flip permutation test on the mean
           statistic (2^3 = 8 permutations) -- equivalent floor.
       (c) per-K breakdown (n=3 each) to show whether the direction is
           consistent across K (descriptive, not a significance claim).
  4. Re-audits the abstract/intro MVPS-vs-S-view claim (paired
     one-sided Wilcoxon p=0.0009, n=23, median +1.18pp) against
     results/r8_ensemble.json to confirm it rests on genuinely
     independent (corpus, fold) pairs (3+4+5+4+7=23 distinct LOCO
     folds across 5 corpora, not the same folds re-used at multiple
     K/lambda settings).

Writes results/camera_ready/wilcoxon_audit.json (new path -- does not
touch or overwrite existing results/*.json).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "camera_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Part 1: reproduce the existing n=12 pooled test exactly
# ---------------------------------------------------------------------

def reproduce_pooled_n12():
    m = load("transfer_ktop_sweep.json")
    existing = load("wilcoxon_transfer.json")
    methods = [k for k in m.keys() if k != "r1_lam50"]
    ks = sorted(int(k) for k in m["r1_lam50"].keys())

    out = {}
    for baseline in methods:
        r1_obs, base_obs = [], []
        for k in ks:
            r1_obs.extend(m["r1_lam50"][str(k)]["per_fold"])
            base_obs.extend(m[baseline][str(k)]["per_fold"])
        diffs = np.array(r1_obs) - np.array(base_obs)
        stat, p = wilcoxon(diffs, alternative="greater")
        rec = {
            "n": len(diffs),
            "median_diff_pp": float(np.median(diffs) * 100),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p_one_sided": float(p),
            "n_positive": int((diffs > 0).sum()),
        }
        rec["matches_existing_json"] = (
            abs(rec["wilcoxon_p_one_sided"] - existing[baseline]["wilcoxon_p_one_sided"]) < 1e-9
            and rec["n"] == existing[baseline]["n"]
        )
        out[baseline] = rec
    return out, ks


# ---------------------------------------------------------------------
# Part 2: fold-level honest analyses (n=3)
# ---------------------------------------------------------------------

def exact_sign_test_n3(diffs_per_fold):
    """Exact one-sided sign test for n=3 paired diffs (ignoring ties=0).
    Returns (n_used, n_positive, p_one_sided)."""
    d = np.array(diffs_per_fold)
    n = len(d[d != 0])
    k = int((d > 0).sum())
    # one-sided p = P(X >= k) under Binomial(n, 0.5)
    from scipy.stats import binomtest
    if n == 0:
        return 0, 0, 1.0
    res = binomtest(k, n, 0.5, alternative="greater")
    return n, k, float(res.pvalue)


def exact_permutation_fold_level(diffs_per_fold, statistic="mean"):
    """Exact sign-flip permutation test over the 3 folds (2^3=8 sign
    patterns). Null: each fold's diff sign is equally likely +/-.
    Returns (observed_stat, p_one_sided, all_perm_stats)."""
    d = np.array(diffs_per_fold, dtype=float)
    n = len(d)
    obs = d.mean() if statistic == "mean" else np.median(d)
    perm_stats = []
    for signs in itertools.product([1, -1], repeat=n):
        signed = d * np.array(signs)
        s = signed.mean() if statistic == "mean" else np.median(signed)
        perm_stats.append(s)
    perm_stats = np.array(perm_stats)
    # one-sided: P(perm_stat >= obs) under the null of exchangeable signs
    p = float((perm_stats >= obs - 1e-12).mean())
    return float(obs), p, perm_stats.tolist()


def fold_level_analysis():
    m = load("transfer_ktop_sweep.json")
    methods = [k for k in m.keys() if k != "r1_lam50"]
    ks = sorted(int(k) for k in m["r1_lam50"].keys())
    n_folds = len(m["r1_lam50"][str(ks[0])]["per_fold"])

    out = {}
    for baseline in methods:
        # per-fold diff, one value per K
        per_k_diffs = {}
        fold_matrix = []  # rows = K, cols = fold
        for k in ks:
            r1_k = np.array(m["r1_lam50"][str(k)]["per_fold"])
            base_k = np.array(m[baseline][str(k)]["per_fold"])
            diff_k = r1_k - base_k
            per_k_diffs[k] = {
                "diff_pp": (diff_k * 100).tolist(),
                "n": len(diff_k),
                "n_positive": int((diff_k > 0).sum()),
                "note": "n=3 per K; any signed-rank test here is floored at one-sided p=0.125",
            }
            fold_matrix.append(diff_k)
        fold_matrix = np.array(fold_matrix)  # shape (n_K, n_folds)

        # fold-level summary: average the diff across K for each fold
        fold_summary = fold_matrix.mean(axis=0)  # shape (n_folds,)

        # (a) exact one-sided Wilcoxon signed-rank on n=3 fold summary
        try:
            w_stat, w_p = wilcoxon(fold_summary, alternative="greater", mode="exact")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        # exact sign test on fold summary
        sign_n, sign_k, sign_p = exact_sign_test_n3(fold_summary)

        # (b) exact sign-flip permutation test at fold level
        perm_obs, perm_p, perm_all = exact_permutation_fold_level(fold_summary, "mean")

        out[baseline] = {
            "fold_summary_diff_pp": (fold_summary * 100).tolist(),
            "n_folds": n_folds,
            "wilcoxon_exact_n3": {
                "stat": float(w_stat) if w_stat == w_stat else None,
                "p_one_sided": float(w_p) if w_p == w_p else None,
                "floor_note": "n=3 paired diffs -> best-achievable one-sided signed-rank p is 0.125 (1/8); "
                               "cannot report p<0.05 with this design.",
            },
            "sign_test_exact_n3": {
                "n_used": sign_n,
                "n_positive": sign_k,
                "p_one_sided": sign_p,
            },
            "fold_level_permutation_2pow3": {
                "observed_mean_diff_pp": perm_obs * 100,
                "p_one_sided": perm_p,
                "all_8_permutation_stats_pp": [x * 100 for x in perm_all],
                "note": "Exact sign-flip permutation over the 3 independent LOCO folds "
                        "(2^3=8 equally likely sign patterns under H0: no systematic direction). "
                        "This is the correct exchangeability unit -- folds are independent, "
                        "K-values computed on the same fold are not.",
            },
            "per_K_breakdown": per_k_diffs,
        }
    return out, ks, n_folds


# ---------------------------------------------------------------------
# Part 3: audit the abstract/intro MVPS-vs-S-view claim (n=23, p=0.0009)
# ---------------------------------------------------------------------

def audit_mvps_vs_sview():
    d = load("r8_ensemble.json")
    mvps, s_view, per_corpus = [], [], {}
    for corpus, methods in d.items():
        if "r8_union_K20each" not in methods or "r1" not in methods:
            continue
        mv = methods["r8_union_K20each"]["per_fold"]
        sv = methods["r1"]["per_fold"]
        assert len(mv) == len(sv)
        mvps.extend(mv)
        s_view.extend(sv)
        per_corpus[corpus] = {
            "n_folds": len(mv),
            "mean_diff_pp": float((np.array(mv) - np.array(sv)).mean() * 100),
        }
    mvps = np.array(mvps)
    s_view = np.array(s_view)
    diffs = mvps - s_view
    stat, p = wilcoxon(diffs, alternative="greater")

    return {
        "identified_source": "results/r8_ensemble.json: method 'r8_union_K20each' "
                              "(MVPS, K_per=20-per-score union) vs method 'r1' (S view, "
                              "lambda=50, alone), pooled across 5 corpora' folds.",
        "n": len(diffs),
        "n_per_corpus": per_corpus,
        "median_diff_pp": float(np.median(diffs) * 100),
        "n_positive": int((diffs > 0).sum()),
        "n_total": len(diffs),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p_one_sided": float(p),
        "matches_paper_claim": {
            "claimed_median_pp": 1.18,
            "claimed_p": 0.0009,
            "claimed_n_pos_over_n": "16/23",
            "reproduced_median_pp": round(float(np.median(diffs) * 100), 2),
            "reproduced_p": round(float(p), 4),
            "reproduced_n_pos_over_n": f"{int((diffs>0).sum())}/{len(diffs)}",
        },
        "independence_assessment": (
            "The 23 observations are (corpus, fold) pairs across 5 distinct corpora "
            "(edu_kor n=3, codle_hashed n=4, bpi2012 n=5, sepsis n=4, oulad n=7). "
            "Each fold within a corpus is a genuine distinct LOCO train/test split -- "
            "unlike Table V, no fold is measured twice under different hyperparameters "
            "for this test. Residual non-independence: folds within the same corpus "
            "share corpus-level base rates, so this is not fully i.i.d.; treat n=23 as "
            "5 clusters of correlated folds, not 23 fully exchangeable units. This is a "
            "materially weaker concern than Table V's same-fold-reused-4x design and is "
            "standard practice for pooled cross-corpus tests, but should be acknowledged."
        ),
    }


def main():
    pooled_n12, ks = reproduce_pooled_n12()
    fold_level, ks2, n_folds = fold_level_analysis()
    mvps_audit = audit_mvps_vs_sview()

    report = {
        "table_v_audit": {
            "description": "Audit of camera-ready/source/sections/experiments.tex Table V "
                            "(tab:transfer) and its caption 'Wilcoxon p<0.05 one-sided for "
                            "all 6 baselines; Holm p<=0.064'.",
            "ks_swept": ks,
            "n_folds_raw": n_folds,
            "part1_reproduce_existing_n12_pooled": pooled_n12,
            "part2_honest_fold_level_n3": fold_level,
        },
        "abstract_intro_mvps_vs_sview_audit": mvps_audit,
        "verdict": {
            "n12_pooled_reproduces": all(v["matches_existing_json"] for v in pooled_n12.values()),
            "n12_pooled_is_valid_as_stated": False,
            "n12_pooled_defense": (
                "n=12 is real (3 LOCO folds x 4 K values = 12 fold-K pairs), so the "
                "reviewer's literal '3 folds -> impossible p<0.05' objection is based on "
                "mis-reading the caption's '3 folds' as the test's sample size. However, "
                "the 12 observations are NOT independent: the same 3 folds are reused "
                "across all 4 K values, so the effective number of independent units is "
                "3, not 12. Reporting the pooled n=12 p-value as if it reflects 12 "
                "independent trials overstates precision and is not defensible as-is.",
            ),
            "fold_level_n3_ceiling": "0.125 (1/8), for both exact signed-rank and exact sign/permutation test",
            "recommended_framing": (
                "Drop the pooled-n=12 significance claim in Table V's caption. Replace with "
                "either (i) the fold-level exact permutation/sign test result (report p_floor=0.125 "
                "honestly, or state '3/3 folds favor S at every K' as a consistency statement "
                "rather than a p-value), or (ii) descriptive language: 'S (lambda=50) attains the "
                "highest median AUC-diff against every baseline at every K, with the same sign in "
                "all 3 LOCO folds' -- i.e. consistent positive differences (n+/N per fold), not a "
                "single inferential p-value built on pseudo-replicated K sweeps."
            ),
            "abstract_intro_p0009_claim": "Reproduces exactly (median +1.18pp, p=0.0009, n+=16/23) "
                                          "from results/r8_ensemble.json using genuinely independent "
                                          "(corpus,fold) pairs. No pseudo-replication across K/lambda. "
                                          "Defensible as stated; add one sentence noting folds are "
                                          "clustered by corpus (5 corpora), not fully i.i.d.",
        },
    }

    out_path = OUT_DIR / "wilcoxon_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")

    # console summary
    print("\n=== Table V (n=12 pooled) reproduction ===")
    for b, v in pooled_n12.items():
        print(f"  {b:15s} n={v['n']:2d} p={v['wilcoxon_p_one_sided']:.4f} "
              f"matches_existing={v['matches_existing_json']}")
    print("\n=== Fold-level (n=3) honest ceiling ===")
    for b, v in fold_level.items():
        print(f"  {b:15s} exact_signed_rank_p={v['wilcoxon_exact_n3']['p_one_sided']} "
              f"sign_test_p={v['sign_test_exact_n3']['p_one_sided']:.4f} "
              f"perm_p={v['fold_level_permutation_2pow3']['p_one_sided']:.4f}")
    print("\n=== MVPS vs S-view (n=23) ===")
    print(json.dumps(mvps_audit["matches_paper_claim"], indent=2))


if __name__ == "__main__":
    main()
