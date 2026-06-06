"""
Calibrate R1: sweep lambda and try alternative aggregations to find a
regime where R1 vs pooled-v1 differ in non-trivial ways AND the joint
bound vs naive bound gap is sizable on real data.

Variants tested:
  S_R1_mv(p, lam) = mean_c IG_c(p) - lam * Var_c IG_c(p)
  S_R1_min(p)     = min_c IG_c(p)                                  (most conservative)
  S_R1_gm(p)      = (prod_c (IG_c(p) + eps))^(1/K) - eps           (geometric mean)
  S_R1_cv(p, lam) = mean_c IG_c(p) * (1 - lam * std/mean)          (CV-penalised)

For each, we compute on the non_separability_scan and report
ranking divergence from S_v1 plus bound gaps on a 100-pattern sample.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import polars as pl
from scipy.stats import spearmanr, kendalltau

from src.c2dpm import load_dataset, count_atomic, C2DPMConfig
from src.config import RESULTS, VOCAB_SIZE, ID_TO_TOKEN
from src.scoring import (
    stability, discrim, joint_score, cohort_min_support,
)
from src.scoring_r1 import per_cohort_ig
from src.joint_bound_r1 import joint_upper_bound_r1, naive_separable_bound_r1
from src.joint_bound import joint_upper_bound, naive_product_bound

EPS = 1e-12


def s_r1_min(ig_c):
    return float(np.min(ig_c))


def s_r1_gm(ig_c):
    eps = 1e-6
    return float(np.exp(np.mean(np.log(ig_c + eps))) - eps)


def s_r1_cv(ig_c, lam):
    mean_ = float(ig_c.mean())
    if mean_ <= 0:
        return 0.0
    cv = float(ig_c.std() / max(mean_, 1e-9))
    return mean_ * max(0.0, 1.0 - lam * cv)


def s_r1_mv(ig_c, lam):
    return float(ig_c.mean() - lam * ig_c.var())


def compute_all_scores(rows_path: Path):
    df = pl.read_parquet(rows_path)
    df = df.with_columns(
        pl.col("IG_c").map_elements(lambda l: np.asarray(l),
                                     return_dtype=pl.Object).alias("ig_c")
    )
    s_v1 = df["S_v1"].to_numpy()

    lams = [1.0, 5.0, 10.0, 50.0]
    out = {"N": int(df.height)}
    for lam in lams:
        scores = np.array([s_r1_mv(np.asarray(r), lam)
                           for r in df["ig_c"].to_list()])
        sp, _ = spearmanr(s_v1, scores)
        kt, _ = kendalltau(s_v1, scores)
        out[f"mv_lam={lam}"] = {
            "spearman_vs_v1": float(sp),
            "kendall_vs_v1": float(kt),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }

    # min variant
    scores_min = np.array([s_r1_min(np.asarray(r)) for r in df["ig_c"].to_list()])
    sp, _ = spearmanr(s_v1, scores_min)
    out["min"] = {
        "spearman_vs_v1": float(sp),
        "score_mean": float(scores_min.mean()),
        "score_std": float(scores_min.std()),
    }

    # geometric mean
    scores_gm = np.array([s_r1_gm(np.asarray(r)) for r in df["ig_c"].to_list()])
    sp, _ = spearmanr(s_v1, scores_gm)
    out["geomean"] = {
        "spearman_vs_v1": float(sp),
        "score_mean": float(scores_gm.mean()),
        "score_std": float(scores_gm.std()),
    }

    # CV-penalised variant
    for lam in [0.5, 1.0]:
        scores_cv = np.array([s_r1_cv(np.asarray(r), lam)
                              for r in df["ig_c"].to_list()])
        sp, _ = spearmanr(s_v1, scores_cv)
        out[f"cv_lam={lam}"] = {
            "spearman_vs_v1": float(sp),
            "score_mean": float(scores_cv.mean()),
            "score_std": float(scores_cv.std()),
        }
    return out, df


def bound_gap_sample(n_samples=80, lam=10.0, seed=42):
    """Sample ~80 random patterns up to L=2; compute v1 bound gap and R1
    bound gap. Report median + percentiles."""
    cfg = C2DPMConfig(theta_sup=0.02, max_len=2)
    sequences, cohorts, clusters, N_cz = load_dataset("edu_kor")
    K, M = N_cz.shape
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        L = int(rng.integers(1, 3))
        p = tuple(rng.integers(0, VOCAB_SIZE, size=L).tolist())
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if n_cz.sum() == 0:
            continue
        # v1 bounds
        v1_naive = naive_product_bound(n_cz, N_cz)
        v1_joint = joint_upper_bound(n_cz, N_cz)
        v1_actual = joint_score(n_cz, N_cz)
        # R1 bounds (mv variant)
        r1_naive = naive_separable_bound_r1(n_cz, N_cz, lam)
        r1_joint = joint_upper_bound_r1(n_cz, N_cz, lam)
        ig = per_cohort_ig(n_cz, N_cz)
        r1_actual = float(ig.mean() - lam * ig.var())
        rows.append({
            "pattern_str": " ".join(ID_TO_TOKEN[t] for t in p),
            "v1_actual": v1_actual,
            "v1_naive": v1_naive,
            "v1_joint": v1_joint,
            "v1_gap_ratio": v1_joint / max(v1_naive, EPS),
            "r1_actual": r1_actual,
            "r1_naive": r1_naive,
            "r1_joint": r1_joint,
            "r1_gap_abs": r1_naive - r1_joint,
            "r1_gap_ratio": (r1_joint - 0.0) / max(r1_naive, EPS),  # closer to 0 = tighter
        })
    return pl.from_dicts(rows)


def main():
    scan_path = RESULTS / "non_separability_scan.parquet"
    if not scan_path.exists():
        raise FileNotFoundError("Run non_separability_demo first")

    print("=" * 65)
    print("R1 calibration on Edu (296 candidate patterns)")
    print("=" * 65)
    out, df = compute_all_scores(scan_path)
    print(json.dumps(out, indent=2))

    print("\n" + "=" * 65)
    print("Bound gap (v1 vs R1) on 80 random patterns, lam=10")
    print("=" * 65)
    sample = bound_gap_sample(n_samples=80, lam=10.0)
    print(f"  v1 joint/naive median:        {sample['v1_gap_ratio'].median():.3f}")
    print(f"  v1 joint/naive min:           {sample['v1_gap_ratio'].min():.3f}")
    print(f"  R1 naive bound median:        {sample['r1_naive'].median():.4f}")
    print(f"  R1 joint bound median:        {sample['r1_joint'].median():.4f}")
    print(f"  R1 abs gap (naive-joint) median: {sample['r1_gap_abs'].median():.4f}")
    print(f"  R1 abs gap (naive-joint) p90:    {sample['r1_gap_abs'].quantile(0.9):.4f}")
    print(f"  R1 abs gap (naive-joint) max:    {sample['r1_gap_abs'].max():.4f}")

    # Save
    with open(RESULTS / "r1_calibration.json", "w") as f:
        json.dump({"score_correlations": out,
                   "bound_gap_n": int(sample.height)}, f, indent=2)
    sample.write_csv(RESULTS / "r1_bound_gap.csv")
    print(f"\nWrote results/r1_calibration.json + r1_bound_gap.csv")


if __name__ == "__main__":
    main()
