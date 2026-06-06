"""
Empirically demonstrate R1 vs pooled-v1 separability gap on Edu.

For every pattern in the Edu top-50 (mined under either score), we
compute both S_v1 and S_R1 and look at the ranking divergence.

Output:
  - Spearman / Kendall rank correlation between S_v1 and S_R1
  - Patterns ranked top-10 under S_R1 but missed by S_v1 top-50
  - Patterns ranked top-10 under S_v1 but missed by S_R1 top-50

These are the empirical counterexample sets A (pooled-only) and
D (cohort-specific-aligned-but-pooled-misses).
"""
from __future__ import annotations

import time
import numpy as np
import polars as pl
from scipy.stats import spearmanr, kendalltau

from src.c2dpm import load_dataset, count_atomic, C2DPMConfig
from src.config import RESULTS, VOCAB_SIZE, ID_TO_TOKEN
from src.scoring import cohort_min_support, stability, discrim, joint_score
from src.scoring_r1 import s_r1, r1_components


def enumerate_all_up_to(max_len: int, cfg: C2DPMConfig):
    sequences, cohorts, clusters, N_cz = load_dataset("edu_kor")
    K, M = N_cz.shape
    print(f"loaded {len(sequences)} sequences, K={K}, M={M}")

    rows = []
    # L=1
    frontier = []
    for tok in range(VOCAB_SIZE):
        p = (tok,)
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            continue
        s_pool = stability(n_cz, N_cz)
        d_pool = discrim(n_cz, N_cz)
        S_v1 = joint_score(n_cz, N_cz)
        mean_ig, var_ig, ig_c = r1_components(n_cz, N_cz)
        S_R1 = s_r1(n_cz, N_cz, lam=1.0)
        rows.append({
            "pattern": " ".join(ID_TO_TOKEN[t] for t in p),
            "length": 1,
            "S_v1": S_v1,
            "S_R1": S_R1,
            "stability_pooled": s_pool,
            "discrim_pooled": d_pool,
            "mean_IG_per_cohort": mean_ig,
            "var_IG_per_cohort": var_ig,
            "IG_c": ig_c.tolist(),
        })
        frontier.append((p, n_cz))
    print(f"L=1: {len(frontier)} survivors")

    # Higher L
    for L in range(2, max_len + 1):
        next_f = []
        for prev, _ in frontier:
            for tok in range(VOCAB_SIZE):
                p = prev + (tok,)
                n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    continue
                s_pool = stability(n_cz, N_cz)
                d_pool = discrim(n_cz, N_cz)
                S_v1 = joint_score(n_cz, N_cz)
                mean_ig, var_ig, ig_c = r1_components(n_cz, N_cz)
                S_R1 = s_r1(n_cz, N_cz, lam=1.0)
                rows.append({
                    "pattern": " ".join(ID_TO_TOKEN[t] for t in p),
                    "length": L,
                    "S_v1": S_v1,
                    "S_R1": S_R1,
                    "stability_pooled": s_pool,
                    "discrim_pooled": d_pool,
                    "mean_IG_per_cohort": mean_ig,
                    "var_IG_per_cohort": var_ig,
                    "IG_c": ig_c.tolist(),
                })
                next_f.append((p, n_cz))
        print(f"L={L}: {len(next_f)} survivors")
        if not next_f:
            break
        frontier = next_f
    return rows


def main():
    cfg = C2DPMConfig(theta_sup=0.05, max_len=2)  # L=2 only for speed
    t0 = time.time()
    rows = enumerate_all_up_to(cfg.max_len, cfg)
    print(f"\nenumerated {len(rows)} patterns in {time.time()-t0:.1f}s")

    df = pl.from_dicts(rows)
    df.write_parquet(RESULTS / "non_separability_scan.parquet")
    print(f"wrote {RESULTS / 'non_separability_scan.parquet'}")

    # Rank correlation
    v1 = df["S_v1"].to_numpy()
    r1 = df["S_R1"].to_numpy()
    sp_corr, sp_p = spearmanr(v1, r1)
    kt_corr, kt_p = kendalltau(v1, r1)
    print(f"\nRanking correlation S_v1 vs S_R1 (N={len(rows)}):")
    print(f"  Spearman rho = {sp_corr:.3f}  p={sp_p:.2e}")
    print(f"  Kendall tau  = {kt_corr:.3f}  p={kt_p:.2e}")

    # Top-10 disagreement
    K_top = 30
    v1_top = set(df.sort("S_v1", descending=True).head(K_top)["pattern"].to_list())
    r1_top = set(df.sort("S_R1", descending=True).head(K_top)["pattern"].to_list())
    print(f"\nTop-{K_top} symmetric difference:")
    only_v1 = v1_top - r1_top
    only_r1 = r1_top - v1_top
    print(f"  S_v1 only ({len(only_v1)}):  {list(only_v1)[:5]}")
    print(f"  S_R1 only ({len(only_r1)}):  {list(only_r1)[:5]}")

    # Find Counterexample-A type patterns (S_v1 high but driven by 1 cohort)
    print("\n=== Counterexample A type (high S_v1 but dominated by 1 cohort) ===")
    df_v1_top = df.sort("S_v1", descending=True).head(50)
    # max IG_c / mean IG_c ratio: high if dominated by 1 cohort
    def domination_ratio(row):
        ig = np.asarray(row["IG_c"])
        if ig.mean() < 1e-6:
            return 0.0
        return float(ig.max() / max(ig.mean(), 1e-9))
    rats = [domination_ratio(r) for r in df_v1_top.iter_rows(named=True)]
    df_v1_top = df_v1_top.with_columns(pl.Series("dom_ratio", rats))
    A_candidates = df_v1_top.filter(pl.col("dom_ratio") > 1.6).sort(
        "dom_ratio", descending=True
    ).head(5)
    for row in A_candidates.iter_rows(named=True):
        print(f"  {row['pattern']:55s}  S_v1={row['S_v1']:.4f}  "
              f"S_R1={row['S_R1']:.4f}  IG_c={[round(x,3) for x in row['IG_c']]}")

    # Find Counterexample D type (high S_R1 but low S_v1)
    print("\n=== Counterexample D type (R1 finds, pooled v1 misses) ===")
    df_r1_top = df.sort("S_R1", descending=True).head(50)
    df_r1_top = df_r1_top.with_columns(
        (pl.col("S_R1") - pl.col("S_v1")).alias("R1_minus_v1")
    )
    D_candidates = df_r1_top.sort("R1_minus_v1", descending=True).head(5)
    for row in D_candidates.iter_rows(named=True):
        print(f"  {row['pattern']:55s}  S_v1={row['S_v1']:.4f}  "
              f"S_R1={row['S_R1']:.4f}  diff={row['R1_minus_v1']:.4f}  "
              f"IG_c={[round(x,3) for x in row['IG_c']]}")


if __name__ == "__main__":
    main()
