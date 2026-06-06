"""
Empirical tightness comparison: joint upper bound vs naive product bound.

Procedure:
  1. Sample N candidate patterns (random + frequent-token-prefixed).
  2. For each, compute actual S, naive bound, joint bound.
  3. Report: avg tightness ratios, % patterns where joint < naive.
  4. Emit CSV + plot.

Why this matters:
  Paper Section "Theoretical Analysis" needs empirical evidence that the
  joint bound is in practice tighter than the naive product. The current
  vertex-enumeration implementation gives a 2-5% gap in synthetic tests;
  this experiment quantifies the gap on real data and identifies regimes
  where the joint bound dominates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.config import DATA_PROC, RESULTS, VOCAB_SIZE, ID_TO_TOKEN, COHORTS, COHORT_LABEL_TO_ID
from src.c2dpm import load_dataset, count_atomic
from src.joint_bound import joint_upper_bound, naive_product_bound, empirical_tightness


def sample_patterns(n_samples: int = 200, max_len: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    patterns = []
    # Mix: 1/3 length-1, 1/3 length-2, 1/3 length-3
    for _ in range(n_samples // 3):
        patterns.append(tuple(rng.integers(0, VOCAB_SIZE, size=1)))
    for _ in range(n_samples // 3):
        patterns.append(tuple(rng.integers(0, VOCAB_SIZE, size=2)))
    for _ in range(n_samples - 2 * (n_samples // 3)):
        patterns.append(tuple(rng.integers(0, VOCAB_SIZE, size=3)))
    return patterns


def run(n_samples: int = 200, max_len: int = 3, seed: int = 42):
    print(f"[tightness] loading dataset...")
    sequences, cohorts, clusters, N_cz = load_dataset()
    K, M = N_cz.shape
    print(f"[tightness] sequences={len(sequences)}  K={K}  M={M}")

    patterns = sample_patterns(n_samples, max_len, seed)
    print(f"[tightness] evaluating {len(patterns)} patterns...")

    rows = []
    t0 = time.time()
    for i, p in enumerate(patterns):
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if n_cz.sum() == 0:
            continue
        r = empirical_tightness(n_cz, N_cz)
        rows.append({
            "pattern_id": i,
            "pattern": " ".join(ID_TO_TOKEN[t] for t in p),
            "length": len(p),
            "total_count": int(n_cz.sum()),
            **r,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(patterns)}  elapsed={time.time()-t0:.1f}s")

    df = pl.from_dicts(rows)
    out = RESULTS / "tightness_experiment.csv"
    df.write_csv(out)
    print(f"\n[tightness] wrote {out}  ({df.height} valid patterns)")

    # Summary
    print("\n=== Summary ===")
    print(f"actual S range: {df['actual'].min():.4f} ~ {df['actual'].max():.4f}")
    print(f"naive bound avg: {df['naive_bound'].mean():.4f}  median: {df['naive_bound'].median():.4f}")
    print(f"joint bound avg: {df['joint_bound'].mean():.4f}  median: {df['joint_bound'].median():.4f}")

    # joint < naive count
    joint_tighter = (df["joint_bound"] < df["naive_bound"]).sum()
    joint_equal = (df["joint_bound"] == df["naive_bound"]).sum()
    print(f"joint < naive: {joint_tighter}/{df.height}  "
          f"({100.0*joint_tighter/df.height:.1f}%)")
    print(f"joint = naive: {joint_equal}/{df.height}  "
          f"({100.0*joint_equal/df.height:.1f}%)")

    # ratio joint/naive distribution
    ratio = df["joint_vs_naive"].to_numpy()
    print(f"\njoint / naive ratio:")
    print(f"  min:    {ratio.min():.4f}")
    print(f"  p10:    {np.percentile(ratio, 10):.4f}")
    print(f"  median: {np.median(ratio):.4f}")
    print(f"  p90:    {np.percentile(ratio, 90):.4f}")
    print(f"  max:    {ratio.max():.4f}")

    # By length
    print("\nBy pattern length:")
    for L in sorted(df["length"].unique().to_list()):
        sub = df.filter(pl.col("length") == L)
        if sub.height == 0:
            continue
        sub_ratio = sub["joint_vs_naive"].to_numpy()
        print(f"  L={L}  n={sub.height}  "
              f"joint/naive median={np.median(sub_ratio):.3f}  "
              f"tighter_pct={100.0*(sub['joint_bound'] < sub['naive_bound']).sum()/sub.height:.1f}%")

    summary = {
        "n_samples": int(df.height),
        "joint_tighter_pct": float(100.0 * joint_tighter / df.height),
        "joint_naive_ratio_median": float(np.median(ratio)),
        "joint_naive_ratio_mean": float(ratio.mean()),
        "joint_naive_ratio_p10": float(np.percentile(ratio, 10)),
        "joint_naive_ratio_p90": float(np.percentile(ratio, 90)),
    }
    with open(RESULTS / "tightness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[tightness] summary -> {RESULTS / 'tightness_summary.json'}")
    return df, summary


if __name__ == "__main__":
    run()
