"""
C2DPM-R1: level-wise miner using the cohort-conditional R1 objective.

Score:      S_R1(p) = mean_c IG_c(p) - lambda * Var_c IG_c(p)
Threshold:  S_R1(p) >= tau                                    (single threshold)

Pruning:
  Apriori:   sup_min(p) < theta_sup  =>  prune (anti-monotone in sup_c)
  Joint:     joint_upper_bound_r1(p) < tau  =>  prune
              (anti-monotone in box B(p))

Naive separable bound (for comparison only): naive_separable_bound_r1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.c2dpm import (
    DatasetSpec, DATASET_REGISTRY, load_dataset, count_atomic,
)
from src.config import RESULTS
from src.scoring import cohort_min_support
from src.scoring_r1 import per_cohort_ig
from src.joint_bound_r1 import (
    joint_upper_bound_r1, naive_separable_bound_r1,
)


@dataclass
class C2DPMR1Config:
    theta_sup: float = 0.02
    tau: float = 0.0       # threshold on S_R1
    lam: float = 50.0      # variance penalty weight
    max_len: int = 3
    use_joint_bound: bool = True
    verbose: bool = True


@dataclass
class MiningStatsR1:
    explored: int = 0
    pruned_apriori: int = 0
    pruned_joint_bound: int = 0
    qualified: int = 0
    level_times: list = field(default_factory=list)


def mine_r1(cfg: C2DPMR1Config | None = None,
            spec: DatasetSpec | str = "edu_kor") -> tuple[pl.DataFrame, MiningStatsR1]:
    if cfg is None:
        cfg = C2DPMR1Config()
    if isinstance(spec, str):
        spec = DATASET_REGISTRY[spec]()
    stats = MiningStatsR1()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    V = len(spec.vocab)
    vocab = spec.vocab
    if cfg.verbose:
        print(f"[c2dpm-r1] dataset={spec.name}  V={V}  N={len(sequences)}  "
              f"K={K}  M={M}  lam={cfg.lam}  tau={cfg.tau}")

    qualified = []

    # L=1
    t0 = time.time()
    frontier = []
    for tok in range(V):
        p = (tok,)
        stats.explored += 1
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            stats.pruned_apriori += 1
            continue
        ig = per_cohort_ig(n_cz, N_cz)
        S = float(ig.mean() - cfg.lam * ig.var())
        if S >= cfg.tau:
            qualified.append({
                "pattern": p,
                "pattern_str": " ".join(vocab[t] for t in p),
                "length": 1,
                "S": S,
                "mean_ig": float(ig.mean()),
                "var_ig": float(ig.var()),
                "ig_c": ig.tolist(),
                "sup_min": float(cohort_min_support(n_cz, N_cz)),
            })
            stats.qualified += 1
        if cfg.use_joint_bound:
            ub = joint_upper_bound_r1(n_cz, N_cz, cfg.lam)
            if ub < cfg.tau:
                stats.pruned_joint_bound += 1
                continue
        frontier.append((p, n_cz))
    elapsed = time.time() - t0
    stats.level_times.append(elapsed)
    if cfg.verbose:
        print(f"[c2dpm-r1] L=1  explored={V}  frontier={len(frontier)}  "
              f"qualified={stats.qualified}  t={elapsed:.1f}s")

    for L in range(2, cfg.max_len + 1):
        t0 = time.time()
        next_f = []
        for prev, _ in frontier:
            for tok in range(V):
                p = prev + (tok,)
                stats.explored += 1
                n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    stats.pruned_apriori += 1
                    continue
                ig = per_cohort_ig(n_cz, N_cz)
                S = float(ig.mean() - cfg.lam * ig.var())
                if S >= cfg.tau:
                    qualified.append({
                        "pattern": p,
                        "pattern_str": " ".join(vocab[t] for t in p),
                        "length": L,
                        "S": S,
                        "mean_ig": float(ig.mean()),
                        "var_ig": float(ig.var()),
                        "ig_c": ig.tolist(),
                        "sup_min": float(cohort_min_support(n_cz, N_cz)),
                    })
                    stats.qualified += 1
                if cfg.use_joint_bound:
                    ub = joint_upper_bound_r1(n_cz, N_cz, cfg.lam)
                    if ub < cfg.tau:
                        stats.pruned_joint_bound += 1
                        continue
                next_f.append((p, n_cz))
        elapsed = time.time() - t0
        stats.level_times.append(elapsed)
        if cfg.verbose:
            print(f"[c2dpm-r1] L={L}  frontier_in={len(frontier)}  "
                  f"frontier_out={len(next_f)}  "
                  f"qualified_so_far={stats.qualified}  t={elapsed:.1f}s")
        if not next_f:
            break
        frontier = next_f

    df = pl.from_dicts(qualified) if qualified else pl.DataFrame()
    if df.height > 0:
        df = df.sort("S", descending=True)
    return df, stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="edu_kor")
    ap.add_argument("--max_len", type=int, default=3)
    ap.add_argument("--theta_sup", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--no_joint_bound", action="store_true")
    args = ap.parse_args()

    cfg = C2DPMR1Config(
        theta_sup=args.theta_sup,
        tau=args.tau,
        lam=args.lam,
        max_len=args.max_len,
        use_joint_bound=not args.no_joint_bound,
    )
    df, stats = mine_r1(cfg, spec=args.dataset)
    print(f"\n=== Results [{args.dataset}] ===")
    print(f"qualified: {df.height}")
    print(f"explored: {stats.explored}")
    print(f"pruned Apriori: {stats.pruned_apriori}")
    print(f"pruned joint bound: {stats.pruned_joint_bound}")
    print(f"per-level times: {[round(t,2) for t in stats.level_times]}")

    bound_suffix = "" if cfg.use_joint_bound else "_nobound"
    out = RESULTS / f"c2dpm_r1_{args.dataset}_lam{int(cfg.lam)}{bound_suffix}.parquet"
    df.write_parquet(out)
    print(f"wrote {out}")
    if df.height > 0:
        print(df.head(15).select(["pattern_str","length","S","mean_ig","var_ig"]))


if __name__ == "__main__":
    main()
