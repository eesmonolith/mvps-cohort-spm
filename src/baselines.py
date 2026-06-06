"""
Baselines for C2DPM comparison.

  apriori_only:        frequent sequential pattern mining only (sup_min ≥ θ).
                       No stability, no discrim filter. PrefixSpan analogue.
  stability_only:      Apriori + filter by stability ≥ τ_s. Single-axis (cohort).
  discrim_only:        Apriori + filter by discrim ≥ τ_d. Single-axis (cluster).
                       Cheng 2007 / DDPMine sequential analogue.
  intersect:           run stability_only and discrim_only independently, then
                       output their intersection. The "trivially compose two
                       single-axis miners" baseline that any reviewer will ask
                       about.
  c2dpm_no_bound:      C2DPM joint filter but without joint upper bound prune
                       (Apriori only). Tests the value of the bound.

Each baseline returns (DataFrame of patterns, MiningStats).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import polars as pl

from src.c2dpm import C2DPMConfig, MiningStats, count_atomic, load_dataset
from src.config import VOCAB_SIZE, ID_TO_TOKEN
from src.scoring import (
    cohort_min_support, stability, discrim, total_support,
)


def _record(pattern, length, n_cz, N_cz):
    s = stability(n_cz, N_cz)
    d = discrim(n_cz, N_cz)
    return {
        "pattern": pattern,
        "pattern_str": " ".join(ID_TO_TOKEN[t] for t in pattern),
        "length": length,
        "S": s * d,
        "stability": s,
        "discrim": d,
        "sup_min": cohort_min_support(n_cz, N_cz),
        "support_total": total_support(n_cz, N_cz),
    }


def baseline_apriori_only(cfg: C2DPMConfig) -> tuple[pl.DataFrame, MiningStats]:
    """Frequent SPM only — sup_min ≥ θ_sup filter."""
    stats = MiningStats()
    seqs, cohs, clus, N_cz = load_dataset()
    K, M = N_cz.shape
    qualified = []
    frontier = []
    # L=1
    t0 = time.time()
    for tok in range(VOCAB_SIZE):
        p = (tok,)
        stats.explored += 1
        n_cz = count_atomic(seqs, cohs, clus, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            stats.pruned_apriori += 1
            continue
        qualified.append(_record(p, 1, n_cz, N_cz))
        stats.qualified += 1
        frontier.append((p, n_cz))
    stats.level_times.append(time.time() - t0)
    # Higher L
    for L in range(2, cfg.max_len + 1):
        t0 = time.time()
        next_f = []
        for prev_p, _ in frontier:
            for tok in range(VOCAB_SIZE):
                p = prev_p + (tok,)
                stats.explored += 1
                n_cz = count_atomic(seqs, cohs, clus, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    stats.pruned_apriori += 1
                    continue
                qualified.append(_record(p, L, n_cz, N_cz))
                stats.qualified += 1
                next_f.append((p, n_cz))
        stats.level_times.append(time.time() - t0)
        if not next_f:
            break
        frontier = next_f
    df = pl.from_dicts(qualified) if qualified else pl.DataFrame()
    if df.height > 0:
        df = df.sort("S", descending=True)
    return df, stats


def baseline_stability_only(cfg: C2DPMConfig) -> tuple[pl.DataFrame, MiningStats]:
    """Apriori + stability filter (single-axis cohort)."""
    df, stats = baseline_apriori_only(cfg)
    if df.height == 0:
        return df, stats
    df_f = df.filter(pl.col("stability") >= cfg.tau_s)
    return df_f, stats


def baseline_discrim_only(cfg: C2DPMConfig) -> tuple[pl.DataFrame, MiningStats]:
    """Apriori + discrim filter (single-axis cluster — Cheng 2007 analogue)."""
    df, stats = baseline_apriori_only(cfg)
    if df.height == 0:
        return df, stats
    df_f = df.filter(pl.col("discrim") >= cfg.tau_d)
    return df_f, stats


def baseline_intersect(cfg: C2DPMConfig) -> tuple[pl.DataFrame, MiningStats]:
    """Stability-only ∩ Discrim-only (compose 2 single-axis miners)."""
    df, stats = baseline_apriori_only(cfg)
    if df.height == 0:
        return df, stats
    df_f = df.filter(
        (pl.col("stability") >= cfg.tau_s) & (pl.col("discrim") >= cfg.tau_d)
    )
    return df_f, stats


def baseline_c2dpm_no_bound(cfg: C2DPMConfig) -> tuple[pl.DataFrame, MiningStats]:
    """C2DPM joint filter, but extension uses only Apriori (no joint UB)."""
    cfg2 = C2DPMConfig(
        theta_sup=cfg.theta_sup,
        tau_s=cfg.tau_s,
        tau_d=cfg.tau_d,
        max_len=cfg.max_len,
        use_joint_bound=False,
        verbose=cfg.verbose,
    )
    from src.c2dpm import mine
    return mine(cfg2)


def compare_all(cfg: C2DPMConfig | None = None):
    if cfg is None:
        cfg = C2DPMConfig(max_len=2, verbose=True)
    print(f"Config: {cfg}\n")

    runs = {}

    print("=== apriori_only ===")
    t0 = time.time()
    df, st = baseline_apriori_only(cfg)
    runs["apriori_only"] = {
        "qualified": df.height, "explored": st.explored,
        "pruned_apriori": st.pruned_apriori, "time_s": time.time() - t0,
    }
    print(f"  qualified={df.height}  explored={st.explored}  "
          f"t={time.time()-t0:.1f}s")

    print("\n=== stability_only ===")
    t0 = time.time()
    df, st = baseline_stability_only(cfg)
    runs["stability_only"] = {
        "qualified": df.height, "explored": st.explored,
        "time_s": time.time() - t0,
    }
    print(f"  qualified={df.height}  t={time.time()-t0:.1f}s")

    print("\n=== discrim_only ===")
    t0 = time.time()
    df, st = baseline_discrim_only(cfg)
    runs["discrim_only"] = {
        "qualified": df.height, "explored": st.explored,
        "time_s": time.time() - t0,
    }
    print(f"  qualified={df.height}  t={time.time()-t0:.1f}s")

    print("\n=== intersect ===")
    t0 = time.time()
    df, st = baseline_intersect(cfg)
    runs["intersect"] = {
        "qualified": df.height, "explored": st.explored,
        "time_s": time.time() - t0,
    }
    print(f"  qualified={df.height}  t={time.time()-t0:.1f}s")

    return runs


if __name__ == "__main__":
    compare_all()
