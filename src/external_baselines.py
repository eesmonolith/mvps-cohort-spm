"""
External-baseline wall-time + pattern-count comparison.

Implements PrefixSpan, single-axis discriminative (Cheng 2007 analogue),
v1 C2DPM, and R1 C2DPM, all sharing the same Apriori candidate
generation and subsequence-with-gaps containment definition. Measures
wall-clock running time and total qualifying patterns on Edu at
L_max=2 (so the slow R1+bound variant fits within a reasonable budget).
"""
from __future__ import annotations

import time
import json
import resource

import numpy as np
import polars as pl

from src.c2dpm import (
    C2DPMConfig, load_dataset, count_atomic,
)
from src.config import RESULTS, VOCAB_SIZE, ID_TO_TOKEN
from src.scoring import (
    cohort_min_support, stability, discrim, joint_score, total_support,
)
from src.scoring_r1 import per_cohort_ig
from src.joint_bound import joint_upper_bound as v1_joint_upper
from src.joint_bound_r1 import joint_upper_bound_r1


def _mem_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_prefixspan(seqs, cohs, clus, N_cz, cfg, K, M):
    """PrefixSpan analogue: frequency-only with Apriori prune."""
    stats = {"explored": 0, "qualified": 0, "pruned_apriori": 0}
    qualified = []
    frontier = []
    t0 = time.time()
    for tok in range(VOCAB_SIZE):
        p = (tok,)
        stats["explored"] += 1
        n_cz = count_atomic(seqs, cohs, clus, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            stats["pruned_apriori"] += 1
            continue
        qualified.append((p, total_support(n_cz, N_cz)))
        stats["qualified"] += 1
        frontier.append((p, n_cz))
    for L in range(2, cfg.max_len + 1):
        next_f = []
        for prev, _ in frontier:
            for tok in range(VOCAB_SIZE):
                p = prev + (tok,)
                stats["explored"] += 1
                n_cz = count_atomic(seqs, cohs, clus, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    stats["pruned_apriori"] += 1
                    continue
                qualified.append((p, total_support(n_cz, N_cz)))
                stats["qualified"] += 1
                next_f.append((p, n_cz))
        if not next_f:
            break
        frontier = next_f
    stats["time_s"] = time.time() - t0
    return qualified, stats


def run_discrim_only(seqs, cohs, clus, N_cz, cfg, K, M, tau_d=0.05):
    """Single-axis discriminative (Cheng 2007 sequential analogue)."""
    stats = {"explored": 0, "qualified": 0, "pruned_apriori": 0}
    qualified = []
    frontier = []
    t0 = time.time()
    for tok in range(VOCAB_SIZE):
        p = (tok,)
        stats["explored"] += 1
        n_cz = count_atomic(seqs, cohs, clus, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            stats["pruned_apriori"] += 1
            continue
        d = discrim(n_cz, N_cz)
        if d >= tau_d:
            qualified.append((p, d))
            stats["qualified"] += 1
        frontier.append((p, n_cz))
    for L in range(2, cfg.max_len + 1):
        next_f = []
        for prev, _ in frontier:
            for tok in range(VOCAB_SIZE):
                p = prev + (tok,)
                stats["explored"] += 1
                n_cz = count_atomic(seqs, cohs, clus, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    stats["pruned_apriori"] += 1
                    continue
                d = discrim(n_cz, N_cz)
                if d >= tau_d:
                    qualified.append((p, d))
                    stats["qualified"] += 1
                next_f.append((p, n_cz))
        if not next_f:
            break
        frontier = next_f
    stats["time_s"] = time.time() - t0
    return qualified, stats


def run_v1_with_bound(seqs, cohs, clus, N_cz, cfg, K, M, tau_s=0.7, tau_d=0.05):
    """v1 multiplicative S = stab * disc, with joint bound."""
    stats = {"explored": 0, "qualified": 0, "pruned_apriori": 0, "pruned_bound": 0}
    qualified = []
    frontier = []
    threshold = tau_s * tau_d
    t0 = time.time()
    for tok in range(VOCAB_SIZE):
        p = (tok,)
        stats["explored"] += 1
        n_cz = count_atomic(seqs, cohs, clus, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            stats["pruned_apriori"] += 1
            continue
        s = stability(n_cz, N_cz); d = discrim(n_cz, N_cz)
        if s >= tau_s and d >= tau_d:
            qualified.append((p, s * d))
            stats["qualified"] += 1
        if v1_joint_upper(n_cz, N_cz) < threshold:
            stats["pruned_bound"] += 1
            continue
        frontier.append((p, n_cz))
    for L in range(2, cfg.max_len + 1):
        next_f = []
        for prev, _ in frontier:
            for tok in range(VOCAB_SIZE):
                p = prev + (tok,)
                stats["explored"] += 1
                n_cz = count_atomic(seqs, cohs, clus, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    stats["pruned_apriori"] += 1
                    continue
                s = stability(n_cz, N_cz); d = discrim(n_cz, N_cz)
                if s >= tau_s and d >= tau_d:
                    qualified.append((p, s * d))
                    stats["qualified"] += 1
                if v1_joint_upper(n_cz, N_cz) < threshold:
                    stats["pruned_bound"] += 1
                    continue
                next_f.append((p, n_cz))
        if not next_f:
            break
        frontier = next_f
    stats["time_s"] = time.time() - t0
    return qualified, stats


def run_r1_with_bound(seqs, cohs, clus, N_cz, cfg, K, M, lam=50.0, tau=0.06):
    """R1 mean-var, with joint bound."""
    stats = {"explored": 0, "qualified": 0, "pruned_apriori": 0, "pruned_bound": 0}
    qualified = []
    frontier = []
    t0 = time.time()
    for tok in range(VOCAB_SIZE):
        p = (tok,)
        stats["explored"] += 1
        n_cz = count_atomic(seqs, cohs, clus, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            stats["pruned_apriori"] += 1
            continue
        ig = per_cohort_ig(n_cz, N_cz)
        S = float(ig.mean() - lam * ig.var())
        if S >= tau:
            qualified.append((p, S))
            stats["qualified"] += 1
        if joint_upper_bound_r1(n_cz, N_cz, lam) < tau:
            stats["pruned_bound"] += 1
            continue
        frontier.append((p, n_cz))
    for L in range(2, cfg.max_len + 1):
        next_f = []
        for prev, _ in frontier:
            for tok in range(VOCAB_SIZE):
                p = prev + (tok,)
                stats["explored"] += 1
                n_cz = count_atomic(seqs, cohs, clus, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    stats["pruned_apriori"] += 1
                    continue
                ig = per_cohort_ig(n_cz, N_cz)
                S = float(ig.mean() - lam * ig.var())
                if S >= tau:
                    qualified.append((p, S))
                    stats["qualified"] += 1
                if joint_upper_bound_r1(n_cz, N_cz, lam) < tau:
                    stats["pruned_bound"] += 1
                    continue
                next_f.append((p, n_cz))
        if not next_f:
            break
        frontier = next_f
    stats["time_s"] = time.time() - t0
    return qualified, stats


def main():
    cfg = C2DPMConfig(theta_sup=0.02, max_len=2)
    seqs, cohs, clus, N_cz = load_dataset("edu_kor")
    K, M = N_cz.shape
    print(f"loaded {len(seqs)} entities, K={K}, M={M}, L_max={cfg.max_len}")

    summary = {}

    print("\n[PrefixSpan analogue]")
    _, st = run_prefixspan(seqs, cohs, clus, N_cz, cfg, K, M)
    summary["PrefixSpan"] = st
    print(st)

    print("\n[Discrim-only (Cheng 2007 sequential)]")
    _, st = run_discrim_only(seqs, cohs, clus, N_cz, cfg, K, M)
    summary["DiscrimOnly"] = st
    print(st)

    print("\n[v1 C2DPM + bound]")
    _, st = run_v1_with_bound(seqs, cohs, clus, N_cz, cfg, K, M)
    summary["v1+bound"] = st
    print(st)

    print("\n[R1 C2DPM + bound]")
    _, st = run_r1_with_bound(seqs, cohs, clus, N_cz, cfg, K, M)
    summary["R1+bound"] = st
    print(st)

    with open(RESULTS / "external_baselines.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS / 'external_baselines.json'}")


if __name__ == "__main__":
    main()
