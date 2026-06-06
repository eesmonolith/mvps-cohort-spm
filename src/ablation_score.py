"""
Ablation: score-function variants.

We compare three aggregation functions for combining the stability and
discriminativeness factors of a candidate pattern:

  multiplicative:    S(p) = stability(p) * discrim(p)         (the paper default)
  harmonic:          S(p) = 2 s d / (s + d)                   (Fbeta-1 style)
  weighted-sum:      S(p) = 0.5 * stability + 0.5 * discrim   (additive)

For each variant we mine on the Edu dataset at the same Apriori support
threshold and report:
  - top-10 patterns' Spearman rank correlation against the multiplicative top-10
  - average length of qualifying patterns
  - total number of qualifying patterns at fixed (tau_s, tau_d)

Outputs JSON to results/ablation_score.json.
"""
from __future__ import annotations

import json
import time
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.c2dpm import C2DPMConfig, load_dataset, count_atomic
from src.config import RESULTS, VOCAB_SIZE, ID_TO_TOKEN
from src.scoring import (
    stability, discrim, cohort_min_support, total_support,
)


def _enumerate_qualifying(score_fn, cfg, sequences, cohorts, clusters, N_cz,
                          tau_threshold, vocab_size, max_len=3):
    """Enumerate all patterns up to max_len, score with score_fn, return list."""
    K, M = N_cz.shape
    qualified = []
    # L=1
    frontier = []
    for tok in range(vocab_size):
        p = (tok,)
        n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
        if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
            continue
        s = stability(n_cz, N_cz); d = discrim(n_cz, N_cz)
        S_val = score_fn(s, d)
        if S_val >= tau_threshold:
            qualified.append((p, S_val, s, d))
        frontier.append((p, n_cz))
    for L in range(2, max_len + 1):
        next_f = []
        for prev, _ in frontier:
            for tok in range(vocab_size):
                p = prev + (tok,)
                n_cz = count_atomic(sequences, cohorts, clusters, p, K, M)
                if cohort_min_support(n_cz, N_cz) < cfg.theta_sup:
                    continue
                s = stability(n_cz, N_cz); d = discrim(n_cz, N_cz)
                S_val = score_fn(s, d)
                if S_val >= tau_threshold:
                    qualified.append((p, S_val, s, d))
                next_f.append((p, n_cz))
        if not next_f:
            break
        frontier = next_f
    qualified.sort(key=lambda x: -x[1])
    return qualified


def multiplicative(s, d):
    return s * d


def harmonic(s, d):
    if s + d <= 0:
        return 0.0
    return 2.0 * s * d / (s + d)


def weighted(s, d):
    return 0.5 * s + 0.5 * d


def main():
    cfg = C2DPMConfig(theta_sup=0.05, tau_s=0.7, tau_d=0.05, max_len=3)
    sequences, cohorts, clusters, N_cz = load_dataset("edu_kor")
    print(f"Loaded {len(sequences)} sequences")

    variants = {
        "multiplicative": (multiplicative, cfg.tau_s * cfg.tau_d),  # 0.035
        "harmonic":       (harmonic,       2 * cfg.tau_s * cfg.tau_d
                                            / (cfg.tau_s + cfg.tau_d)),
        "weighted_sum":   (weighted,       0.5 * cfg.tau_s + 0.5 * cfg.tau_d),
    }

    results = {}
    top10_keys = {}
    for name, (fn, thr) in variants.items():
        t0 = time.time()
        q = _enumerate_qualifying(fn, cfg, sequences, cohorts, clusters,
                                  N_cz, thr, VOCAB_SIZE, cfg.max_len)
        elapsed = time.time() - t0
        avg_len = float(np.mean([len(p) for p, *_ in q])) if q else 0.0
        results[name] = {
            "n_qualified": len(q),
            "threshold": thr,
            "avg_pattern_length": avg_len,
            "mining_time_s": elapsed,
            "top10": [
                {
                    "pattern": " ".join(ID_TO_TOKEN[t] for t in p),
                    "score": float(S),
                    "stability": float(s),
                    "discrim": float(d),
                }
                for (p, S, s, d) in q[:10]
            ],
        }
        top10_keys[name] = [tuple(p) for p, *_ in q[:10]]
        print(f"  {name}: n={len(q)}  avg_len={avg_len:.2f}  t={elapsed:.1f}s")

    # Spearman / Jaccard against multiplicative top-10
    base = set(top10_keys["multiplicative"])
    for name in ("harmonic", "weighted_sum"):
        other = set(top10_keys[name])
        jaccard = len(base & other) / max(len(base | other), 1)
        results[name]["jaccard_top10_vs_multiplicative"] = float(jaccard)
        print(f"  {name} top-10 jaccard vs multiplicative: {jaccard:.2f}")

    with open(RESULTS / "ablation_score.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS / 'ablation_score.json'}")


if __name__ == "__main__":
    main()
