"""
C2DPM main mining algorithm.

Inputs:
  - sequences_DB:   dict[user_id] -> list[int] event tokens
  - cohort_labels:  dict[user_id] -> int (cohort index 0..K-1)
  - cluster_labels: dict[user_id] -> int (cluster index 0..M-1)

Hyperparameters:
  - theta_sup:  min cohort-conditional support
  - tau_s:      stability threshold
  - tau_d:      discriminativeness threshold
  - max_len:    max pattern length

Outputs:
  - DataFrame of (pattern, S, stability, discrim, sup_min) for qualifying patterns
  - Statistics (#explored, #pruned by each rule, mining time)

Pattern model: SUBSEQUENCE with gaps (standard SPM definition).
  s = [a, b, c, d, e]  contains pattern p = [a, c] iff there exist i < j
  such that s[i] == a and s[j] == c.

Performance: level-wise (Apriori) + projected-DB hint via prefix index.
This is a clean reference implementation prioritizing correctness for the
paper baseline comparison. Production-level speed left to future work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import polars as pl

from src.config import (
    DATA_PROC, COHORTS, VOCAB_SIZE, ID_TO_TOKEN, COHORT_LABEL_TO_ID,
    RESULTS,
)
from src.joint_bound import joint_upper_bound
from src.scoring import (
    cohort_min_support, stability, discrim, joint_score, total_support,
)


# ============================================================
# Dataset registry — supports multi-domain mining
# ============================================================

@dataclass
class DatasetSpec:
    name: str
    vocab: list
    cohort_keys: list                # ordered cohort labels
    seq_files: dict                  # cohort_label -> parquet path
    cluster_file: str | Path         # combined cluster labels file
    join_on: str = "user_id"


def kor_spec() -> DatasetSpec:
    return DatasetSpec(
        name="edu_kor",
        vocab=list(ID_TO_TOKEN.values()),
        cohort_keys=[c["label"] for c in COHORTS.values()],
        seq_files={
            c["label"]: DATA_PROC / f"sequences_{c['label']}.parquet"
            for c in COHORTS.values()
        },
        cluster_file=DATA_PROC / "cluster_labels.parquet",
    )


def retailrocket_spec() -> DatasetSpec:
    from src.retailrocket_adapter import VOCAB_RR, COHORTS_RR
    return DatasetSpec(
        name="retailrocket",
        vocab=VOCAB_RR,
        cohort_keys=COHORTS_RR,
        seq_files={  # single combined file in our adapter output
            "_all": DATA_PROC / "sequences_retailrocket.parquet"
        },
        cluster_file=DATA_PROC / "cluster_labels_retailrocket.parquet",
    )


def codle_hashed_spec() -> DatasetSpec:
    from src.codle_hashed_adapter import QUARTERS
    return DatasetSpec(
        name="codle_hashed",
        vocab=list(ID_TO_TOKEN.values()),  # same vocab as Kor
        cohort_keys=QUARTERS,
        seq_files={"_all": DATA_PROC / "sequences_codle_hashed.parquet"},
        cluster_file=DATA_PROC / "cluster_labels_codle_hashed.parquet",
    )


def bpi2012_spec() -> DatasetSpec:
    import json
    with open(RESULTS / "bpi2012_vocab.json") as f:
        meta = json.load(f)
    return DatasetSpec(
        name="bpi2012",
        vocab=meta["vocab"],
        cohort_keys=["2011-10", "2011-11", "2011-12",
                     "2012-01", "2012-02", "2012-03"],
        seq_files={"_all": DATA_PROC / "sequences_bpi2012.parquet"},
        cluster_file=DATA_PROC / "cluster_labels_bpi2012.parquet",
    )


def sepsis_spec() -> DatasetSpec:
    import json
    with open(RESULTS / "sepsis_vocab.json") as f:
        meta = json.load(f)
    return DatasetSpec(
        name="sepsis",
        vocab=meta["vocab"],
        cohort_keys=["q1_2014", "q2_2014", "q3_2014", "q4_2014"],
        seq_files={"_all": DATA_PROC / "sequences_sepsis.parquet"},
        cluster_file=DATA_PROC / "cluster_labels_sepsis.parquet",
    )


def oulad_spec() -> DatasetSpec:
    import json
    with open(RESULTS / "oulad_vocab.json") as f:
        meta = json.load(f)
    return DatasetSpec(
        name="oulad",
        vocab=meta["vocab"],
        cohort_keys=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        seq_files={"_all": DATA_PROC / "sequences_oulad.parquet"},
        cluster_file=DATA_PROC / "cluster_labels_oulad.parquet",
    )


def hospital_billing_spec() -> DatasetSpec:
    import json
    with open(RESULTS / "hospital_billing_vocab.json") as f:
        meta = json.load(f)
    quarters = [f"{y}-q{q}" for y in (2013, 2014, 2015) for q in (1, 2, 3, 4)]
    return DatasetSpec(
        name="hospital_billing",
        vocab=meta["vocab"],
        cohort_keys=quarters,
        seq_files={"_all": DATA_PROC / "sequences_hospital_billing.parquet"},
        cluster_file=DATA_PROC / "cluster_labels_hospital_billing.parquet",
    )


def bpi2012_amount_spec() -> DatasetSpec:
    import json
    with open(RESULTS / "bpi2012_amount_vocab.json") as f:
        meta = json.load(f)
    return DatasetSpec(
        name="bpi2012_amount",
        vocab=meta["vocab"],
        cohort_keys=["small", "medium", "large"],
        seq_files={"_all": DATA_PROC / "sequences_bpi2012_amount.parquet"},
        cluster_file=DATA_PROC / "cluster_labels_bpi2012_amount.parquet",
    )


DATASET_REGISTRY = {
    "edu_kor": kor_spec,
    "retailrocket": retailrocket_spec,
    "codle_hashed": codle_hashed_spec,
    "bpi2012": bpi2012_spec,
    "bpi2012_amount": bpi2012_amount_spec,
    "sepsis": sepsis_spec,
    "hospital_billing": hospital_billing_spec,
    "oulad": oulad_spec,
}


@dataclass
class MiningStats:
    explored: int = 0
    pruned_apriori: int = 0
    pruned_joint_bound: int = 0
    qualified: int = 0
    level_times: list = field(default_factory=list)


@dataclass
class C2DPMConfig:
    theta_sup: float = 0.02   # min cohort support 2%
    tau_s: float = 0.7        # stability threshold
    tau_d: float = 0.05       # discrim threshold (IG normalized)
    max_len: int = 5          # max pattern length
    use_joint_bound: bool = True   # toggle for ablation
    verbose: bool = True


# ============================================================
# Sequence DB
# ============================================================

def load_dataset(spec: DatasetSpec | str = "edu_kor"):
    """Load sequences + cohort/cluster labels for the given dataset spec.

    Returns:
        sequences: list of token-id sequences
        cohorts:   shape (N,) cohort index 0..K-1
        clusters:  shape (N,) cluster index 0..M-1
        N_cz:      shape (K, M) population matrix
        spec:      the DatasetSpec used (for vocab lookup)
    """
    if isinstance(spec, str):
        spec = DATASET_REGISTRY[spec]()
    # cohort_label -> id
    cohort_to_id = {label: i for i, label in enumerate(spec.cohort_keys)}

    cluster_df = pl.read_parquet(spec.cluster_file)
    if "cohort_label" in cluster_df.columns:
        cluster_map = {
            (row[spec.join_on], row["cohort_label"]): row["cluster_id"]
            for row in cluster_df.iter_rows(named=True)
        }
        use_cohort_in_key = True
    else:
        cluster_map = dict(zip(
            cluster_df[spec.join_on].to_list(),
            cluster_df["cluster_id"].to_list(),
        ))
        use_cohort_in_key = False

    sequences = []
    cohorts = []
    clusters = []

    for cohort_label, seq_path in spec.seq_files.items():
        seq_df = pl.read_parquet(seq_path)
        for row in seq_df.iter_rows(named=True):
            user = row[spec.join_on]
            # determine effective cohort id
            row_cohort_label = row.get("cohort_label", cohort_label)
            if row_cohort_label not in cohort_to_id:
                continue
            c_id = cohort_to_id[row_cohort_label]
            # lookup cluster id
            if use_cohort_in_key:
                z = cluster_map.get((user, row_cohort_label))
            else:
                z = cluster_map.get(user)
            if z is None:
                continue
            sequences.append(row["token_ids"])
            cohorts.append(c_id)
            clusters.append(z)

    cohorts = np.asarray(cohorts, dtype=np.int32)
    clusters = np.asarray(clusters, dtype=np.int32)
    if cohorts.size == 0:
        raise ValueError(f"No sequences matched for spec {spec.name}")
    K = int(cohorts.max()) + 1
    M = int(clusters.max()) + 1
    N_cz = np.zeros((K, M), dtype=np.int64)
    for c, z in zip(cohorts, clusters):
        N_cz[c, z] += 1
    return sequences, cohorts, clusters, N_cz


# ============================================================
# Pattern matching (subseq w/ gaps)
# ============================================================

def contains(seq: list[int], pattern: tuple[int, ...]) -> bool:
    """Subsequence-with-gaps containment check.

    Returns True iff there exist i_1 < i_2 < ... < i_k in seq such that
    seq[i_j] == pattern[j] for all j.
    """
    j = 0
    plen = len(pattern)
    for tok in seq:
        if tok == pattern[j]:
            j += 1
            if j == plen:
                return True
    return False


def count_atomic(sequences, cohorts, clusters, pattern,
                 K: int, M: int) -> np.ndarray:
    """Return n_{c,z}(p) shape (K, M)."""
    n_cz = np.zeros((K, M), dtype=np.int64)
    for s, c, z in zip(sequences, cohorts, clusters):
        if contains(s, pattern):
            n_cz[c, z] += 1
    return n_cz


# ============================================================
# Mining loop
# ============================================================

def mine(cfg: C2DPMConfig | None = None,
         spec: DatasetSpec | str = "edu_kor") -> tuple[pl.DataFrame, MiningStats]:
    if cfg is None:
        cfg = C2DPMConfig()
    if isinstance(spec, str):
        spec = DATASET_REGISTRY[spec]()
    stats = MiningStats()

    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape
    N = len(sequences)
    vocab = spec.vocab
    V = len(vocab)
    if cfg.verbose:
        print(f"[c2dpm] dataset={spec.name}  V={V}  sequences={N}  K={K}  M={M}")
        print(f"[c2dpm] N_cz=\n{N_cz}")

    qualified: list[dict] = []

    # ---- Level 1: single-token patterns ----
    t0 = time.time()
    frontier: list[tuple[tuple[int, ...], np.ndarray]] = []
    for tok in range(V):
        pattern = (tok,)
        stats.explored += 1
        n_cz = count_atomic(sequences, cohorts, clusters, pattern, K, M)
        sup_min = cohort_min_support(n_cz, N_cz)
        if sup_min < cfg.theta_sup:
            stats.pruned_apriori += 1
            continue
        s = stability(n_cz, N_cz)
        d = discrim(n_cz, N_cz)
        S = s * d
        if s >= cfg.tau_s and d >= cfg.tau_d:
            qualified.append({
                "pattern": pattern,
                "pattern_str": " ".join(vocab[t] for t in pattern),
                "length": 1,
                "S": S,
                "stability": s,
                "discrim": d,
                "sup_min": sup_min,
                "support_total": total_support(n_cz, N_cz),
            })
            stats.qualified += 1
        # Decide if extension is worthwhile
        if cfg.use_joint_bound:
            ub = joint_upper_bound(n_cz, N_cz)
            if ub < cfg.tau_s * cfg.tau_d:
                stats.pruned_joint_bound += 1
                continue
        frontier.append((pattern, n_cz))
    elapsed = time.time() - t0
    stats.level_times.append(elapsed)
    if cfg.verbose:
        print(f"[c2dpm] L=1  explored={V}  "
              f"frontier={len(frontier)}  qualified={stats.qualified}  "
              f"t={elapsed:.2f}s")

    # ---- Higher levels ----
    for L in range(2, cfg.max_len + 1):
        t0 = time.time()
        next_frontier: list[tuple[tuple[int, ...], np.ndarray]] = []
        for prev_pattern, _ in frontier:
            for tok in range(V):
                pattern = prev_pattern + (tok,)
                stats.explored += 1
                n_cz = count_atomic(sequences, cohorts, clusters, pattern, K, M)
                sup_min = cohort_min_support(n_cz, N_cz)
                if sup_min < cfg.theta_sup:
                    stats.pruned_apriori += 1
                    continue
                s = stability(n_cz, N_cz)
                d = discrim(n_cz, N_cz)
                S = s * d
                if s >= cfg.tau_s and d >= cfg.tau_d:
                    qualified.append({
                        "pattern": pattern,
                        "pattern_str": " ".join(vocab[t] for t in pattern),
                        "length": L,
                        "S": S,
                        "stability": s,
                        "discrim": d,
                        "sup_min": sup_min,
                        "support_total": total_support(n_cz, N_cz),
                    })
                    stats.qualified += 1
                if cfg.use_joint_bound:
                    ub = joint_upper_bound(n_cz, N_cz)
                    if ub < cfg.tau_s * cfg.tau_d:
                        stats.pruned_joint_bound += 1
                        continue
                next_frontier.append((pattern, n_cz))
        elapsed = time.time() - t0
        stats.level_times.append(elapsed)
        if cfg.verbose:
            print(f"[c2dpm] L={L}  frontier_in={len(frontier)}  "
                  f"frontier_out={len(next_frontier)}  "
                  f"qualified_so_far={stats.qualified}  t={elapsed:.2f}s")
        if not next_frontier:
            break
        frontier = next_frontier

    df = pl.from_dicts(qualified) if qualified else pl.DataFrame()
    if df.height > 0:
        df = df.sort("S", descending=True)
    return df, stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="edu_kor",
                    choices=list(DATASET_REGISTRY.keys()))
    ap.add_argument("--max_len", type=int, default=3)
    ap.add_argument("--theta_sup", type=float, default=0.05)
    ap.add_argument("--tau_s", type=float, default=0.7)
    ap.add_argument("--tau_d", type=float, default=0.05)
    ap.add_argument("--no_joint_bound", action="store_true")
    ap.add_argument("--out", default=None,
                    help="Output parquet path. Default uses results/c2dpm_{dataset}.parquet")
    args = ap.parse_args()

    cfg = C2DPMConfig(
        theta_sup=args.theta_sup,
        tau_s=args.tau_s,
        tau_d=args.tau_d,
        max_len=args.max_len,
        use_joint_bound=not args.no_joint_bound,
        verbose=True,
    )
    print(f"Config: {cfg}  dataset={args.dataset}")
    df, stats = mine(cfg, spec=args.dataset)
    print(f"\n=== Results [{args.dataset}] ===")
    print(f"qualified: {df.height}")
    print(f"explored: {stats.explored}")
    print(f"pruned by Apriori: {stats.pruned_apriori}")
    print(f"pruned by joint bound: {stats.pruned_joint_bound}")
    print(f"per-level times: {[round(t, 2) for t in stats.level_times]}")

    if df.height > 0:
        from src.config import RESULTS
        out = Path(args.out) if args.out else RESULTS / f"c2dpm_{args.dataset}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        print(f"\nwrote {out}")
        print(f"\nTop-15:")
        print(df.head(15).select(["pattern_str", "length", "S", "stability",
                                  "discrim", "sup_min"]))


if __name__ == "__main__":
    from pathlib import Path
    main()
