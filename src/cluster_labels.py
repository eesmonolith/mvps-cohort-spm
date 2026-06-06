"""
Build per-student cluster labels.

Phase-1 placeholder: behavioral-feature KMeans (cheap, deterministic).
Later (if time) replace with multimodal-encoder + HDBSCAN.

Features per student:
  - submit_pass_rate     = #submit_pass / #submissions
  - error_rate           = #(submit_error + submit_fail) / #submissions
  - llm_dependency       = #llm_help_request / #events
  - error_diversity      = entropy over exec_error_* tokens
  - syntax_error_share   = #exec_SyntaxError / #exec_*
  - runtime_error_share  = #exec_(Type|Value|Key|Index|Attribute) / #exec_*
  - activity_volume      = log(n_events)
  - session_breaks       = #session_break

KMeans with K=4 (target 4 behavioral modes).
"""
from __future__ import annotations

import json
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PROC, RESULTS, SEED, COHORTS, TOKEN_TO_ID, VOCAB,
)

# Token-id constants
T_PASS = TOKEN_TO_ID["submit_pass"]
T_FAIL = TOKEN_TO_ID["submit_fail"]
T_SUBE = TOKEN_TO_ID["submit_error"]
T_LLM = TOKEN_TO_ID["llm_help_request"]
T_SB = TOKEN_TO_ID["session_break"]
T_CLEAN = TOKEN_TO_ID["exec_clean"]

EXEC_ERR_TOKEN_IDS = [
    TOKEN_TO_ID[t] for t in VOCAB if t.startswith("exec_") and t != "exec_clean"
]
RUNTIME_ERR_NAMES = {
    "exec_TypeError", "exec_ValueError", "exec_KeyError",
    "exec_IndexError", "exec_AttributeError",
}
RUNTIME_ERR_IDS = [TOKEN_TO_ID[n] for n in RUNTIME_ERR_NAMES]
T_SYNTAX = TOKEN_TO_ID["exec_SyntaxError"]

N_CLUSTERS = 4
FEATURE_NAMES = [
    "pass_rate",
    "error_rate",
    "llm_dependency",
    "error_diversity",
    "syntax_error_share",
    "runtime_error_share",
    "log_activity",
    "session_breaks_norm",
]


def _shannon_entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s == 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def features_from_token_ids(token_ids: list[int]) -> dict:
    arr = np.asarray(token_ids, dtype=np.int64)
    n = arr.size
    if n == 0:
        return {f: 0.0 for f in FEATURE_NAMES}

    n_pass = int((arr == T_PASS).sum())
    n_fail = int((arr == T_FAIL).sum())
    n_sub_err = int((arr == T_SUBE).sum())
    n_sub_total = n_pass + n_fail + n_sub_err + 1  # +1 smooth
    pass_rate = n_pass / n_sub_total
    error_rate = (n_fail + n_sub_err) / n_sub_total

    n_llm = int((arr == T_LLM).sum())
    llm_dependency = n_llm / n

    # Error token distribution (excluding exec_clean)
    err_counts = np.array(
        [int((arr == tid).sum()) for tid in EXEC_ERR_TOKEN_IDS],
        dtype=np.float64,
    )
    error_diversity = _shannon_entropy(err_counts)
    err_total = err_counts.sum() + 1.0
    n_syntax = int((arr == T_SYNTAX).sum())
    syntax_error_share = n_syntax / err_total
    n_runtime = sum(int((arr == tid).sum()) for tid in RUNTIME_ERR_IDS)
    runtime_error_share = n_runtime / err_total

    log_activity = float(np.log1p(n))
    session_breaks_norm = int((arr == T_SB).sum()) / max(n, 1)

    return {
        "pass_rate": pass_rate,
        "error_rate": error_rate,
        "llm_dependency": llm_dependency,
        "error_diversity": error_diversity,
        "syntax_error_share": syntax_error_share,
        "runtime_error_share": runtime_error_share,
        "log_activity": log_activity,
        "session_breaks_norm": session_breaks_norm,
    }


def build_features(seq_df: pl.DataFrame) -> pl.DataFrame:
    """Compute behavioral feature vector per student."""
    feats = []
    for row in seq_df.iter_rows(named=True):
        f = features_from_token_ids(row["token_ids"])
        f["user_id"] = row["user_id"]
        f["cohort_label"] = row["cohort_label"]
        feats.append(f)
    return pl.from_dicts(feats)


def cluster_all_cohorts(k: int = N_CLUSTERS, seed: int = SEED) -> pl.DataFrame:
    """Fit KMeans on combined 3-cohort features, emit cluster labels per student."""
    cohort_labels = [c["label"] for c in COHORTS.values()]
    all_feats = []
    for cl in cohort_labels:
        seq = pl.read_parquet(DATA_PROC / f"sequences_{cl}.parquet")
        all_feats.append(build_features(seq))
    feats = pl.concat(all_feats, how="vertical")
    print(f"Feature matrix: {feats.shape}")

    X = feats.select(FEATURE_NAMES).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(Xs)

    feats = feats.with_columns(pl.Series("cluster_id", labels.astype(np.int32)))

    # Save per-student cluster labels
    out_path = DATA_PROC / "cluster_labels.parquet"
    feats.write_parquet(out_path)
    print(f"Wrote {out_path}  ({feats.shape[0]} students, {k} clusters)")

    # Per-cluster summary
    print("\nCluster sizes + centroid (z-score space):")
    centroids = km.cluster_centers_
    for cid in range(k):
        size = int((labels == cid).sum())
        print(f"  cluster {cid}  n={size}  centroid={centroids[cid].round(2)}")

    # Per-cohort × cluster cross-tab
    ct = (
        feats.group_by(["cohort_label", "cluster_id"])
        .agg(pl.len().alias("n"))
        .sort(["cohort_label", "cluster_id"])
    )
    print("\nCohort × Cluster:")
    print(ct)

    # Save scaler + KMeans params for reproducibility
    meta = {
        "feature_names": FEATURE_NAMES,
        "n_clusters": k,
        "seed": seed,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "kmeans_centroids": km.cluster_centers_.tolist(),
        "kmeans_inertia": float(km.inertia_),
    }
    with open(RESULTS / "cluster_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {RESULTS / 'cluster_metadata.json'}")
    return feats


if __name__ == "__main__":
    cluster_all_cohorts()
