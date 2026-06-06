"""
codle_K12_python_hashed adapter — second education-domain dataset.

Different student population (anonymised 2025 year, 14,687 students) on
same Codle platform as Kor 3-semester data. Uses same event vocabulary
as the main Edu dataset to enable transfer comparisons.

Cohort axis: quarter of 2025 (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep,
              Q4=Oct-Dec) → K=4.
Cluster axis: KMeans on per-student behavioural features → M=4
              (same feature set as main Edu).
"""
from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    DATA_RAW, DATA_PROC, RESULTS, VOCAB, TOKEN_TO_ID, MAX_TRAJECTORY_LEN,
    MIN_TRAJECTORY_LEN, SESSION_BREAK_THRESHOLD_MIN,
    error_name_to_token, submission_status_to_token,
)

# Private corpus; override via HPIC_CODLE_RAW, else relative to HPIC_DATA_RAW.
RAW = Path(os.environ.get("HPIC_CODLE_RAW", str(DATA_RAW / "codle_K12_python")))

# Quarter labels
QUARTERS = ["q1_2025", "q2_2025", "q3_2025", "q4_2025"]
QUARTER_MONTH_MAP = {1: 0, 2: 0, 3: 0,  # Q1
                     4: 1, 5: 1, 6: 1,  # Q2
                     7: 2, 8: 2, 9: 2,  # Q3
                     10: 3, 11: 3, 12: 3}  # Q4

SESSION_BREAK_MIN = SESSION_BREAK_THRESHOLD_MIN


def _ts_to_quarter_label(ts: pd.Timestamp) -> str | None:
    if pd.isna(ts):
        return None
    if ts.year != 2025:
        return None
    return QUARTERS[QUARTER_MONTH_MAP[ts.month]]


def load_codle_events() -> pl.DataFrame:
    """Merge submissions + executions + error_help into one event stream."""
    def _read(path, cols):
        return pd.read_csv(path, engine="python", on_bad_lines="skip",
                           quoting=1, usecols=cols)

    print("loading codle hashed CSVs...")
    sub = _read(RAW / "submissions_hashed.csv",
                ["problem_id", "user_id", "status", "timestamp"])
    ex = _read(RAW / "executions_hashed.csv",
               ["execution_id", "user_id", "problem_id", "error_name", "timestamp"])
    eh = _read(RAW / "error_help_hashed.csv",
               ["user_id", "execution_id", "timestamp"])

    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    ex["timestamp"] = pd.to_datetime(ex["timestamp"], errors="coerce")
    eh["timestamp"] = pd.to_datetime(eh["timestamp"], errors="coerce")

    print(f"  sub {len(sub)}  ex {len(ex)}  eh {len(eh)}")

    # submission → submit_* tokens
    sub_evt = pd.DataFrame({
        "user_id": sub["user_id"].astype(str),
        "problem_id": sub["problem_id"].astype(str),
        "timestamp": sub["timestamp"],
        "token": sub["status"].apply(submission_status_to_token),
    })
    # execution → exec_* tokens
    ex_evt = pd.DataFrame({
        "user_id": ex["user_id"].astype(str),
        "problem_id": ex["problem_id"].astype(str),
        "timestamp": ex["timestamp"],
        "token": ex["error_name"].apply(error_name_to_token),
    })
    # error_help → llm_help_request token
    eh_join = eh.merge(ex[["execution_id", "problem_id"]],
                       on="execution_id", how="left")
    eh_evt = pd.DataFrame({
        "user_id": eh_join["user_id"].astype(str),
        "problem_id": eh_join["problem_id"].astype(str),
        "timestamp": eh_join["timestamp"],
        "token": "llm_help_request",
    })
    all_evt = pd.concat([sub_evt, ex_evt, eh_evt], ignore_index=True)
    all_evt = all_evt.dropna(subset=["timestamp", "user_id"])
    all_evt["timestamp"] = pd.to_datetime(all_evt["timestamp"], errors="coerce")
    all_evt = all_evt.dropna(subset=["timestamp"])

    # Restrict to 2025
    all_evt = all_evt[(all_evt["timestamp"].dt.year == 2025)]
    print(f"  events in 2025: {len(all_evt):,}")

    df = pl.from_pandas(all_evt)
    df = df.sort(["user_id", "timestamp"])
    return df


def insert_session_breaks(events: pl.DataFrame) -> pl.DataFrame:
    events = events.with_columns(
        pl.col("timestamp").diff().over("user_id").alias("dt_prev")
    )
    threshold = timedelta(minutes=SESSION_BREAK_MIN)
    breaks = events.filter(pl.col("dt_prev") > threshold).select(
        pl.col("user_id"),
        pl.col("problem_id"),
        (pl.col("timestamp") - pl.duration(seconds=1)).alias("timestamp"),
        pl.lit("session_break").alias("token"),
    )
    if breaks.height > 0:
        events = pl.concat(
            [events.drop("dt_prev"), breaks],
            how="vertical_relaxed",
        ).sort(["user_id", "timestamp"])
    else:
        events = events.drop("dt_prev")
    return events


def build_trajectories(events: pl.DataFrame) -> pl.DataFrame:
    # token id
    events = events.with_columns(
        pl.col("token").map_elements(
            lambda t: TOKEN_TO_ID.get(t, -1),
            return_dtype=pl.Int32,
            skip_nulls=False,
        ).alias("token_id")
    )
    events = events.filter(pl.col("token_id") >= 0)

    # quarter cohort label from first event timestamp per user
    user_quarter = events.group_by("user_id").agg(
        pl.col("timestamp").min().alias("first_ts")
    )
    user_quarter_df = user_quarter.to_pandas()
    user_quarter_df["cohort_label"] = user_quarter_df["first_ts"].apply(
        _ts_to_quarter_label
    )
    user_quarter_pl = pl.from_pandas(
        user_quarter_df[["user_id", "cohort_label"]].dropna()
    )
    events = events.join(user_quarter_pl, on="user_id", how="inner")

    grouped = events.group_by("user_id").agg(
        pl.col("token_id").alias("token_ids"),
        pl.col("cohort_label").first().alias("cohort_label"),
        pl.col("timestamp").min().alias("ts_start"),
        pl.col("timestamp").max().alias("ts_end"),
    )
    grouped = grouped.with_columns(
        pl.col("token_ids").list.len().alias("n_events"),
    ).filter(pl.col("n_events") >= MIN_TRAJECTORY_LEN)

    grouped = grouped.with_columns(
        pl.when(pl.col("n_events") > MAX_TRAJECTORY_LEN)
        .then(pl.col("token_ids").list.tail(MAX_TRAJECTORY_LEN))
        .otherwise(pl.col("token_ids"))
        .alias("token_ids")
    )
    llm_id = TOKEN_TO_ID["llm_help_request"]
    grouped = grouped.with_columns(
        pl.col("token_ids").list.eval(pl.element() == llm_id)
        .list.sum().alias("n_llm_calls")
    )
    return grouped.select(["user_id", "cohort_label", "token_ids",
                           "n_events", "n_llm_calls", "ts_start", "ts_end"])


def build_clusters(traj: pl.DataFrame, k: int = 4, seed: int = 42) -> pl.DataFrame:
    """Recompute KMeans behavioural clusters on this dataset only."""
    from src.cluster_labels import features_from_token_ids, FEATURE_NAMES
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print(f"computing behavioural features for {traj.height} users...")
    feats = []
    for row in traj.iter_rows(named=True):
        f = features_from_token_ids(row["token_ids"])
        f["user_id"] = row["user_id"]
        f["cohort_label"] = row["cohort_label"]
        feats.append(f)
    df = pl.from_dicts(feats)

    X = df.select(FEATURE_NAMES).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(Xs)
    df = df.with_columns(pl.Series("cluster_id", labels.astype(np.int32)))
    return df


def main():
    events = load_codle_events()
    events = insert_session_breaks(events)
    traj = build_trajectories(events)
    print(f"trajectories: {traj.height}")
    if traj.height == 0:
        raise RuntimeError("No trajectories — check timestamp filtering")

    seq_out = DATA_PROC / "sequences_codle_hashed.parquet"
    traj.write_parquet(seq_out)
    print(f"wrote {seq_out}")

    # Cohort × user count
    print("\ncohort distribution:")
    print(traj.group_by("cohort_label").agg(pl.len()).sort("cohort_label"))

    # Clusters
    clu = build_clusters(traj)
    cluster_out = DATA_PROC / "cluster_labels_codle_hashed.parquet"
    clu.select(["user_id", "cohort_label", "cluster_id"]).write_parquet(cluster_out)
    print(f"wrote {cluster_out}")

    # Cohort × cluster
    joined = traj.join(clu.select(["user_id", "cluster_id"]),
                       on="user_id", how="inner")
    print("\ncohort × cluster:")
    print(joined.group_by(["cohort_label", "cluster_id"]).agg(pl.len())
          .sort(["cohort_label", "cluster_id"]))


if __name__ == "__main__":
    main()
