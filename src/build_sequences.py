"""
Build per-student event sequences per cohort.

For each (user_id, cohort) emit:
  - sorted event tokens (vocab IDs)
  - timestamp deltas (seconds since previous event)
  - per-event metadata (problem_id, event_type)

Output: data/processed/sequences_{cohort_label}.parquet
Columns: user_id, cohort_label, token_ids (list[int]), n_events, n_llm_calls, ts_start, ts_end
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.config import (
    COHORTS, DATA_PROC, TOKEN_TO_ID, VOCAB,
    SESSION_BREAK_THRESHOLD_MIN, MAX_TRAJECTORY_LEN, MIN_TRAJECTORY_LEN,
    error_name_to_token, submission_status_to_token,
)
from src.data_loader import (
    load_submissions, load_executions, load_error_help, load_problem_views,
)


def build_event_table(cohort_key: str) -> pl.DataFrame:
    """
    Merge submissions + executions + error_help + problem_views into a single
    long-format event stream per (user_id, timestamp).
    """
    print(f"[{cohort_key}] loading raw tables...")
    sub = load_submissions(cohort_key)
    ex = load_executions(cohort_key)
    eh = load_error_help(cohort_key)
    pv = load_problem_views(cohort_key)

    print(f"  sub={sub.shape}  ex={ex.shape}  eh={eh.shape}  pv={pv.shape}")

    # Submissions -> token
    sub_events = sub.select(
        pl.col("user_id").cast(pl.Utf8),
        pl.col("problem_id").cast(pl.Utf8),
        pl.col("timestamp"),
        pl.col("status")
        .map_elements(submission_status_to_token, return_dtype=pl.Utf8)
        .alias("token"),
    ).with_columns(pl.lit("submission").alias("event_source"))

    # Executions -> token
    # NOTE: skip_nulls=False so error_name=null rows map to exec_clean
    ex_events = ex.select(
        pl.col("user_id").cast(pl.Utf8),
        pl.col("problem_id").cast(pl.Utf8),
        pl.col("timestamp"),
        pl.col("error_name")
        .map_elements(error_name_to_token, return_dtype=pl.Utf8, skip_nulls=False)
        .alias("token"),
    ).with_columns(pl.lit("execution").alias("event_source"))

    # error_help -> llm_help_request token (problem_id via execution_id join)
    eh_with_problem = eh.join(
        ex.select(["execution_id", "problem_id"]),
        on="execution_id",
        how="left",
    )
    eh_events = eh_with_problem.select(
        pl.col("user_id").cast(pl.Utf8),
        pl.col("problem_id").cast(pl.Utf8),
        pl.col("timestamp"),
        pl.lit("llm_help_request").alias("token"),
        pl.lit("error_help").alias("event_source"),
    )

    # problem_views -> problem_view token
    pv_events = pv.select(
        pl.col("user_id").cast(pl.Utf8),
        pl.col("problem_id").cast(pl.Utf8),
        pl.col("timestamp"),
        pl.lit("problem_view").alias("token"),
        pl.lit("problem_view").alias("event_source"),
    )

    all_events = pl.concat(
        [sub_events, ex_events, eh_events, pv_events],
        how="vertical_relaxed",
    )

    # Filter to cohort period
    start, end = COHORTS[cohort_key]["period"]
    start_dt = pl.lit(start).str.to_datetime()
    end_dt = pl.lit(end).str.to_datetime() + pl.duration(days=1)

    all_events = all_events.filter(
        (pl.col("timestamp") >= start_dt)
        & (pl.col("timestamp") < end_dt)
        & pl.col("timestamp").is_not_null()
        & pl.col("user_id").is_not_null()
    )

    print(f"  merged events: {all_events.shape}")
    return all_events.sort(["user_id", "timestamp"])


def insert_session_breaks(events: pl.DataFrame) -> pl.DataFrame:
    """For each user, insert a session_break token where consecutive events
    are more than SESSION_BREAK_THRESHOLD_MIN apart."""
    # Compute time delta within user
    events = events.with_columns(
        pl.col("timestamp")
        .diff()
        .over("user_id")
        .alias("dt_prev")
    )
    threshold = timedelta(minutes=SESSION_BREAK_THRESHOLD_MIN)
    breaks = events.filter(pl.col("dt_prev") > threshold).select(
        pl.col("user_id"),
        pl.col("problem_id"),
        (pl.col("timestamp") - pl.duration(seconds=1)).alias("timestamp"),
        pl.lit("session_break").alias("token"),
        pl.lit("synthetic").alias("event_source"),
    )
    if len(breaks) > 0:
        breaks = breaks.with_columns(pl.lit(None, dtype=pl.Duration).alias("dt_prev"))
        events = pl.concat([events, breaks], how="vertical_relaxed")
    return events.drop("dt_prev").sort(["user_id", "timestamp"])


def aggregate_to_trajectories(events: pl.DataFrame, cohort_key: str) -> pl.DataFrame:
    """Group events by user_id and emit one trajectory row per user."""
    cohort_label = COHORTS[cohort_key]["label"]

    # Convert tokens to integer IDs (drop tokens not in vocab — should not happen)
    events = events.with_columns(
        pl.col("token")
        .map_elements(lambda t: TOKEN_TO_ID.get(t, -1), return_dtype=pl.Int32)
        .alias("token_id")
    )
    n_unknown = events.filter(pl.col("token_id") == -1).height
    if n_unknown > 0:
        print(f"  WARN: {n_unknown} unknown tokens dropped")
        events = events.filter(pl.col("token_id") >= 0)

    grouped = events.group_by("user_id").agg(
        pl.col("token_id").alias("token_ids"),
        pl.col("timestamp").min().alias("ts_start"),
        pl.col("timestamp").max().alias("ts_end"),
        pl.col("token").alias("tokens"),
    )

    grouped = grouped.with_columns(
        pl.col("token_ids").list.len().alias("n_events"),
        pl.lit(cohort_label).alias("cohort_label"),
    )

    # Count LLM calls per user
    llm_id = TOKEN_TO_ID["llm_help_request"]
    grouped = grouped.with_columns(
        pl.col("token_ids")
        .list.eval(pl.element() == llm_id)
        .list.sum()
        .alias("n_llm_calls")
    )

    # Filter min length
    grouped = grouped.filter(pl.col("n_events") >= MIN_TRAJECTORY_LEN)

    # Truncate from end (keep most recent MAX_TRAJECTORY_LEN events)
    grouped = grouped.with_columns(
        pl.when(pl.col("n_events") > MAX_TRAJECTORY_LEN)
        .then(pl.col("token_ids").list.tail(MAX_TRAJECTORY_LEN))
        .otherwise(pl.col("token_ids"))
        .alias("token_ids")
    )
    return grouped.select(
        ["user_id", "cohort_label", "token_ids",
         "n_events", "n_llm_calls", "ts_start", "ts_end"]
    )


def build_cohort_sequences(cohort_key: str) -> pl.DataFrame:
    events = build_event_table(cohort_key)
    events = insert_session_breaks(events)
    trajectories = aggregate_to_trajectories(events, cohort_key)

    cohort_label = COHORTS[cohort_key]["label"]
    out_path = DATA_PROC / f"sequences_{cohort_label}.parquet"
    trajectories.write_parquet(out_path)
    print(f"[{cohort_key}] wrote {out_path}  ({trajectories.shape[0]} students)")

    # Quick stats
    print(f"  n_events: mean={trajectories['n_events'].mean():.1f}  "
          f"median={trajectories['n_events'].median():.1f}  "
          f"max={trajectories['n_events'].max()}")
    print(f"  n_llm_calls: mean={trajectories['n_llm_calls'].mean():.2f}  "
          f"max={trajectories['n_llm_calls'].max()}")
    return trajectories


def main():
    print(f"VOCAB ({len(VOCAB)}): {VOCAB}\n")
    all_stats = []
    for cohort_key in COHORTS:
        traj = build_cohort_sequences(cohort_key)
        all_stats.append({
            "cohort": cohort_key,
            "n_students": traj.shape[0],
            "total_events": int(traj["n_events"].sum()),
            "avg_events_per_student": float(traj["n_events"].mean()),
            "total_llm_calls": int(traj["n_llm_calls"].sum()),
        })
        print()
    print("=" * 60)
    print("SUMMARY")
    for s in all_stats:
        print(f"  {s['cohort']:12s}  students={s['n_students']:5d}  "
              f"events={s['total_events']:7d}  "
              f"avg/stu={s['avg_events_per_student']:6.1f}  "
              f"llm={s['total_llm_calls']:6d}")


if __name__ == "__main__":
    main()
