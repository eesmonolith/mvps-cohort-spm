"""
RetailRocket adapter for C2DPM cross-domain validation.

Domain mapping:
  sequence:  per-visitor event tokens, time-ordered (filter visitors with >=5 events)
  cohort:    month of visitor's first event (May/Jun/Jul/Aug/Sep 2015 -> 5 cohorts)
  cluster:   behavioral outcome
      0: viewer-only       (no addtocart, no transaction)
      1: cart_abandoner    (>=1 addtocart, 0 transaction)
      2: converter         (>=1 transaction)

Event vocab (4 tokens — small for fast mining):
  0: view
  1: addtocart
  2: transaction
  3: session_break

Output: data/processed/sequences_retailrocket.parquet
        data/processed/cluster_labels_retailrocket.parquet
Both compatible with `src/c2dpm.load_dataset` if loader is adapted.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.config import DATA_EXT, DATA_PROC, RESULTS, MIN_TRAJECTORY_LEN, MAX_TRAJECTORY_LEN

RAW_EVENTS = DATA_EXT / "retailrocket" / "events.csv"

VOCAB_RR = ["view", "addtocart", "transaction", "session_break"]
TOKEN_RR = {t: i for i, t in enumerate(VOCAB_RR)}

# Class labels
CLUSTER_NAMES_RR = {
    0: "viewer",
    1: "cart_abandoner",
    2: "converter",
}

# Cohort: month label
COHORTS_RR = ["may2015", "jun2015", "jul2015", "aug2015", "sep2015"]
COHORT_RR_TO_ID = {c: i for i, c in enumerate(COHORTS_RR)}

SESSION_BREAK_MIN = 30
MIN_EVENTS = 5


def load_events() -> pl.DataFrame:
    df = pl.read_csv(
        RAW_EVENTS,
        schema_overrides={"transactionid": pl.Float64},
    )
    # Convert timestamp ms-since-epoch -> datetime
    df = df.with_columns(
        (pl.col("timestamp") * 1000).cast(pl.Datetime("us"))
        .alias("ts_dt"),
    )
    return df


def build_visitor_classes(df: pl.DataFrame) -> pl.DataFrame:
    """Assign each visitor a class label based on overall behavior."""
    agg = (
        df.group_by("visitorid")
        .agg(
            (pl.col("event") == "view").sum().alias("n_view"),
            (pl.col("event") == "addtocart").sum().alias("n_addtocart"),
            (pl.col("event") == "transaction").sum().alias("n_transaction"),
            pl.col("ts_dt").min().alias("first_event"),
        )
    )
    agg = agg.with_columns(
        pl.when(pl.col("n_transaction") > 0)
        .then(2)
        .when(pl.col("n_addtocart") > 0)
        .then(1)
        .otherwise(0)
        .alias("cluster_id"),
        # Cohort = month of first event
        pl.col("first_event").dt.strftime("%b%Y").str.to_lowercase().alias("cohort_label"),
    )
    return agg


def build_sequences(df: pl.DataFrame, classes: pl.DataFrame,
                    min_events: int = MIN_EVENTS) -> pl.DataFrame:
    """Build per-visitor sequence of token IDs with session breaks."""
    # Token mapping
    df = df.with_columns(
        pl.col("event").replace(
            {"view": 0, "addtocart": 1, "transaction": 2}
        ).cast(pl.Int32).alias("token_id"),
    )
    df = df.sort(["visitorid", "ts_dt"])

    # Compute previous timestamp per visitor for session_break detection
    df = df.with_columns(
        pl.col("ts_dt").diff().over("visitorid").alias("dt_prev"),
    )
    threshold = timedelta(minutes=SESSION_BREAK_MIN)
    breaks = df.filter(pl.col("dt_prev") > threshold).select(
        pl.col("visitorid"),
        (pl.col("ts_dt") - pl.duration(seconds=1)).alias("ts_dt"),
        pl.lit(TOKEN_RR["session_break"]).cast(pl.Int32).alias("token_id"),
    )
    df_keep = df.select(["visitorid", "ts_dt", "token_id"])
    all_events = pl.concat([df_keep, breaks], how="vertical_relaxed").sort(
        ["visitorid", "ts_dt"]
    )

    grouped = all_events.group_by("visitorid").agg(
        pl.col("token_id").alias("token_ids"),
    )
    grouped = grouped.with_columns(
        pl.col("token_ids").list.len().alias("n_events")
    ).filter(pl.col("n_events") >= min_events)

    # Truncate
    grouped = grouped.with_columns(
        pl.when(pl.col("n_events") > MAX_TRAJECTORY_LEN)
        .then(pl.col("token_ids").list.tail(MAX_TRAJECTORY_LEN))
        .otherwise(pl.col("token_ids"))
        .alias("token_ids")
    )

    # Join class + cohort
    grouped = grouped.join(
        classes.select(["visitorid", "cluster_id", "cohort_label"]),
        on="visitorid",
        how="left",
    )

    # Keep only cohorts in valid set (drop None / unexpected)
    grouped = grouped.filter(pl.col("cohort_label").is_in(COHORTS_RR))
    return grouped


def main():
    print(f"[rr] loading events from {RAW_EVENTS}...")
    df = load_events()
    print(f"[rr] events: {df.shape}")
    print(f"[rr] visitors: {df['visitorid'].n_unique():,}")

    print(f"\n[rr] computing visitor classes...")
    classes = build_visitor_classes(df)
    print(classes.select("cluster_id").to_series().value_counts().sort("cluster_id"))
    print(f"\ncohort distribution:")
    print(classes.select("cohort_label").to_series().value_counts().sort("cohort_label"))

    print(f"\n[rr] building sequences (min_events={MIN_EVENTS})...")
    sequences = build_sequences(df, classes)
    print(f"[rr] valid sequences: {sequences.shape[0]:,}")
    print(f"[rr] events per visitor: mean={sequences['n_events'].mean():.1f}  "
          f"median={sequences['n_events'].median():.1f}  "
          f"max={sequences['n_events'].max()}")

    # Cohort x cluster
    ct = (
        sequences.group_by(["cohort_label", "cluster_id"])
        .agg(pl.len().alias("n"))
        .sort(["cohort_label", "cluster_id"])
    )
    print("\nCohort × Cluster (RetailRocket):")
    print(ct)

    # Save
    seq_out = DATA_PROC / "sequences_retailrocket.parquet"
    sequences.rename({"visitorid": "user_id"}).select(
        ["user_id", "cohort_label", "token_ids", "n_events"]
    ).write_parquet(seq_out)
    print(f"\nwrote {seq_out}")

    cluster_out = DATA_PROC / "cluster_labels_retailrocket.parquet"
    cluster_df = sequences.rename({"visitorid": "user_id"}).select(
        ["user_id", "cohort_label", "cluster_id"]
    )
    cluster_df.write_parquet(cluster_out)
    print(f"wrote {cluster_out}")

    # Metadata
    meta = {
        "vocab": VOCAB_RR,
        "cluster_names": CLUSTER_NAMES_RR,
        "cohorts": COHORTS_RR,
        "n_sequences": int(sequences.shape[0]),
        "min_events": MIN_EVENTS,
    }
    with open(RESULTS / "retailrocket_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {RESULTS / 'retailrocket_metadata.json'}")


if __name__ == "__main__":
    main()
