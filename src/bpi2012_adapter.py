"""
BPI Challenge 2012 adapter.

Loan-application process event log from a Dutch financial institution.
~262K events across ~13K loan-application cases.

Sequence: per-case ordered events (activity tokens).
Cohort axis: month of case start (Oct 2011 - Mar 2012 → K=6 months).
Cluster axis: case outcome (APPROVED / DECLINED / CANCELLED) → M=3.

The BPI event log uses three sub-processes: A (application), O (offer),
W (work-item). We tokenise activity names with a coarse vocabulary
(~ 24 distinct activity tokens).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pm4py

from src.config import DATA_EXT, DATA_PROC, RESULTS, MIN_TRAJECTORY_LEN, MAX_TRAJECTORY_LEN

RAW_XES = DATA_EXT / "bpi2012" / "BPIC2012.xes"

# 6 month cohorts
COHORTS_BPI = ["2011-10", "2011-11", "2011-12",
               "2012-01", "2012-02", "2012-03"]

# Outcome labels
OUTCOMES = {"A_APPROVED": 0, "A_DECLINED": 1, "A_CANCELLED": 2}


def load_xes():
    print(f"loading {RAW_XES}...")
    log = pm4py.read_xes(str(RAW_XES))
    print(f"  {len(log):,} events")
    return log


def build_vocab_and_outcome(df: pd.DataFrame):
    """Map each (case_id) → sequence of activity tokens + outcome label.

    NOTE: The three outcome activities (A_APPROVED/A_DECLINED/A_CANCELLED)
    are extracted as cluster labels but EXCLUDED from the event sequence,
    to avoid trivial event-token-defined cluster prediction (the failure
    mode demonstrated by RetailRocket).
    """
    activities = df["concept:name"].unique().tolist()
    # Build vocab excluding outcome tokens (to avoid leakage into sequence)
    non_outcome_acts = [a for a in activities if a not in OUTCOMES]
    print(f"  {len(activities)} distinct activities ({len(non_outcome_acts)} after dropping outcome)")
    activity_to_id = {a: i for i, a in enumerate(sorted(non_outcome_acts))}

    cases = df.groupby("case:concept:name", sort=False)
    rows = []
    for case_id, gdf in cases:
        gdf = gdf.sort_values("time:timestamp")
        names = gdf["concept:name"].tolist()
        # tokens: ALL non-outcome activities
        tokens = [activity_to_id[a] for a in names if a in activity_to_id]
        # outcome: last A_APPROVED / A_DECLINED / A_CANCELLED in trace
        outcome = None
        for a in reversed(names):
            if a in OUTCOMES:
                outcome = OUTCOMES[a]
                break
        if outcome is None:
            continue
        start_ts = gdf["time:timestamp"].iloc[0]
        cohort_label = f"{start_ts.year}-{start_ts.month:02d}"
        if cohort_label not in COHORTS_BPI:
            continue
        rows.append({
            "user_id": str(case_id),
            "cohort_label": cohort_label,
            "cluster_id": outcome,
            "token_ids": tokens,
            "n_events": len(tokens),
            "ts_start": start_ts,
            "ts_end": gdf["time:timestamp"].iloc[-1],
        })

    df_out = pl.from_dicts(rows)
    # Filter length, truncate
    df_out = df_out.filter(pl.col("n_events") >= MIN_TRAJECTORY_LEN)
    df_out = df_out.with_columns(
        pl.when(pl.col("n_events") > MAX_TRAJECTORY_LEN)
        .then(pl.col("token_ids").list.tail(MAX_TRAJECTORY_LEN))
        .otherwise(pl.col("token_ids"))
        .alias("token_ids")
    )
    return df_out, activity_to_id


def main():
    log = load_xes()
    df = pm4py.convert_to_dataframe(log)
    print(f"DataFrame: {df.shape}")
    print(f"cols: {df.columns.tolist()[:8]}")

    traj, vocab_map = build_vocab_and_outcome(df)
    print(f"\nvalid traces: {traj.height}")
    if traj.height == 0:
        raise RuntimeError("No traces — check outcome / cohort filtering")

    # Cohort × cluster distribution
    print("\ncohort × cluster:")
    print(traj.group_by(["cohort_label", "cluster_id"]).agg(pl.len())
          .sort(["cohort_label", "cluster_id"]))

    seq_out = DATA_PROC / "sequences_bpi2012.parquet"
    traj.select(["user_id", "cohort_label", "token_ids",
                 "n_events", "ts_start", "ts_end"]).write_parquet(seq_out)
    print(f"\nwrote {seq_out}")

    cluster_out = DATA_PROC / "cluster_labels_bpi2012.parquet"
    traj.select(["user_id", "cohort_label", "cluster_id"]).write_parquet(cluster_out)
    print(f"wrote {cluster_out}")

    # Vocab
    vocab = sorted(vocab_map.keys(), key=lambda k: vocab_map[k])
    with open(RESULTS / "bpi2012_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "n": len(vocab)}, f, indent=2)
    print(f"wrote vocab ({len(vocab)} tokens)")


if __name__ == "__main__":
    main()
