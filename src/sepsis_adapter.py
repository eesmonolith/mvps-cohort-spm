"""
Sepsis event log adapter (4TU 2016).

~1050 sepsis-patient cases, ~15K clinical events from a Dutch hospital.
16 distinct activities (ER Registration, IV Antibiotics, CRP, etc.).

Cohort axis: case-start quarter (4 quarters, 2014).
Cluster axis: outcome — ICU admission (Admission IC) vs returned-to-ER
              vs released. We use 3 clusters from the 3 release types
              versus IC vs Return.

Outcome activities EXCLUDED from sequence to avoid event-token leakage.
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

RAW_XES = DATA_EXT / "sepsis" / "Sepsis_Cases.xes"

# Outcome activities (used for cluster label, removed from sequence)
OUTCOME_ACTIVITIES = {
    "Admission IC": 0,        # ICU admission (severe)
    "Return ER": 1,           # Returned to ER (re-admission within 28 days)
    "Release A": 2, "Release B": 2, "Release C": 2,
    "Release D": 2, "Release E": 2,  # any release (non-ICU, non-return)
}

# Coarser 3-cluster mapping: 0=ICU, 1=Return, 2=Release


def main():
    print(f"loading {RAW_XES}")
    log = pm4py.read_xes(str(RAW_XES))
    df = pm4py.convert_to_dataframe(log)
    print(f"  events: {len(df)}")

    # Build vocab (exclude outcome activities)
    activities = df["concept:name"].unique().tolist()
    non_outcome = [a for a in activities if a not in OUTCOME_ACTIVITIES]
    activity_to_id = {a: i for i, a in enumerate(sorted(non_outcome))}
    print(f"  activities: {len(activities)} total, {len(non_outcome)} non-outcome")

    cases = df.groupby("case:concept:name", sort=False)
    rows = []
    for case_id, gdf in cases:
        gdf = gdf.sort_values("time:timestamp")
        names = gdf["concept:name"].tolist()
        tokens = [activity_to_id[a] for a in names if a in activity_to_id]
        # outcome
        outcome = None
        for a in names:
            if a in OUTCOME_ACTIVITIES:
                outcome = OUTCOME_ACTIVITIES[a]
                if outcome == 0:  # ICU = most severe, takes priority
                    break
        if outcome is None:
            continue
        start_ts = gdf["time:timestamp"].iloc[0]
        # Quarter cohort
        m = start_ts.month
        q = (m - 1) // 3  # 0..3
        cohort = f"q{q+1}_{start_ts.year}"
        if start_ts.year != 2014:
            continue
        rows.append({
            "user_id": str(case_id),
            "cohort_label": cohort,
            "cluster_id": outcome,
            "token_ids": tokens,
            "n_events": len(tokens),
            "ts_start": start_ts,
        })

    df_out = pl.from_dicts(rows)
    df_out = df_out.filter(pl.col("n_events") >= MIN_TRAJECTORY_LEN)
    df_out = df_out.with_columns(
        pl.when(pl.col("n_events") > MAX_TRAJECTORY_LEN)
        .then(pl.col("token_ids").list.tail(MAX_TRAJECTORY_LEN))
        .otherwise(pl.col("token_ids"))
        .alias("token_ids")
    )
    print(f"  valid traces: {df_out.height}")
    print("\ncohort × cluster:")
    print(df_out.group_by(["cohort_label", "cluster_id"]).agg(pl.len())
          .sort(["cohort_label", "cluster_id"]))

    seq_out = DATA_PROC / "sequences_sepsis.parquet"
    df_out.select(["user_id", "cohort_label", "token_ids",
                   "n_events", "ts_start"]).write_parquet(seq_out)
    print(f"\nwrote {seq_out}")

    cluster_out = DATA_PROC / "cluster_labels_sepsis.parquet"
    df_out.select(["user_id", "cohort_label", "cluster_id"]).write_parquet(cluster_out)
    print(f"wrote {cluster_out}")

    vocab = sorted(activity_to_id.keys(), key=lambda k: activity_to_id[k])
    with open(RESULTS / "sepsis_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "n": len(vocab)}, f, indent=2)
    print(f"wrote vocab ({len(vocab)} tokens)")


if __name__ == "__main__":
    main()
