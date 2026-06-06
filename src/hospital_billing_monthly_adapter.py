"""
Hospital Billing monthly adapter (K=36 months Dec 2012 - Nov 2015).

Same as hospital_billing_adapter but cohort = year-month (K=36)
instead of year-quarter (K=12). Enables high-K real-corpus demo
where K*M = 36*3 = 108 makes vertex enumeration (2^108) and
face-wise (3^36 ≈ 1.5 * 10^17) both infeasible, only polynomial
bound (~26μs/call) is viable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pm4py

from src.config import DATA_EXT, DATA_PROC, RESULTS, MIN_TRAJECTORY_LEN, MAX_TRAJECTORY_LEN

RAW_XES = DATA_EXT / "hospital_billing" / "HospitalBilling.xes"

OUTCOME_ACTS = {"BILLED", "REJECT", "STORNO"}


def case_outcome(activities, is_cancelled):
    if "BILLED" in activities and not is_cancelled: return 0
    if "STORNO" in activities or "REJECT" in activities or is_cancelled: return 1
    return 2


def main():
    print(f"loading {RAW_XES}")
    log = pm4py.read_xes(str(RAW_XES))
    df = pm4py.convert_to_dataframe(log)
    print(f"  events: {len(df)}")

    activities = df["concept:name"].unique().tolist()
    non_outcome = [a for a in activities if a not in OUTCOME_ACTS]
    activity_to_id = {a: i for i, a in enumerate(sorted(non_outcome))}
    print(f"  vocab: {len(non_outcome)}")

    cases = df.groupby("case:concept:name", sort=False)
    rows = []
    skipped = 0
    for case_id, gdf in cases:
        gdf = gdf.sort_values("time:timestamp")
        names = gdf["concept:name"].tolist()
        tokens = [activity_to_id[a] for a in names if a in activity_to_id]
        is_cancelled = False
        if "isCancelled" in gdf.columns:
            vals = gdf["isCancelled"].dropna().tolist()
            is_cancelled = any(str(v).lower() == "true" for v in vals)
        outcome = case_outcome(set(names), is_cancelled)
        start_ts = pd.to_datetime(gdf["time:timestamp"].iloc[0])
        y = start_ts.year
        if y < 2013 or y > 2015:
            skipped += 1; continue
        m = start_ts.month
        cohort = f"{y}-{m:02d}"
        rows.append({
            "user_id": str(case_id),
            "cohort_label": cohort,
            "cluster_id": outcome,
            "token_ids": tokens,
            "n_events": len(tokens),
            "ts_start": start_ts,
        })

    print(f"  skipped: {skipped}")
    out = pl.from_dicts(rows)
    out = out.filter(pl.col("n_events") >= MIN_TRAJECTORY_LEN)
    out = out.with_columns(
        pl.when(pl.col("n_events") > MAX_TRAJECTORY_LEN)
        .then(pl.col("token_ids").list.tail(MAX_TRAJECTORY_LEN))
        .otherwise(pl.col("token_ids")).alias("token_ids")
    )
    print(f"\nvalid traces: {out.height}")
    cohort_counts = out.group_by("cohort_label").agg(pl.len()).sort("cohort_label")
    print(f"K={cohort_counts.height} cohorts")

    seq_out = DATA_PROC / "sequences_hospital_billing_monthly.parquet"
    out.select(["user_id", "cohort_label", "token_ids",
                "n_events", "ts_start"]).write_parquet(seq_out)
    cl_out = DATA_PROC / "cluster_labels_hospital_billing_monthly.parquet"
    out.select(["user_id", "cohort_label", "cluster_id"]).write_parquet(cl_out)

    vocab = sorted(activity_to_id.keys(), key=lambda k: activity_to_id[k])
    with open(RESULTS / "hospital_billing_monthly_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "n": len(vocab)}, f, indent=2)
    print(f"wrote {seq_out}, {cl_out}, vocab")


if __name__ == "__main__":
    main()
