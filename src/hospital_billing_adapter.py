"""
Hospital Billing event log adapter (4TU 2017, Mannhardt).

100K loan-billing cases from Dutch hospital ERP financial module.
Spans Dec 2012 – Jan 2016 (~3 years), 18 distinct activities.

Cohort axis: case-start year-quarter (target K=12, span 2013-Q1 to
             2015-Q4 + 2012-Q4/2016-Q1 if dense enough).
Cluster axis: outcome — BILLED (successful) / CANCELLED (isCancelled
             true) / OTHER (closed without billing).
Outcome activities (BILLED, REJECT, STORNO) excluded from sequence
vocab to avoid event-token leakage.
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

# Outcome-defining activities (used for cluster, removed from sequence)
OUTCOME_ACTS = {"BILLED", "REJECT", "STORNO"}


def case_outcome(activities, is_cancelled):
    """Return cluster id 0/1/2."""
    if "BILLED" in activities and not is_cancelled:
        return 0  # successful bill
    if "STORNO" in activities or "REJECT" in activities or is_cancelled:
        return 1  # cancelled / rejected
    return 2  # other / incomplete


def main():
    print(f"loading {RAW_XES}")
    log = pm4py.read_xes(str(RAW_XES))
    df = pm4py.convert_to_dataframe(log)
    print(f"  events: {len(df)}")

    activities = df["concept:name"].unique().tolist()
    non_outcome = [a for a in activities if a not in OUTCOME_ACTS]
    activity_to_id = {a: i for i, a in enumerate(sorted(non_outcome))}
    print(f"  activities: {len(activities)} total, {len(non_outcome)} non-outcome")

    cases = df.groupby("case:concept:name", sort=False)
    rows = []
    skipped = 0
    for case_id, gdf in cases:
        gdf = gdf.sort_values("time:timestamp")
        names = gdf["concept:name"].tolist()
        tokens = [activity_to_id[a] for a in names if a in activity_to_id]
        # outcome
        is_cancelled = False
        if "isCancelled" in gdf.columns:
            vals = gdf["isCancelled"].dropna().tolist()
            is_cancelled = any(str(v).lower() == "true" for v in vals)
        outcome = case_outcome(set(names), is_cancelled)
        start_ts = pd.to_datetime(gdf["time:timestamp"].iloc[0])
        y = start_ts.year
        q = (start_ts.month - 1) // 3 + 1
        if y < 2013 or y > 2015:
            skipped += 1; continue
        cohort = f"{y}-q{q}"
        rows.append({
            "user_id": str(case_id),
            "cohort_label": cohort,
            "cluster_id": outcome,
            "token_ids": tokens,
            "n_events": len(tokens),
            "ts_start": start_ts,
        })

    print(f"  skipped (out-of-range year): {skipped}")
    out = pl.from_dicts(rows)
    out = out.filter(pl.col("n_events") >= MIN_TRAJECTORY_LEN)
    out = out.with_columns(
        pl.when(pl.col("n_events") > MAX_TRAJECTORY_LEN)
        .then(pl.col("token_ids").list.tail(MAX_TRAJECTORY_LEN))
        .otherwise(pl.col("token_ids")).alias("token_ids")
    )
    print(f"\nvalid traces: {out.height}")
    print("\ncohort × cluster:")
    print(out.group_by(["cohort_label", "cluster_id"]).agg(pl.len())
          .sort(["cohort_label", "cluster_id"]))

    seq_out = DATA_PROC / "sequences_hospital_billing.parquet"
    out.select(["user_id", "cohort_label", "token_ids",
                "n_events", "ts_start"]).write_parquet(seq_out)
    print(f"\nwrote {seq_out}")
    cl_out = DATA_PROC / "cluster_labels_hospital_billing.parquet"
    out.select(["user_id", "cohort_label", "cluster_id"]).write_parquet(cl_out)
    print(f"wrote {cl_out}")

    vocab = sorted(activity_to_id.keys(), key=lambda k: activity_to_id[k])
    with open(RESULTS / "hospital_billing_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "n": len(vocab)}, f, indent=2)


if __name__ == "__main__":
    main()
