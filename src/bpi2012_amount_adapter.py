"""
BPI 2012 amount-based cohort adapter.

Cohort axis: loan-amount tertile (small / medium / large) — orthogonal to
outcome so no temporal/outcome leakage.
Cluster axis: case outcome (APPROVED / DECLINED / CANCELLED).

Sequence: same activity tokens as bpi2012_adapter, outcome activities
excluded from vocab.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pm4py

from src.config import DATA_EXT, DATA_PROC, RESULTS, MIN_TRAJECTORY_LEN, MAX_TRAJECTORY_LEN

RAW_XES = DATA_EXT / "bpi2012" / "BPIC2012.xes"
OUTCOMES = {"A_APPROVED": 0, "A_DECLINED": 1, "A_CANCELLED": 2}


def main():
    print(f"loading {RAW_XES}")
    log = pm4py.read_xes(str(RAW_XES))
    df = pm4py.convert_to_dataframe(log)
    df["case:AMOUNT_REQ"] = pd.to_numeric(df["case:AMOUNT_REQ"], errors="coerce")

    activities = df["concept:name"].unique().tolist()
    non_outcome = [a for a in activities if a not in OUTCOMES]
    activity_to_id = {a: i for i, a in enumerate(sorted(non_outcome))}
    print(f"  vocab: {len(non_outcome)}")

    # Per-case amount
    case_amount = df.groupby("case:concept:name", sort=False)["case:AMOUNT_REQ"].first()
    amount_vals = case_amount.dropna().values
    q33, q66 = np.percentile(amount_vals, [33.33, 66.67])
    print(f"  amount tertiles: q33={q33:.0f}  q66={q66:.0f}")

    def amount_to_cohort(a):
        if pd.isna(a): return None
        if a <= q33: return "small"
        if a <= q66: return "medium"
        return "large"

    rows = []
    for case_id, gdf in df.groupby("case:concept:name", sort=False):
        gdf = gdf.sort_values("time:timestamp")
        names = gdf["concept:name"].tolist()
        tokens = [activity_to_id[a] for a in names if a in activity_to_id]
        outcome = None
        for a in reversed(names):
            if a in OUTCOMES:
                outcome = OUTCOMES[a]; break
        if outcome is None: continue
        amount = gdf["case:AMOUNT_REQ"].iloc[0]
        cohort = amount_to_cohort(amount)
        if cohort is None: continue
        rows.append({
            "user_id": str(case_id),
            "cohort_label": cohort,
            "cluster_id": outcome,
            "token_ids": tokens,
            "n_events": len(tokens),
            "ts_start": gdf["time:timestamp"].iloc[0],
            "ts_end": gdf["time:timestamp"].iloc[-1],
            "amount": float(amount),
        })

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

    seq_out = DATA_PROC / "sequences_bpi2012_amount.parquet"
    out.select(["user_id", "cohort_label", "token_ids", "n_events",
                "ts_start", "ts_end"]).write_parquet(seq_out)
    print(f"\nwrote {seq_out}")
    cl_out = DATA_PROC / "cluster_labels_bpi2012_amount.parquet"
    out.select(["user_id", "cohort_label", "cluster_id"]).write_parquet(cl_out)
    print(f"wrote {cl_out}")

    vocab = sorted(activity_to_id.keys(), key=lambda k: activity_to_id[k])
    with open(RESULTS / "bpi2012_amount_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "n": len(vocab),
                   "tertiles": {"q33": float(q33), "q66": float(q66)}}, f, indent=2)


if __name__ == "__main__":
    main()
