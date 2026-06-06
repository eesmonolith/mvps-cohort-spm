"""
OULAD (Open University Learning Analytics Dataset) adapter.

Cohort axis: course module (7 distinct: AAA, BBB, CCC, DDD, EEE, FFF, GGG)
            — each module is a different subject area, so cross-module
            heterogeneity is high (the reviewer-requested
            high-heterogeneity public corpus).
Cluster axis: final_result {Pass, Withdrawn, Fail, Distinction} → M=4.

Sequence: per-student VLE clicks, tokenised by activity_type (15-20
distinct types). Date-ordered. Trace is the course-presentation
specific click stream for that student in the module they enrolled.

We use only the "click" activity (not the outcome) so no event-token
leakage of final_result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.config import DATA_EXT, DATA_PROC, RESULTS, MIN_TRAJECTORY_LEN, MAX_TRAJECTORY_LEN

RAW = DATA_EXT / "OULAD"

MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
OUTCOMES = {"Distinction": 0, "Pass": 1, "Fail": 2, "Withdrawn": 3}


def main():
    print("loading OULAD...")
    si = pd.read_csv(RAW / "studentInfo.csv")
    vle = pd.read_csv(RAW / "vle.csv")
    sv = pd.read_csv(RAW / "studentVle.csv")
    print(f"  studentInfo: {len(si)}  vle: {len(vle)}  studentVle: {len(sv)}")

    # build activity vocab
    activity_types = sorted(vle["activity_type"].unique())
    activity_to_id = {a: i for i, a in enumerate(activity_types)}
    print(f"  activity vocab: {len(activity_to_id)}")

    # map (id_site, module, presentation) -> activity_type
    vle_map = vle.set_index(["id_site", "code_module", "code_presentation"])["activity_type"].to_dict()

    # student-level outcome (one row per (student, module, presentation))
    si["outcome"] = si["final_result"].map(OUTCOMES)
    si = si.dropna(subset=["outcome"])
    si["outcome"] = si["outcome"].astype(int)

    # per (student, module, presentation), sequence of activity_type tokens ordered by date
    # join sv with vle to get activity_type per click
    sv["activity_type"] = sv.apply(
        lambda r: vle_map.get((r["id_site"], r["code_module"], r["code_presentation"])), axis=1,
    )
    sv = sv.dropna(subset=["activity_type"])
    sv["token"] = sv["activity_type"].map(activity_to_id)

    # group by (student, module, presentation) ordered by date
    print("  building sequences...")
    sv_sorted = sv.sort_values(["id_student", "code_module", "code_presentation", "date"])
    grouped = sv_sorted.groupby(["id_student", "code_module", "code_presentation"])["token"].apply(list)

    rows = []
    si_map = si.set_index(["id_student", "code_module", "code_presentation"])["outcome"].to_dict()
    for (sid, mod, pres), tokens in grouped.items():
        if mod not in MODULES: continue
        out = si_map.get((sid, mod, pres))
        if out is None: continue
        rows.append({
            "user_id": f"{sid}_{mod}_{pres}",
            "cohort_label": mod,
            "cluster_id": int(out),
            "token_ids": tokens,
            "n_events": len(tokens),
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

    seq_out = DATA_PROC / "sequences_oulad.parquet"
    out.select(["user_id", "cohort_label", "token_ids", "n_events"]).write_parquet(seq_out)
    print(f"\nwrote {seq_out}")
    cl_out = DATA_PROC / "cluster_labels_oulad.parquet"
    out.select(["user_id", "cohort_label", "cluster_id"]).write_parquet(cl_out)
    print(f"wrote {cl_out}")

    vocab = sorted(activity_to_id.keys(), key=lambda k: activity_to_id[k])
    with open(RESULTS / "oulad_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "n": len(vocab)}, f, indent=2)
    print(f"wrote vocab ({len(vocab)} tokens)")


if __name__ == "__main__":
    main()
