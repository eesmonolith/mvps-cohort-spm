"""
Load raw Kor 3-semester CSVs with python-engine quoted parsing
(multi-line `code` columns break naive CSV).

Each loader returns a polars DataFrame with normalized column names.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import polars as pl

from src.config import COHORTS


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse heterogeneous timestamp strings (with or without tz) to
    tz-naive UTC pandas Timestamps. Handles ISO + space-separated formats."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None)


# Known column-name fixups (raw data has occasional typos like "timestam")
COLUMN_RENAMES = {
    "timestam": "timestamp",
    "problems_answers ": "problems_answers",
}


def _read_quoted_csv(path: Path, usecols=None,
                     ts_cols: tuple[str, ...] = ("timestamp",)) -> pl.DataFrame:
    """python-engine + quoting=1 needed for multi-line code columns.

    Reads all columns first, renames known typos, normalizes any timestamp
    columns to tz-naive UTC, then subsets to ``usecols`` if requested.
    Handles raw header inconsistencies across the 3 Kor semesters.
    """
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        quoting=1,
    )
    df = df.rename(columns=COLUMN_RENAMES)
    for col in ts_cols:
        if col in df.columns:
            df[col] = _to_naive_datetime(df[col])
    if usecols is not None:
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise ValueError(
                f"After rename, columns still missing from {path.name}: {missing}. "
                f"Available: {list(df.columns)}"
            )
        df = df[list(usecols)]
    return pl.from_pandas(df)


def load_submissions(cohort_key: str) -> pl.DataFrame:
    cfg = COHORTS[cohort_key]
    fname = f"submissions{cfg['file_suffix']}.csv"
    cols = ["problem_id", "user_id", "status", "timestamp"]
    df = _read_quoted_csv(cfg["raw_dir"] / fname, usecols=cols)
    return df.with_columns(pl.col("status").cast(pl.Int32, strict=False))


def load_executions(cohort_key: str) -> pl.DataFrame:
    cfg = COHORTS[cohort_key]
    fname = f"executions{cfg['file_suffix']}.csv"
    cols = ["execution_id", "user_id", "problem_id", "error_name", "timestamp"]
    df = _read_quoted_csv(cfg["raw_dir"] / fname, usecols=cols)
    return df.with_columns(pl.col("execution_id").cast(pl.Int64, strict=False))


def load_error_help(cohort_key: str) -> pl.DataFrame:
    cfg = COHORTS[cohort_key]
    fname = f"error_help{cfg['file_suffix']}.csv"
    cols = ["user_id", "execution_id", "timestamp"]
    df = _read_quoted_csv(cfg["raw_dir"] / fname, usecols=cols)
    return df.with_columns(pl.col("execution_id").cast(pl.Int64, strict=False))


def load_problem_views(cohort_key: str) -> pl.DataFrame:
    cfg = COHORTS[cohort_key]
    fname = f"problem_views{cfg['file_suffix']}.csv"
    cols = ["problem_id", "user_id", "timestamp"]
    return _read_quoted_csv(cfg["raw_dir"] / fname, usecols=cols)


def load_classroom_students(cohort_key: str) -> pl.DataFrame:
    cfg = COHORTS[cohort_key]
    fname = f"classroom_students{cfg['file_suffix']}.csv"
    return _read_quoted_csv(cfg["raw_dir"] / fname)


def load_problems(cohort_key: str) -> pl.DataFrame:
    cfg = COHORTS[cohort_key]
    fname = f"problems{cfg['file_suffix']}.csv"
    cols = ["classroom_id", "problem_id", "title"]
    return _read_quoted_csv(cfg["raw_dir"] / fname, usecols=cols)
