"""Shared helpers that keep tool output small, aggregate, and
JSON-serializable. Every tool in tools_*.py should funnel its return value
through these before handing it back to the agent -- this is the single
place the "no raw data to the LLM" rule is enforced.
"""

from __future__ import annotations

import pandas as pd

from credit_agent.config import (
    MAX_CATEGORIES_SHOWN,
    MAX_SAMPLE_ROWS,
    STAT_DECIMALS,
)


def round_floats(obj):
    """Recursively round floats in nested dict/list structures so numbers
    stay compact and diff-friendly instead of dumping 17 significant digits
    into the model's context."""
    if isinstance(obj, float):
        return round(obj, STAT_DECIMALS)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def cap_sample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Return at most `n` rows (hard-capped at MAX_SAMPLE_ROWS), sampled
    rather than head-sliced so the model doesn't over-index on row order."""
    n = max(1, min(n, MAX_SAMPLE_ROWS))
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def top_k_value_counts(series: pd.Series, k: int = MAX_CATEGORIES_SHOWN) -> dict:
    """Value counts collapsed to the top-k categories plus an 'other' bucket,
    so a high-cardinality column can't flood the context window."""
    counts = series.value_counts(dropna=False)
    top = counts.head(k)
    result = {str(idx): int(cnt) for idx, cnt in top.items()}
    remainder = counts.iloc[k:].sum()
    if remainder > 0:
        result["__other__"] = int(remainder)
    return result


def describe_numeric(df: pd.DataFrame, columns: list[str]) -> dict:
    """Aggregate numeric summary stats only -- min/max/mean/std/quantiles.
    Never touches individual cell values."""
    desc = df[columns].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    return round_floats(desc)


def dataset_shape(df: pd.DataFrame) -> dict:
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
