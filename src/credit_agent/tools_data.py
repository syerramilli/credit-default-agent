"""Tools for the data-profiler sub-agent.

Every tool here does its work on the real DataFrame in Python, then returns
only schema/dtype/aggregate-statistic/small-sample JSON -- never the raw
table. See guardrails.py for the caps that get enforced.
"""

from __future__ import annotations

from langchain_core.tools import tool

from credit_agent import data_registry as reg
from credit_agent.config import MAX_CORRELATION_COLUMNS
from credit_agent.guardrails import (
    cap_sample,
    describe_numeric,
    round_floats,
    top_k_value_counts,
)


@tool
def load_dataset(name: str = "raw") -> dict:
    """Load the UCI Default of Credit Card Clients dataset (downloading and
    caching it on first run) and register it under the given name so other
    tools can reference it. Returns row/column counts and the target column
    name -- never the data itself."""
    return reg.load_raw_dataset(name)


@tool
def get_schema(dataset_name: str) -> dict:
    """Get the column names and dtypes for a registered dataset, plus its
    shape. This is the right first call before requesting any statistics."""
    df = reg.get_dataset(dataset_name)
    return {
        "rows": int(df.shape[0]),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


@tool
def get_missing_report(dataset_name: str) -> dict:
    """Get per-column missing-value counts and percentages for a registered
    dataset. Aggregate only."""
    df = reg.get_dataset(dataset_name)
    missing = df.isna().sum()
    pct = (missing / len(df) * 100) if len(df) else missing
    return round_floats(
        {
            col: {"missing_count": int(missing[col]), "missing_pct": float(pct[col])}
            for col in df.columns
            if missing[col] > 0
        }
        or {"missing_count": 0, "note": "no missing values in any column"}
    )


@tool
def get_summary_stats(dataset_name: str, columns: list[str] | None = None) -> dict:
    """Get aggregate summary statistics (count/mean/std/min/25%/50%/75%/max)
    for numeric columns in a registered dataset. Pass `columns` to limit the
    scope; omit it to summarize every numeric column. Returns statistics
    only, never row-level values."""
    df = reg.get_dataset(dataset_name)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target_cols = [c for c in (columns or numeric_cols) if c in numeric_cols]
    if not target_cols:
        return {"error": "No matching numeric columns found.", "numeric_columns": numeric_cols}
    return describe_numeric(df, target_cols)


@tool
def get_categorical_breakdown(dataset_name: str, column: str) -> dict:
    """Get a value-count breakdown for a single categorical/discrete column,
    collapsed to the top categories (see guardrails) plus an '__other__'
    bucket. Use this instead of sampling to understand a column's
    distribution."""
    df = reg.get_dataset(dataset_name)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found.", "available_columns": list(df.columns)}
    return top_k_value_counts(df[column])


@tool
def get_correlation_matrix(dataset_name: str, columns: list[str] | None = None) -> dict:
    """Get a Pearson correlation matrix over numeric columns of a registered
    dataset. Pass `columns` to scope it; without scoping, datasets with more
    than a small number of numeric columns will be rejected -- ask for a
    subset instead of the whole matrix."""
    df = reg.get_dataset(dataset_name)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cols = columns or numeric_cols
    cols = [c for c in cols if c in numeric_cols]
    if not cols:
        return {"error": "No matching numeric columns found.", "numeric_columns": numeric_cols}
    if len(cols) > MAX_CORRELATION_COLUMNS:
        return {
            "error": (
                f"{len(cols)} columns requested, cap is {MAX_CORRELATION_COLUMNS}. "
                "Pass a smaller `columns` list."
            )
        }
    corr = df[cols].corr(method="pearson")
    return round_floats(corr.to_dict())


@tool
def get_sample_rows(dataset_name: str, n: int = 5) -> dict:
    """Get a small random sample of rows (hard-capped, see guardrails) from a
    registered dataset, as a last resort when schema/stats alone aren't
    enough to understand the data shape. Never request this for anything
    other than a quick sanity check."""
    df = reg.get_dataset(dataset_name)
    sample = cap_sample(df, n)
    return {"n_returned": len(sample), "records": round_floats(sample.to_dict(orient="records"))}
