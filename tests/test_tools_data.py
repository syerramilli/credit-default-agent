"""data-profiler tools: schema, missingness, stats, categorical breakdowns,
correlations, samples. Each check pins the tool's output shape and the
guardrail it's supposed to respect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_agent import data_registry as reg
from credit_agent.config import MAX_CORRELATION_COLUMNS, MAX_SAMPLE_ROWS
from credit_agent.tools_data import (
    get_categorical_breakdown,
    get_correlation_matrix,
    get_missing_report,
    get_sample_rows,
    get_schema,
    get_summary_stats,
)


def test_get_schema(registered_df):
    result = get_schema.invoke({"dataset_name": "raw"})
    assert result["rows"] == len(registered_df)
    assert set(result["columns"]) == set(registered_df.columns)


def test_get_missing_report_no_missing(registered_df):
    result = get_missing_report.invoke({"dataset_name": "raw"})
    assert result["note"] == "no missing values in any column"


def test_get_missing_report_counts_missing():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    reg.register_dataset("with_nulls", df)
    result = get_missing_report.invoke({"dataset_name": "with_nulls"})
    assert result["a"]["missing_count"] == 2
    assert result["a"]["missing_pct"] == 50.0
    assert "b" not in result  # only columns with missing values are reported


def test_get_summary_stats_scopes_to_requested_columns(registered_df):
    result = get_summary_stats.invoke({"dataset_name": "raw", "columns": ["AGE"]})
    assert set(result) == {"AGE"}


def test_get_summary_stats_no_matching_columns_errors(registered_df):
    result = get_summary_stats.invoke({"dataset_name": "raw", "columns": ["not_a_column"]})
    assert "error" in result


def test_get_categorical_breakdown_unknown_column_errors(registered_df):
    result = get_categorical_breakdown.invoke({"dataset_name": "raw", "column": "nope"})
    assert "error" in result
    assert "available_columns" in result


def test_get_categorical_breakdown_known_column(registered_df):
    result = get_categorical_breakdown.invoke({"dataset_name": "raw", "column": "EDUCATION"})
    assert sum(result.values()) == len(registered_df)


def test_get_correlation_matrix_default_scope(registered_df):
    result = get_correlation_matrix.invoke({"dataset_name": "raw"})
    assert "LIMIT_BAL" in result
    assert result["LIMIT_BAL"]["LIMIT_BAL"] == 1.0  # self-correlation


def test_get_correlation_matrix_rejects_too_many_columns(registered_df):
    wide_df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(20, MAX_CORRELATION_COLUMNS + 1))
    )
    wide_df.columns = [str(c) for c in wide_df.columns]
    reg.register_dataset("wide", wide_df)
    result = get_correlation_matrix.invoke({"dataset_name": "wide"})
    assert "error" in result


def test_get_correlation_matrix_no_matching_columns_errors(registered_df):
    result = get_correlation_matrix.invoke({"dataset_name": "raw", "columns": ["not_a_column"]})
    assert "error" in result


def test_get_sample_rows_hard_capped(registered_df):
    result = get_sample_rows.invoke({"dataset_name": "raw", "n": 10_000})
    assert result["n_returned"] == MAX_SAMPLE_ROWS
    assert len(result["records"]) == MAX_SAMPLE_ROWS
    assert set(result["records"][0]) == set(registered_df.columns)


def test_get_sample_rows_smaller_than_dataset(registered_df):
    result = get_sample_rows.invoke({"dataset_name": "raw", "n": 3})
    assert result["n_returned"] == 3
