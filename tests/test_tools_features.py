"""feature-engineer tools: encode/scale/drop, then the stratified split that
model-trainer depends on. Each transform tool must register its output under
`new_dataset_name` and return only a schema/shape, never the data."""

from __future__ import annotations

import pytest

from credit_agent import data_registry as reg
from credit_agent.tools_features import (
    drop_columns,
    one_hot_encode,
    scale_numeric,
    train_test_split_dataset,
)

TARGET = reg.TARGET_COLUMN


def test_one_hot_encode_registers_dummies_and_drops_original_column(registered_df):
    result = one_hot_encode.invoke(
        {"dataset_name": "raw", "columns": ["EDUCATION"], "new_dataset_name": "encoded"}
    )
    encoded = reg.get_dataset("encoded")
    assert "EDUCATION" not in encoded.columns
    assert any(c.startswith("EDUCATION_") for c in encoded.columns)
    assert result["dataset_name"] == "encoded"
    assert result["rows"] == len(registered_df)


def test_one_hot_encode_missing_column_errors(registered_df):
    result = one_hot_encode.invoke(
        {"dataset_name": "raw", "columns": ["nope"], "new_dataset_name": "encoded"}
    )
    assert "error" in result
    assert "encoded" not in reg.DATASETS  # failed call must not register anything


def test_scale_numeric_produces_zero_mean_unit_variance(registered_df):
    result = scale_numeric.invoke(
        {"dataset_name": "raw", "columns": ["LIMIT_BAL"], "new_dataset_name": "scaled"}
    )
    scaled = reg.get_dataset("scaled")
    assert scaled["LIMIT_BAL"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled["LIMIT_BAL"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)
    assert len(result["mean_"]) == 1
    assert len(result["scale_"]) == 1
    # original dataset must be untouched (tool copies before mutating)
    assert reg.get_dataset("raw")["LIMIT_BAL"].mean() != pytest.approx(0.0, abs=1e-9)


def test_scale_numeric_missing_column_errors(registered_df):
    result = scale_numeric.invoke(
        {"dataset_name": "raw", "columns": ["nope"], "new_dataset_name": "scaled"}
    )
    assert "error" in result


def test_drop_columns(registered_df):
    result = drop_columns.invoke(
        {"dataset_name": "raw", "columns": ["AGE"], "new_dataset_name": "dropped"}
    )
    dropped = reg.get_dataset("dropped")
    assert "AGE" not in dropped.columns
    assert result["columns"] == list(dropped.columns)


def test_drop_columns_missing_column_errors(registered_df):
    result = drop_columns.invoke(
        {"dataset_name": "raw", "columns": ["nope"], "new_dataset_name": "dropped"}
    )
    assert "error" in result


def test_train_test_split_is_stratified(registered_df):
    result = train_test_split_dataset.invoke(
        {
            "dataset_name": "raw",
            "target_column": TARGET,
            "test_size": 0.3,
            "new_train_name": "train",
            "new_test_name": "test",
        }
    )
    train_df, test_df = reg.get_dataset("train"), reg.get_dataset("test")

    assert result["train_rows"] == len(train_df) == 140
    assert result["test_rows"] == len(test_df) == 60

    full_rate = registered_df[TARGET].mean()
    train_rate = train_df[TARGET].mean()
    test_rate = test_df[TARGET].mean()
    # stratified split -- class balance should track the full dataset closely
    assert train_rate == pytest.approx(full_rate, abs=0.05)
    assert test_rate == pytest.approx(full_rate, abs=0.05)


def test_train_test_split_missing_target_errors(registered_df):
    result = train_test_split_dataset.invoke({"dataset_name": "raw", "target_column": "nope"})
    assert "error" in result
