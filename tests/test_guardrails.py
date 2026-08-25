"""guardrails.py is the single place the "no raw data to the LLM" rule is
enforced -- these tests pin down the caps directly against config.py so a
future change to either can't silently drift out of sync."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_agent.config import MAX_CATEGORIES_SHOWN, MAX_SAMPLE_ROWS, STAT_DECIMALS
from credit_agent.guardrails import (
    cap_sample,
    dataset_shape,
    describe_numeric,
    round_floats,
    top_k_value_counts,
)


def test_round_floats_rounds_nested_floats_only():
    obj = {
        "a": 1.123456789,
        "b": [1.987654321, {"c": 2.5000001}],
        "d": "unchanged",
        "e": 3,
    }
    result = round_floats(obj)
    assert result["a"] == round(1.123456789, STAT_DECIMALS)
    assert result["b"][0] == round(1.987654321, STAT_DECIMALS)
    assert result["b"][1]["c"] == round(2.5000001, STAT_DECIMALS)
    assert result["d"] == "unchanged"
    assert result["e"] == 3  # int untouched, not coerced to float


def test_cap_sample_never_exceeds_max_sample_rows():
    df = pd.DataFrame({"x": range(1000)})
    sample = cap_sample(df, n=1000)
    assert len(sample) == MAX_SAMPLE_ROWS


def test_cap_sample_returns_everything_if_df_smaller_than_n():
    df = pd.DataFrame({"x": range(3)})
    sample = cap_sample(df, n=5)
    assert len(sample) == 3


def test_top_k_value_counts_collapses_beyond_cap():
    # one more distinct category than the cap, each with a distinct count so
    # collapsing is unambiguous
    values = []
    for i in range(MAX_CATEGORIES_SHOWN + 1):
        values += [f"cat_{i}"] * (i + 1)
    series = pd.Series(values)

    result = top_k_value_counts(series)

    assert len(result) == MAX_CATEGORIES_SHOWN + 1  # top-k plus __other__
    assert "__other__" in result
    assert result["__other__"] == 1  # the smallest category (count=1) got bumped


def test_top_k_value_counts_no_other_bucket_when_under_cap():
    series = pd.Series(["a", "a", "b"])
    result = top_k_value_counts(series)
    assert "__other__" not in result
    assert result == {"a": 2, "b": 1}


def test_describe_numeric_has_expected_stat_keys():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10, 20, 30, 40]})
    result = describe_numeric(df, ["x", "y"])
    assert set(result.keys()) == {"x", "y"}
    assert set(result["x"].keys()) >= {"count", "mean", "std", "min", "50%", "max"}


def test_dataset_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert dataset_shape(df) == {"rows": 3, "columns": 2}


@pytest.mark.parametrize("n", [-5, 0])
def test_cap_sample_handles_nonpositive_n(n):
    df = pd.DataFrame({"x": range(10)})
    sample = cap_sample(df, n=n)
    assert len(sample) == 1  # clamped up to at least 1 row, never 0 or negative
