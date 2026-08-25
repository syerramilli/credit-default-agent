"""The registry is the trick that keeps raw data out of the LLM's context:
these tests check the name-based addressing contract and the RegistryError
messages tools rely on to let the agent self-correct."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_agent import data_registry as reg


def test_register_and_get_dataset_roundtrip():
    df = pd.DataFrame({"a": [1, 2, 3]})
    reg.register_dataset("mine", df)
    assert reg.get_dataset("mine") is df


def test_get_dataset_missing_raises_with_available_names_listed():
    reg.register_dataset("known", pd.DataFrame({"a": [1]}))
    with pytest.raises(reg.RegistryError) as exc_info:
        reg.get_dataset("unknown")
    message = str(exc_info.value)
    assert "unknown" in message
    assert "known" in message  # lists what *does* exist, per the docstring


def test_get_dataset_missing_when_registry_empty_says_so():
    with pytest.raises(reg.RegistryError, match=r"\(none loaded yet\)"):
        reg.get_dataset("anything")


def test_register_and_get_model_roundtrip():
    estimator = object()  # registry doesn't care what the estimator is
    reg.register_model("m1", estimator, ["a", "b"], "target")
    model = reg.get_model("m1")
    assert model["estimator"] is estimator
    assert model["feature_columns"] == ["a", "b"]
    assert model["target"] == "target"


def test_get_model_missing_raises_with_hint():
    with pytest.raises(reg.RegistryError, match=r"train_classifier"):
        reg.get_model("nope")


def test_dataset_summary_reports_rows_and_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    summary = reg.dataset_summary(df)
    assert summary == {"rows": 2, "columns": ["a", "b"]}


def test_list_datasets_covers_every_registered_dataset():
    reg.register_dataset("d1", pd.DataFrame({"a": [1]}))
    reg.register_dataset("d2", pd.DataFrame({"b": [1, 2]}))
    listing = reg.list_datasets()
    assert set(listing) == {"d1", "d2"}
    assert listing["d2"]["rows"] == 2


def test_load_raw_dataset_reads_and_registers_local_cache(tmp_path, monkeypatch):
    """Network-free: points DATA_PATH at a small local CSV instead of hitting
    ucimlrepo, and checks load_raw_dataset reads + registers it correctly."""
    csv_path = tmp_path / "credit_default.csv"
    df = pd.DataFrame({"LIMIT_BAL": [1000, 2000, 3000], reg.TARGET_COLUMN: [0, 1, 0]})
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(reg, "DATA_PATH", csv_path)

    summary = reg.load_raw_dataset("raw")

    assert summary == {
        "dataset_name": "raw",
        "rows": 3,
        "columns": 2,
        "target_column": reg.TARGET_COLUMN,
        "source": str(csv_path),
    }
    assert list(reg.get_dataset("raw").columns) == ["LIMIT_BAL", reg.TARGET_COLUMN]
