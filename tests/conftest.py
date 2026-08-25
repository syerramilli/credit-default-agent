"""Shared fixtures.

Everything here is network-free and fast: no test touches the UCI repo or
the Anthropic API. Tools address data/models by name in the process-global
registries in `data_registry.py`, so the `_clean_registry` fixture resets
those between tests -- without it, a dataset registered in one test would
leak into the next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_agent import data_registry as reg

TARGET = reg.TARGET_COLUMN


@pytest.fixture(autouse=True)
def _clean_registry():
    reg.DATASETS.clear()
    reg.MODELS.clear()
    yield
    reg.DATASETS.clear()
    reg.MODELS.clear()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, deterministic dataset shaped like the real one: numeric and
    categorical feature columns plus a binary target that's genuinely
    predictable from them, so trained models score meaningfully above chance
    in tests that check for that."""
    rng = np.random.default_rng(42)
    n = 200
    limit_bal = rng.uniform(10_000, 500_000, n)
    pay_0 = rng.integers(-2, 8, n)
    age = rng.integers(21, 70, n)
    education = rng.integers(1, 5, n)
    logit = -0.00002 * limit_bal + 0.9 * pay_0 + rng.normal(0, 1, n)
    target = (logit > np.median(logit)).astype(int)
    return pd.DataFrame(
        {
            "LIMIT_BAL": limit_bal,
            "PAY_0": pay_0,
            "AGE": age,
            "EDUCATION": education,
            TARGET: target,
        }
    )


@pytest.fixture
def registered_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """`sample_df`, already registered under the name 'raw'."""
    reg.register_dataset("raw", sample_df)
    return sample_df


@pytest.fixture
def split_names(registered_df: pd.DataFrame) -> tuple[str, str]:
    """Registers a stratified train/test split of `registered_df` under
    'train'/'test' via the real tool (not a hand-rolled split), so tests
    exercise the same code path the agent does."""
    from credit_agent.tools_features import train_test_split_dataset

    train_test_split_dataset.invoke(
        {
            "dataset_name": "raw",
            "target_column": TARGET,
            "test_size": 0.3,
            "new_train_name": "train",
            "new_test_name": "test",
        }
    )
    return "train", "test"
